// Main entry point: imports, event listeners, and initialization
import {
    ROWS, COLS, PLAYER_A, PLAYER_B, NORMAL, SWAPPED, NUM_DIRECTIONS, JS_DIRECTIONS, initialPosition
} from './game-constants.js';
import {
    parseStartingPosition, cloneBoard, serializeBoardState, boardToKey, convertBoardToBinaryJS
} from './game-board.js';
import {
    renderBoard, getPieceImage
} from './game-render.js';
import {
    selectPiece, deselectPiece, calculateLegalMoves, makeMove, unmarkSwapped, switchPlayer, checkWinCondition
} from './game-logic.js';
import {
    boardToNNInput, neuralNetworkPredict
} from './game-ai.js';
import {
    initializeMCTS
} from './game-mcts-wrapper.js';
import {
    showOverlay, hideOverlay, hideAllOverlays
} from './game-overlays.js';
import {
    findBestAIMove, moveToActionIndex, calculateLegalMovesForState, 
    allowsOpponentWin, evaluateBoardState, countFriendlyNeighbors, checkWinConditionForState
} from './game-ai-advanced.js';
import {
    runBoardPositionTest
} from './test-board-position.js';

// --- Main Orchestration ---
// This code is adapted from script.js and is compatible with game_logic_adapter.js and mcts_js.js

// Global state (mirrors script.js)
let board = [];
let currentPlayer = PLAYER_A;
let selectedPiece = null;
let legalMoves = [];
let moveHistory = [];
let gameOver = false;
let winner = null;
let winPath = [];
let playerAScore = 0;
let playerBScore = 0;
let currentHistoryIndex = undefined;
let startingPosition = null;
let startingPositionIndex = 0;
let isInSelfPlay = false;
let selfPlayTimeoutId = null;
let tfModel = null;
let AI_DIFFICULTY = 'easy';
let AI_DEPTH = 1;
let ANALYSIS_MODE = false;
let MCTS_ENABLED = false;
let MCTS_SIMULATIONS = 50;
let MCTS_TEMPERATURE = 0.1; // Better default for gameplay (was 0.01)
let MCTS_PUCT_CONSTANT = 1.0;
let MCTS_DIRICHLET_ALPHA = 0.3;
let MCTS_DIRICHLET_EPSILON = 0.25;
let MCTS_VERBOSE = false; // Can be enabled via browser console: window.MCTS_VERBOSE = true
let mctsSearch = null;
let gameLogic = null;
let AI_PLAYS_FIRST = false;

window.analysisMode = ANALYSIS_MODE; // For compatibility with game_logic_adapter.js


// DOM references
const boardElement = document.getElementById('game-board');
const resetBtn = document.getElementById('reset-btn');
const infoBtn = document.getElementById('info-btn');
const historyBtn = document.getElementById('history-btn');
const startBtn = document.getElementById('start-btn');
const infoOverlay = document.getElementById('info-overlay');
const historyOverlay = document.getElementById('history-overlay');
const winOverlay = document.getElementById('win-overlay');
const historyList = document.getElementById('history-list');
const winMessage = document.getElementById('win-message');
const overlayCloseButtons = document.querySelectorAll('.close-overlay-btn');
const overlays = document.querySelectorAll('.overlay');
const aiSpinnerOverlay = document.getElementById('ai-spinner-overlay');
const mctsBtn = document.getElementById('mcts-btn');
const mctsOverlay = document.getElementById('mcts-overlay');

// --- TensorFlow.js Backend Selection ---
(async function setupTFBackend() {
    if (typeof tf !== 'undefined') {
        const backends = Object.keys(tf.engine().registryFactory);
        console.log("Available TF.js backends:", backends);
        if (backends.includes('webgl')) {
            try { await tf.setBackend('webgl'); console.log("TensorFlow.js using WebGL backend for best performance"); } catch (webglError) { console.warn("WebGL backend failed, falling back to alternatives:", webglError); }
        } else if (backends.includes('wasm')) {
            try { await tf.setBackend('wasm'); console.log("TensorFlow.js using WASM backend for improved CPU performance"); } catch (wasmError) { console.warn("WASM backend failed, falling back to CPU:", wasmError); await tf.setBackend('cpu'); console.log("TensorFlow.js using CPU backend"); }
        } else {
            await tf.setBackend('cpu');
            console.log("TensorFlow.js using CPU backend");
        }
        await tf.ready();
        console.log("Active backend:", tf.getBackend());
    } else {
        console.warn("tf object not available at the time of setting backend.");
    }
})();

// --- Wait for TensorFlow.js to be Available ---
async function waitForTensorFlow(maxWaitTime = 30000) {
    console.log("Waiting for TensorFlow.js to become available...");
    const startTime = Date.now();
    let attempts = 0;
    
    while (typeof tf === 'undefined') {
        attempts++;
        const elapsed = Date.now() - startTime;
        
        if (elapsed > maxWaitTime) {
            console.error(`TensorFlow.js failed to load within ${maxWaitTime}ms timeout after ${attempts} attempts`);
            throw new Error("TensorFlow.js failed to load within timeout");
        }
        
        if (attempts % 10 === 0) {
            console.log(`Still waiting for TensorFlow.js... Attempt ${attempts}, elapsed: ${elapsed}ms`);
        }
        
        await new Promise(resolve => setTimeout(resolve, 200)); // Wait 200ms between checks
    }
    
    console.log(`TensorFlow.js is now available after ${attempts} attempts (${Date.now() - startTime}ms)`);
    return true;
}

// --- Model Loading ---
async function loadTFJSModel() {
    try {
        console.log("=== Model Loading Debug Start ===");
        console.log("Loading TensorFlow.js model...");
        console.log("TensorFlow.js available:", typeof tf !== 'undefined');
        console.log("TensorFlow.js version:", typeof tf !== 'undefined' ? tf.version?.tfjs : 'N/A');
        
        if (typeof tf === 'undefined') {
            throw new Error("TensorFlow.js not available yet");
        }
        
        // Wait for TensorFlow.js to be ready
        console.log("Waiting for TensorFlow.js backend to be ready...");
        await tf.ready();
        console.log("TensorFlow.js backend ready, active backend:", tf.getBackend());
        
        console.log("Loading model from: ./switcharoo_tfjs_model/model.json");
        console.log("Starting tf.loadGraphModel call...");
        
        // Add detailed error handling for model loading
        let loadedModel;
        try {
            loadedModel = await tf.loadGraphModel('./switcharoo_tfjs_model/model.json');
            console.log("tf.loadGraphModel completed successfully");
        } catch (modelLoadError) {
            console.error("Error during tf.loadGraphModel:", modelLoadError);
            console.error("Model load error type:", modelLoadError.constructor.name);
            console.error("Model load error message:", modelLoadError.message);
            console.error("Model load error stack:", modelLoadError.stack);
            throw modelLoadError;
        }
        
        tfModel = loadedModel;
        console.log("Model loaded successfully:", tfModel);
        console.log("Model constructor:", tfModel.constructor.name);
        console.log("Model inputs:", tfModel.inputs?.length || 'undefined');
        console.log("Model outputs:", tfModel.outputs?.length || 'undefined');
        
        // Test the model with a dummy prediction
        try {
            console.log("Testing model with dummy input...");
            const dummyInput = tf.zeros([1, 192]);
            const inputNodeName = tfModel.inputs[0].name;
            console.log("Input node name:", inputNodeName);
            
            const prediction = tfModel.execute({[inputNodeName]: dummyInput});
            console.log("Dummy prediction successful, output tensors:", prediction.length);
            
            // Clean up test tensors
            dummyInput.dispose();
            if (Array.isArray(prediction)) {
                prediction.forEach(tensor => tensor.dispose());
            } else {
                prediction.dispose();
            }
            console.log("Model test completed successfully");
        } catch (testError) {
            console.warn("Model test failed:", testError);
            console.warn("Proceeding anyway as model loaded successfully");
        }
        
        // Make tfModel available globally for testing
        window.tfModel = tfModel;
        console.log("Model assigned to window.tfModel:", window.tfModel);
        console.log("=== Model Loading Debug End ===");
        
        return true;
    } catch (error) {
        console.error("=== Model Loading Failed ===");
        console.error("Could not load pre-trained graph model. Error object:", error);
        console.error("Error type:", error.constructor.name);
        console.error("Error message:", error.message);
        console.error("Error stack:", error.stack);
        console.warn("The Neural Network AI option will be disabled as it requires pre-trained weights to work properly.");
        tfModel = null;
        window.tfModel = null;
        console.log("Model loading failed, window.tfModel set to null");
        console.log("=== Model Loading Debug End ===");
        return false;
    }
}

// --- MCTS Initialization ---
function setupMCTS() {
    if (typeof MCTSSearch !== 'undefined' && typeof SwitcharooGameLogic !== 'undefined') {
        gameLogic = new SwitcharooGameLogic({
            ROWS, COLS, PLAYER_A, PLAYER_B, NORMAL, SWAPPED, NUM_DIRECTIONS, JS_DIRECTIONS
        });
        mctsSearch = new MCTSSearch({
            numSimulations: MCTS_SIMULATIONS,
            cPuct: MCTS_PUCT_CONSTANT,
            temperature: MCTS_TEMPERATURE,
            dirichletAlpha: MCTS_DIRICHLET_ALPHA,
            dirichletEpsilon: MCTS_DIRICHLET_EPSILON,
            enabled: MCTS_ENABLED,
            verbose: MCTS_VERBOSE,
            logSearchStats: true
        });
        
        // Make MCTS objects available globally for testing
        window.mctsSearch = mctsSearch;
        window.gameLogic = gameLogic;
        
        console.log("MCTS initialized with settings:", {
            numSimulations: MCTS_SIMULATIONS,
            cPuct: MCTS_PUCT_CONSTANT,
            temperature: MCTS_TEMPERATURE,
            dirichletAlpha: MCTS_DIRICHLET_ALPHA,
            dirichletEpsilon: MCTS_DIRICHLET_EPSILON,
            enabled: MCTS_ENABLED,
            verbose: MCTS_VERBOSE
        });
    } else {
        console.warn("MCTS classes not available - MCTS features disabled");
        window.mctsSearch = null;
        window.gameLogic = null;
    }
}
document.addEventListener('DOMContentLoaded', async () => {
    console.log("=== DOM Content Loaded ===");
    console.log("TensorFlow.js available at DOMContentLoaded:", typeof tf !== 'undefined');
    
    // Initialize MCTS first
    setupMCTS();
    
    // Add a small delay to allow deferred scripts to finish loading
    await new Promise(resolve => setTimeout(resolve, 500));
    console.log("After 500ms delay, TensorFlow.js available:", typeof tf !== 'undefined');
    
    // Wait for TensorFlow.js to be available, then load the neural network model
    try {
        console.log("Starting TensorFlow.js wait and model loading...");
        await waitForTensorFlow();
        console.log("TensorFlow.js ready, starting model load...");
        const modelLoaded = await loadTFJSModel();
        console.log("Model loading result:", modelLoaded);
        console.log("Final window.tfModel state:", window.tfModel);
    } catch (error) {
        console.warn("Failed to wait for TensorFlow.js or load model:", error);
        console.warn("Error details:", error.message, error.stack);
        window.tfModel = null;
    }
    
    // Set up AI plays first checkbox
    const aiPlaysFirstCheckbox = document.getElementById('ai-plays-first');
    if (aiPlaysFirstCheckbox) {
        aiPlaysFirstCheckbox.checked = AI_PLAYS_FIRST;
        aiPlaysFirstCheckbox.addEventListener('change', () => {
            AI_PLAYS_FIRST = aiPlaysFirstCheckbox.checked;
            console.log('AI plays first:', AI_PLAYS_FIRST);
        });
    }
    
    // Initialize the game
    console.log("Initializing game...");
    initGame();
    console.log("Game initialization complete");
});

// --- Game Initialization ---
function initGame() {
    if (isInSelfPlay) stopSelfPlay();
    console.log(`initGame called with startingPositionIndex: ${startingPositionIndex}`);
    startingPosition = parseStartingPosition(initialPosition[startingPositionIndex]);
    board = startingPosition.map(row => row.map(cell => cell ? { ...cell } : null));
    currentPlayer = PLAYER_A;
    selectedPiece = null;
    legalMoves = [];
    moveHistory = [];
    gameOver = false;
    winner = null;
    winPath = [];
    historyBtn.disabled = true;
    renderBoard({
        board,
        selectedPiece,
        legalMoves,
        gameOver,
        winner,
        winPath,
        moveHistory,
        currentHistoryIndex,
        playerAScore,
        playerBScore,
        boardElement,
        currentPlayer
    });
    hideAllOverlays();

    // If AI plays first is enabled, and no moves have been made, trigger AI move as Player B
    if (AI_PLAYS_FIRST && moveHistory.length === 0) {
        currentPlayer = PLAYER_B;
        setTimeout(async () => {
            await triggerAIMove();
            // After AI move, set currentPlayer to Player A for human
            currentPlayer = PLAYER_A;
            renderBoard({
                board,
                selectedPiece,
                legalMoves,
                gameOver,
                winner,
                winPath,
                moveHistory,
                currentHistoryIndex,
                playerAScore,
                playerBScore,
                boardElement,
                currentPlayer
            });
        }, 300); // Small delay for UI
    }

    console.log("Game Initialized. Player A's turn (Bottom). AI plays first?", AI_PLAYS_FIRST);
}

// --- Score Display Update ---
function updateScoreDisplay() {
    // Update control scores
    const controlScoreA = document.getElementById('control-score-a');
    const controlScoreB = document.getElementById('control-score-b');
    if (controlScoreA) controlScoreA.textContent = `${playerAScore}`;
    if (controlScoreB) controlScoreB.textContent = `${playerBScore}`;
    // Update game-over scores if present
    const scoreA = document.getElementById('score-a');
    const scoreB = document.getElementById('score-b');
    if (scoreA && scoreB) {
        scoreA.textContent = `Player A: ${playerAScore}`;
        scoreB.textContent = `Player B: ${playerBScore}`;
    }
}

// --- Move History Overlay ---
function displayMoveHistory() {
    historyList.innerHTML = '';
    if (moveHistory.length === 0) {
        historyList.textContent = 'No moves made yet.';
        return;
    }
    // Display in descending order (most recent first)
    for (let idx = moveHistory.length - 1; idx >= 0; idx--) {
        const move = moveHistory[idx];
        const moveNumber = idx + 1;
        const startRowDisplay = ROWS - move.start.row;
        const endRowDisplay = ROWS - move.end.row;
        const startColDisplay = String.fromCharCode('A'.charCodeAt(0) + move.start.col);
        const endColDisplay = String.fromCharCode('A'.charCodeAt(0) + move.end.col);
        const moveDiv = document.createElement('div');
        moveDiv.textContent = `${moveNumber}. Player ${move.player}: ${startColDisplay}${startRowDisplay} → ${endColDisplay}${endRowDisplay}`;
        moveDiv.classList.add('history-move');
        if (currentHistoryIndex === moveNumber) {
            moveDiv.classList.add('selected-move');
        }
        moveDiv.addEventListener('click', () => {
            currentHistoryIndex = moveNumber;
            if (move.boardAfter) {
                renderBoardWithWinPath({
                    board: move.boardAfter,
                    selectedPiece: null,
                    legalMoves: [],
                    gameOver,
                    winner,
                    winPath,
                    moveHistory,
                    currentHistoryIndex,
                    playerAScore,
                    playerBScore,
                    boardElement,
                    currentPlayer
                });
            } else {
                console.warn('No boardAfter for move', moveNumber);
            }
            updateMoveCounter();
            hideOverlay(historyOverlay);
        });
        historyList.appendChild(moveDiv);
    }
    // Add option to view initial state
    const initialStateDiv = document.createElement('div');
    initialStateDiv.textContent = `0. Initial State`;
    initialStateDiv.classList.add('history-move');
    if (currentHistoryIndex === 0) {
        initialStateDiv.classList.add('selected-move');
    }
    initialStateDiv.addEventListener('click', () => {
        const initialBoard = startingPosition.map(row => row.map(cell => cell ? { ...cell } : null));
        currentHistoryIndex = 0;
        renderBoardWithWinPath({
            board: initialBoard,
            selectedPiece: null,
            legalMoves: [],
            gameOver,
            winner,
            winPath,
            moveHistory,
            currentHistoryIndex,
            playerAScore,
            playerBScore,
            boardElement,
            currentPlayer
        });
        updateMoveCounter();
        hideOverlay(historyOverlay);
    });
    historyList.insertBefore(initialStateDiv, historyList.firstChild);
}

// --- Board Navigation After Win ---
function navigateHistory(direction) {
    if (!gameOver) return;
    if (currentHistoryIndex === undefined) {
        currentHistoryIndex = moveHistory.length;
    }
    const newIndex = Math.max(0, Math.min(moveHistory.length, currentHistoryIndex + direction));
    if (newIndex === currentHistoryIndex) return;
    currentHistoryIndex = newIndex;
    if (currentHistoryIndex === moveHistory.length) {
        renderBoardWithWinPath({
            board,
            selectedPiece,
            legalMoves,
            gameOver,
            winner,
            winPath,
            moveHistory,
            currentHistoryIndex,
            playerAScore,
            playerBScore,
            boardElement,
            currentPlayer
        });
    } else if (currentHistoryIndex === 0) {
        const initialBoard = startingPosition.map(row => row.map(cell => cell ? { ...cell } : null));
        renderBoardWithWinPath({
            board: initialBoard,
            selectedPiece: null,
            legalMoves: [],
            gameOver,
            winner,
            winPath,
            moveHistory,
            currentHistoryIndex,
            playerAScore,
            playerBScore,
            boardElement,
            currentPlayer
        });
    } else {
        renderBoardWithWinPath({
            board: moveHistory[currentHistoryIndex - 1].boardAfter,
            selectedPiece: null,
            legalMoves: [],
            gameOver,
            winner,
            winPath,
            moveHistory,
            currentHistoryIndex,
            playerAScore,
            playerBScore,
            boardElement,
            currentPlayer
        });
    }
    updateMoveCounter();
}

function updateMoveCounter() {
    if (!gameOver) return;
    let counter = document.querySelector('.move-counter');
    if (!counter) {
        counter = document.createElement('div');
        counter.classList.add('move-counter');
        boardElement.appendChild(counter);
    }
    counter.textContent = currentHistoryIndex === undefined || currentHistoryIndex === moveHistory.length ?
        'Final Position' :
        `Move ${currentHistoryIndex} of ${moveHistory.length}`;
}

function restoreFinalState() {
    currentHistoryIndex = undefined;
    renderBoard(board);
}

// --- Patch: Record move history for navigation and overlay ---
function recordMoveHistory(player, start, end, boardBefore, boardAfter) {
    moveHistory.push({
        player,
        start: { ...start },
        end: { ...end },
        boardBefore: cloneBoard(boardBefore),
        boardAfter: cloneBoard(boardAfter)
    });
}

// --- Event Listeners ---
resetBtn.addEventListener('click', () => {
    startingPositionIndex = 0;
    initGame();
});
infoBtn.addEventListener('click', () => showOverlay(infoOverlay));
historyBtn.addEventListener('click', () => {
    if (!historyBtn.disabled) {
        displayMoveHistory();
        showOverlay(historyOverlay);
    }
});
startBtn.addEventListener('click', () => {
    startingPositionIndex = (startingPositionIndex + 1) % initialPosition.length;
    initGame();
});

overlayCloseButtons.forEach(button => {
    button.addEventListener('click', async () => {
        const overlayId = button.getAttribute('data-overlay');
        const overlayElem = document.getElementById(overlayId);
        hideOverlay(overlayElem);
        // If closing the MCTS overlay, AI plays first is checked, and no moves have been made, trigger AI move for Player B
        if (
            overlayId === 'mcts-overlay' &&
            typeof AI_PLAYS_FIRST !== 'undefined' && AI_PLAYS_FIRST &&
            Array.isArray(moveHistory) && moveHistory.length === 0 &&
            typeof currentPlayer !== 'undefined' && currentPlayer === PLAYER_A
        ) {
            // Set currentPlayer to Player B, let AI move, then set back to Player A
            currentPlayer = PLAYER_B;
            await triggerAIMove();
            currentPlayer = PLAYER_A;
            renderBoard({
                board,
                selectedPiece,
                legalMoves,
                gameOver,
                winner,
                winPath,
                moveHistory,
                currentHistoryIndex,
                playerAScore,
                playerBScore,
                boardElement,
                currentPlayer
            });
        }
    });
});
overlays.forEach(overlay => {
    overlay.addEventListener('click', (event) => {
        if (event.target === overlay) hideOverlay(overlay);
    });
});
if (mctsBtn && mctsOverlay) {
    mctsBtn.addEventListener('click', () => {
        showOverlay(mctsOverlay);
        // Optionally update MCTS UI here if needed
    });
}

// --- AI Difficulty Select Handler ---
const aiDifficultySelect = document.getElementById('ai-difficulty-select');
function updateMCTSControlsVisibility() {
    const mctsOnlyElements = document.querySelectorAll('.mcts-only');
    if (aiDifficultySelect && aiDifficultySelect.value === 'hard_ai') {
        mctsOnlyElements.forEach(el => el.style.display = '');
        // Enable MCTS by default when switching to hard_ai
        MCTS_ENABLED = true;
    } else {
        mctsOnlyElements.forEach(el => el.style.display = 'none');
        // Disable MCTS when not in AI Agent mode
        MCTS_ENABLED = false;
    }
    // Update UI to reflect the MCTS_ENABLED state
    updateMCTSUI();
}
if (aiDifficultySelect) {
    aiDifficultySelect.addEventListener('change', () => {
        // Update AI_DIFFICULTY variable
        AI_DIFFICULTY = aiDifficultySelect.value;
        console.log(`AI Difficulty changed to: ${AI_DIFFICULTY}`);
        
        // Update MCTS controls visibility
        updateMCTSControlsVisibility();
    });
    // Run once on load
    updateMCTSControlsVisibility();
}

// --- MCTS Slider Event Listeners ---
function setupMCTSSliders() {
    const mctsSimulationsSlider = document.getElementById('mcts-simulations');
    const mctsSimulationsValue = document.getElementById('mcts-simulations-value');
    const mctsTemperatureSlider = document.getElementById('mcts-temperature');
    const mctsTemperatureValue = document.getElementById('mcts-temperature-value');
    const mctsEnabledCheckbox = document.getElementById('mcts-enabled');
    const mctsResetBtn = document.getElementById('mcts-reset-defaults');

    // Simulations slider
    if (mctsSimulationsSlider && mctsSimulationsValue) {
        mctsSimulationsSlider.addEventListener('input', (e) => {
            const value = parseInt(e.target.value);
            MCTS_SIMULATIONS = Math.max(1, Math.min(200, value));
            mctsSimulationsValue.textContent = MCTS_SIMULATIONS;
            // Update MCTS instance if it exists
            if (mctsSearch) {
                mctsSearch.setNumSimulations(MCTS_SIMULATIONS);
            }
        });
    }

    // Temperature slider
    if (mctsTemperatureSlider && mctsTemperatureValue) {
        mctsTemperatureSlider.addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            MCTS_TEMPERATURE = Math.max(0.0, Math.min(2.0, value));
            mctsTemperatureValue.textContent = MCTS_TEMPERATURE.toFixed(2);
            // Update MCTS instance if it exists
            if (mctsSearch) {
                mctsSearch.setTemperature(MCTS_TEMPERATURE);
            }
        });
    }

    // MCTS enabled checkbox
    if (mctsEnabledCheckbox) {
        mctsEnabledCheckbox.addEventListener('change', (e) => {
            if (aiDifficultySelect && aiDifficultySelect.value === 'hard_ai') {
                MCTS_ENABLED = e.target.checked;
                if (mctsSearch) {
                    mctsSearch.setEnabled(MCTS_ENABLED);
                }
            } else {
                // Force disable if not in AI Agent mode
                e.target.checked = false;
                MCTS_ENABLED = false;
            }
        });
    }

    // Reset to defaults button
    if (mctsResetBtn) {
        mctsResetBtn.addEventListener('click', () => {
            MCTS_ENABLED = true;
            MCTS_SIMULATIONS = 50;
            MCTS_TEMPERATURE = 0.1; // Better default for gameplay
            MCTS_VERBOSE = false;
            
            // Update UI elements
            updateMCTSUI();
            
            // Reinitialize MCTS with new settings
            setupMCTS();
        });
    }
}

// --- Update MCTS UI Elements ---
function updateMCTSUI() {
    const mctsEnabledCheckbox = document.getElementById('mcts-enabled');
    const mctsSimulationsSlider = document.getElementById('mcts-simulations');
    const mctsSimulationsValue = document.getElementById('mcts-simulations-value');
    const mctsTemperatureSlider = document.getElementById('mcts-temperature');
    const mctsTemperatureValue = document.getElementById('mcts-temperature-value');

    if (mctsEnabledCheckbox) mctsEnabledCheckbox.checked = MCTS_ENABLED;
    if (mctsSimulationsSlider) mctsSimulationsSlider.value = MCTS_SIMULATIONS;
    if (mctsSimulationsValue) mctsSimulationsValue.textContent = MCTS_SIMULATIONS;
    if (mctsTemperatureSlider) mctsTemperatureSlider.value = MCTS_TEMPERATURE;
    if (mctsTemperatureValue) mctsTemperatureValue.textContent = MCTS_TEMPERATURE.toFixed(2);
}

// Initialize MCTS sliders when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    setupMCTSSliders();
    updateMCTSUI();
});

// --- MCTS Button Image Update ---
function updateMCTSButtonImage() {
    const mctsBtn = document.getElementById('mcts-btn');
    const mctsImg = mctsBtn?.querySelector('img');
    if (!mctsImg) return;
    switch (aiDifficultySelect.value) {
        case 'easy':
            mctsImg.src = 'images/happy-outline.svg';
            break;
        case 'hard1':
            mctsImg.src = 'images/hardware-chip-outline.svg';
            break;
        case 'hard_ai':
            mctsImg.src = 'images/skull-outline.svg';
            break;
        default:
            mctsImg.src = 'images/happy-outline.svg';
    }
}
if (aiDifficultySelect) {
    aiDifficultySelect.addEventListener('change', updateMCTSButtonImage);
    // Run once on load
    updateMCTSButtonImage();
}

// --- Board Click Handler ---
if (boardElement) {
    boardElement.addEventListener('click', (event) => {
        // Find the clicked cell
        const cell = event.target.closest('.cell');
        if (!cell) return;
        const row = parseInt(cell.dataset.row, 10);
        const col = parseInt(cell.dataset.col, 10);
        if (isNaN(row) || isNaN(col)) return;
        // Only allow moves if game is not over and not in self-play
        if (gameOver || isInSelfPlay) return;
        // Only allow Player A to move if not AI's turn
        if (currentPlayer === PLAYER_B && !isInSelfPlay) return;
        // Selection logic
        if (selectedPiece) {
            // Check if clicking a legal move target
            if (legalMoves.some(move => move.row === row && move.col === col)) {
                // Make the move
                const prevSelected = selectedPiece;
                const playerWhoMoved = currentPlayer; // Capture the player making the move before switching
                const boardBefore = cloneBoard(board);
                board = makeMove(selectedPiece.row, selectedPiece.col, row, col, board, selectedPiece, legalMoves, currentPlayer);
                recordMoveHistory(currentPlayer, selectedPiece, { row, col }, boardBefore, board);
                selectedPiece = null;
                legalMoves = [];
                // Check for win
                const winA = checkWinCondition(board, PLAYER_A);
                const winB = checkWinCondition(board, PLAYER_B);
                if (winA.win || winB.win) {
                    console.log(`Win detected for Player ${winA.win ? PLAYER_A : PLAYER_B}. Path:`, winA.win ? winA.path : winB.path);
                    gameOver = true;
                    winner = winA.win && winB.win ? 'both' : (winA.win ? PLAYER_A : PLAYER_B);
                    winPath = winA.win ? winA.path : winB.path;
                    // --- Enhancement: Show win overlay, update stats, enable history button ---
                    if (winner === PLAYER_A) {
                        playerAScore++;
                    } else if (winner === PLAYER_B) {
                        playerBScore++;
                    }
                    if (winOverlay && winMessage) {
                        winMessage.textContent = winner === 'both' ?
                            'Both players win simultaneously!' :
                            `Player ${winner === PLAYER_A ? 'A' : 'B'} wins!`;
                        showOverlay(winOverlay);
                    }
                    if (historyBtn) historyBtn.disabled = false;
                } else {
                    // Switch player
                    currentPlayer = switchPlayer(currentPlayer);
                    // If it's now Player B's turn and not in self-play, trigger AI
                    if (currentPlayer === PLAYER_B && !isInSelfPlay) {
                        setTimeout(triggerAIMove, 100);
                    }
                }
                renderBoardWithWinPath({
                    board,
                    selectedPiece,
                    legalMoves,
                    gameOver,
                    winner,
                    winPath,
                    moveHistory,
                    currentHistoryIndex,
                    playerAScore,
                    playerBScore,
                    boardElement,
                    currentPlayer
                });
                updateScoreDisplay();
                console.log(`Player ${playerWhoMoved} moved from (${prevSelected.row},${prevSelected.col}) to (${row},${col})`);
            } else if (row === selectedPiece.row && col === selectedPiece.col) {
                // Deselect
                selectedPiece = null;
                legalMoves = [];
                renderBoardWithWinPath({
                    board,
                    selectedPiece,
                    legalMoves,
                    gameOver,
                    winner,
                    winPath,
                    moveHistory,
                    currentHistoryIndex,
                    playerAScore,
                    playerBScore,
                    boardElement,
                    currentPlayer
                });
                console.log("Deselected piece at", row, col);
            } else {
                // Try selecting another of own pieces
                if (board[row][col] && board[row][col].player === currentPlayer) {
                    selectedPiece = { row, col };
                    legalMoves = calculateLegalMoves(row, col, board, currentPlayer);
                    renderBoardWithWinPath({
                        board,
                        selectedPiece,
                        legalMoves,
                        gameOver,
                        winner,
                        winPath,
                        moveHistory,
                        currentHistoryIndex,
                        playerAScore,
                        playerBScore,
                        boardElement,
                        currentPlayer
                    });
                    console.log(`Selected piece at (${row}, ${col}). Legal moves:`, legalMoves);
                } else {
                    // Deselect
                    selectedPiece = null;
                    legalMoves = [];
                    renderBoardWithWinPath({
                        board,
                        selectedPiece,
                        legalMoves,
                        gameOver,
                        winner,
                        winPath,
                        moveHistory,
                        currentHistoryIndex,
                        playerAScore,
                        playerBScore,
                        boardElement,
                        currentPlayer
                    });
                    console.log("Deselected piece (clicked invalid target)");
                }
            }
        } else {
            // No piece selected, try selecting if it's the current player's piece
            if (board[row][col] && board[row][col].player === currentPlayer) {
                selectedPiece = { row, col };
                legalMoves = calculateLegalMoves(row, col, board, currentPlayer);
                renderBoardWithWinPath({
                    board,
                    selectedPiece,
                    legalMoves,
                    gameOver,
                    winner,
                    winPath,
                    moveHistory,
                    currentHistoryIndex,
                    playerAScore,
                    playerBScore,
                    boardElement,
                    currentPlayer
                });
                console.log(`Selected piece at (${row}, ${col}). Legal moves:`, legalMoves);
            }
        }
    });
}

// --- AI Move Trigger ---

async function triggerAIMove() {
    if (gameOver) return;
    
    // Use the sophisticated AI from the advanced module
    const bestMove = await findBestAIMove(
        board, 
        currentPlayer, 
        AI_DIFFICULTY, 
        AI_DEPTH, 
        window.analysisMode, // analysisMode
        MCTS_ENABLED, 
        MCTS_SIMULATIONS, 
        tfModel, 
        mctsSearch, 
        gameLogic
    );

    if (bestMove) {
        const boardBefore = cloneBoard(board);
        board = makeMove(bestMove.start.row, bestMove.start.col, bestMove.end.row, bestMove.end.col, board, bestMove.start, [bestMove.end], currentPlayer);
        recordMoveHistory(currentPlayer, bestMove.start, bestMove.end, boardBefore, board);
        // Check for win
        const winA = checkWinCondition(board, PLAYER_A);
        const winB = checkWinCondition(board, PLAYER_B);
        if (winA.win || winB.win) {
            console.log(`Win detected for Player ${winA.win ? PLAYER_A : PLAYER_B}. Path:`, winA.win ? winA.path : winB.path);
            gameOver = true;
            winner = winA.win && winB.win ? 'both' : (winA.win ? PLAYER_A : PLAYER_B);
            winPath = winA.win ? winA.path : winB.path;
            // --- Enhancement: Show win overlay, update stats, enable history button ---
            if (winner === PLAYER_A) {
                playerAScore++;
            } else if (winner === PLAYER_B) {
                playerBScore++;
            }
            if (winOverlay && winMessage) {
                winMessage.textContent = winner === 'both' ?
                    'Both players win simultaneously!' :
                    `Player ${winner === PLAYER_A ? 'A' : 'B'} wins!`;
                showOverlay(winOverlay);
            }
            if (historyBtn) historyBtn.disabled = false;
        } else {
            currentPlayer = switchPlayer(currentPlayer);
        }
        renderBoardWithWinPath({
            board,
            selectedPiece,
            legalMoves,
            gameOver,
            winner,
            winPath,
            moveHistory,
            currentHistoryIndex,
            playerAScore,
            playerBScore,
            boardElement,
            currentPlayer
        });
        updateScoreDisplay();
        console.log(`AI (Player ${switchPlayer(currentPlayer)}) moved from (${bestMove.start.row},${bestMove.start.col}) to (${bestMove.end.row},${bestMove.end.col})`);
    } else {
        // AI has no valid moves - switch turn back to other player
        console.log(`AI (Player ${currentPlayer}) has no valid moves. Switching turn.`);
        currentPlayer = switchPlayer(currentPlayer);
        renderBoardWithWinPath({
            board,
            selectedPiece,
            legalMoves,
            gameOver,
            winner,
            winPath,
            moveHistory,
            currentHistoryIndex,
            playerAScore,
            playerBScore,
            boardElement,
            currentPlayer
        });
        updateScoreDisplay();
    }
}

// --- Board scroll/keyboard/touch navigation after win ---
boardElement.addEventListener('wheel', (event) => {
    if (!gameOver) return;
    event.preventDefault();
    const direction = event.deltaY > 0 ? -1 : 1; // Descending order: up = next, down = previous
    navigateHistory(direction);
});

let touchStartY = null;
boardElement.addEventListener('touchstart', (event) => {
    if (!gameOver) return;
    if (event.touches.length === 1) {
        touchStartY = event.touches[0].clientY;
    }
});
boardElement.addEventListener('touchend', (event) => {
    if (!gameOver || touchStartY === null) return;
    const touchEndY = event.changedTouches[0].clientY;
    const deltaY = touchEndY - touchStartY;
    if (Math.abs(deltaY) > 30) {
        const direction = deltaY < 0 ? 1 : -1; // Swipe up = next, down = previous
        navigateHistory(direction);
    }
    touchStartY = null;
});

// --- DEBUG Mode (from script.js) ---
var DEBUG = true; // Set to false to disable debug logs
var old_console_log = console.log;
console.log = function() {
    if (DEBUG) {
        old_console_log.apply(this, arguments);
    }
};
var old_console_warn = console.warn;
console.warn = function() {
    if (DEBUG) {
        old_console_warn.apply(this, arguments);
    }
};
// leave in console.error

// --- Export for Debug/Adapter Compatibility ---
window.SwitcharooGlobals = {
    board,
    currentPlayer,
    selectedPiece,
    legalMoves,
    moveHistory,
    gameOver,
    winner,
    winPath,
    playerAScore,
    playerBScore,
    currentHistoryIndex,
    startingPosition,
    startingPositionIndex,
    isInSelfPlay,
    selfPlayTimeoutId,
    tfModel,
    AI_DIFFICULTY,
    AI_DEPTH,
    ANALYSIS_MODE,
    MCTS_ENABLED,
    MCTS_SIMULATIONS,
    MCTS_TEMPERATURE,
    MCTS_PUCT_CONSTANT,
    MCTS_DIRICHLET_ALPHA,
    MCTS_DIRICHLET_EPSILON,
    MCTS_VERBOSE,
    mctsSearch,
    gameLogic,

    parseStartingPosition,
    cloneBoard,
    serializeBoardState,
    boardToKey,
    convertBoardToBinaryJS,
    renderBoard,
    getPieceImage,
    selectPiece,
    deselectPiece,
    calculateLegalMoves,
    makeMove,
    unmarkSwapped,
    switchPlayer,
    checkWinCondition,
    boardToNNInput,
    neuralNetworkPredict,
    initializeMCTS: setupMCTS,
    showOverlay,
    hideOverlay,
    hideAllOverlays,
    initGame,
    findBestAIMove,
    moveToActionIndex,
    calculateLegalMovesForState,
    allowsOpponentWin,
    evaluateBoardState,
    countFriendlyNeighbors,
    checkWinConditionForState,
    analyzeHistoricalMove, 
    startSelfPlay,
    stopSelfPlay
};

// --- Expose analyzeHistoricalMove to console ---
window.analyzeHistoricalMove = analyzeHistoricalMove;

// --- Patch: Only highlight win path if viewing the final (winning) position
// This requires passing a flag to renderBoard to indicate if win path should be shown
// We'll wrap renderBoard to add this flag based on currentHistoryIndex
function renderBoardWithWinPath(options) {
    // Only show win path if viewing the final position
    const showWinPath = (currentHistoryIndex === undefined || currentHistoryIndex === moveHistory.length);
    renderBoard({
        ...options,
        showWinPath,
        boardElement // PATCH: always pass boardElement
    });
}

// --- Historical Move Analysis Function ---
async function analyzeHistoricalMove() {
    if (currentHistoryIndex !== undefined && currentHistoryIndex > 0) {
        const historicalBoard = moveHistory[currentHistoryIndex - 1].boardAfter;

        // Determine whose turn it is for this historical board state
        // The boardAfter represents the state after a move was made
        // So the next player to move is the opposite of who made that move
        const playerWhoMadePreviousMove = moveHistory[currentHistoryIndex - 1].player;
        const playerToMoveNext = playerWhoMadePreviousMove === PLAYER_A ? PLAYER_B : PLAYER_A;

        ANALYSIS_MODE = true;
        const bestMove = await findBestAIMove(
            historicalBoard, 
            playerToMoveNext, 
            AI_DIFFICULTY, 
            AI_DEPTH, 
            true, // analysisMode
            MCTS_ENABLED, 
            MCTS_SIMULATIONS, 
            tfModel, 
            mctsSearch, 
            gameLogic
        );
        ANALYSIS_MODE = false;
        console.log(`Best move for this board state (Player ${playerToMoveNext}'s turn):`, bestMove);
    } else {
        console.log('No historical move selected.');
    }
}

// --- Self-Play Functions ---
function startSelfPlay() {
    if (gameOver) {
        console.log("Cannot start self-play: game is over");
        return;
    }

    isInSelfPlay = true;
    console.log("Self-play started");

    // Start the self-play loop
    scheduleSelfPlayMove();
}

function stopSelfPlay() {
    isInSelfPlay = false;

    // Clear any pending self-play move
    if (selfPlayTimeoutId) {
        clearTimeout(selfPlayTimeoutId);
        selfPlayTimeoutId = null;
    }
    console.log("Self-play stopped");
}

function scheduleSelfPlayMove() {
    if (!isInSelfPlay || gameOver) {
        return;
    }

    // 500ms pause between player moves for better visualization
    const delay = 500; // 500ms between moves

    selfPlayTimeoutId = setTimeout(() => {
        makeSelfPlayMove();
    }, delay);
}

async function makeSelfPlayMove() {
    if (!isInSelfPlay || gameOver) {
        return;
    }

    try {
        // Make an AI move for the current player
        await triggerAIMove();

        // Schedule the next move if game is still ongoing
        if (!gameOver && isInSelfPlay) {
            scheduleSelfPlayMove();
        } else if (gameOver) {
            stopSelfPlay();
        }
    } catch (error) {
        console.error("Error during self-play move:", error); stopSelfPlay();
    }
}

// --- DEBUG: Log current board state ---
import { logBoardForDebug } from './game-ai-advanced.js';

// Expose a function to print the current board state in text format from the browser console
window.printBoardState = function() {
    if (typeof board === 'undefined') {
        console.warn("Global board variable is not defined.");
        return;
    }
    logBoardForDebug(board);
    return board;
};
