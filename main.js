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
    selectPiece, deselectPiece, calculateLegalMoves, makeMove, unmarkPlayerSwapped, switchPlayer, checkWinCondition
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
let ANALYSIS_MODE = true;
let MCTS_ENABLED = false;
let MCTS_SIMULATIONS = 50;
let MCTS_TEMPERATURE = 0.01;
let MCTS_PUCT_CONSTANT = 1.0;
let MCTS_DIRICHLET_ALPHA = 0.3;
let MCTS_DIRICHLET_EPSILON = 0.25;
let MCTS_VERBOSE = false;
let mctsSearch = null;
let gameLogic = null;
const transpositionTable = new Map();

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

// --- Model Loading ---
async function loadTFJSModel() {
    try {
        console.log("Loading TensorFlow.js model...");
        tfModel = await tf.loadGraphModel('./switcharoo_tfjs_model/model.json');
        console.log("Model loaded successfully:", tfModel);
        return true;
    } catch (error) {
        console.warn("Could not load pre-trained graph model. Error object:", error);
        console.warn("The Neural Network AI option will be disabled as it requires pre-trained weights to work properly.");
        tfModel = null;
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
    }
}
document.addEventListener('DOMContentLoaded', () => {
    initGame();
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
    transpositionTable.clear();
    console.log("Game Initialized. Player A's turn (Bottom).");
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
    button.addEventListener('click', () => {
        const overlayId = button.getAttribute('data-overlay');
        hideOverlay(document.getElementById(overlayId));
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
    } else {
        mctsOnlyElements.forEach(el => el.style.display = 'none');
    }
}
if (aiDifficultySelect) {
    aiDifficultySelect.addEventListener('change', updateMCTSControlsVisibility);
    // Run once on load
    updateMCTSControlsVisibility();
}

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
                console.log(`Player ${currentPlayer} moved from (${prevSelected.row},${prevSelected.col}) to (${row},${col})`);
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
// Helper: filter out moves that allow immediate opponent win
function getSafeMoves(possibleMoves, board, player, opponent, excludeForcingMoves = false) {
    const safeMoves = [];
    for (const move of possibleMoves) {
        let tempBoard = cloneBoard(board);
        const movingPiece = tempBoard[move.start.row][move.start.col];
        const targetPiece = tempBoard[move.end.row][move.end.col];
        if (!targetPiece) {
            tempBoard[move.end.row][move.end.col] = { ...movingPiece };
            tempBoard[move.start.row][move.start.col] = null;
            unmarkPlayerSwapped(player, tempBoard);
        } else {
            tempBoard[move.end.row][move.end.col] = { ...movingPiece, state: SWAPPED };
            tempBoard[move.start.row][move.start.col] = { ...targetPiece, state: SWAPPED };
        }
        // Check if any opponent response is an immediate win
        let opponentCanWin = false;
        let opponentCanForce = false;
        for (let r2 = 0; r2 < ROWS && !opponentCanWin; r2++) {
            for (let c2 = 0; c2 < COLS && !opponentCanWin; c2++) {
                if (tempBoard[r2][c2] && tempBoard[r2][c2].player === opponent) {
                    const oppMoves = calculateLegalMoves(r2, c2, tempBoard, opponent);
                    for (const oppMove of oppMoves) {
                        let oppBoard = cloneBoard(tempBoard);
                        const oppMovingPiece = oppBoard[r2][c2];
                        const oppTargetPiece = oppBoard[oppMove.row][oppMove.col];
                        if (!oppTargetPiece) {
                            oppBoard[oppMove.row][oppMove.col] = { ...oppMovingPiece };
                            oppBoard[r2][c2] = null;
                            unmarkPlayerSwapped(opponent, oppBoard);
                        } else {
                            oppBoard[oppMove.row][oppMove.col] = { ...oppMovingPiece, state: SWAPPED };
                            oppBoard[r2][c2] = { ...oppTargetPiece, state: SWAPPED };
                        }
                        const win = checkWinCondition(oppBoard, opponent);
                        if (win.win) {
                            opponentCanWin = true;
                            break;
                        }
                        // Forcing move check: after this opponent move, does player have any safe moves?
                        if (excludeForcingMoves && !opponentCanWin) {
                            // Find all possible player moves after this opponent move
                            let playerMoves = [];
                            for (let r3 = 0; r3 < ROWS; r3++) {
                                for (let c3 = 0; c3 < COLS; c3++) {
                                    if (oppBoard[r3][c3] && oppBoard[r3][c3].player === player) {
                                        const moves = calculateLegalMoves(r3, c3, oppBoard, player);
                                        moves.forEach(mv => {
                                            playerMoves.push({ start: { row: r3, col: c3 }, end: { row: mv.row, col: mv.col } });
                                        });
                                    }
                                }
                            }
                            // Are there any safe moves for player?
                            let foundSafe = false;
                            for (const pmove of playerMoves) {
                                let testBoard = cloneBoard(oppBoard);
                                const pmovingPiece = testBoard[pmove.start.row][pmove.start.col];
                                const ptargetPiece = testBoard[pmove.end.row][pmove.end.col];
                                if (!ptargetPiece) {
                                    testBoard[pmove.end.row][pmove.end.col] = { ...pmovingPiece };
                                    testBoard[pmove.start.row][pmove.start.col] = null;
                                    unmarkPlayerSwapped(player, testBoard);
                                } else {
                                    testBoard[pmove.end.row][pmove.end.col] = { ...pmovingPiece, state: SWAPPED };
                                    testBoard[pmove.start.row][pmove.start.col] = { ...ptargetPiece, state: SWAPPED };
                                }
                                // Check if opponent can win immediately after this move
                                let oppCanWin = false;
                                for (let r4 = 0; r4 < ROWS && !oppCanWin; r4++) {
                                    for (let c4 = 0; c4 < COLS && !oppCanWin; c4++) {
                                        if (testBoard[r4][c4] && testBoard[r4][c4].player === opponent) {
                                            const oppMoves2 = calculateLegalMoves(r4, c4, testBoard, opponent);
                                            for (const om2 of oppMoves2) {
                                                let tb2 = cloneBoard(testBoard);
                                                const omPiece = tb2[r4][c4];
                                                const omTarget = tb2[om2.row][om2.col];
                                                if (!omTarget) {
                                                    tb2[om2.row][om2.col] = { ...omPiece };
                                                    tb2[r4][c4] = null;
                                                    unmarkPlayerSwapped(opponent, tb2);
                                                } else {
                                                    tb2[om2.row][om2.col] = { ...omPiece, state: SWAPPED };
                                                    tb2[r4][c4] = { ...omTarget, state: SWAPPED };
                                                }
                                                const win2 = checkWinCondition(tb2, opponent);
                                                if (win2.win) {
                                                    oppCanWin = true;
                                                    break;
                                                }
                                            }
                                        }
                                    }
                                }
                                if (!oppCanWin) {
                                    foundSafe = true;
                                    break;
                                }
                            }
                            if (!foundSafe) {
                                opponentCanForce = true;
                                break;
                            }
                        }
                    }
                }
            }
        }
        if (!opponentCanWin && (!excludeForcingMoves || !opponentCanForce)) safeMoves.push(move);
    }
    return safeMoves.length > 0 ? safeMoves : possibleMoves;
}

async function triggerAIMove() {
    if (gameOver) return;
    let bestMove = null;
    // Gather all possible moves for Player B
    let possibleMoves = [];
    for (let r = 0; r < ROWS; r++) {
        for (let c = 0; c < COLS; c++) {
            if (board[r][c] && board[r][c].player === PLAYER_B) {
                const moves = calculateLegalMoves(r, c, board, PLAYER_B);
                moves.forEach(move => {
                    possibleMoves.push({ start: { row: r, col: c }, end: { row: move.row, col: move.col } });
                });
            }
        }
    }
    if (possibleMoves.length === 0) return;

    // Always check for immediate win first
    let immediateWin = null;
    for (const move of possibleMoves) {
        let tempBoard = cloneBoard(board);
        const movingPiece = tempBoard[move.start.row][move.start.col];
        const targetPiece = tempBoard[move.end.row][move.end.col];
        if (!targetPiece) {
            tempBoard[move.end.row][move.end.col] = { ...movingPiece };
            tempBoard[move.start.row][move.start.col] = null;
            unmarkPlayerSwapped(PLAYER_B, tempBoard);
        } else {
            tempBoard[move.end.row][move.end.col] = { ...movingPiece, state: SWAPPED };
            tempBoard[move.start.row][move.start.col] = { ...targetPiece, state: SWAPPED };
        }
        const winB = checkWinCondition(tempBoard, PLAYER_B);
        if (winB.win) {
            immediateWin = move;
            break;
        }
    }
    if (immediateWin) {
        bestMove = immediateWin;
    } else {
        // Filter out own-goal moves (allowing immediate opponent win)
        let excludeForcing = (AI_DIFFICULTY === 'hard1' || (AI_DIFFICULTY === 'hard_ai'));
        const safeMoves = getSafeMoves(possibleMoves, board, PLAYER_B, PLAYER_A, excludeForcing);
        if (AI_DIFFICULTY === 'easy') {
            bestMove = safeMoves[Math.floor(Math.random() * safeMoves.length)];
        } else if (AI_DIFFICULTY === 'hard1' || (AI_DIFFICULTY === 'hard_ai' && !tfModel)) {
            // Heuristic: prefer advancing pieces among safe moves
            let bestScore = -Infinity;
            let bestMoves = [];
            for (const move of safeMoves) {
                let tempBoard = cloneBoard(board);
                const movingPiece = tempBoard[move.start.row][move.start.col];
                const targetPiece = tempBoard[move.end.row][move.end.col];
                if (!targetPiece) {
                    tempBoard[move.end.row][move.end.col] = { ...movingPiece };
                    tempBoard[move.start.row][move.start.col] = null;
                    unmarkPlayerSwapped(PLAYER_B, tempBoard);
                } else {
                    tempBoard[move.end.row][move.end.col] = { ...movingPiece, state: SWAPPED };
                    tempBoard[move.start.row][move.start.col] = { ...targetPiece, state: SWAPPED };
                }
                let score = 0;
                for (let r = 0; r < ROWS; r++) {
                    for (let c = 0; c < COLS; c++) {
                        if (tempBoard[r][c] && tempBoard[r][c].player === PLAYER_B) {
                            score += r;
                        }
                    }
                }
                if (score > bestScore) {
                    bestScore = score;
                    bestMoves = [move];
                } else if (score === bestScore) {
                    bestMoves.push(move);
                }
            }
            bestMove = bestMoves[Math.floor(Math.random() * bestMoves.length)];
        } else if (AI_DIFFICULTY === 'hard_ai' && tfModel) {
            // Use MCTS or NN policy among safe moves
            if (MCTS_ENABLED && mctsSearch && gameLogic) {
                const actionProbs = await mctsSearch.search(board, PLAYER_B, neuralNetworkPredict, gameLogic);
                let bestProb = -1;
                for (const move of safeMoves) {
                    const actionIndex = (move.start.row * COLS + move.start.col) * (ROWS * COLS) + (move.end.row * COLS + move.end.col);
                    if (actionProbs[actionIndex] > bestProb) {
                        bestProb = actionProbs[actionIndex];
                        bestMove = move;
                    }
                }
                if (!bestMove) {
                    bestMove = safeMoves[Math.floor(Math.random() * safeMoves.length)];
                }
            } else {
                bestMove = safeMoves[Math.floor(Math.random() * safeMoves.length)];
            }
        } else {
            bestMove = safeMoves[Math.floor(Math.random() * safeMoves.length)];
        }
    }

    if (bestMove) {
        const boardBefore = cloneBoard(board);
        board = makeMove(bestMove.start.row, bestMove.start.col, bestMove.end.row, bestMove.end.col, board, bestMove.start, [bestMove.end], PLAYER_B);
        recordMoveHistory(PLAYER_B, bestMove.start, bestMove.end, boardBefore, board);
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
        console.log(`AI (Player B) moved from (${bestMove.start.row},${bestMove.start.col}) to (${bestMove.end.row},${bestMove.end.col})`);
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
    transpositionTable,
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
    unmarkPlayerSwapped,
    switchPlayer,
    checkWinCondition,
    boardToNNInput,
    neuralNetworkPredict,
    initializeMCTS: setupMCTS,
    showOverlay,
    hideOverlay,
    hideAllOverlays,
    initGame
};

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
