// The code in this file has been refactored into ES modules:
//   - game-constants.js
//   - game-board.js
//   - game-render.js
//   - game-logic.js
//   - game-ai.js
//   - game-mcts-wrapper.js
//   - game-overlays.js
//   - main.js (entry point)
// See those files for the new modular code structure.

document.addEventListener('DOMContentLoaded', async () => {    // Try best available backend: WebGL > WASM > CPU
    try {
        if (typeof tf !== 'undefined') {
            // Check available backends
            const backends = Object.keys(tf.engine().registryFactory);
            console.log("Available TF.js backends:", backends);

            // Try WebGL first for better performance on devices that support it
            if (backends.includes('webgl')) {
                try {
                    await tf.setBackend('webgl');
                    console.log("TensorFlow.js using WebGL backend for best performance");
                } catch (webglError) {
                    console.warn("WebGL backend failed, falling back to alternatives:", webglError);
                }
            }
            // If WebGL fails or isn't available, try WASM for better CPU performance
            else if (backends.includes('wasm')) {
                try {
                    await tf.setBackend('wasm');
                    console.log("TensorFlow.js using WASM backend for improved CPU performance");
                } catch (wasmError) {
                    console.warn("WASM backend failed, falling back to CPU:", wasmError);
                    await tf.setBackend('cpu');
                    console.log("TensorFlow.js using CPU backend");
                }
            }
            // Default to CPU as last resort
            else {
                await tf.setBackend('cpu');
                console.log("TensorFlow.js using CPU backend");
            }

            // Ensure backend is ready before loading model
            await tf.ready();

            console.log("Active backend:", tf.getBackend());
        } else {
            console.warn("tf object not available at the time of setting backend.");
        }
    } catch (e) {
        console.error("Error setting TF.js backend:", e);
    }

    const ROWS = 8;
    const COLS = 4;
    const PLAYER_A = 'A'; // White
    const PLAYER_B = 'B'; // Black
    const NORMAL = 'normal';
    const SWAPPED = 'swapped';
    const NUM_DIRECTIONS = 8;
    const JS_DIRECTIONS = [
        { dr: -1, dc: -1 }, // 0
        { dr: -1, dc: 0 }, // 1
        { dr: -1, dc: 1 }, // 2
        { dr: 0, dc: -1 }, // 3
        { dr: 0, dc: 1 }, // 4
        { dr: 1, dc: -1 }, // 5
        { dr: 1, dc: 0 }, // 6
        { dr: 1, dc: 1 }  // 7
    ]; let AI_DIFFICULTY = 'easy'; // easy, hard1, hard2, hard_ai
    let AI_DEPTH = 1; // Will be set based on difficulty
    let ANALYSIS_MODE = true; // Ensure this is true for detailed NN logging
    let tfModel = null; // TensorFlow.js model    let startingPosition = null; // Starting position for the game
    let startingPositionIndex = 0; // Index for the initial position in the array    // MCTS Configuration
    let MCTS_ENABLED = false; // Enable/disable MCTS for performance comparison - disabled by default for easy mode
    let MCTS_SIMULATIONS = 50; // Number of MCTS simulations per move (1-1000)
    let MCTS_TEMPERATURE = 0.1; // Temperature for action selection (0.0-2.0) - better default for gameplay
    let MCTS_PUCT_CONSTANT = 1.0; // PUCT exploration constant
    let MCTS_DIRICHLET_ALPHA = 0.3; // Dirichlet noise alpha
    let MCTS_DIRICHLET_EPSILON = 0.25; // Dirichlet noise epsilon
    let MCTS_VERBOSE = false; // Enable detailed MCTS logging

    // MCTS instances
    let mctsSearch = null;
    let gameLogic = null;

    const initialPosition = [
        "BBBB\nBBBB\n....\n....\n....\n....\nAAAA\nAAAA",
        "....\n....\nBBBB\nBBBB\nAAAA\nAAAA\n....\n....",
        "BB..\nBB..\nBB..\nBB..\n..AA\n..AA\n..AA\n..AA",
        "B...\nBB..\nBB..\nBBB.\n.AAA\n..AA\n..AA\n...A",
        "B..B\n.BB.\n.BB.\nB..B\nA..A\n.AA.\n.AA.\nA..A",
        "....\n....\nBABA\nABAB\nBABA\nABAB\n....\n...."
    ];



    // Load TensorFlow.js model
    async function loadTFJSModel() {
        try {
            console.log("Loading TensorFlow.js model...");
            console.log("TensorFlow.js version:", tf.version.tfjs); // Log TFJS version

            // Try direct model loading first
            try {
                console.info("Attempting to load GRAPH model from: ./switcharoo_tfjs_model/model.json");
                console.log("BEFORE await tf.loadGraphModel");
                // Switched back to loadGraphModel
                const loadedModel = await tf.loadGraphModel('./switcharoo_tfjs_model/model.json');
                console.log("AFTER await tf.loadGraphModel. Model object:", loadedModel);
                console.info("Graph model loaded successfully!");

                // Use the loaded model directly
                tfModel = loadedModel;

                // GraphModel does not have .summary() or .inputs/.outputs in the same way as LayersModel
                // console.log("Model summary:"); // tfModel.summary(); // This would error for GraphModel
                // console.log("Model inputs:", tfModel.inputs); // Different structure for GraphModel
                // console.log("Model outputs:", tfModel.outputs);                console.log("Model signature:", tfModel.signature);


                return true;
            } catch (directLoadError) {
                console.warn("Could not load pre-trained graph model. Error object:", directLoadError); // Log the error object
                console.warn("The Neural Network AI option will be disabled as it requires pre-trained weights to work properly.");

                // Don't create a fallback model with random weights as it's not useful for gameplay
                console.warn("Using standard heuristic AI only (easy/hard modes).");
                return false;
            } finally { // Added finally block
                console.log("Finally block of model loading reached.");
            }

        } catch (error) {
            console.error("Failed to initialize TensorFlow.js:", error);
            return false;
        }
    }

    // Try to load the model
    try {
        if (typeof tf !== 'undefined') {
            loadTFJSModel();
        } else {
            console.warn("TensorFlow.js is not available. 'AI' difficulty option will not use the neural network model.");
        }
    } catch (e) {
        console.warn("Error initializing TensorFlow.js:", e);
    }    // Initialize MCTS components
    function initializeMCTS() {
        if (typeof MCTSSearch !== 'undefined' && typeof SwitcharooGameLogic !== 'undefined') {
            gameLogic = new SwitcharooGameLogic({
                ROWS: ROWS,
                COLS: COLS,
                PLAYER_A: PLAYER_A,
                PLAYER_B: PLAYER_B,
                NORMAL: NORMAL,
                SWAPPED: SWAPPED,
                NUM_DIRECTIONS: NUM_DIRECTIONS,
                JS_DIRECTIONS: JS_DIRECTIONS
            }); mctsSearch = new MCTSSearch({
                numSimulations: MCTS_SIMULATIONS,
                cPuct: MCTS_PUCT_CONSTANT,
                temperature: MCTS_TEMPERATURE,
                dirichletAlpha: MCTS_DIRICHLET_ALPHA,
                dirichletEpsilon: MCTS_DIRICHLET_EPSILON,
                enabled: MCTS_ENABLED,
                verbose: MCTS_VERBOSE,
                logSearchStats: true // Force enable for testing
            });
            console.log("MCTS initialized successfully");
        } else {
            console.warn("MCTS classes not available - MCTS features disabled");
        }
    }

    // Initialize MCTS when DOM is ready
    document.addEventListener('DOMContentLoaded', () => {
        initializeMCTS();
    }); const boardElement = document.getElementById('game-board');
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
    const aiSpinnerOverlay = document.getElementById('ai-spinner-overlay');  // Add this line

    let board = []; // 2D array: null or { player: PLAYER_A/B, state: NORMAL/SWAPPED }
    let currentPlayer = PLAYER_A;
    let selectedPiece = null; // { row, col }
    let legalMoves = []; // Array of { row, col }
    let moveHistory = []; // Array of { player, start, end, boardBefore }
    let gameOver = false;
    let winner = null;
    let winPath = []; // Stores cells [{row, col}] of the winning path
    let playerAScore = 0;
    let playerBScore = 0; let currentHistoryIndex = undefined;
    let touchStartY = 0;
    const transpositionTable = new Map();

    // Self-play variables
    let isInSelfPlay = false;
    let selfPlayTimeoutId = null;

    var DEBUG = true; // Set to false to disable
    var old_console_log = console.log;
    console.log = function () {
        if (DEBUG) {
            old_console_log.apply(this, arguments);
        }
    };
    var old_console_warn = console.warn;
    console.warn = function () {
        if (DEBUG) {
            old_console_warn.apply(this, arguments);
        }
    };
    // leave in console.error
    // --- Initialization ---

    function parseStartingPosition(positionStr) {
        const board = Array(ROWS).fill(null).map(() => Array(COLS).fill(null));
        const rows = positionStr.split('\n');
        for (let r = 0; r < ROWS; r++) {
            for (let c = 0; c < COLS; c++) {
                const piece = rows[r][c];
                if (piece === 'A') {
                    board[r][c] = { player: PLAYER_A, state: NORMAL };
                } else if (piece === 'B') {
                    board[r][c] = { player: PLAYER_B, state: NORMAL };
                } else if (piece === 'a') {
                    board[r][c] = { player: PLAYER_A, state: SWAPPED };
                } else if (piece === 'b') {
                    board[r][c] = { player: PLAYER_B, state: SWAPPED };
                }
            }
        }
        return board;
    } function initGame() {
        // Stop self-play if it's running
        if (isInSelfPlay) {
            stopSelfPlay();
        }

        // Use the updated startingPositionIndex to set the starting position
        console.log(`initGame called with startingPositionIndex: ${startingPositionIndex}`); // Add log
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
        renderBoard();
        updateStatusMessage(); // Optional: Indicate whose turn
        hideAllOverlays();
        updateScoreDisplay();
        transpositionTable.clear();
        console.log("Game Initialized. Player A's turn (Bottom)."); // Updated log
    }

    // --- Rendering ---
    // (renderBoard function remains the same, it renders based on the board array)
    function renderBoard(boardState = board) {
        boardElement.innerHTML = '';
        boardElement.classList.remove('game-over');

        for (let r = 0; r < ROWS; r++) {
            for (let c = 0; c < COLS; c++) {
                const cell = document.createElement('div');
                cell.classList.add('cell');
                cell.dataset.row = r;
                cell.dataset.col = c;

                // Checkerboard pattern
                cell.classList.add(((r + c) % 2 === 0) ? 'light' : 'dark');

                // Add piece if present
                const pieceData = boardState[r][c];
                if (pieceData) {
                    const pieceImg = document.createElement('img');
                    pieceImg.classList.add('piece');
                    pieceImg.src = getPieceImage(pieceData.player, pieceData.state);
                    pieceImg.alt = `Player ${pieceData.player} ${pieceData.state}`;
                    cell.appendChild(pieceImg);
                }

                // Add highlights
                if (selectedPiece && selectedPiece.row === r && selectedPiece.col === c) {
                    cell.classList.add('selected');
                }

                if (legalMoves.some(move => move.row === r && move.col === c)) {
                    cell.classList.add('legal-move');
                    if (boardState[r][c] && boardState[r][c].player !== currentPlayer) {
                        cell.classList.add('swap-target');
                    }
                }

                // Win path highlighting - only show on final move
                if (gameOver && winner &&
                    (currentHistoryIndex === undefined || currentHistoryIndex === moveHistory.length)) {
                    // Get current state win paths from move history
                    const lastMove = moveHistory[moveHistory.length - 1];
                    if (lastMove && lastMove.winPaths) {
                        if (winner === 'both') {
                            // Check both paths
                            if (lastMove.winPaths.current.some(pos => pos.row === r && pos.col === c)) {
                                cell.classList.add('win-path');
                            }
                            if (lastMove.winPaths.other.some(pos => pos.row === r && pos.col === c)) {
                                if (cell.classList.contains('win-path')) {
                                    cell.classList.remove('win-path');
                                    cell.classList.add('win-path-second');
                                } else {
                                    cell.classList.add('win-path-second');
                                }
                            }
                        } else {
                            // Single winner
                            const winningPath = lastMove.winPaths[winner];
                            if (winningPath && winningPath.some(pos => pos.row === r && pos.col === c)) {
                                cell.classList.add('win-path');
                            }
                        }
                    }
                }

                cell.addEventListener('click', () => handleCellClick(r, c));
                boardElement.appendChild(cell);
            }
        }

        // Add score display and move counter if game is over
        if (gameOver) {
            boardElement.classList.add('game-over');

            if (currentHistoryIndex !== undefined) {
                const moveCount = document.createElement('div');
                moveCount.classList.add('move-counter');
                moveCount.textContent = currentHistoryIndex === moveHistory.length ?
                    'Final Position' :
                    `Move ${currentHistoryIndex} of ${moveHistory.length}`;
                boardElement.appendChild(moveCount);
            }

            const scoreDisplay = document.createElement('div');
            scoreDisplay.classList.add('score-display');
            scoreDisplay.innerHTML = `
                <div id="score-a">Player A: ${playerAScore}</div>
                <div id="score-b">Player B: ${playerBScore}</div>
            `;
            boardElement.appendChild(scoreDisplay);
        }

        // Update control scores
        const controlScoreA = document.getElementById('control-score-a');
        const controlScoreB = document.getElementById('control-score-b');
        if (controlScoreA) controlScoreA.textContent = `${playerAScore}`;
        if (controlScoreB) controlScoreB.textContent = `${playerBScore}`;

        if (gameOver) {
            updateMoveCounter();
        }
    }


    function getPieceImage(player, state) {
        const color = player === PLAYER_A ? 'white' : 'black';
        const type = state === NORMAL ? 'normal' : 'swapped';
        return `images/${color}_${type}.png`;
    }

    function updateStatusMessage() {
        // Optional: Implement a status display element if needed
        // e.g., document.getElementById('status').textContent = `Turn: Player ${currentPlayer}`;
    }     // --- Event Handlers ---
    // (No changes needed in handleCellClick, resetBtn, infoBtn, historyBtn listeners,
    // overlayCloseButtons, or overlay backdrop listeners)
    function handleCellClick(row, col) {
        if (gameOver) return; // No moves after game ends (unless viewing history)
        if (currentPlayer === PLAYER_B && !isInSelfPlay) return; // Block human clicks during AI turn (except in self-play)
        if (isInSelfPlay) return; // Block all human interaction during self-play

        const clickedCellPiece = board[row][col];

        if (selectedPiece) {
            // Check if clicking a legal move target
            const isLegalMove = legalMoves.some(move => move.row === row && move.col === col);
            if (isLegalMove) {
                makeMove(selectedPiece.row, selectedPiece.col, row, col);
            } else if (row === selectedPiece.row && col === selectedPiece.col) {
                // Clicking the selected piece again deselects it
                deselectPiece();
                renderBoard();
            } else if (clickedCellPiece && clickedCellPiece.player === currentPlayer) {
                // Clicking another of own pieces selects the new one
                deselectPiece(); // Deselect previous first
                selectPiece(row, col);
                renderBoard();
            } else {
                // Clicking an invalid spot deselects
                deselectPiece();
                renderBoard();
            }
        } else {
            // No piece selected, try selecting if it's the current player's piece
            if (clickedCellPiece && clickedCellPiece.player === currentPlayer) {
                selectPiece(row, col);
                renderBoard();
            }
        }
    } resetBtn.addEventListener('click', () => {
        console.log("Reset button clicked. Resetting to initial position 0."); // Add log
        startingPositionIndex = 0; // Explicitly reset index to 0
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
        // Cycle through starting positions
        startingPositionIndex = (startingPositionIndex + 1) % initialPosition.length;
        console.log(`Starting board button clicked. New position index: ${startingPositionIndex}`);
        initGame();
    });

    // AI Difficulty Selection (moved to MCTS overlay)
    function updateAIDifficulty(difficulty) {
        switch (difficulty) {
            case 'easy':
                AI_DIFFICULTY = 'easy';
                AI_DEPTH = 1;
                break;
            case 'hard1':
                AI_DIFFICULTY = 'hard1';
                AI_DEPTH = 3;
                break;
            case 'hard_ai':
                // Only allow hard_ai if model is loaded
                if (tfModel) {
                    AI_DIFFICULTY = 'hard_ai';
                    AI_DEPTH = 1; // Depth doesn't matter for neural network
                    MCTS_ENABLED = true; // Enable MCTS by default in AI Agent mode
                } else {
                    AI_DIFFICULTY = 'easy';
                    AI_DEPTH = 1;
                    console.warn("Neural network model not loaded, falling back to Easy");
                }
                break;
        }
        // Update MCTS controls visibility
        updateMCTSControlsVisibility();

        // Update MCTS button image
        updateMCTSButtonImage();

        console.log(`AI Difficulty switched to: ${AI_DIFFICULTY} (depth: ${AI_DEPTH})`);
    }

    function updateMCTSButtonImage() {
        const mctsBtn = document.getElementById('mcts-btn');
        const mctsImg = mctsBtn?.querySelector('img');

        if (mctsImg) {
            switch (AI_DIFFICULTY) {
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
                    break;
            }
        }
    } function updateMCTSControlsVisibility() {
        const mctsOnlyElements = document.querySelectorAll('.mcts-only');
        if (AI_DIFFICULTY === 'hard_ai') {
            mctsOnlyElements.forEach(element => {
                element.classList.remove('disabled');
            });
        } else {
            mctsOnlyElements.forEach(element => {
                element.classList.add('disabled');
            });
            // Disable MCTS when not in AI Agent mode
            MCTS_ENABLED = false;
            updateMCTSSettings();
        }

        // Update the UI to reflect current MCTS settings
        updateMCTSUI();
    } overlayCloseButtons.forEach(button => {
        button.addEventListener('click', () => {
            const overlayId = button.getAttribute('data-overlay');
            hideOverlay(document.getElementById(overlayId));
        });
    });

    // Close overlay by clicking outside content
    overlays.forEach(overlay => {
        overlay.addEventListener('click', (event) => {
            if (event.target === overlay) { // Check if click is on the backdrop itself
                hideOverlay(overlay);
            }
        });
    });


    // --- Game Logic ---
    // (selectPiece, deselectPiece, calculateLegalMoves, unmarkAllSwapped, switchPlayer remain the same)
    function selectPiece(row, col) {
        selectedPiece = { row, col };
        legalMoves = calculateLegalMoves(row, col);
        console.log(`Selected piece at (${row}, ${col}). Legal moves:`, legalMoves);
    }

    function deselectPiece() {
        selectedPiece = null;
        legalMoves = [];
    }

    function calculateLegalMoves(r, c) {
        const moves = [];
        const piece = board[r][c];
        if (!piece) return moves; // Should not happen if called correctly

        const pieceState = piece.state;
        const opponent = piece.player === PLAYER_A ? PLAYER_B : PLAYER_A;

        for (let dr = -1; dr <= 1; dr++) {
            for (let dc = -1; dc <= 1; dc++) {
                if (dr === 0 && dc === 0) continue; // Skip self

                const nr = r + dr;
                const nc = c + dc;

                // Check bounds
                if (nr >= 0 && nr < ROWS && nc >= 0 && nc < COLS) {
                    const targetCell = board[nr][nc];

                    if (targetCell === null) {
                        // Move to empty cell
                        moves.push({ row: nr, col: nc });
                    } else if (targetCell.player === opponent) {
                        // Can swap only with opponent's NORMAL piece
                        if (targetCell.state === NORMAL) {
                            moves.push({ row: nr, col: nc, isSwap: true });
                        }
                    }
                    // Cannot move to a cell occupied by own piece
                    // Cannot move to a cell occupied by opponent's SWAPPED piece
                }
            }
        }
        return moves;
    }

    function makeMove(startRow, startCol, endRow, endCol) {
        if (!selectedPiece || startRow !== selectedPiece.row || startCol !== selectedPiece.col) {
            console.error("Move error: Invalid start piece.");
            return;
        }

        const move = legalMoves.find(m => m.row === endRow && m.col === endCol);
        if (!move) {
            console.error("Move error: Invalid target cell.");
            deselectPiece();
            renderBoard();
            return;
        }

        // Record state before the move for history
        const boardBefore = cloneBoard(board);
        moveHistory.push({
            player: currentPlayer,
            start: { row: startRow, col: startCol },
            end: { row: endRow, col: endCol },
            boardBefore: boardBefore
        });

        // Apply the move
        const movingPiece = board[startRow][startCol];
        const targetPiece = board[endRow][endCol];

        if (targetPiece === null) {
            // Move to empty cell
            board[endRow][endCol] = movingPiece;
            board[startRow][startCol] = null;
            unmarkPlayerSwapped(currentPlayer); // Only unmark current player's pieces
        } else {
            // Swap move
            board[endRow][endCol] = { ...movingPiece, state: SWAPPED };
            board[startRow][startCol] = { ...targetPiece, state: SWAPPED };
        }

        // Store the state after the move
        moveHistory[moveHistory.length - 1].boardAfter = cloneBoard(board);
        deselectPiece();

        // Check for win conditions for BOTH players
        const currentPlayerCheck = checkWinConditionForState(board, currentPlayer);
        const otherPlayer = currentPlayer === PLAYER_A ? PLAYER_B : PLAYER_A;
        const otherPlayerCheck = checkWinConditionForState(board, otherPlayer);

        console.log("Win check results:", {
            currentPlayer,
            currentPlayerWin: currentPlayerCheck.win,
            otherPlayerWin: otherPlayerCheck.win,
            currentPath: currentPlayerCheck.path,
            otherPath: otherPlayerCheck.path
        });

        if (currentPlayerCheck.win || otherPlayerCheck.win) {
            gameOver = true;

            if (currentPlayerCheck.win && otherPlayerCheck.win) {
                // Simultaneous win scenario
                winner = 'both';
                // Store both winning paths in the move history
                moveHistory[moveHistory.length - 1].winPaths = {
                    current: currentPlayerCheck.path,
                    other: otherPlayerCheck.path
                };

                // First render the board without highlights
                renderBoard();

                // Add BOTH winning paths
                if (currentPlayerCheck.path) {
                    currentPlayerCheck.path.forEach(pos => {
                        const cell = boardElement.querySelector(`[data-row="${pos.row}"][data-col="${pos.col}"]`);
                        if (cell) {
                            cell.classList.add('win-path');
                            console.log('Adding win-path to', pos.row, pos.col);
                        }
                    });
                }

                if (otherPlayerCheck.path) {
                    otherPlayerCheck.path.forEach(pos => {
                        const cell = boardElement.querySelector(`[data-row="${pos.row}"][data-col="${pos.col}"]`);
                        if (cell) {
                            if (cell.classList.contains('win-path')) {
                                // If cell is already part of first path, use second style
                                cell.classList.remove('win-path');
                                cell.classList.add('win-path-second');
                            } else {
                                cell.classList.add('win-path-second');
                            }
                            console.log('Adding win-path-second to', pos.row, pos.col);
                        }
                    });
                }
            } else {
                // Single player win
                winner = currentPlayerCheck.win ? currentPlayer : otherPlayer;
                winPath = currentPlayerCheck.win ? currentPlayerCheck.path : otherPlayerCheck.path;
                moveHistory[moveHistory.length - 1].winPaths = {
                    [winner]: winPath
                };
                // Render board with single win path
                renderBoard();
                // Add highlights after rendering
                winPath.forEach(pos => {
                    const cell = boardElement.querySelector(`[data-row="${pos.row}"][data-col="${pos.col}"]`);
                    if (cell) {
                        cell.classList.add('win-path');
                    }
                });
            }
            handleWin(winner);
        } else {
            renderBoard();
            switchPlayer();
            // Trigger AI move for Player B in normal mode, but NOT for self-play (self-play handles its own timing)
            if (currentPlayer === PLAYER_B && !isInSelfPlay) {
                setTimeout(() => {
                    triggerAIMove();
                }, 50);
            }
        }
    }

    // reverting back to swapping all pieces
    function unmarkPlayerSwapped(player, boardState = board) {
        for (let r = 0; r < ROWS; r++) {
            for (let c = 0; c < COLS; c++) {
                if (boardState[r][c] && boardState[r][c].state === SWAPPED) {
                    boardState[r][c].state = NORMAL;
                }
            }
        }
    }

    function switchPlayer() {
        currentPlayer = (currentPlayer === PLAYER_A) ? PLAYER_B : PLAYER_A;
        updateStatusMessage();
        console.log(`Turn switched. Player ${currentPlayer}'s turn.`);    }

    // --- Win Condition ---

    function checkWinCondition(player) {
        // Define starting and target areas for each player
        const startRows = (player === PLAYER_A) ? [ROWS - 2, ROWS - 1] : [0, 1];  // A: rows 6,7; B: rows 0,1
        const targetRows = (player === PLAYER_A) ? [0, 1] : [ROWS - 2, ROWS - 1]; // A: rows 0,1; B: rows 6,7

        const visited = Array(ROWS).fill(null).map(() => Array(COLS).fill(false));
        const queue = []; // Queue for BFS: stores {row, col, path}

        // Find all pieces in the starting area and add them to the queue
        for (const startRow of startRows) {
            for (let c = 0; c < COLS; c++) {
                if (board[startRow] && board[startRow][c] && board[startRow][c].player === player) {
                    queue.push({
                        row: startRow,
                        col: c,
                        path: [{ row: startRow, col: c }]
                    });
                    visited[startRow][c] = true;
                }
            }
        }

        while (queue.length > 0) {
            const current = queue.shift();
            const { row, col, path } = current;

            // Check if we reached any target row
            if (targetRows.includes(row)) {
                console.log(`Win detected for Player ${player}. Path:`, path);
                return { win: true, path: path }; // Found a path
            }

            // Explore neighbors
            for (let dr = -1; dr <= 1; dr++) {
                for (let dc = -1; dc <= 1; dc++) {
                    if (dr === 0 && dc === 0) continue;

                    const nr = row + dr;
                    const nc = col + dc;

                    // Check bounds, if visited, and if it's the player's piece
                    if (nr >= 0 && nr < ROWS && nc >= 0 && nc < COLS &&
                        !visited[nr][nc] &&
                        board[nr] && board[nr][nc] && board[nr][nc].player === player) // Added check for board[nr] existence
                    {
                        visited[nr][nc] = true;
                        const newPath = [...path, { row: nr, col: nc }];
                        queue.push({ row: nr, col: nc, path: newPath });
                    }
                }
            }
        }

        return { win: false, path: [] }; // No path found
    }
    function handleWin(winningPlayer) {
        console.log(`Game Over! ${winningPlayer === 'both' ? 'Both players win!' : 'Player ' + winningPlayer + ' wins!'}`);

        // Stop self-play if it's running and update play button
        if (isInSelfPlay) {
            stopSelfPlay();
        }

        // Update scores
        if (winningPlayer === 'both') {
            playerAScore++;
            playerBScore++;
        } else if (winningPlayer === PLAYER_A) {
            playerAScore++;
        } else {
            playerBScore++;
        }

        updateScoreDisplay();

        winMessage.textContent = winningPlayer === 'both' ?
            "Hey, You Both Win!" :
            `Player ${winningPlayer} Wins!`;

        showOverlay(winOverlay);
        historyBtn.disabled = false;
        currentHistoryIndex = moveHistory.length;

        // Auto-close win overlay after 5 seconds
        setTimeout(() => {
            if (winOverlay.classList.contains('active')) {
                hideOverlay(winOverlay);
            }
        }, 5000);
    }

    // --- AI Opponent (Player B) ---

    // Function to convert a move object to an action index
    function moveToActionIndex(move) {
        const startCellIndex = move.start.row * COLS + move.start.col;
        const dr = move.end.row - move.start.row;
        const dc = move.end.col - move.start.col;

        let directionIndex = -1;
        for (let i = 0; i < JS_DIRECTIONS.length; i++) {
            if (JS_DIRECTIONS[i].dr === dr && JS_DIRECTIONS[i].dc === dc) {
                directionIndex = i;
                break;
            }
        }

        if (directionIndex === -1) {
            console.error("Could not find direction for move:", JSON.stringify(move)); // Log input move
            return null;
        }
        const actionIndex = startCellIndex * NUM_DIRECTIONS + directionIndex;
        if (DEBUG) console.log(`moveToActionIndex: move: ${JSON.stringify(move)}, startCellIndex: ${startCellIndex}, directionIndex: ${directionIndex}, actionIndex: ${actionIndex}`); // Log details
        return actionIndex;
    }    // Function to convert board state to neural network input
    // This function replicates the Python _get_state() method from game_env.py
    // Returns a 6-channel representation: each channel is a flattened ROWS x COLS board (32 elements each)
    // Total: 6 * 32 = 192 elements to match AZ_NN_INPUT_DEPTH
    function boardToNNInput(boardState, currentPlayer = PLAYER_B) {
        if (DEBUG) console.log("boardToNNInput: input boardState (first row):", JSON.stringify(boardState[0])); // Log part of input board

        const flatBoardSize = ROWS * COLS; // 8 * 4 = 32
        const totalSize = 6 * flatBoardSize; // 6 * 32 = 192
        const nnInput = new Float32Array(totalSize);

        // Initialize all channels to 0
        for (let i = 0; i < totalSize; i++) {
            nnInput[i] = 0.0;
        }

        // Process each cell of the board
        for (let r = 0; r < ROWS; r++) {
            for (let c = 0; c < COLS; c++) {
                const pos = r * COLS + c; // Position in flattened board (0-31)
                const pieceData = boardState[r][c];

                if (pieceData) {
                    if (pieceData.player === PLAYER_A) {
                        if (pieceData.state === NORMAL) {
                            // Channel 0: Player A Normal pieces
                            nnInput[pos] = 1.0;
                        } else { // SWAPPED
                            // Channel 1: Player A Swapped pieces  
                            nnInput[flatBoardSize + pos] = 1.0;
                        }
                    } else { // PLAYER_B
                        if (pieceData.state === NORMAL) {
                            // Channel 2: Player B Normal pieces
                            nnInput[2 * flatBoardSize + pos] = 1.0;
                        } else { // SWAPPED
                            // Channel 3: Player B Swapped pieces
                            nnInput[3 * flatBoardSize + pos] = 1.0;
                        }
                    }
                } else {
                    // Channel 4: Empty cells
                    nnInput[4 * flatBoardSize + pos] = 1.0;
                }
            }
        }

        // Channel 5: Current player (all positions)
        // Fill all positions in channel 5 with 0.0 for Player A, 1.0 for Player B
        const playerValue = (currentPlayer === PLAYER_A) ? 0.0 : 1.0;
        for (let pos = 0; pos < flatBoardSize; pos++) {
            nnInput[5 * flatBoardSize + pos] = playerValue;
        }
        if (DEBUG) {
            console.log("boardToNNInput: output nnInput length:", nnInput.length);
            console.log("boardToNNInput: first 10 values:", Array.from(nnInput.slice(0, 10)));
            console.log("boardToNNInput: last 10 values:", Array.from(nnInput.slice(-10)));
        }
        return nnInput;
    }

    // Neural Network Prediction Function for MCTS
    // Takes NN input array and returns {policy, value}
    async function neuralNetworkPredict(nnInput) {
        if (!tfModel) {
            throw new Error("TensorFlow model not loaded");
        }

        if (!nnInput || nnInput.length !== 192) {
            throw new Error(`Invalid NN input: expected 192 elements, got ${nnInput?.length}`);
        }

        let inputTensor;
        let outputTensors;

        try {
            // Create tensor with batch dimension [1, 192]
            inputTensor = tf.tensor2d([nnInput]);

            // Use executeAsync for GraphModel prediction
            const inputNodeName = tfModel.inputs[0].name;
            outputTensors = await tfModel.executeAsync({ [inputNodeName]: inputTensor });

            if (outputTensors.length < 2) {
                throw new Error(`Expected 2 output tensors (value, policy), got ${outputTensors.length}`);
            }

            const valueOutputTensor = outputTensors[0]; // Shape [1, 1]
            const policyOutputTensor = outputTensors[1]; // Shape [1, 256]

            // Extract value and policy data
            const valueData = await valueOutputTensor.array();
            const policyData = await policyOutputTensor.array();

            const value = valueData[0][0]; // Single value
            const policy = policyData[0]; // Array of 256 policy values

            // Clean up tensors
            inputTensor.dispose();
            outputTensors.forEach(tensor => tensor.dispose());

            return {
                value: value,
                policy: policy
            };

        } catch (error) {
            // Clean up tensors in case of error
            if (inputTensor && !inputTensor.isDisposed) {
                inputTensor.dispose();
            }
            if (outputTensors) {
                outputTensors.forEach(tensor => {
                    if (tensor && !tensor.isDisposed) {
                        tensor.dispose();
                    }
                });
            }
            throw error;
        }
    } async function triggerAIMove() {
        if (gameOver) return;
        console.log(`AI (Player ${currentPlayer}) is thinking...`);

        const spinnerDelay = 1500; // Time before showing spinner
        const spinnerMinDisplay = 1000; // Minimum time spinner is shown
        let spinnerShowTimeout;
        let spinnerIsVisible = false;
        const startTime = performance.now();

        // Schedule spinner to show after the delay
        spinnerShowTimeout = setTimeout(() => {
            showOverlay(aiSpinnerOverlay);
            spinnerIsVisible = true;
            console.log("Showing AI spinner...");
        }, spinnerDelay);

        // Use a Promise to simulate the async nature of findBestAIMove
        // In a real scenario, findBestAIMove might be async or use a Worker
        new Promise((resolve, reject) => { // Added reject for error handling
            // Simulate AI thinking time (replace with actual call)
            // Using setTimeout to ensure it's async and doesn't block the main thread
            setTimeout(() => {
                try { // Added try...catch for errors within findBestAIMove
                    const bestMove = findBestAIMove(board, currentPlayer);
                    resolve(bestMove);
                } catch (err) {
                    reject(err); // Reject promise if findBestAIMove throws error
                }
            }, 10); // Small delay to yield thread, actual calculation happens in findBestAIMove
        }).then(bestMove => {
            const endTime = performance.now();
            const duration = endTime - startTime;
            console.log(`AI finished in ${duration.toFixed(0)}ms.`);

            // Clear the timeout that would show the spinner (it might not have run yet)
            clearTimeout(spinnerShowTimeout);

            const executeMoveAction = () => {
                if (bestMove) {
                    console.log("AI chooses move:", bestMove);
                    selectPiece(bestMove.start.row, bestMove.start.col);
                    const currentLegalMoves = calculateLegalMoves(bestMove.start.row, bestMove.start.col);
                    if (currentLegalMoves.some(m => m.row === bestMove.end.row && m.col === bestMove.end.col)) {
                        makeMove(bestMove.start.row, bestMove.start.col, bestMove.end.row, bestMove.end.col);
                        return;
                    } else {
                        console.error("AI Logic Error: Chosen move is not legal after re-selection?", bestMove, currentLegalMoves);
                        deselectPiece();
                        renderBoard();
                        console.warn("AI failed to make a legal move. Switching back to Player A.");
                        switchPlayer();
                    }
                } else {
                    console.warn("AI has no legal moves!");
                    switchPlayer();
                }
            };

            // Decide whether to show/hide spinner based on actual duration
            const timeSpinnerShouldHaveStarted = startTime + spinnerDelay;

            if (duration >= spinnerDelay) {
                // AI took longer than the delay. Spinner should show or was shown.
                console.log(`AI took ${duration.toFixed(0)}ms (>= ${spinnerDelay}ms delay). Spinner should show.`);

                // Show the spinner now if it wasn't shown already (i.e., AI finished before initial timeout fired)
                if (!spinnerIsVisible) { // Check the flag, not the classList yet
                    showOverlay(aiSpinnerOverlay);
                    spinnerIsVisible = true; // Mark as visible now
                    console.log("Showing AI spinner retroactively...");
                    // Since it's shown now, it needs to stay for the full min display time
                    setTimeout(() => {
                        hideOverlay(aiSpinnerOverlay);
                        console.log("Hiding AI spinner after minimum display time (retroactive).");
                        executeMoveAction();
                    }, spinnerMinDisplay);
                } else {
                    // Spinner was already shown by the initial timeout because duration > spinnerDelay and timeout fired
                    const alreadyShownDuration = endTime - timeSpinnerShouldHaveStarted;
                    const remainingSpinnerTime = Math.max(0, spinnerMinDisplay - alreadyShownDuration);
                    console.log(`Spinner already shown for ${alreadyShownDuration.toFixed(0)}ms. Remaining display time: ${remainingSpinnerTime.toFixed(0)}ms`);

                    // Ensure spinner stays for the required remaining time
                    setTimeout(() => {
                        hideOverlay(aiSpinnerOverlay);
                        console.log("Hiding AI spinner after minimum display time (proactive).");
                        executeMoveAction();
                    }, remainingSpinnerTime);
                }

            } else {
                // AI finished before the delay, spinner timeout was cleared, spinner never shown.
                console.log(`AI finished in ${duration.toFixed(0)}ms (< ${spinnerDelay}ms delay). Executing move immediately.`);
                // Ensure spinner is hidden just in case (shouldn't be needed as timeout was cleared)
                hideOverlay(aiSpinnerOverlay);
                executeMoveAction();
            }
        }).catch(error => {
            console.error("Error during AI move calculation:", error);
            // Handle error: maybe hide spinner, switch player, show error message?
            clearTimeout(spinnerShowTimeout); // Ensure spinner timeout is cleared
            if (spinnerIsVisible) { // Use the flag, not classList, as it might not be added yet
                hideOverlay(aiSpinnerOverlay);
            }
            // Optionally switch back to player A
            console.warn("Error in AI turn. Switching back to Player A.");
            switchPlayer();
        });
    } async function findBestAIMove(boardState = board, player = PLAYER_B) {
        let possibleMoves = [];
        for (let r = 0; r < ROWS; r++) {
            for (let c = 0; c < COLS; c++) {
                if (boardState[r][c] && boardState[r][c].player === player) {
                    const moves = calculateLegalMovesForState(boardState, r, c);
                    moves.forEach(move => {
                        possibleMoves.push({
                            start: { row: r, col: c },
                            end: { row: move.row, col: move.col },
                            isSwap: !!move.isSwap
                        });
                    });
                }
            }
        }

        if (possibleMoves.length === 0) return null;        // First, check for immediate winning moves
        for (const move of possibleMoves) {
            const tempBoard = cloneBoard(boardState);
            const { start, end } = move; // Ensure `start` and `end` are destructured from `move`
            const movingPiece = tempBoard[start.row][start.col];
            const targetPiece = tempBoard[end.row][end.col];

            if (!targetPiece) {
                tempBoard[end.row][end.col] = { ...movingPiece };
                tempBoard[start.row][start.col] = null;
                unmarkPlayerSwapped(player, tempBoard);
            } else {
                tempBoard[end.row][end.col] = { ...movingPiece, state: SWAPPED };
                tempBoard[start.row][start.col] = { ...targetPiece, state: SWAPPED };
            } if (checkWinConditionForState(tempBoard, player).win) {
                if (ANALYSIS_MODE) {
                    console.log(`Found immediate winning move: ${start.row},${start.col} to ${end.row},${end.col}`);
                }
                return move;
            }
        }        // If using neural network model
        if (AI_DIFFICULTY === 'hard_ai' && tfModel) {
            console.log("Using neural network for move selection");

            // Use MCTS if enabled and available
            if (MCTS_ENABLED && mctsSearch && gameLogic) {
                console.log(`Using MCTS with ${MCTS_SIMULATIONS} simulations`);
                try {
                    const actionProbs = await mctsSearch.search(boardState, player, neuralNetworkPredict, gameLogic);

                    // Convert action probabilities to move
                    const legalMoves = [];
                    for (let r = 0; r < ROWS; r++) {
                        for (let c = 0; c < COLS; c++) {
                            if (boardState[r][c] && boardState[r][c].player === player) {
                                const moves = calculateLegalMovesForState(boardState, r, c);
                                moves.forEach(move => {
                                    legalMoves.push({
                                        start: { row: r, col: c },
                                        end: { row: move.row, col: move.col },
                                        isSwap: !!move.isSwap
                                    });
                                });
                            }
                        }
                    }

                    if (legalMoves.length > 0) {
                        // Select move based on MCTS action probabilities
                        let bestMove = null;
                        let bestProb = -1;

                        for (const move of legalMoves) {
                            const actionIndex = moveToActionIndex(move);
                            if (actionIndex !== null && actionIndex >= 0 && actionIndex < actionProbs.length) {
                                const prob = actionProbs[actionIndex];
                                if (prob > bestProb) {
                                    bestProb = prob;
                                    bestMove = move;
                                }
                            }
                        }

                        if (bestMove && ANALYSIS_MODE) {
                            console.log(`MCTS selected move: ${bestMove.start.row},${bestMove.start.col} to ${bestMove.end.row},${bestMove.end.col} (prob: ${bestProb.toFixed(4)})`);
                        }

                        if (bestMove) {
                            return bestMove;
                        }
                    }
                } catch (error) {
                    console.error("MCTS error, falling back to direct NN policy:", error);
                    // Fall through to direct neural network policy
                }
            }

            console.log("Using direct neural network policy");
            let bestScore = -Infinity;
            let scoredMoves = [];
            let topMoves = []; // Initialize topMoves// Get the neural network input from the CURRENT board state
            const currentNNInput = boardToNNInput(boardState, player); // Include currentPlayer parameter

            if (!currentNNInput || currentNNInput.length !== 192) {
                console.error("Failed to generate valid NN input from current board state. Expected 192 elements but got:", currentNNInput?.length, "Falling back.");
                // Fallback to heuristic AI if NN input generation fails
                AI_DIFFICULTY = 'hard1'; // Temporarily change difficulty
                const fallbackMove = await findBestAIMove(boardState); // Recurse with heuristic
                AI_DIFFICULTY = 'hard_ai'; // Change back
                return fallbackMove;
            }

            let inputTensor;
            let outputTensors;
            let policyQValues;

            try {                // Create a single tensor for the current board state input
                // Reshape to [1, 192] as the model expects a batch dimension
                inputTensor = tf.tensor2d([currentNNInput]);

                if (ANALYSIS_MODE) console.log("NN Input Tensor Shape:", inputTensor.shape);

                // Use executeAsync for GraphModel prediction
                const inputNodeName = tfModel.inputs[0].name;
                outputTensors = await tfModel.executeAsync({ [inputNodeName]: inputTensor });

                if (ANALYSIS_MODE) {
                    console.log("--- Model Output Inspection (Current State Prediction) ---");
                    console.log("tfModel.outputs:", tfModel.outputs);
                    if (tfModel.outputs && tfModel.outputs.length > 0) {
                        tfModel.outputs.forEach((outputNode, index) => {
                            console.log(`Model Output Node ${index}: name='${outputNode.name}', shape=${JSON.stringify(outputNode.shape)}`);
                        });
                    }
                    console.log(`Actual number of output tensors: ${outputTensors.length}`);
                    outputTensors.forEach((tensor, index) => {
                        console.log(`Output Tensor ${index}: shape=${JSON.stringify(tensor.shape)}, rank=${tensor.rank}, dtype=${tensor.dtype}`);
                    });
                    console.log("--- End Model Output Inspection ---");
                }

                const valueOutputTensor = outputTensors[0]; // Shape [1, 1]
                const policyOutputTensor = outputTensors[1]; // Shape [1, 256]

                if (!policyOutputTensor) {
                    console.error("Policy output tensor (outputTensors[1]) is undefined. Cannot proceed with NN.");
                    throw new Error("Policy tensor is missing, cannot use NN.");
                }

                // policyQValues will be an array like [[q0, q1, ..., q255]]
                const policyData = await policyOutputTensor.array();
                policyQValues = policyData[0]; // Extract the actual Q-values array

                if (ANALYSIS_MODE) {
                    console.log(`Predicted Value from NN: ${(await valueOutputTensor.array())[0][0]}`);
                    console.log(`Policy Q-Values (first 10 of ${policyQValues.length}):`, policyQValues.slice(0, 10));
                }

                // Dispose tensors as soon as their data is extracted
                if (valueOutputTensor && !valueOutputTensor.isDisposed) valueOutputTensor.dispose();
                if (policyOutputTensor && !policyOutputTensor.isDisposed) policyOutputTensor.dispose();
                if (inputTensor && !inputTensor.isDisposed) inputTensor.dispose();
                // Ensure all original outputTensors are disposed if not already
                outputTensors.forEach(t => { if (t && !t.isDisposed) t.dispose(); });


                // --- Process Scores for all possible moves from the CURRENT state ---
                for (const move of possibleMoves) {
                    const { start, end } = move;

                    // Create a temporary board to check if the move allows an opponent win
                    const tempBoard = cloneBoard(boardState);
                    const movingPiece = tempBoard[start.row][start.col];
                    const targetPiece = tempBoard[end.row][end.col]; if (!targetPiece) {
                        tempBoard[end.row][end.col] = { ...movingPiece };
                        tempBoard[start.row][start.col] = null;
                        unmarkPlayerSwapped(player, tempBoard);
                    } else {
                        tempBoard[end.row][end.col] = { ...movingPiece, state: SWAPPED };
                        tempBoard[start.row][start.col] = { ...targetPiece, state: SWAPPED };
                    }                    // Check if this move allows opponent to win
                    const opponent = player === PLAYER_A ? PLAYER_B : PLAYER_A;
                    if (allowsOpponentWin(tempBoard, opponent, AI_DEPTH)) { // Use proper AI_DEPTH for thorough checking
                        if (ANALYSIS_MODE) {
                            console.log(`NN: Skipping move ${start.row},${start.col} to ${end.row},${end.col}: allows opponent win`);
                        }
                        continue; // Skip moves that allow opponent to win
                    }

                    const actionIndex = moveToActionIndex(move);

                    if (actionIndex === null || actionIndex < 0 || actionIndex >= policyQValues.length) {
                        if (ANALYSIS_MODE) console.warn(`NN: Skipping move due to invalid actionIndex (${actionIndex}) for move:`, move);
                        continue;
                    }

                    const score = policyQValues[actionIndex];

                    if (typeof score === 'number' && !isNaN(score)) {
                        scoredMoves.push({ move, score });
                        if (score > bestScore) {
                            bestScore = score;
                        }
                        if (ANALYSIS_MODE) {
                            console.log(`NN Move ${start.row},${start.col} to ${end.row},${end.col} (Action ${actionIndex}): Q-value ${score.toFixed(4)}`);
                        }
                    } else {
                        if (ANALYSIS_MODE) {
                            console.warn(`NN Move ${start.row},${start.col} to ${end.row},${end.col} (Action ${actionIndex}): Q-value is invalid (Score: ${score}, Type: ${typeof score})`);
                        }
                    }
                }

                if (scoredMoves.length > 0 && bestScore > -Infinity) {
                    // Use a threshold relative to the best score.
                    // Example: moves within 20% of the best score are considered.
                    // Adjust the 0.2 (20%) factor as needed.
                    // If bestScore is negative, threshold should be higher (less negative).
                    // If bestScore is positive, threshold should be lower.
                    let dynamicScoreThreshold;
                    if (bestScore >= 0) {
                        dynamicScoreThreshold = bestScore * 0.8; // e.g. if best is 1.0, threshold is 0.8
                    } else {
                        dynamicScoreThreshold = bestScore * 1.2; // e.g. if best is -1.0, threshold is -1.2 (scores > -1.2)
                    }
                    // A simpler way: bestScore - Math.abs(bestScore * 0.20)
                    // For very small scores, this might not be ideal. Let's use a small absolute tolerance too.
                    const scoreTolerance = 0.05; // Add a small absolute tolerance
                    dynamicScoreThreshold = bestScore - Math.abs(bestScore * 0.20) - scoreTolerance;


                    if (ANALYSIS_MODE) console.log(`NN: Final bestScore: ${bestScore.toFixed(4)}, dynamicScoreThreshold: ${dynamicScoreThreshold.toFixed(4)}`);

                    topMoves = scoredMoves
                        .filter(sm => typeof sm.score === 'number' && !isNaN(sm.score) && sm.score >= dynamicScoreThreshold)
                        .map(({ move }) => move);

                    if (ANALYSIS_MODE && topMoves.length === 0 && scoredMoves.length > 0) {
                        console.log("NN: No moves met the dynamic threshold, but scored moves exist. Considering all scored moves as top moves.");
                        // Fallback: if no moves meet the threshold but there are scored moves, consider all of them.
                        // Or, more simply, take the one(s) with the absolute bestScore.
                        topMoves = scoredMoves.filter(sm => sm.score === bestScore).map(({ move }) => move);
                    }

                } else {
                    if (ANALYSIS_MODE) console.log("NN: No moves scored or bestScore remained -Infinity.");
                }

                if (topMoves.length > 0) {
                    const randomIndex = Math.floor(Math.random() * topMoves.length);
                    if (ANALYSIS_MODE) console.log(`NN chose from ${topMoves.length} top moves. Best Q-value: ${bestScore.toFixed(4)}`);
                    return topMoves[randomIndex];
                } else {
                    console.warn("Neural network failed to find any moves above threshold or any valid moves. Falling back to heuristic.");
                    // Fall through to heuristic AI logic
                }

            } catch (error) {
                console.error("Error in neural network prediction or processing:", error);
                // Dispose any tensors that might still be around from the try block
                if (inputTensor && !inputTensor.isDisposed) inputTensor.dispose();
                if (outputTensors) {
                    outputTensors.forEach(t => { if (t && !t.isDisposed) t.dispose(); });
                }
                // Fall through to heuristic AI logic
            }
        }

        // --- Fallback Heuristic AI Logic (easy, hard1, or if NN fails) ---
        console.log(`Falling back to heuristic AI: ${AI_DIFFICULTY}`);

        if (AI_DIFFICULTY === 'easy') {
            // Filter out moves that allow immediate opponent win
            let safeMoves = [];
            for (const move of possibleMoves) {
                // Test each move on a cloned board
                const tempBoard = cloneBoard(boardState);
                const { start, end } = move;

                // Apply move to temporary board
                const movingPiece = tempBoard[start.row][start.col];
                const targetPiece = tempBoard[end.row][end.col]; if (!targetPiece) {
                    tempBoard[end.row][end.col] = { ...movingPiece };
                    tempBoard[start.row][start.col] = null;
                    unmarkPlayerSwapped(player, tempBoard);
                } else {
                    tempBoard[end.row][end.col] = { ...movingPiece, state: SWAPPED };
                    tempBoard[start.row][start.col] = { ...targetPiece, state: SWAPPED };
                }                // Check if this move allows opponent to win on their next turn(s)
                // Use allowsOpponentWin with proper AI_DEPTH for consistency
                const opponent = player === PLAYER_A ? PLAYER_B : PLAYER_A;
                const isSafe = !allowsOpponentWin(tempBoard, opponent, AI_DEPTH);
                if (ANALYSIS_MODE) {
                    console.log(`Evaluating move: ${start.row},${start.col} to ${end.row},${end.col} - Safe: ${isSafe}`);
                }
                if (isSafe) {
                    safeMoves.push(move);
                }
            }
            if (ANALYSIS_MODE) {
                console.log('Safe moves:', safeMoves.map(m => `${m.start.row},${m.start.col} to ${m.end.row},${m.end.col}`));
                if (safeMoves.length === 0) {
                    console.log('No safe moves found; selecting from all possible moves.');
                }
            }
            const movesToChoose = safeMoves.length > 0 ? safeMoves : possibleMoves;
            return movesToChoose[Math.floor(Math.random() * movesToChoose.length)];
        }

        let bestScore = -Infinity;
        let bestMoves = []; // Store moves with the best score

        for (const move of possibleMoves) {
            // Create proper deep clone of board
            const tempBoard = board.map(row =>
                row.map(cell => cell ? { ...cell } : null)
            );

            const movingPiece = tempBoard[move.start.row][move.start.col];
            const targetPiece = tempBoard[move.end.row][move.end.col];

            if (targetPiece === null) {
                // Empty cell move
                tempBoard[move.end.row][move.end.col] = { ...movingPiece }; // Use spread to clone
                tempBoard[move.start.row][move.start.col] = null; // Fix: use move.start instead of start                unmarkPlayerSwapped(player, tempBoard); 
            } else {
                // Swap move
                tempBoard[move.end.row][move.end.col] = { ...movingPiece, state: SWAPPED };
                tempBoard[move.start.row][move.start.col] = { ...targetPiece, state: SWAPPED };
            }

            const score = evaluateBoardState(tempBoard, player, move); // << Uses updated evaluateBoardState             // Check if this move allows opponent to win
            const opponent = player === PLAYER_A ? PLAYER_B : PLAYER_A;
            if (!allowsOpponentWin(tempBoard, opponent, AI_DEPTH)) { // Use proper AI_DEPTH for thorough checking
                if (score > bestScore) {
                    bestScore = score;
                    bestMoves = [move];
                } else if (score === bestScore) {
                    bestMoves.push(move);
                }
            } else {
                console.log(`AI Avoids move: ${move.start.row},${move.start.col} -> ${move.end.row},${move.end.col} (allows Player A win)`);
                // If all moves lead to a loss, the AI will have to pick one eventually
                // We could give these moves a massive penalty instead of excluding them
                // For now, let's try excluding them unless no other options exist.
                // Revisit this if AI gets stuck. Let's add them back with a huge penalty.
                const lossPenalty = -100000;
                if (lossPenalty > bestScore) {
                    bestScore = lossPenalty;
                    bestMoves = [move];
                } else if (lossPenalty === bestScore) {
                    bestMoves.push(move);
                }
            }
        }

        // Check if all evaluated moves lead to a loss
        const lossPenalty = -100000; // Ensure this matches the penalty value used above
        if (bestMoves.length > 0 && bestScore === lossPenalty) {
            console.log("AI: All moves lead to a loss. Forced move.");
        }

        // If bestMoves is empty (maybe all moves led to immediate loss?), pick any move?
        // This shouldn't happen if we add the losing moves with a penalty.
        if (bestMoves.length === 0) {
            console.warn("AI couldn't find a non-losing move, or evaluation error. Picking random move.");
            // Ensure possibleMoves isn't empty before picking random
            if (possibleMoves.length > 0) {
                return possibleMoves[Math.floor(Math.random() * possibleMoves.length)];
            } else {
                return null; // Truly no moves available
            }
        }


        // Choose randomly among the best moves
        return bestMoves[Math.floor(Math.random() * bestMoves.length)];
    }


    // (allowsOpponentWin simulation logic remains the same, but calls updated checkWinConditionForState)
    function allowsOpponentWin(boardState, opponentPlayer, depth = AI_DEPTH, alpha = -Infinity, beta = Infinity) {
        if (depth <= 0) return false;

        // Early exit - check immediate win
        if (checkWinConditionForState(boardState, opponentPlayer).win) {
            if (ANALYSIS_MODE) {
                console.log('Opponent has already won in this state.');
            }
            return true;
        }

        // Check transposition table using serialized key
        const boardKey = serializeBoardState(boardState);
        if (transpositionTable.has(boardKey)) {
            const cachedEntry = transpositionTable.get(boardKey);
            if (cachedEntry.depth >= depth) {
                if (ANALYSIS_MODE) {
                    console.log(`Using cached result for board state: ${boardKey}, depth: ${cachedEntry.depth}, value: ${cachedEntry.value}`);
                }
                return cachedEntry.value;
            }
        }

        // Get all opponent's possible moves
        let opponentMoves = [];
        for (let r = 0; r < ROWS; r++) {
            for (let c = 0; c < COLS; c++) {
                if (boardState[r][c]?.player === opponentPlayer) {
                    const moves = calculateLegalMovesForState(boardState, r, c);
                    moves.forEach(move =>
                        opponentMoves.push({ start: { row: r, col: c }, end: move })
                    );
                }
            }
        }

        // Move ordering optimization: Prioritize moves that might lead to a win
        let immediateWinMoves = [];
        let otherMoves = [];
        let moveStates = new Map();

        for (const move of opponentMoves) {
            const { start, end } = move;
            const afterOpponentMove = cloneBoard(boardState);
            const movingPiece = afterOpponentMove[start.row][start.col];
            const targetPiece = afterOpponentMove[end.row][end.col];

            if (!targetPiece) {
                afterOpponentMove[end.row][end.col] = { ...movingPiece };
                afterOpponentMove[start.row][start.col] = null;
                unmarkPlayerSwapped(opponentPlayer, afterOpponentMove);
            } else {
                afterOpponentMove[end.row][end.col] = { ...movingPiece, state: SWAPPED };
                afterOpponentMove[start.row][start.col] = { ...targetPiece, state: SWAPPED };
            }
            moveStates.set(move, afterOpponentMove);

            if (checkWinConditionForState(afterOpponentMove, opponentPlayer).win) {
                immediateWinMoves.push(move);
            } else {
                otherMoves.push(move);
            }
        }

        const orderedOpponentMoves = [...immediateWinMoves, ...otherMoves];

        // Check each move using the ordered list
        for (const move of orderedOpponentMoves) {
            const afterOpponentMove = moveStates.get(move);
            if (checkWinConditionForState(afterOpponentMove, opponentPlayer).win) {
                if (ANALYSIS_MODE) {
                    console.log(`Opponent can win with move: ${move.start.row},${move.start.col} to ${move.end.row},${move.end.col}`);
                }
                transpositionTable.set(boardKey, { value: true, depth: depth });
                return true;
            }

            // Recursive search for harder difficulties
            if ((AI_DIFFICULTY === 'hard1' || AI_DIFFICULTY === 'hard2') && depth > 1) {
                const currentPlayer = opponentPlayer === PLAYER_A ? PLAYER_B : PLAYER_A;
                if (!hasValidResponse(afterOpponentMove, currentPlayer, depth - 1, -beta, -alpha)) {
                    beta = Math.min(beta, 1);
                    if (beta <= alpha) {
                        transpositionTable.set(boardKey, { value: true, depth: depth });
                        return true;
                    }
                } else {
                    alpha = Math.max(alpha, -1);
                }
            }
        }

        transpositionTable.set(boardKey, { value: false, depth: depth });
        return false;
    }

    function hasValidResponse(boardState, currentPlayer, depth, alpha, beta) {
        if (depth <= 0) return true; // Assume safe if we've reached max depth
        const opponent = currentPlayer === PLAYER_A ? PLAYER_B : PLAYER_A;

        // Check immediate win conditions
        if (checkWinConditionForState(boardState, opponent).win) {
            return false; // Opponent can win, no valid response
        }

        // Generate all possible moves for current player
        let possibleMoves = [];
        for (let r = 0; r < ROWS; r++) {
            for (let c = 0; c < COLS; c++) {
                if (boardState[r][c]?.player === currentPlayer) {
                    const moves = calculateLegalMovesForState(boardState, r, c);
                    moves.forEach(move =>
                        possibleMoves.push({ start: { row: r, col: c }, end: move })
                    );
                }
            }
        }

        // Try each move and look deeper
        for (const move of possibleMoves) {
            const nextState = cloneBoard(boardState);
            const { start, end } = move;

            // Apply move
            const movingPiece = nextState[start.row][start.col];
            const targetPiece = nextState[end.row][end.col];

            if (!targetPiece) {
                nextState[end.row][end.col] = { ...movingPiece };
                nextState[start.row][start.col] = null;
                unmarkPlayerSwapped(currentPlayer, nextState);
            } else {
                nextState[end.row][end.col] = { ...movingPiece, state: SWAPPED };
                nextState[start.row][start.col] = { ...targetPiece, state: SWAPPED };
            }

            // Look at opponent's response
            if (!allowsOpponentWin(nextState, opponent, depth - 1, -beta, -alpha)) {
                return true; // Found at least one valid defensive line
            }
        }

        return false; // No valid defensive moves found
    }

    // (calculateLegalMovesForState remains the same)
    function calculateLegalMovesForState(boardState, r, c) {
        const moves = [];
        const piece = boardState[r][c];
        if (!piece) return moves;

        const opponent = piece.player === PLAYER_A ? PLAYER_B : PLAYER_A;

        for (let dr = -1; dr <= 1; dr++) {
            for (let dc = -1; dc <= 1; dc++) {
                if (dr === 0 && dc === 0) continue;
                const nr = r + dr;
                const nc = c + dc;

                if (nr >= 0 && nr < ROWS && nc >= 0 && nc < COLS) {
                    const targetCell = boardState[nr][nc];
                    if (targetCell === null) {
                        moves.push({ row: nr, col: nc });
                    } else if (targetCell.player === opponent && targetCell.state === NORMAL) {
                        moves.push({ row: nr, col: nc, isSwap: true });
                    }
                }
            }
        }
        return moves;
    }    // Helper to check win condition for a given board state - UPDATED start/target rows
    function checkWinConditionForState(boardState, player) {
        // Define starting and target areas for each player
        const startRows = (player === PLAYER_A) ? [ROWS - 2, ROWS - 1] : [0, 1];  // A: rows 6,7; B: rows 0,1
        const targetRows = (player === PLAYER_A) ? [0, 1] : [ROWS - 2, ROWS - 1]; // A: rows 0,1; B: rows 6,7

        const visited = Array(ROWS).fill(null).map(() => Array(COLS).fill(false));
        const queue = []; // Queue for BFS: stores {row, col, path}

        // Find all pieces in the starting area and add them to the queue
        for (const startRow of startRows) {
            for (let c = 0; c < COLS; c++) {
                if (boardState[startRow] && boardState[startRow][c] && boardState[startRow][c].player === player) {
                    queue.push({
                        row: startRow,
                        col: c,
                        path: [{ row: startRow, col: c }]
                    });
                    visited[startRow][c] = true;
                }
            }
        }

        while (queue.length > 0) {
            const current = queue.shift();
            const { row, col, path } = current;

            // Check if we reached any target row
            if (targetRows.includes(row)) {
                console.log(`Win detected for Player ${player}. Path:`, path);
                return { win: true, path: path }; // Found a path
            }

            // Explore neighbors
            for (let dr = -1; dr <= 1; dr++) {
                for (let dc = -1; dc <= 1; dc++) {
                    if (dr === 0 && dc === 0) continue;

                    const nr = row + dr;
                    const nc = col + dc;

                    // Check bounds, if visited, and if it's the player's piece
                    if (nr >= 0 && nr < ROWS && nc >= 0 && nc < COLS &&
                        !visited[nr][nc] &&
                        boardState[nr] && boardState[nr][nc] && boardState[nr][nc].player === player) {
                        visited[nr][nc] = true;
                        queue.push({ row: nr, col: nc, path: [...path, { row: nr, col: nc }] }); // Include complete path to current position
                    }
                }
            }
        }

        return { win: false, path: [] };
    }


    // --- Heuristic Evaluation --- UPDATED for new orientation
    function evaluateBoardState(boardState, player, move) {
        let score = 0;
        const opponent = player === PLAYER_A ? PLAYER_B : PLAYER_A;

        // 1. Check for immediate win for AI (highest priority)
        if (checkWinConditionForState(boardState, player).win) {
            return 1000000; // Very high score for winning move
        }
        // Check for immediate win for opponent (if AI allows it - this check is slightly redundant
        // with allowsOpponentWin but can be a quick penalty if needed, though allowsOpponentWin is better)
        // if (checkWinConditionForState(boardState, opponent).win) {
        //     return -1000000; // Massive penalty if the move leads to immediate loss
        // }


        // Heuristic components (Weights need tuning!)
        const ADVANCE_WEIGHT = 10;
        const BACK_ROW_PENALTY = -2; // Penalty for Player B being on row 0
        const CONNECTIVITY_WEIGHT = 5;
        const FOUR_IN_ROW_PENALTY = -4;
        const SWAP_BONUS = 2; // Bonus for making a swap
        const RESTRICT_OPPONENT_WEIGHT = 3;
        const CENTER_CONTROL_WEIGHT = 1; // Slight bonus for pieces near center cols (1, 2)
        const CENTER_SQUARES_WEIGHT = 1; // Slight bonus rows (3,4)
        const ROW_COUNT_WEIGHT = 2;

        let playerCount = 0;
        let opponentCount = 0;
        let playerConnectivity = 0;
        let opponentConnectivity = 0; // Could be used for defense

        for (let r = 0; r < ROWS; r++) {
            let playerHorizontalRow = 0; // Track horizontal pieces for penalty

            for (let c = 0; c < COLS; c++) {
                const piece = boardState[r][c];
                if (piece) {
                    if (piece.player === player) { // Evaluating for Player B
                        playerCount++;
                        playerHorizontalRow++;

                        // Advancement Score (for Player B, HIGHER row index is better)
                        score += ADVANCE_WEIGHT * r; // <<<< CHANGED

                        // Back Row Penalty (for Player B, row index 0)
                        if (r === 0) { // <<<< CHANGED
                            score += BACK_ROW_PENALTY;
                        }

                        // Connectivity Score only do this count for pieces mid-board
                        if (r > 1 && r < ROWS - 1) {
                            playerConnectivity += countFriendlyNeighbors(boardState, r, c, player);
                        }

                        // Center Control
                        if (c === 1 || c === 2) {
                            score += CENTER_CONTROL_WEIGHT;
                        }

                        if (r === 3 || r === 4) {
                            score += CENTER_SQUARES_WEIGHT;
                        }

                    } else { // Opponent piece (Player A)
                        opponentCount++;
                        playerHorizontalRow = 0; // Reset count on opponent piece
                        if (r > 1 && r < ROWS - 1) {
                            opponentConnectivity += countFriendlyNeighbors(boardState, r, c, opponent);
                        }
                        // Could add opponent advancement penalty here (A wants lower index)
                        // score -= ADVANCE_WEIGHT * (ROWS - 1 - r); // Penalize AI if opponent advances
                    }
                } else {
                    playerHorizontalRow = 0; // Reset count on empty cell
                }

                // 4-in-a-row Penalty (only check if count reaches 4)
                if (playerHorizontalRow === 4) {
                    score += FOUR_IN_ROW_PENALTY;
                }

            }
            playerHorizontalRow = 0; // Reset at end of row
        }

        // Add connectivity scores
        score += CONNECTIVITY_WEIGHT * playerConnectivity;
        // score -= CONNECTIVITY_WEIGHT * opponentConnectivity;

        // Swap Bonus/Utility
        if (move.isSwap) {
            score += SWAP_BONUS;
            // Potential further analysis: did the swap block a key opponent piece?
            // Or did it open up a path for the AI? (More complex analysis)
        }


        // Opponent Restriction (Approximate) - Count opponent's legal moves
        let opponentMovesCount = 0;
        for (let r = 0; r < ROWS; r++) {
            for (let c = 0; c < COLS; c++) {
                if (boardState[r][c] && boardState[r][c].player === opponent) {
                    opponentMovesCount += calculateLegalMovesForState(boardState, r, c).length;
                }
            }
        }
        score -= RESTRICT_OPPONENT_WEIGHT * opponentMovesCount;


        let rowsOccupiedCount = 0;
        for (let r = 0; r < ROWS; r++) {
            for (let c = 0; c < COLS; c++) {
                if (boardState[r][c] && boardState[r][c].player === player) {
                    rowsOccupiedCount += 1;
                }
            }
        }
        // score -= ROW_COUNT_WEIGHT * rowsOccupiedCount;

        // Add small random factor to break ties sometimes
        score += Math.random() * 0.1;

        return score;
    }

    // (countFriendlyNeighbors remains the same)
    function countFriendlyNeighbors(boardState, r, c, player) {
        let count = 0;
        for (let dr = -1; dr <= 1; dr++) {
            for (let dc = -1; dc <= 1; dc++) {
                if (dr === 0 && dc === 0) continue;
                const nr = r + dr;
                const nc = c + dc;
                if (nr >= 0 && nr < ROWS && nc >= 0 && nc < COLS &&
                    boardState[nr][nc] && boardState[nr][nc].player === player) {
                    count++;
                }
            }
        }
        return count;
    }


    // --- History --- UPDATED row display calculation

    function displayMoveHistory() {
        historyList.innerHTML = ''; // Clear previous list
        if (moveHistory.length === 0) {
            historyList.textContent = 'No moves made yet.';
            return;
        }

        // Display in reverse order (most recent first)
        [...moveHistory].reverse().forEach((move, index) => {
            const moveDiv = document.createElement('div');
            const moveNumber = moveHistory.length - index;
            // Convert internal row (0-7) to display row (8-1) - REVERSED
            const startRowDisplay = ROWS - move.start.row; // <<<< CHANGED
            const endRowDisplay = ROWS - move.end.row;     // <<<< CHANGED
            // Convert col (0-3) to display col (A-D)
            const startColDisplay = String.fromCharCode('A'.charCodeAt(0) + move.start.col);
            const endColDisplay = String.fromCharCode('A'.charCodeAt(0) + move.end.col);

            moveDiv.textContent = `${moveNumber}. Player ${move.player}: ${startColDisplay}${startRowDisplay} -> ${endColDisplay}${endRowDisplay}`;
            moveDiv.classList.add('history-move');
            // Highlight current move being viewed
            if (currentHistoryIndex === moveHistory.length - index) {
                moveDiv.classList.add('selected-move');
            }
            moveDiv.addEventListener('click', () => {
                currentHistoryIndex = moveHistory.length - index;
                renderBoard(move.boardAfter);
                if (currentHistoryIndex === moveHistory.length) {
                    winPath = checkWinCondition(currentPlayer).path || [];
                }
                updateMoveCounter();
                hideOverlay(historyOverlay);
            });
            historyList.appendChild(moveDiv);
        });

        // Add option to view initial state
        const initialStateDiv = document.createElement('div');
        initialStateDiv.textContent = `0. Initial State`;
        initialStateDiv.classList.add('history-move');
        // Highlight initial state if it's currently viewed
        if (currentHistoryIndex === 0) {
            initialStateDiv.classList.add('selected-move');
        }
        initialStateDiv.addEventListener('click', () => {
            const initialBoard = startingPosition.map(row => row.map(cell => cell ? { ...cell } : null));
            currentHistoryIndex = 0;
            renderBoard(initialBoard);
            updateMoveCounter();
            hideOverlay(historyOverlay);
        });
        historyList.appendChild(initialStateDiv); // Add initial state option at the end
    }

    function navigateHistory(direction) {
        if (!gameOver) return; // Exit if not in history mode

        if (currentHistoryIndex === undefined) {
            currentHistoryIndex = moveHistory.length;
        }

        const newIndex = Math.max(0, Math.min(moveHistory.length, currentHistoryIndex + direction));
        if (newIndex === currentHistoryIndex) return; // No change needed

        currentHistoryIndex = newIndex;

        if (currentHistoryIndex === moveHistory.length) {
            renderBoard(board);
        } else if (currentHistoryIndex === 0) {
            // Show initial board state
            const initialBoard = startingPosition.map(row => row.map(cell => cell ? { ...cell } : null));
            renderBoard(initialBoard);
        } else {
            renderBoard(moveHistory[currentHistoryIndex - 1].boardAfter);
        }
        updateMoveCounter();
    }

    function updateMoveCounter() {
        if (!gameOver) return;

        const counter = document.createElement('div');
        counter.classList.add('move-counter');
        counter.textContent = currentHistoryIndex === undefined || currentHistoryIndex === moveHistory.length ?
            'Final Position' :
            `Move ${currentHistoryIndex} of ${moveHistory.length}`;

        // Remove any existing counter
        const existingCounter = boardElement.querySelector('.move-counter');
        if (existingCounter) {
            existingCounter.remove();
        }
        boardElement.appendChild(counter);
    }

    function restoreFinalState() {
        currentHistoryIndex = undefined;
        boardElement.style.opacity = '1';
        renderBoard(board);
    }

    // Update board click handler for history mode
    boardElement.addEventListener('click', () => {
        if (currentHistoryIndex !== undefined) {
            restoreFinalState();
        }
    });

    // --- Overlay Management ---
    // (showOverlay, hideOverlay, hideAllOverlays remain the same)
    function showOverlay(overlayElement) {
        overlayElement.classList.add('active');
    }

    function hideOverlay(overlayElement) {
        overlayElement.classList.remove('active');
    }
    function hideAllOverlays() {
        overlays.forEach(hideOverlay);
    }

    // --- Start Game ---
    initGame();

    // Add scroll wheel handler
    boardElement.addEventListener('wheel', (event) => {
        if (!gameOver) return;

        event.preventDefault();
        const direction = event.deltaY > 0 ? 1 : -1;
        navigateHistory(direction);
    });

    // Add touch handlers for mobile
    boardElement.addEventListener('touchstart', (event) => {
        if (!gameOver) return;
        touchStartY = event.touches[0].clientY;
    });

    boardElement.addEventListener('touchmove', (event) => {
        if (!gameOver) return;
        event.preventDefault();

        const touchEndY = event.touches[0].clientY;
        const deltaY = touchEndY - touchStartY;

        if (Math.abs(deltaY) > 30) { // Minimum swipe distance
            const direction = deltaY < 0 ? 1 : -1;
            navigateHistory(direction);
            touchStartY = touchEndY;
        }
    });

    function navigateHistory(direction) {
        if (currentHistoryIndex === undefined) {
            currentHistoryIndex = moveHistory.length;
        }

        currentHistoryIndex = Math.max(0, Math.min(moveHistory.length, currentHistoryIndex + direction));

        if (currentHistoryIndex === moveHistory.length) {
            restoreFinalState();
        } else if (currentHistoryIndex === 0) {
            // Show initial board state with experimental positions:
            const initialBoard = startingPosition.map(row => row.map(cell => cell ? { ...cell } : null));
            renderBoard(initialBoard);
        } else {
            renderBoard(moveHistory[currentHistoryIndex - 1].boardAfter);
        }
        updateMoveCounter();
    }

    function updateScoreDisplay() {
        // Update control scores
        const controlScoreA = document.getElementById('control-score-a');
        const controlScoreB = document.getElementById('control-score-b'); // Corrected typo: getElementById
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

    // Utility to deep clone boardState
    function cloneBoard(boardState) {
        // Use structuredClone if available (modern browsers), else fallback to JSON method
        return (typeof structuredClone === 'function') ?
            structuredClone(boardState) :
            JSON.parse(JSON.stringify(boardState));
    } function boardToKey(boardState) {
        // Create a simpler string representation that's JSON-safe
        return boardState.map(row =>
            row.map(cell =>
                cell ? `${cell.player}${cell.state[0]}` : '_'  // Use first letter of state
            ).join('')
        ).join('|');
    }

    // --- You MUST implement this function based on your Python code ---
    function convertBoardToBinaryJS(boardState) {
        // This function needs to take the 8x4 boardState and produce
        // an array of 5 numbers (integers) exactly like the Python
        // board_to_binary function does (likely using bitwise operations).

        // Initialize 5 numbers (unsigned 32-bit integers)
        // JavaScript bitwise operations operate on 32-bit signed integers,
        // but for setting bits with OR, this should be fine.
        // The `>>> 0` operation can be used to treat a number as unsigned 32-bit if needed,
        // but it's mainly for the final result if it could be negative.
        let emptyMask = 0;
        let playerANormalMask = 0;
        let playerASwappedMask = 0;
        let playerBNormalMask = 0;
        let playerBSwappedMask = 0;

        for (let r = 0; r < ROWS; r++) {
            for (let c = 0; c < COLS; c++) {
                const pos = r * COLS + c; // COLS is 4, so r * 4 + c
                const piece = boardState[r][c];

                if (piece === null) { // Empty
                    emptyMask |= (1 << pos);
                } else if (piece.player === PLAYER_A) {
                    if (piece.state === NORMAL) { // Player A normal
                        playerANormalMask |= (1 << pos);
                    } else { // Player A swapped
                        playerASwappedMask |= (1 << pos);
                    }
                } else if (piece.player === PLAYER_B) {
                    if (piece.state === NORMAL) { // Player B normal
                        playerBNormalMask |= (1 << pos);
                    } else { // Player B swapped
                        playerBSwappedMask |= (1 << pos);
                    }
                }
            }
        }

        // Ensure results are treated as unsigned 32-bit integers if necessary,
        // though for positive results from ORing bits, it might not change much.
        return [
            emptyMask >>> 0,
            playerANormalMask >>> 0,
            playerASwappedMask >>> 0,
            playerBNormalMask >>> 0,
            playerBSwappedMask >>> 0
        ];
    }

    function serializeBoardState(boardState) {
        // Create a simple string representation that avoids JSON.parse/stringify errors
        return boardState.map(row =>
            row.map(cell => {
                if (!cell) return '_';
                return `${cell.player}${cell.state === NORMAL ? 'N' : 'S'}`;
            }).join('')
        ).join('|');
    } async function analyzeHistoricalMove() {
        if (currentHistoryIndex !== undefined && currentHistoryIndex > 0) {
            const historicalBoard = moveHistory[currentHistoryIndex - 1].boardAfter;

            // Determine whose turn it is for this historical board state
            // The boardAfter represents the state after a move was made
            // So the next player to move is the opposite of who made that move
            const playerWhoMadePreviousMove = moveHistory[currentHistoryIndex - 1].player;
            const playerToMoveNext = playerWhoMadePreviousMove === PLAYER_A ? PLAYER_B : PLAYER_A;

            ANALYSIS_MODE = true;
            const bestMove = await findBestAIMove(historicalBoard, playerToMoveNext);
            ANALYSIS_MODE = false;
            console.log(`Best move for this board state (Player ${playerToMoveNext}'s turn):`, bestMove);
        } else {
            console.log('No historical move selected.');
        }
    }
    window.analyzeHistoricalMove = analyzeHistoricalMove;

    function setupTestPosition() {
        // Clear the board first
        board = Array(ROWS).fill(null).map(() => Array(COLS).fill(null));

        // Setup the position from bottom to top (row 7 to 0)
        const position = [
            '....',  // row 0 (top)
            '.Bw.',
            '.wbb',
            'wbb.',
            'wbw.',
            'bbw.',
            'WW..',
            '....'   // row 7 (bottom)
        ];

        // Fill the board based on the test position
        for (let r = 0; r < ROWS; r++) {
            for (let c = 0; c < COLS; c++) {
                const piece = position[r][c];
                switch (piece) {
                    case 'w':
                        board[r][c] = { player: PLAYER_A, state: SWAPPED };
                        break;
                    case 'W':
                        board[r][c] = { player: PLAYER_A, state: NORMAL };
                        break;
                    case 'b':
                        board[r][c] = { player: PLAYER_B, state: SWAPPED };
                        break;
                    case 'B':
                        board[r][c] = { player: PLAYER_B, state: NORMAL };
                        break;
                }
            }
        }

        currentPlayer = PLAYER_B; // Set to Player B's turn
        selectedPiece = null;
        legalMoves = [];
        moveHistory = [];
        gameOver = false;
        winner = null;
        winPath = [];
        historyBtn.disabled = true;
        DEBUG = true; // Enable debug logging
        AI_DIFFICULTY = 'easy'; // Force easy mode for testing
        AI_DEPTH = 1;

        renderBoard();
        updateStatusMessage();
        hideAllOverlays();
        updateScoreDisplay();
        transpositionTable.clear();

        // Log the initial win condition check for both players
        const playerAWin = checkWinConditionForState(board, PLAYER_A);
        const playerBWin = checkWinConditionForState(board, PLAYER_B);
        console.log('Initial win check:', {
            playerA: playerAWin,
            playerB: playerBWin
        });

        // Trigger AI move since it's Player B's turn
        setTimeout(() => {
            triggerAIMove();
        }, 100);
    }

    // For testing in console
    window.setupTestPosition = setupTestPosition;    // --- MCTS configuration button
    const mctsBtn = document.getElementById('mcts-btn');
    const mctsOverlay = document.getElementById('mcts-overlay');

    mctsBtn.addEventListener('click', () => {
        showOverlay(mctsOverlay);
        updateMCTSUI();
    });

    // MCTS control functions
    function toggleMCTS() {
        MCTS_ENABLED = !MCTS_ENABLED;
        updateMCTSSettings();
        console.log(`MCTS ${MCTS_ENABLED ? 'enabled' : 'disabled'}`);
    }

    function setMCTSSimulations(value) {
        MCTS_SIMULATIONS = Math.max(1, Math.min(1000, parseInt(value)));
        updateMCTSSettings();
        if (MCTS_VERBOSE) {
            console.log(`MCTS simulations set to ${MCTS_SIMULATIONS}`);
        }
    }

    function setMCTSTemperature(value) {
        MCTS_TEMPERATURE = Math.max(0.0, Math.min(2.0, parseFloat(value)));
        updateMCTSSettings();
        if (MCTS_VERBOSE) {
            console.log(`MCTS temperature set to ${MCTS_TEMPERATURE}`);
        }
    }

    function updateMCTSSettings() {
        // Reinitialize MCTS with new settings if it exists
        if (typeof initializeMCTS === 'function') {
            initializeMCTS();
        }

        // Update window globals for debugging
        window.MCTS_ENABLED = MCTS_ENABLED;
        window.MCTS_SIMULATIONS = MCTS_SIMULATIONS;
        window.MCTS_TEMPERATURE = MCTS_TEMPERATURE;
        window.MCTS_VERBOSE = MCTS_VERBOSE;
    }    // MCTS control event handlers
    function setupMCTSControls() {
        const aiDifficultySelect = document.getElementById('ai-difficulty-select');
        const mctsEnabledCheckbox = document.getElementById('mcts-enabled');
        const mctsSimulationsSlider = document.getElementById('mcts-simulations');
        const mctsSimulationsValue = document.getElementById('mcts-simulations-value');
        const mctsTemperatureSlider = document.getElementById('mcts-temperature');
        const mctsTemperatureValue = document.getElementById('mcts-temperature-value');
        const mctsVerboseCheckbox = document.getElementById('mcts-verbose');
        const mctsResetBtn = document.getElementById('mcts-reset-defaults');

        // AI Difficulty selector
        aiDifficultySelect.addEventListener('change', (e) => {
            updateAIDifficulty(e.target.value);
        });

        // MCTS enabled/disabled
        mctsEnabledCheckbox.addEventListener('change', (e) => {
            if (AI_DIFFICULTY === 'hard_ai') {
                MCTS_ENABLED = e.target.checked;
                updateMCTSSettings();
            } else {
                // Force disable if not in AI Agent mode
                e.target.checked = false;
                MCTS_ENABLED = false;
            }
        });        // Simulations slider
        mctsSimulationsSlider.addEventListener('input', (e) => {
            const value = parseInt(e.target.value);
            setMCTSSimulations(value);
            if (mctsSimulationsValue) mctsSimulationsValue.textContent = value;
        });

        // Temperature slider
        mctsTemperatureSlider.addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            setMCTSTemperature(value);
            if (mctsTemperatureValue) mctsTemperatureValue.textContent = value.toFixed(2);
        });

        // Verbose logging
        mctsVerboseCheckbox.addEventListener('change', (e) => {
            MCTS_VERBOSE = e.target.checked;
            updateMCTSSettings();
        });        // Reset to defaults
        mctsResetBtn.addEventListener('click', () => {
            MCTS_ENABLED = true;
            MCTS_SIMULATIONS = 50;
            MCTS_TEMPERATURE = 0.1; // Better default for gameplay
            MCTS_VERBOSE = false; // Reset verbose to disabled by default
            updateMCTSSettings();
            updateMCTSUI(); // Update all UI elements including sliders
        });
    }

    // Update MCTS UI to reflect current settings
    function updateMCTSUI() {
        const aiDifficultySelect = document.getElementById('ai-difficulty-select');
        const mctsEnabledCheckbox = document.getElementById('mcts-enabled');
        const mctsSimulationsSlider = document.getElementById('mcts-simulations');
        const mctsSimulationsValue = document.getElementById('mcts-simulations-value');
        const mctsTemperatureSlider = document.getElementById('mcts-temperature');
        const mctsTemperatureValue = document.getElementById('mcts-temperature-value');
        const mctsVerboseCheckbox = document.getElementById('mcts-verbose');

        // Update AI difficulty selector
        if (aiDifficultySelect) aiDifficultySelect.value = AI_DIFFICULTY;

        // Update MCTS controls
        if (mctsEnabledCheckbox) mctsEnabledCheckbox.checked = MCTS_ENABLED;
        if (mctsSimulationsSlider) mctsSimulationsSlider.value = MCTS_SIMULATIONS;
        if (mctsSimulationsValue) mctsSimulationsValue.textContent = MCTS_SIMULATIONS; if (mctsTemperatureSlider) mctsTemperatureSlider.value = MCTS_TEMPERATURE;
        if (mctsTemperatureValue) mctsTemperatureValue.textContent = MCTS_TEMPERATURE.toFixed(2);
        if (mctsVerboseCheckbox) mctsVerboseCheckbox.checked = MCTS_VERBOSE;

        // Note: updateMCTSControlsVisibility() handles the visibility logic and calls updateMCTSUI()
        // So we don't call it here to avoid infinite recursion
    }// Initialize MCTS controls when DOM is ready
    setupMCTSControls();

    // Initialize MCTS controls visibility
    updateMCTSControlsVisibility();

    // Initialize MCTS button image
    updateMCTSButtonImage();

    // --- Debugging and Analysis Tools ---
    window.DEBUG = DEBUG;    // DISABLED: Legacy script.js global variables - using modular system instead
    // window.tfModel = tfModel;
    // window.AI_DIFFICULTY = AI_DIFFICULTY;
    // window.AI_DEPTH = AI_DEPTH;
    // window.MCTS_ENABLED = MCTS_ENABLED;
    // window.MCTS_SIMULATIONS = MCTS_SIMULATIONS;
    // window.MCTS_TEMPERATURE = MCTS_TEMPERATURE;
    // window.MCTS_PUCT_CONSTANT = MCTS_PUCT_CONSTANT;
    // window.MCTS_DIRICHLET_ALPHA = MCTS_DIRICHLET_ALPHA;
    // window.MCTS_DIRICHLET_EPSILON = MCTS_DIRICHLET_EPSILON;
    // window.MCTS_VERBOSE = MCTS_VERBOSE;
    // DISABLED: Legacy script.js global functions - using modular system instead
    // window.moveHistory = moveHistory;
    // window.currentPlayer = currentPlayer;
    // window.board = board;
    // window.selectPiece = selectPiece;
    // window.deselectPiece = deselectPiece;
    // window.makeMove = makeMove;
    // window.triggerAIMove = triggerAIMove;
    // window.findBestAIMove = findBestAIMove;
    // window.neuralNetworkPredict = neuralNetworkPredict;
    // window.allowsOpponentWin = allowsOpponentWin;
    // window.checkWinCondition = checkWinCondition;
    // window.serializeBoardState = serializeBoardState;
    // window.convertBoardToBinaryJS = convertBoardToBinaryJS;
    // window.cloneBoard = cloneBoard;
    // window.boardToKey = boardToKey;
    // window.parseStartingPosition = parseStartingPosition;
    // window.initGame = initGame;
    // window.renderBoard = renderBoard;
    // window.updateScoreDisplay = updateScoreDisplay;
    // window.displayMoveHistory = displayMoveHistory;
    // window.navigateHistory = navigateHistory;
    // window.toggleMCTS = toggleMCTS;
    // window.setMCTSSimulations = setMCTSSimulations;
    // window.setMCTSTemperature = setMCTSTemperature;
    // window.analyzeHistoricalMove = analyzeHistoricalMove;
    // window.setupTestPosition = setupTestPosition;
    // window.startSelfPlay = startSelfPlay;
    // window.stopSelfPlay = stopSelfPlay;    // Convenience toggle function - DISABLED
    // window.selfPlay = function () {
    //     if (isInSelfPlay) {
    //         stopSelfPlay();
    //     } else {
    //         startSelfPlay();
    //     }
    // };

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
    }    // Console commands info - DISABLED (using modular system)
    // console.log("🎮 Self-play console commands available:");
    // console.log("  startSelfPlay() - Start AI vs AI self-play");
    // console.log("  stopSelfPlay()  - Stop AI vs AI self-play");
    // console.log("  selfPlay()      - Toggle self-play on/off");
});