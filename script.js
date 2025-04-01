document.addEventListener('DOMContentLoaded', () => {
    const ROWS = 8;
    const COLS = 4;
    const PLAYER_A = 'A'; // White
    const PLAYER_B = 'B'; // Black
    const NORMAL = 'normal';
    const SWAPPED = 'swapped';
    let AI_DIFFICULTY = 'easy'; // easy, hard1, hard2
    let AI_DEPTH = 5; // Will be set based on difficulty
    let ANALYSIS_MODE = false;

    const boardElement = document.getElementById('game-board');
    const resetBtn = document.getElementById('reset-btn');
    const infoBtn = document.getElementById('info-btn');
    const historyBtn = document.getElementById('history-btn');
    const difficultyBtn = document.getElementById('difficulty-btn'); // Add this line
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
    let playerBScore = 0;
    let currentHistoryIndex = undefined;
    let touchStartY = 0;
    const transpositionTable = new Map();

    var DEBUG = false; // Set to false to disable
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
    // --- Initialization ---

    function initGame() {
        board = Array(ROWS).fill(null).map(() => Array(COLS).fill(null));
        currentPlayer = PLAYER_A;
        selectedPiece = null;
        legalMoves = [];
        moveHistory = [];
        gameOver = false;
        winner = null;
        winPath = [];

        // Place pieces - REVERSED ORIENTATION
        for (let r = 0; r < ROWS; r++) {
            for (let c = 0; c < COLS; c++) {
                if (r < 2) { // Player B (Black) at top (rows 0, 1 -> displayed as 8, 7)
                    board[r][c] = { player: PLAYER_B, state: NORMAL };
                } else if (r >= ROWS - 2) { // Player A (White) at bottom (rows 6, 7 -> displayed as 2, 1)
                    board[r][c] = { player: PLAYER_A, state: NORMAL };
                }
            }
        }

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
    }

     // --- Event Handlers ---
     // (No changes needed in handleCellClick, resetBtn, infoBtn, historyBtn listeners,
     // overlayCloseButtons, or overlay backdrop listeners)
    function handleCellClick(row, col) {
        if (gameOver) return; // No moves after game ends (unless viewing history)
        if (currentPlayer === PLAYER_B) return; // Block human clicks during AI turn

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
    }

    resetBtn.addEventListener('click', initGame);
    infoBtn.addEventListener('click', () => showOverlay(infoOverlay));
    historyBtn.addEventListener('click', () => {
        if (!historyBtn.disabled) {
            displayMoveHistory();
            showOverlay(historyOverlay);
        }
    });

    // Replace the difficultyBtn click handler with this new version
    difficultyBtn.addEventListener('click', () => {
        switch(AI_DIFFICULTY) {
            case 'easy':
                AI_DIFFICULTY = 'hard1';
                AI_DEPTH = 3;
                difficultyBtn.classList.add('hard');
                const img = difficultyBtn.querySelector('img');
                img.src = 'images/hardware-chip-outline.svg';
                difficultyBtn.title = 'AI Difficulty: Hard Level 1';
                break;
            // case 'hard1':
            //     AI_DIFFICULTY = 'hard2';
            //     AI_DEPTH = 5;
            //     const img2 = difficultyBtn.querySelector('img');
            //     img2.src = 'images/skull-outline.svg';
            //     difficultyBtn.title = 'AI Difficulty: Hard Level 2';
            //     break;
            // case 'hard2':
            case 'hard1':
                AI_DIFFICULTY = 'easy';
                AI_DEPTH = 1;
                difficultyBtn.classList.remove('hard');
                const img3 = difficultyBtn.querySelector('img');
                img3.src = 'images/happy-outline.svg';
                difficultyBtn.title = 'AI Difficulty: Easy';
                break;
        }
        console.log(`AI Difficulty switched to: ${AI_DIFFICULTY} (depth: ${AI_DEPTH})`);
    });

    overlayCloseButtons.forEach(button => {
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
            unmarkAllSwapped();
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
            if (currentPlayer === PLAYER_B && !gameOver) {
                setTimeout(() => {
                    triggerAIMove();
                }, 50);
            }
        }
    }

    function unmarkAllSwapped() {
        let changed = false;
        for (let r = 0; r < ROWS; r++) {
            for (let c = 0; c < COLS; c++) {
                if (board[r][c] && board[r][c].state === SWAPPED) {
                    board[r][c].state = NORMAL;
                    changed = true;
                }
            }
        }
        //if (changed) console.log("Unmarked swapped pieces.");
    }

    function switchPlayer() {
        currentPlayer = (currentPlayer === PLAYER_A) ? PLAYER_B : PLAYER_A;
        updateStatusMessage();
        console.log(`Turn switched. Player ${currentPlayer}'s turn.`);
    }

    // --- Win Condition ---

    function checkWinCondition(player) {
        // REVERSED ORIENTATION: Define start/target based on player and new orientation
        const startRow = (player === PLAYER_A) ? ROWS - 2 : 1; // A starts near bottom (idx 6), B near top (idx 1)
        const targetRow = (player === PLAYER_A) ? 1 : ROWS - 2; // A targets near top (idx 1), B targets near bottom (idx 6)

        const visited = Array(ROWS).fill(null).map(() => Array(COLS).fill(false));
        const queue = []; // Queue for BFS: stores {row, col, path}

        // Find starting pieces for the player in their designated 'start' row
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


        while (queue.length > 0) {
            const current = queue.shift();
            const { row, col, path } = current;

            // Check if we reached the target row
            if (row === targetRow) {
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

    function triggerAIMove() {
        if (gameOver) return;
        console.log("AI (Player B) is thinking...");

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
                     const bestMove = findBestAIMove();
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
                // AI took longer than the delay. Spinner should be shown or was shown.
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
    }

    // (findBestAIMove simulation logic remains the same, but calls updated evaluateBoardState and allowsOpponentWin)
    function findBestAIMove(boardState = board) {
        let possibleMoves = [];
        for (let r = 0; r < ROWS; r++) {
            for (let c = 0; c < COLS; c++) {
                if (boardState[r][c] && boardState[r][c].player === PLAYER_B) {
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

        if (possibleMoves.length === 0) return null;

        // First, check for immediate winning moves
        for (const move of possibleMoves) {
            const tempBoard = cloneBoard(boardState);
            const { start, end } = move;
            const movingPiece = tempBoard[start.row][start.col];
            const targetPiece = tempBoard[end.row][end.col];
    
            if (!targetPiece) {
                tempBoard[end.row][end.col] = { ...movingPiece };
                tempBoard[start.row][start.col] = null;
                for (let r = 0; r < ROWS; r++) {
                    for (let c = 0; c < COLS; c++) {
                        if (tempBoard[r][c]?.state === SWAPPED) {
                            tempBoard[r][c].state = NORMAL;
                        }
                    }
                }
            } else {
                tempBoard[end.row][end.col] = { ...movingPiece, state: SWAPPED };
                tempBoard[start.row][start.col] = { ...targetPiece, state: SWAPPED };
            }
    
            if (checkWinConditionForState(tempBoard, PLAYER_B).win) {
                if (ANALYSIS_MODE) {
                    console.log(`Found immediate winning move: ${start.row},${start.col} to ${end.row},${end.col}`);
                }
                return move;
            }
        }

        // If no immediate win, proceed with normal evaluation
        if (AI_DIFFICULTY === 'easy') {
            // Filter out moves that allow immediate opponent win
            let safeMoves = [];
            for (const move of possibleMoves) {
                // Test each move on a cloned board
                const tempBoard = cloneBoard(boardState);
                const {start, end} = move;
                
                // Apply move to temporary board
                const movingPiece = tempBoard[start.row][start.col];
                const targetPiece = tempBoard[end.row][end.col];

                if (!targetPiece) {
                    tempBoard[end.row][end.col] = {...movingPiece};
                    tempBoard[start.row][start.col] = null;
                    // Reset swapped pieces
                    for (let r = 0; r < ROWS; r++) {
                        for (let c = 0; c < COLS; c++) {
                            if (tempBoard[r][c]?.state === SWAPPED) {
                                tempBoard[r][c].state = NORMAL;
                            }
                        }
                    }
                } else {
                    tempBoard[end.row][end.col] = {...movingPiece, state: SWAPPED};
                    tempBoard[start.row][start.col] = {...targetPiece, state: SWAPPED};
                }

                // Check if this move allows Player A to win on their *next* turn
                // Use allowsOpponentWin with depth 1 for easy difficulty
                const isSafe = !allowsOpponentWin(tempBoard, PLAYER_A, 1);
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
                row.map(cell => cell ? {...cell} : null)
            );
            
            const movingPiece = tempBoard[move.start.row][move.start.col];
            const targetPiece = tempBoard[move.end.row][move.end.col];

            if (targetPiece === null) {
                // Empty cell move
                tempBoard[move.end.row][move.end.col] = {...movingPiece}; // Use spread to clone
                tempBoard[move.start.row][move.start.col] = null;
                // Simulate unmarking swapped pieces
                for (let r = 0; r < ROWS; r++) {
                    for (let c = 0; c < COLS; c++) {
                        if (tempBoard[r][c] && tempBoard[r][c].state === SWAPPED) {
                            tempBoard[r][c] = {...tempBoard[r][c], state: NORMAL};
                        }
                    }
                }
            } else {
                // Swap move
                tempBoard[move.end.row][move.end.col] = {...movingPiece, state: SWAPPED};
                tempBoard[move.start.row][move.start.col] = {...targetPiece, state: SWAPPED};
            }

            const score = evaluateBoardState(tempBoard, PLAYER_B, move); // << Uses updated evaluateBoardState

             // Check if this move allows Player A to win immediately
             if (!allowsOpponentWin(tempBoard, PLAYER_A)) { // << Uses updated allowsOpponentWin
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
                        opponentMoves.push({start: {row: r, col: c}, end: move})
                    );
                }
            }
        }
    
        // Move ordering optimization: Prioritize moves that might lead to a win
        let immediateWinMoves = [];
        let otherMoves = [];
        let moveStates = new Map();
    
        for (const move of opponentMoves) {
            const {start, end} = move;
            const afterOpponentMove = cloneBoard(boardState);
            const movingPiece = afterOpponentMove[start.row][start.col];
            const targetPiece = afterOpponentMove[end.row][end.col];
    
            if (!targetPiece) {
                afterOpponentMove[end.row][end.col] = {...movingPiece};
                afterOpponentMove[start.row][start.col] = null;
                for (let r = 0; r < ROWS; r++) {
                    for (let c = 0; c < COLS; c++) {
                        if (afterOpponentMove[r][c]?.state === SWAPPED) {
                            afterOpponentMove[r][c].state = NORMAL;
                        }
                    }
                }
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
                transpositionTable.set(boardKey, {value: true, depth: depth});
                return true;
            }
    
            // Recursive search for harder difficulties
            if ((AI_DIFFICULTY === 'hard1' || AI_DIFFICULTY === 'hard2') && depth > 1) {
                const currentPlayer = opponentPlayer === PLAYER_A ? PLAYER_B : PLAYER_A;
                if (!hasValidResponse(afterOpponentMove, currentPlayer, depth - 1, -beta, -alpha)) {
                    beta = Math.min(beta, 1);
                    if (beta <= alpha) {
                        transpositionTable.set(boardKey, {value: true, depth: depth});
                        return true;
                    }
                } else {
                    alpha = Math.max(alpha, -1);
                }
            }
        }
    
        transpositionTable.set(boardKey, {value: false, depth: depth});
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
                        possibleMoves.push({start: {row: r, col: c}, end: move})
                    );
                }
            }
        }

        // Try each move and look deeper
        for (const move of possibleMoves) {
            const nextState = cloneBoard(boardState);
            const {start, end} = move;
            
            // Apply move
            const movingPiece = nextState[start.row][start.col];
            const targetPiece = nextState[end.row][end.col];

            if (!targetPiece) {
                nextState[end.row][end.col] = {...movingPiece};
                nextState[start.row][start.col] = null;
                // Reset swapped pieces
                for (let r = 0; r < ROWS; r++) {
                    for (let c = 0; c < COLS; c++) {
                        if (nextState[r][c]?.state === SWAPPED) {
                            nextState[r][c].state = NORMAL;
                        }
                    }
                }
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
    }

    // Helper to check win condition for a given board state - UPDATED start/target rows
     function checkWinConditionForState(boardState, player) {
         // REVERSED ORIENTATION: Define start/target based on player and new orientation
         const startRow = (player === PLAYER_A) ? ROWS - 2 : 1; // A starts near bottom (idx 6), B near top (idx 1)
         const targetRow = (player === PLAYER_A) ? 1 : ROWS - 2; // A targets near top (idx 1), B targets near bottom (idx 6)

         const visited = Array(ROWS).fill(null).map(() => Array(COLS).fill(false));
         const queue = []; // Queue for BFS: stores {row, col, path}

         // Find starting pieces for the player in their designated 'start' row
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

         while (queue.length > 0) {
             const current = queue.shift();
             const { row, col, path } = current;

             // Check if we reached the target row - IMPORTANT: Return the complete path!
             if (row === targetRow) {
                 return { win: true, path: path }; // Return the complete winning path
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
                         boardState[nr] && boardState[nr][nc] && boardState[nr][nc].player === player)
                     {
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
                        if (r > 1 && r < ROWS - 1){
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
                        if (r> 1 && r < ROWS-1) {
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
             const initialBoard = Array(ROWS).fill(null).map(() => Array(COLS).fill(null));
              // Recreate initial state based on NEW orientation
              for (let r = 0; r < ROWS; r++) {
                 for (let c = 0; c < COLS; c++) {
                     if (r < 2) initialBoard[r][c] = { player: PLAYER_B, state: NORMAL }; // B top
                      else if (r >= ROWS - 2) initialBoard[r][c] = { player: PLAYER_A, state: NORMAL }; // A bottom
                 }
             }
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
             const initialBoard = Array(ROWS).fill(null).map(() => Array(COLS).fill(null));
             for (let r = 0; r < ROWS; r++) {
                 for (let c = 0; c < COLS; c++) {
                     if (r < 2) initialBoard[r][c] = { player: PLAYER_B, state: NORMAL };
                     else if (r >= ROWS - 2) initialBoard[r][c] = { player: PLAYER_A, state: NORMAL };
                 }
             }
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
            // Show initial board state
            const initialBoard = Array(ROWS).fill(null).map(() => Array(COLS).fill(null));
            for (let r = 0; r < ROWS; r++) {
                for (let c = 0; c < COLS; c++) {
                    if (r < 2) initialBoard[r][c] = { player: PLAYER_B, state: NORMAL };
                    else if (r >= ROWS - 2) initialBoard[r][c] = { player: PLAYER_A, state: NORMAL };
                }
            }
            renderBoard(initialBoard);
        } else {
            renderBoard(moveHistory[currentHistoryIndex - 1].boardAfter);
        }
        updateMoveCounter();
    }

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

    // Utility to deep clone boardState
    function cloneBoard(boardState) {
        // Use structuredClone if available (modern browsers), else fallback to JSON method
        return (typeof structuredClone === 'function') ? 
               structuredClone(boardState) :
               JSON.parse(JSON.stringify(boardState));
    }

    function boardToKey(boardState) {
        // Create a simpler string representation that's JSON-safe
        return boardState.map(row => 
            row.map(cell => 
                cell ? `${cell.player}${cell.state[0]}` : '_'  // Use first letter of state
            ).join('')
        ).join('|');
    }

    function serializeBoardState(boardState) {
        return boardState.map(row => 
            row.map(cell => {
                if (!cell) return '_';
                return `${cell.player}${cell.state === 'normal' ? 'N' : 'S'}`;
            }).join('')
        ).join('|');
    }
    function analyzeHistoricalMove() {
        if (currentHistoryIndex !== undefined && currentHistoryIndex > 0) {
            const historicalBoard = moveHistory[currentHistoryIndex - 1].boardAfter;
            ANALYSIS_MODE = true;
            const bestMove = findBestAIMove(historicalBoard);
            ANALYSIS_MODE = false;
            console.log('Best move for this board state:', bestMove);
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
    window.setupTestPosition = setupTestPosition;
});