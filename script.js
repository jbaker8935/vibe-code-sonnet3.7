document.addEventListener('DOMContentLoaded', () => {
    const ROWS = 8;
    const COLS = 4;
    const PLAYER_A = 'A'; // White
    const PLAYER_B = 'B'; // Black
    const NORMAL = 'normal';
    const SWAPPED = 'swapped';
    let AI_DIFFICULTY = 'easy'; // Changed to let and default to 'easy'

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
        boardElement.classList.remove('game-over'); // Remove game over visual cues if any

        for (let r = 0; r < ROWS; r++) {
            for (let c = 0; c < COLS; c++) {
                const cell = document.createElement('div');
                cell.classList.add('cell');
                cell.dataset.row = r; // Internal row index (0-7)
                cell.dataset.col = c;

                // Checkerboard pattern
                cell.classList.add(((r + c) % 2 === 0) ? 'light' : 'dark');

                const pieceData = boardState[r][c];
                if (pieceData) {
                    const pieceImg = document.createElement('img');
                    pieceImg.classList.add('piece');
                    pieceImg.src = getPieceImage(pieceData.player, pieceData.state);
                    pieceImg.alt = `Player ${pieceData.player} ${pieceData.state}`;
                    cell.appendChild(pieceImg);
                }

                // Add highlights AFTER adding piece
                if (selectedPiece && selectedPiece.row === r && selectedPiece.col === c) {
                    cell.classList.add('selected');
                }

                if (legalMoves.some(move => move.row === r && move.col === c)) {
                    cell.classList.add('legal-move');
                     // Add specific class if it's a swap target
                    const targetPiece = boardState[r][c];
                    if (targetPiece && targetPiece.player !== currentPlayer) {
                         cell.classList.add('swap-target');
                    }
                }

                // Only show win path if we're viewing the final position
                if (winPath.length > 0 && (currentHistoryIndex === undefined || currentHistoryIndex === moveHistory.length)) {
                    if (winPath.some(pos => pos.row === r && pos.col === c)) {
                        cell.classList.add('win-path');
                    }
                }

                cell.addEventListener('click', () => handleCellClick(r, c));
                boardElement.appendChild(cell);
            }
        }
        
        if (gameOver) {
            boardElement.classList.add('game-over');
            // Always show move counter during history viewing
            if (currentHistoryIndex !== undefined) {
                const moveCount = document.createElement('div');
                moveCount.classList.add('move-counter');
                moveCount.textContent = currentHistoryIndex === undefined || currentHistoryIndex === moveHistory.length ? 
                    'Final Position' : 
                    `Move ${currentHistoryIndex} of ${moveHistory.length}`;
                boardElement.appendChild(moveCount);
            }

            // Add score display
            const scoreDisplay = document.createElement('div');
            scoreDisplay.classList.add('score-display');
            scoreDisplay.innerHTML = `
                <div id="score-a">Player A: ${playerAScore}</div>
                <div id="score-b">Player B: ${playerBScore}</div>
            `;
            boardElement.appendChild(scoreDisplay);

            // Update the scores in the controls division
            const controlScoreA = document.getElementById('control-score-a');
            const controlScoreB = document.getElementById('control-score-b');
            if (controlScoreA) controlScoreA.textContent = ` ${playerAScore}`;
            if (controlScoreB) controlScoreB.textContent = ` ${playerBScore}`;
        }

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

    // Add difficulty button handler
    difficultyBtn.addEventListener('click', () => {
        AI_DIFFICULTY = AI_DIFFICULTY === 'easy' ? 'hard' : 'easy';
        difficultyBtn.classList.toggle('hard', AI_DIFFICULTY === 'hard');
        const img = difficultyBtn.querySelector('img');
        if (AI_DIFFICULTY === 'hard') {
            img.src = 'images/skull-outline.svg';
            difficultyBtn.title = 'AI Difficulty: Hard';
        } else {
            img.src = 'images/happy-outline.svg';
            difficultyBtn.title = 'AI Difficulty: Easy';
        }
        console.log(`AI Difficulty switched to: ${AI_DIFFICULTY}`);
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
             deselectPiece(); // Deselect if move was invalid
             renderBoard();
            return;
        }

        // Record state *before* the move for history
        const boardBefore = cloneBoard(board);
        moveHistory.push({
            player: currentPlayer,
            start: { row: startRow, col: startCol },
            end: { row: endRow, col: endCol },
            boardBefore: boardBefore
        });


        const movingPiece = board[startRow][startCol];
        const targetPiece = board[endRow][endCol];

        if (targetPiece === null) {
            // Move to empty cell
            board[endRow][endCol] = movingPiece;
            board[startRow][startCol] = null;
            unmarkAllSwapped(); // Reset swapped status on empty move
            console.log(`Player ${currentPlayer} moved ${startRow},${startCol} to ${endRow},${endCol}. Swapped pieces reset.`);
        } else if (targetPiece.player !== currentPlayer && targetPiece.state === NORMAL) {
            // Swap move
            board[endRow][endCol] = { ...movingPiece, state: SWAPPED };
            board[startRow][startCol] = { ...targetPiece, state: SWAPPED };
             console.log(`Player ${currentPlayer} swapped ${startRow},${startCol} with ${endRow},${endCol}. Both marked SWAPPED.`);
        } else {
             console.error("Illegal move logic error!"); // Should have been caught by legalMoves check
             return;
        }

        // Store the state after the move
        moveHistory[moveHistory.length - 1].boardAfter = cloneBoard(board);

        deselectPiece();

        // Check for win BEFORE switching player
        const winCheckResult = checkWinCondition(currentPlayer); // << Uses updated checkWinCondition
        if (winCheckResult.win) {
             gameOver = true;
             winner = currentPlayer;
             winPath = winCheckResult.path;
             renderBoard(); // Render final board state with win path highlight
             handleWin(winner);
        } else {
             switchPlayer();
             renderBoard(); // Render board after move, before AI starts thinking
             if (currentPlayer === PLAYER_B && !gameOver) {
                 // Trigger AI move after a short delay for UX
                 setTimeout(triggerAIMove, 500);
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

        // Find all starting pieces for the player in their designated 'start' row
        for (let c = 0; c < COLS; c++) {
            // Ensure the piece exists and belongs to the player before starting BFS from it
            if (board[startRow] && board[startRow][c] && board[startRow][c].player === player) {
                queue.push({ row: startRow, col: c, path: [{ row: startRow, col: c }] });
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
         console.log(`Game Over! Player ${winningPlayer} wins!`);
         // Update scores
         if (winningPlayer === PLAYER_A) {
             playerAScore++;
         } else {
             playerBScore++;
         }
         updateScoreDisplay();
         
         winMessage.textContent = `Player ${winningPlayer} Wins!`;
         showOverlay(winOverlay);
         historyBtn.disabled = false; // Enable history button
         currentHistoryIndex = moveHistory.length; // Initialize at final position

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

        let spinnerTimeout;
        let spinnerMinTimeout;
        let spinnerShown = false;

        // Show spinner after 3 seconds if AI is still thinking
        spinnerTimeout = setTimeout(() => {
            if (AI_DIFFICULTY === 'hard') {
                showOverlay(aiSpinnerOverlay);
                spinnerShown = true;
                // Start minimum display timer when spinner is shown
                spinnerMinTimeout = setTimeout(() => {
                    spinnerMinTimeout = null;
                }, 1000);
            }
        }, 3000);

        // Use Promise to handle AI move computation
        new Promise((resolve) => {
            setTimeout(() => {
                const bestMove = findBestAIMove();
                resolve(bestMove);
            }, 100);
        }).then(bestMove => {
            if (bestMove) {
                console.log("AI chooses move:", bestMove);
                selectPiece(bestMove.start.row, bestMove.start.col);
                if (legalMoves.some(m => m.row === bestMove.end.row && m.col === bestMove.end.col)) {
                    makeMove(bestMove.start.row, bestMove.start.col, bestMove.end.row, bestMove.end.col);
                } else {
                    console.error("AI Logic Error: Chosen move is not legal according to recalculation?");
                    deselectPiece();
                }
            } else {
                console.log("AI has no legal moves!");
            }
            
            // Clear the spinner timeout if it hasn't triggered yet
            if (spinnerTimeout) {
                clearTimeout(spinnerTimeout);
            }

            // If spinner was shown, wait for minimum display time before hiding
            if (spinnerShown) {
                const hideSpinner = () => {
                    hideOverlay(aiSpinnerOverlay);
                };

                if (spinnerMinTimeout) {
                    // If min display timer is still running, wait for it
                    clearTimeout(spinnerMinTimeout);
                    setTimeout(hideSpinner, 1000);
                } else {
                    // Min display time has already elapsed
                    hideSpinner();
                }
            }
        });
    }

    // (findBestAIMove simulation logic remains the same, but calls updated evaluateBoardState and allowsOpponentWin)
     function findBestAIMove() {
        let possibleMoves = [];
        for (let r = 0; r < ROWS; r++) {
            for (let c = 0; c < COLS; c++) {
                if (board[r][c] && board[r][c].player === PLAYER_B) {
                    const moves = calculateLegalMoves(r, c);
                    moves.forEach(move => {
                        possibleMoves.push({
                            start: { row: r, col: c },
                            end: { row: move.row, col: move.col },
                            isSwap: !!move.isSwap // Store if it's a swap
                        });
                    });
                }
            }
        }

        if (possibleMoves.length === 0) return null;

        let bestScore = -Infinity;
        let bestMoves = []; // Store moves with the best score

        for (const move of possibleMoves) {
            // Simulate the move
            const tempBoard = cloneBoard(board);
            const movingPiece = tempBoard[move.start.row][move.start.col];
            const targetPiece = tempBoard[move.end.row][move.end.col];
            let wasEmptyMove = false;

             if (targetPiece === null) {
                 // Empty cell move
                 tempBoard[move.end.row][move.end.col] = movingPiece;
                 tempBoard[move.start.row][move.start.col] = null;
                 // Simulate unmarking swapped pieces
                 for (let r = 0; r < ROWS; r++) {
                    for (let c = 0; c < COLS; c++) {
                         if (tempBoard[r][c] && tempBoard[r][c].state === SWAPPED) {
                             tempBoard[r][c].state = NORMAL;
                         }
                    }
                 }
                 wasEmptyMove = true;
            } else {
                 // Swap move (only possible with NORMAL opponent)
                 tempBoard[move.end.row][move.end.col] = { ...movingPiece, state: SWAPPED };
                 tempBoard[move.start.row][move.start.col] = { ...targetPiece, state: SWAPPED };
            }


            const score = evaluateBoardState(tempBoard, PLAYER_B, move, wasEmptyMove); // << Uses updated evaluateBoardState

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
     function allowsOpponentWin(boardState, opponentPlayer, depth = 3) {
         if (depth <= 0) return false; // Stop recursion if too deep
         
         if (AI_DIFFICULTY === 'hard') {
             // Get all Player A's possible moves
             for (let r = 0; r < ROWS; r++) {
                 for (let c = 0; c < COLS; c++) {
                     if (boardState[r][c] && boardState[r][c].player === opponentPlayer) {
                         const opponentMoves = calculateLegalMovesForState(boardState, r, c);
                         for (const move of opponentMoves) {
                             // Simulate Player A's move
                             const afterOpponentMove = cloneBoard(boardState);
                             const movingPiece = afterOpponentMove[r][c];
                             const targetPiece = afterOpponentMove[move.row][move.col];

                             // Apply Player A's move
                             if (targetPiece === null) {
                                 afterOpponentMove[move.row][move.col] = movingPiece;
                                 afterOpponentMove[r][c] = null;
                                 // Reset swapped pieces
                                 for (let rr = 0; rr < ROWS; rr++) {
                                     for (let cc = 0; cc < COLS; cc++) {
                                         if (afterOpponentMove[rr][cc] && afterOpponentMove[rr][cc].state === SWAPPED) {
                                             afterOpponentMove[rr][cc].state = NORMAL;
                                         }
                                     }
                                 }
                             } else {
                                 afterOpponentMove[move.row][move.col] = { ...movingPiece, state: SWAPPED };
                                 afterOpponentMove[r][c] = { ...targetPiece, state: SWAPPED };
                             }

                             // If Player A wins directly after their move
                             if (checkWinConditionForState(afterOpponentMove, opponentPlayer).win) {
                                 return true;
                             }

                             // Get all of Player B's possible responses
                             let playerBHasValidResponse = false;
                             for (let br = 0; br < ROWS && !playerBHasValidResponse; br++) {
                                 for (let bc = 0; bc < COLS && !playerBHasValidResponse; bc++) {
                                     if (afterOpponentMove[br][bc] && afterOpponentMove[br][bc].player === PLAYER_B) {
                                         const playerBMoves = calculateLegalMovesForState(afterOpponentMove, br, bc);
                                         for (const bMove of playerBMoves) {
                                             const afterPlayerBMove = cloneBoard(afterOpponentMove);
                                             const bMovingPiece = afterPlayerBMove[br][bc];
                                             const bTargetPiece = afterPlayerBMove[bMove.row][bMove.col];

                                             // Apply Player B's response move
                                             if (bTargetPiece === null) {
                                                 afterPlayerBMove[bMove.row][bMove.col] = bMovingPiece;
                                                 afterPlayerBMove[br][bc] = null;
                                                 // Reset swapped pieces
                                                 for (let rr = 0; rr < ROWS; rr++) {
                                                     for (let cc = 0; cc < COLS; cc++) {
                                                         if (afterPlayerBMove[rr][cc] && afterPlayerBMove[rr][cc].state === SWAPPED) {
                                                             afterPlayerBMove[rr][cc].state = NORMAL;
                                                         }
                                                     }
                                                 }
                                             } else {
                                                 afterPlayerBMove[bMove.row][bMove.col] = { ...bMovingPiece, state: SWAPPED };
                                                 afterPlayerBMove[br][bc] = { ...bTargetPiece, state: SWAPPED };
                                             }

                                             // If this response prevents Player A from winning next move
                                             if (!checkWinConditionForState(afterPlayerBMove, opponentPlayer).win &&
                                                 !allowsOpponentWin(afterPlayerBMove, opponentPlayer, depth - 1)) {
                                                 playerBHasValidResponse = true;
                                                 break;
                                             }
                                         }
                                     }
                                 }
                             }
                             
                             // If no valid response was found for this Player A move
                             if (!playerBHasValidResponse) {
                                 return true;
                             }
                         }
                     }
                 }
             }
             return false;
         } else {
             // Original easier logic - just check immediate wins
             for (let r = 0; r < ROWS; r++) {
                 for (let c = 0; c < COLS; c++) {
                     if (boardState[r][c] && boardState[r][c].player === opponentPlayer) {
                         const opponentMoves = calculateLegalMovesForState(boardState, r, c);
                         for (const move of opponentMoves) {
                             const nextBoardState = cloneBoard(boardState);
                             const movingPiece = nextBoardState[r][c];
                             const targetPiece = nextBoardState[move.row][move.col];

                             if (targetPiece === null) {
                                 nextBoardState[move.row][move.col] = movingPiece;
                                 nextBoardState[r][c] = null;
                                 for (let rr = 0; rr < ROWS; rr++) {
                                     for (let cc = 0; cc < COLS; cc++) {
                                         if (nextBoardState[rr][cc] && nextBoardState[rr][cc].state === SWAPPED) {
                                             nextBoardState[rr][cc].state = NORMAL;
                                         }
                                     }
                                 }
                             } else {
                                 nextBoardState[move.row][move.col] = { ...movingPiece, state: SWAPPED };
                                 nextBoardState[r][c] = { ...targetPiece, state: SWAPPED };
                             }

                             if (checkWinConditionForState(nextBoardState, opponentPlayer).win) {
                                 return true;
                             }
                         }
                     }
                 }
             }
         }
         return false;
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
         const queue = []; // {row, col, path} - path not strictly needed here, just for consistency

         for (let c = 0; c < COLS; c++) {
             // Ensure the piece exists and belongs to the player before starting BFS from it
             if (boardState[startRow] && boardState[startRow][c] && boardState[startRow][c].player === player) {
                 queue.push({ row: startRow, col: c, path: [{ row: startRow, col: c }] });
                 visited[startRow][c] = true;
             }
         }

         while (queue.length > 0) {
             const { row, col } = queue.shift(); // Path details ignored for simple win check
             if (row === targetRow) return { win: true };

             for (let dr = -1; dr <= 1; dr++) {
                 for (let dc = -1; dc <= 1; dc++) {
                     if (dr === 0 && dc === 0) continue;
                     const nr = row + dr;
                     const nc = col + dc;

                     // Check bounds, if visited, and if it's the player's piece
                     if (nr >= 0 && nr < ROWS && nc >= 0 && nc < COLS &&
                         !visited[nr][nc] &&
                         boardState[nr] && boardState[nr][nc] && boardState[nr][nc].player === player) // Added check for boardState[nr] existence
                     {
                         visited[nr][nc] = true;
                         queue.push({ row: nr, col: nc, path: [] }); // Path details ignored
                     }
                 }
             }
         }
         return { win: false };
     }


     // --- Heuristic Evaluation --- UPDATED for new orientation
    function evaluateBoardState(boardState, player, move, wasEmptyMove) {
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
        const BACK_ROW_PENALTY = -20; // Penalty for Player B being on row 0
        const CONNECTIVITY_WEIGHT = 5;
        const FOUR_IN_ROW_PENALTY = -30;
        const SWAP_BONUS = 8; // Bonus for making a swap
        const RESTRICT_OPPONENT_WEIGHT = 2;
        const CENTER_CONTROL_WEIGHT = 1; // Slight bonus for pieces near center cols (1, 2)

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

                        // Connectivity Score
                        playerConnectivity += countFriendlyNeighbors(boardState, r, c, player);

                        // Center Control
                        if (c === 1 || c === 2) {
                            score += CENTER_CONTROL_WEIGHT;
                        }

                    } else { // Opponent piece (Player A)
                        opponentCount++;
                        playerHorizontalRow = 0; // Reset count on opponent piece
                        opponentConnectivity += countFriendlyNeighbors(boardState, r, c, opponent);
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
        // Could subtract opponent connectivity for defense: score -= CONNECTIVITY_WEIGHT * opponentConnectivity;

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
            moveDiv.addEventListener('click', () => {
                // View state *before* this move was made
                 currentHistoryIndex = moveHistory.length - index - 1;
                 renderBoard(move.boardAfter);
                 updateMoveCounter();
                 hideOverlay(historyOverlay);
            });
            historyList.appendChild(moveDiv);
        });

         // Add option to view initial state
         const initialStateDiv = document.createElement('div');
         initialStateDiv.textContent = `0. Initial State`;
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
        return boardState.map(row => 
            row.map(cell => 
                cell ? `${cell.player}${cell.state}` : '-'
            ).join('')
        ).join('|');
    }

    function hasValidResponse(boardState, opponentPlayer, depth, alpha, beta) {
        const currentPlayer = opponentPlayer === PLAYER_A ? PLAYER_B : PLAYER_A;
        let hasResponse = false;

        // Quick check for blocking moves first
        for (let r = 0; r < ROWS && !hasResponse; r++) {
            for (let c = 0; c < COLS && !hasResponse; c++) {
                if (boardState[r][c]?.player === currentPlayer) {
                    const moves = calculateLegalMovesForState(boardState, r, c);
                    for (const move of moves) {
                        const nextState = Object.create(Object.getPrototypeOf(boardState));
                        Object.assign(nextState, boardState);
                        // Apply move...
                        if (!allowsOpponentWin(nextState, opponentPlayer, depth - 1, -beta, -alpha)) {
                            hasResponse = true;
                            break;
                        }
                    }
                }
            }
        }
        
        return hasResponse;
    }

    function boardToKey(boardState) {
        return boardState.map(row => 
            row.map(cell => 
                cell ? `${cell.player}${cell.state}` : '-'
            ).join('')
        ).join('|');
    }

    function allowsOpponentWin(boardState, opponentPlayer, depth = 3, alpha = -Infinity, beta = Infinity) {
        if (depth <= 0) return false;

        // Early exit - check immediate win
        if (checkWinConditionForState(boardState, opponentPlayer).win) {
            return true;
        }

        // Check transposition table
        const boardKey = boardToKey(boardState);
        if (transpositionTable.has(boardKey)) {
            return transpositionTable.get(boardKey);
        }

        if (AI_DIFFICULTY === 'hard') {
            // Get and sort moves by potential
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

            // Sort moves - prioritize moves toward opponent's goal
            opponentMoves.sort((a, b) => {
                const targetRow = opponentPlayer === PLAYER_A ? 1 : ROWS - 2;
                return Math.abs(targetRow - a.end.row) - Math.abs(targetRow - b.end.row);
            });

            for (const move of opponentMoves) {
                const {start, end} = move;
                // Use Object.create for faster cloning of similar boards
                const afterOpponentMove = Object.create(Object.getPrototypeOf(boardState));
                Object.assign(afterOpponentMove, boardState);
                
                // Apply move
                const movingPiece = afterOpponentMove[start.row][start.col];
                const targetPiece = afterOpponentMove[end.row][end.col];

                if (!targetPiece) {
                    afterOpponentMove[end.row][end.col] = movingPiece;
                    afterOpponentMove[start.row][start.col] = null;
                    // Reset swapped pieces using more efficient method
                    for (const row of afterOpponentMove) {
                        for (const cell of row) {
                            if (cell?.state === SWAPPED) cell.state = NORMAL;
                        }
                    }
                } else {
                    afterOpponentMove[end.row][end.col] = { ...movingPiece, state: SWAPPED };
                    afterOpponentMove[start.row][start.col] = { ...targetPiece, state: SWAPPED };
                }

                // Check if this leads to a win
                if (checkWinConditionForState(afterOpponentMove, opponentPlayer).win ||
                    !hasValidResponse(afterOpponentMove, opponentPlayer, depth - 1, alpha, beta)) {
                    transpositionTable.set(boardKey, true);
                    return true;
                }

                // Alpha-beta pruning
                alpha = Math.max(alpha, -beta);
                if (alpha >= beta) break;
            }
        }
        
        transpositionTable.set(boardKey, false);
        return false;
    }

    function hasValidResponse(boardState, opponentPlayer, depth, alpha, beta) {
        const currentPlayer = opponentPlayer === PLAYER_A ? PLAYER_B : PLAYER_A;
        let hasResponse = false;

        // Quick check for blocking moves first
        for (let r = 0; r < ROWS && !hasResponse; r++) {
            for (let c = 0; c < COLS && !hasResponse; c++) {
                if (boardState[r][c]?.player === currentPlayer) {
                    const moves = calculateLegalMovesForState(boardState, r, c);
                    for (const move of moves) {
                        const nextState = Object.create(Object.getPrototypeOf(boardState));
                        Object.assign(nextState, boardState);
                        // Apply move...
                        if (!allowsOpponentWin(nextState, opponentPlayer, depth - 1, -beta, -alpha)) {
                            hasResponse = true;
                            break;
                        }
                    }
                }
            }
        }
        
        return hasResponse;
    }

});