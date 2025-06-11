// Game logic: moves, win condition, switching players, etc.
import { ROWS, COLS, PLAYER_A, PLAYER_B, NORMAL, SWAPPED, JS_DIRECTIONS, NUM_DIRECTIONS } from './game-constants.js';
import { cloneBoard } from './game-board.js';

// --- Game Logic Functions ---

// Select a piece and calculate its legal moves
export function selectPiece(row, col, board, currentPlayer) {
    const legalMoves = calculateLegalMoves(row, col, board, currentPlayer);
    return { selectedPiece: { row, col }, legalMoves };
}

export function deselectPiece() {
    return { selectedPiece: null, legalMoves: [] };
}

export function calculateLegalMoves(r, c, board, currentPlayer) {
    const moves = [];
    const piece = board[r][c];
    if (!piece) return moves;
    const opponent = piece.player === PLAYER_A ? PLAYER_B : PLAYER_A;
    for (let dr = -1; dr <= 1; dr++) {
        for (let dc = -1; dc <= 1; dc++) {
            if (dr === 0 && dc === 0) continue;
            const nr = r + dr;
            const nc = c + dc;
            if (nr >= 0 && nr < ROWS && nc >= 0 && nc < COLS) {
                const targetCell = board[nr][nc];
                if (targetCell === null) {
                    moves.push({ row: nr, col: nc });
                } else if (targetCell.player === opponent) {
                    if (targetCell.state === NORMAL) {
                        moves.push({ row: nr, col: nc, isSwap: true });
                    }
                }
            }
        }
    }
    return moves;
}

export function makeMove(startRow, startCol, endRow, endCol, board, selectedPiece, legalMoves, currentPlayer) {
    if (!selectedPiece || startRow !== selectedPiece.row || startCol !== selectedPiece.col) {
        throw new Error("Move error: Invalid start piece.");
    }
    const move = legalMoves.find(m => m.row === endRow && m.col === endCol);
    if (!move) {
        throw new Error("Move error: Invalid target cell.");
    }
    // Apply the move
    const movingPiece = board[startRow][startCol];
    const targetPiece = board[endRow][endCol];
    console.log(`makeMove: Player ${currentPlayer} moves from (${startRow},${startCol}) to (${endRow},${endCol})`);
    if (targetPiece === null) {
        board[endRow][endCol] = movingPiece;
        board[startRow][startCol] = null;
        unmarkSwapped(board);
    } else {
        board[endRow][endCol] = { ...movingPiece, state: SWAPPED };
        board[startRow][startCol] = { ...targetPiece, state: SWAPPED };
    }
    return board;
}

export function unmarkSwapped(boardState) {
    for (let r = 0; r < ROWS; r++) {
        for (let c = 0; c < COLS; c++) {
            if (boardState[r][c] && boardState[r][c].state === SWAPPED) {
                boardState[r][c].state = NORMAL;
            }
        }
    }
}

export function switchPlayer(currentPlayer) {
    const nextPlayer = currentPlayer === PLAYER_A ? PLAYER_B : PLAYER_A;
    console.log(`Switching player: ${currentPlayer} -> ${nextPlayer}`);
    return nextPlayer;
}

export function checkWinCondition(boardState, player) {
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

// Export all logic functions needed by main.js and AI

// ...
