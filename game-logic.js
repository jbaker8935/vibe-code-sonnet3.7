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
        unmarkPlayerSwapped(currentPlayer, board);
    } else {
        board[endRow][endCol] = { ...movingPiece, state: SWAPPED };
        board[startRow][startCol] = { ...targetPiece, state: SWAPPED };
    }
    return board;
}

export function unmarkPlayerSwapped(player, boardState) {
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

export function checkWinCondition(board, player) {
    // Define starting and target areas for each player
    const startRows = (player === PLAYER_A) ? [ROWS - 2, ROWS - 1] : [0, 1];  // A: rows 6,7; B: rows 0,1
    const targetRows = (player === PLAYER_A) ? [0, 1] : [ROWS - 2, ROWS - 1]; // A: rows 0,1; B: rows 6,7
    
    const visited = Array(ROWS).fill(null).map(() => Array(COLS).fill(false));
    const queue = [];
    
    // Find all pieces in the starting area and add them to the queue
    for (const startRow of startRows) {
        for (let c = 0; c < COLS; c++) {
            if (board[startRow] && board[startRow][c] && board[startRow][c].player === player) {
                queue.push({ row: startRow, col: c, path: [{ row: startRow, col: c }] });
                visited[startRow][c] = true;
            }
        }
    }
    
    while (queue.length > 0) {
        const current = queue.shift();
        const { row, col, path } = current;
        
        // Check if we reached any target row
        if (targetRows.includes(row)) {
            return { win: true, path };
        }
        
        // Explore neighbors
        for (let dr = -1; dr <= 1; dr++) {
            for (let dc = -1; dc <= 1; dc++) {
                if (dr === 0 && dc === 0) continue;
                const nr = row + dr;
                const nc = col + dc;
                if (nr >= 0 && nr < ROWS && nc >= 0 && nc < COLS &&
                    !visited[nr][nc] &&
                    board[nr] && board[nr][nc] && board[nr][nc].player === player) {
                    visited[nr][nc] = true;
                    const newPath = [...path, { row: nr, col: nc }];
                    queue.push({ row: nr, col: nc, path: newPath });
                }
            }
        }
    }
    return { win: false, path: [] };
}

// Export all logic functions needed by main.js and AI

// ...
