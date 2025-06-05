// Board state, parsing, cloning, serialization, and helpers
import { ROWS, COLS, PLAYER_A, PLAYER_B, NORMAL, SWAPPED } from './game-constants.js';

export function parseStartingPosition(positionStr) {
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
    console.log("Parsed starting position:", board);
    return board;
}

export function cloneBoard(boardState) {
    const cloned = (typeof structuredClone === 'function') ? structuredClone(boardState) : JSON.parse(JSON.stringify(boardState));
    return cloned;
}

export function serializeBoardState(boardState) {
    const serialized = boardState.map(row => row.map(cell => cell ? `${cell.player}${cell.state === NORMAL ? 'N' : 'S'}` : '_').join('')).join('|');
    console.log("Serialized board state:", serialized);
    return serialized;
}

export function boardToKey(boardState) {
    return boardState.map(row => 
        row.map(cell => 
            cell ? `${cell.player}${cell.state[0]}` : '_'  
        ).join('')
    ).join('|');
}

export function convertBoardToBinaryJS(boardState) {
    let emptyMask = 0;
    let playerANormalMask = 0;
    let playerASwappedMask = 0;
    let playerBNormalMask = 0;
    let playerBSwappedMask = 0;
    for (let r = 0; r < ROWS; r++) {
        for (let c = 0; c < COLS; c++) {
            const pos = r * COLS + c;
            const piece = boardState[r][c];
            if (piece === null) {
                emptyMask |= (1 << pos);
            } else if (piece.player === PLAYER_A) {
                if (piece.state === NORMAL) {
                    playerANormalMask |= (1 << pos);
                } else {
                    playerASwappedMask |= (1 << pos);
                }
            } else if (piece.player === PLAYER_B) {
                if (piece.state === NORMAL) {
                    playerBNormalMask |= (1 << pos);
                } else {
                    playerBSwappedMask |= (1 << pos);
                }
            }
        }
    }
    const result = [emptyMask >>> 0, playerANormalMask >>> 0, playerASwappedMask >>> 0, playerBNormalMask >>> 0, playerBSwappedMask >>> 0];
    console.log("Converted board to binary:", result);
    return result;
}
