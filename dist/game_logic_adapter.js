/**
 * Game Logic Adapter for MCTS Integration
 * Provides standardized interface between MCTS and Switcharoo game logic
 */

class SwitcharooGameLogic {
    constructor(gameConstants) {
        this.ROWS = gameConstants.ROWS || 8;
        this.COLS = gameConstants.COLS || 4;
        this.PLAYER_A = gameConstants.PLAYER_A || 'A';
        this.PLAYER_B = gameConstants.PLAYER_B || 'B';
        this.NORMAL = gameConstants.NORMAL || 'normal';
        this.SWAPPED = gameConstants.SWAPPED || 'swapped';
        this.NUM_DIRECTIONS = gameConstants.NUM_DIRECTIONS || 8;
        this.JS_DIRECTIONS = gameConstants.JS_DIRECTIONS || [
            { dr: -1, dc: -1 }, { dr: -1, dc:  0 }, { dr: -1, dc:  1 },
            { dr:  0, dc: -1 }, { dr:  0, dc:  1 },
            { dr:  1, dc: -1 }, { dr:  1, dc:  0 }, { dr:  1, dc:  1 }
        ];
    }

    /**
     * Get all legal actions for a player in the current board state
     * @param {Array} boardState - 2D board array
     * @param {string} player - PLAYER_A or PLAYER_B
     * @returns {Array} - Array of action indices
     */
    getLegalActions(boardState, player) {
        const actions = [];
        
        for (let r = 0; r < this.ROWS; r++) {
            for (let c = 0; c < this.COLS; c++) {
                const piece = boardState[r][c];
                if (piece && piece.player === player) {
                    const pieceMoves = this.calculateLegalMovesForPiece(boardState, r, c);
                    for (const move of pieceMoves) {
                        const actionIndex = this.moveToActionIndex(
                            { start: { row: r, col: c }, end: { row: move.row, col: move.col } }
                        );
                        if (actionIndex !== null) {
                            actions.push(actionIndex);
                        }
                    }
                }
            }
        }
        
        return actions;
    }

    /**
     * Calculate legal moves for a specific piece
     * @param {Array} boardState - 2D board array
     * @param {number} r - Row of the piece
     * @param {number} c - Column of the piece
     * @returns {Array} - Array of {row, col} move targets
     */
    calculateLegalMovesForPiece(boardState, r, c) {
        const moves = [];
        const piece = boardState[r][c];
        if (!piece) return moves;

        const pieceState = piece.state;
        const opponent = piece.player === this.PLAYER_A ? this.PLAYER_B : this.PLAYER_A;

        for (let dr = -1; dr <= 1; dr++) {
            for (let dc = -1; dc <= 1; dc++) {
                if (dr === 0 && dc === 0) continue;

                const nr = r + dr;
                const nc = c + dc;

                if (nr >= 0 && nr < this.ROWS && nc >= 0 && nc < this.COLS) {
                    const targetCell = boardState[nr][nc];

                    if (targetCell === null) {
                        // Move to empty cell - always legal
                        moves.push({ row: nr, col: nc });
                    } else if (targetCell.player === opponent) {
                        // Swap move - legal based on piece states
                        if (pieceState === this.NORMAL && targetCell.state === this.NORMAL) {
                            moves.push({ row: nr, col: nc });
                        }
                    }
                }
            }
        }

        return moves;
    }

    /**
     * Apply an action to the board state
     * @param {Array} boardState - 2D board array
     * @param {number} actionIndex - Action index
     * @param {string} player - Current player
     * @returns {Object} - {success: boolean, newBoard: Array, error: string}
     */
    applyAction(boardState, actionIndex, player) {
        const move = this.actionIndexToMove(actionIndex);
        if (!move) {
            return { success: false, newBoard: null, error: "Invalid action index" };
        }

        const { start, end } = move;
        const newBoard = this.cloneBoard(boardState);

        // Validate move
        if (!this.isValidPosition(start.row, start.col) || !this.isValidPosition(end.row, end.col)) {
            return { success: false, newBoard: null, error: "Invalid position" };
        }

        const movingPiece = newBoard[start.row][start.col];
        if (!movingPiece || movingPiece.player !== player) {
            return { success: false, newBoard: null, error: "Invalid piece or wrong player" };
        }

        const targetPiece = newBoard[end.row][end.col];

        if (targetPiece === null) {
            // Move to empty cell
            newBoard[end.row][end.col] = movingPiece;
            newBoard[start.row][start.col] = null;
            this.unmarkSwapped(newBoard, player);
        } else {
            // Swap move
            if (targetPiece.player === player) {
                return { success: false, newBoard: null, error: "Cannot swap with own piece" };
            }
            
            // Check if swap is legal (both pieces must be normal)
            if (movingPiece.state !== this.NORMAL || targetPiece.state !== this.NORMAL) {
                return { success: false, newBoard: null, error: "Can only swap normal pieces" };
            }

            newBoard[end.row][end.col] = { ...movingPiece, state: this.SWAPPED };
            newBoard[start.row][start.col] = { ...targetPiece, state: this.SWAPPED };
        }

        return { success: true, newBoard: newBoard, error: null };
    }

    /**
     * Check if the game has ended
     * @param {Array} boardState - 2D board array
     * @returns {Object} - {isTerminal: boolean, winner: string|null}
     */
    checkGameEnd(boardState) {
        const playerAResult = this.checkWinConditionForPlayer(boardState, this.PLAYER_A);
        const playerBResult = this.checkWinConditionForPlayer(boardState, this.PLAYER_B);

        if (playerAResult.win && playerBResult.win) {
            return { isTerminal: true, winner: 'both' };
        } else if (playerAResult.win) {
            return { isTerminal: true, winner: this.PLAYER_A };
        } else if (playerBResult.win) {
            return { isTerminal: true, winner: this.PLAYER_B };
        }

        return { isTerminal: false, winner: null };
    }

    /**
     * Check win condition for a specific player
     * @param {Array} boardState - 2D board array
     * @param {string} player - PLAYER_A or PLAYER_B
     * @returns {Object} - {win: boolean, path: Array}
     */
    checkWinConditionForPlayer(boardState, player) {
        // Define starting and target areas for each player
        const startRows = (player === this.PLAYER_A) ? [this.ROWS - 2, this.ROWS - 1] : [0, 1];  // A: rows 6,7; B: rows 0,1
        const targetRows = (player === this.PLAYER_A) ? [0, 1] : [this.ROWS - 2, this.ROWS - 1]; // A: rows 0,1; B: rows 6,7

        const visited = Array(this.ROWS).fill(null).map(() => Array(this.COLS).fill(false));
        const queue = [];

        // Find all pieces in the starting area and add them to the queue
        for (const startRow of startRows) {
            for (let c = 0; c < this.COLS; c++) {
                if (boardState[startRow] && boardState[startRow][c] && 
                    boardState[startRow][c].player === player) {
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
                return { win: true, path: path };
            }

            // Explore neighbors
            for (let dr = -1; dr <= 1; dr++) {
                for (let dc = -1; dc <= 1; dc++) {
                    if (dr === 0 && dc === 0) continue;

                    const nr = row + dr;
                    const nc = col + dc;

                    if (this.isValidPosition(nr, nc) && !visited[nr][nc] &&
                        boardState[nr] && boardState[nr][nc] && 
                        boardState[nr][nc].player === player) {
                        visited[nr][nc] = true;
                        queue.push({
                            row: nr,
                            col: nc,
                            path: [...path, { row: nr, col: nc }]
                        });
                    }
                }
            }
        }

        return { win: false, path: [] };
    }

    /**
     * Convert board state (or history) to neural network input format (history stacking)
     * @param {Array} historyBoards - Array of up to 3 board states (oldest first, newest last), or a single board
     * @param {string} currentPlayer - Current player
     * @returns {Float32Array} - Neural network input
     */
    boardToNNInput(historyBoards, currentPlayer) {
        // If a single board is provided, wrap it in an array for backward compatibility
        if (!Array.isArray(historyBoards) || (historyBoards.length && !Array.isArray(historyBoards[0]))) {
            historyBoards = [historyBoards];
        }
        const flatBoardSize = this.ROWS * this.COLS; // 32
        const singleBoardSize = 192; // 6 * 32
        const totalSize = 576; // 3 * 192
        const nnInput = new Float32Array(totalSize);
        for (let h = 0; h < 3; h++) {
            const board = historyBoards[h] ? historyBoards[h] : this._createEmptyBoard();
            const offset = h * singleBoardSize;
            for (let r = 0; r < this.ROWS; r++) {
                for (let c = 0; c < this.COLS; c++) {
                    const pos = r * this.COLS + c;
                    const pieceData = board[r][c];
                    if (pieceData) {
                        if (pieceData.player === this.PLAYER_A) {
                            if (pieceData.state === this.NORMAL) {
                                nnInput[offset + pos] = 1.0;
                            } else {
                                nnInput[offset + flatBoardSize + pos] = 1.0;
                            }
                        } else {
                            if (pieceData.state === this.NORMAL) {
                                nnInput[offset + 2 * flatBoardSize + pos] = 1.0;
                            } else {
                                nnInput[offset + 3 * flatBoardSize + pos] = 1.0;
                            }
                        }
                    } else {
                        nnInput[offset + 4 * flatBoardSize + pos] = 1.0;
                    }
                }
            }
            // Player channel for this board (for current board only)
            if (h === 2) {
                const playerValue = (currentPlayer === this.PLAYER_A) ? 0.0 : 1.0;
                for (let pos = 0; pos < flatBoardSize; pos++) {
                    nnInput[offset + 5 * flatBoardSize + pos] = playerValue;
                }
            }
        }
        return nnInput;
    }

    _createEmptyBoard() {
        const board = [];
        for (let r = 0; r < this.ROWS; r++) {
            const row = [];
            for (let c = 0; c < this.COLS; c++) {
                row.push(null);
            }
            board.push(row);
        }
        return board;
    }

    /**
     * Get the next player
     * @param {string} currentPlayer - Current player
     * @returns {string} - Next player
     */
    getNextPlayer(currentPlayer) {
        return (currentPlayer === this.PLAYER_A) ? this.PLAYER_B : this.PLAYER_A;
    }

    /**
     * Clone a board state
     * @param {Array} boardState - 2D board array
     * @returns {Array} - Cloned board
     */
    cloneBoard(boardState) {
        return boardState.map(row => 
            row.map(cell => cell ? { ...cell } : null)
        );
    }

    /**
     * Convert move to action index
     * @param {Object} move - {start: {row, col}, end: {row, col}}
     * @returns {number|null} - Action index or null if invalid
     */
    moveToActionIndex(move) {
        const startCellIndex = move.start.row * this.COLS + move.start.col;
        const dr = move.end.row - move.start.row;
        const dc = move.end.col - move.start.col;

        let directionIndex = -1;
        for (let i = 0; i < this.JS_DIRECTIONS.length; i++) {
            if (this.JS_DIRECTIONS[i].dr === dr && this.JS_DIRECTIONS[i].dc === dc) {
                directionIndex = i;
                break;
            }
        }

        if (directionIndex === -1) {
            return null;
        }

        return startCellIndex * this.NUM_DIRECTIONS + directionIndex;
    }

    /**
     * Convert action index to move
     * @param {number} actionIndex - Action index
     * @returns {Object|null} - Move object or null if invalid
     */
    actionIndexToMove(actionIndex) {
        if (actionIndex < 0 || actionIndex >= 256) return null;

        const startCellIndex = Math.floor(actionIndex / this.NUM_DIRECTIONS);
        const directionIndex = actionIndex % this.NUM_DIRECTIONS;

        if (directionIndex >= this.JS_DIRECTIONS.length) return null;

        const startRow = Math.floor(startCellIndex / this.COLS);
        const startCol = startCellIndex % this.COLS;

        if (!this.isValidPosition(startRow, startCol)) return null;

        const direction = this.JS_DIRECTIONS[directionIndex];
        const endRow = startRow + direction.dr;
        const endCol = startCol + direction.dc;

        if (!this.isValidPosition(endRow, endCol)) return null;

        return {
            start: { row: startRow, col: startCol },
            end: { row: endRow, col: endCol }
        };
    }

    /**
     * Check if position is valid
     * @param {number} row - Row index
     * @param {number} col - Column index
     * @returns {boolean} - True if valid
     */
    isValidPosition(row, col) {
        return row >= 0 && row < this.ROWS && col >= 0 && col < this.COLS;
    }

    /**
     * Unmark swapped pieces for a player
     * @param {Array} boardState - 2D board array (modified in place)
     * @param {string} player - Player whose pieces to unmark
     */
    unmarkSwapped(board, player) {
        for (let r = 0; r < this.ROWS; r++) {
            for (let c = 0; c < this.COLS; c++) {
                if (board[r][c] && 
                    board[r][c].player === player && 
                    board[r][c].state === this.SWAPPED) {
                    board[r][c].state = this.NORMAL;
                }
            }
        }
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { SwitcharooGameLogic };
}
