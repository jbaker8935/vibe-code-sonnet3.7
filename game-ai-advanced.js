// Advanced AI module: Contains sophisticated findBestAIMove function and helper functions
// Extracted from script.js and adapted for modular system

import {
    ROWS, COLS, PLAYER_A, PLAYER_B, NORMAL, SWAPPED, NUM_DIRECTIONS, JS_DIRECTIONS
} from './game-constants.js';
import {
    cloneBoard
} from './game-board.js';
import {
    calculateLegalMoves, unmarkSwapped, checkWinCondition
} from './game-logic.js';
import {
    boardToNNInput, neuralNetworkPredict
} from './game-ai.js';

// Note: TensorFlow.js is assumed to be available globally as 'tf'

// Helper function to convert move to action index
export function moveToActionIndex(move) {
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
        console.error("Could not find direction for move:", JSON.stringify(move));
        return null;
    }
    const actionIndex = startCellIndex * NUM_DIRECTIONS + directionIndex;
    
    // Validate action index range
    if (actionIndex < 0 || actionIndex >= 256) {
        console.error(`INVALID ACTION INDEX: ${actionIndex} for move (${move.start.row},${move.start.col}) → (${move.end.row},${move.end.col})`);
        console.error(`  Details: startCell=${startCellIndex}, direction=${directionIndex}, dr=${dr}, dc=${dc}`);
        return null; // Return null for invalid indices
    }
    
    return actionIndex;
}

// Helper function to calculate legal moves for a given board state
export function calculateLegalMovesForState(boardState, row, col) {
    const moves = [];
    const piece = boardState[row][col];
    if (!piece) return moves;

    // Use the same logic as calculateLegalMoves but for any board state
    for (const direction of JS_DIRECTIONS) {
        const newRow = row + direction.dr;
        const newCol = col + direction.dc;
        
        if (newRow >= 0 && newRow < ROWS && newCol >= 0 && newCol < COLS) {
            const targetCell = boardState[newRow][newCol];
            if (!targetCell) {
                // Empty cell - valid move
                moves.push({ row: newRow, col: newCol, isSwap: false });
            } else if (targetCell.player !== piece.player) {
                // Opponent piece - can only swap with NORMAL pieces, not SWAPPED ones
                if (targetCell.state === NORMAL) {
                    moves.push({ row: newRow, col: newCol, isSwap: true });
                }
            }
        }
    }
    return moves;
}

// Helper function to check if a player can win in one move
export function canWinInOneMove(boardState, player) {
    // console.log(`[canWinInOneMove] Checking for player: ${player}`);
    // console.log('[canWinInOneMove] Initial boardState:');
    // logBoardForDebug(boardState);

    // Check all possible moves for the player
    for (let r = 0; r < ROWS; r++) {
        for (let c = 0; c < COLS; c++) {
            if (boardState[r][c] && boardState[r][c].player === player) {
                const moves = calculateLegalMovesForState(boardState, r, c);
                for (const move of moves) {
                    const tempBoard = cloneBoard(boardState);
                    const movingPiece = tempBoard[r][c];
                    const targetPiece = tempBoard[move.row][move.col];
                    
                    // console.log(`[canWinInOneMove] Player ${player} trying move from (${r},${c}) to (${move.row},${move.col})`);

                    if (!targetPiece) {
                        // Move to empty cell
                        tempBoard[move.row][move.col] = { ...movingPiece };
                        tempBoard[r][c] = null;
                        unmarkSwapped(tempBoard);
                    } else {
                        // Swap move
                        tempBoard[move.row][move.col] = { ...movingPiece, state: SWAPPED };
                        tempBoard[r][c] = { ...targetPiece, state: SWAPPED };
                    }
                    
                    // console.log('[canWinInOneMove] Board before checkWinCondition:');
                    // logBoardForDebug(tempBoard);
                    
                    // Check if this move results in a win
                    const winResult = checkWinCondition(tempBoard, player);
                    // console.log(`[canWinInOneMove] winResult for move (${r},${c})->(${move.row},${move.col}):`, JSON.stringify(winResult));

                    
                    if (winResult.win) {
                        // console.log(`[canWinInOneMove] WIN DETECTED for player ${player} with move from (${r},${c}) to (${move.row},${move.col})`);
                        // console.log('[canWinInOneMove] Winning board state:');
                        // logBoardForDebug(tempBoard);
                        // console.log('[canWinInOneMove] Win path:', JSON.stringify(winResult.path));
                        return {
                            canWin: true,
                            winningMove: {
                                start: { row: r, col: c },
                                end: { row: move.row, col: move.col },
                                isSwap: !!targetPiece
                            },
                            winResult: winResult
                        };
                    }
                }
            }
        }
    }

    
    return { canWin: false };
}

// Helper function to check if a move allows opponent to win (immediate or next move)
export function allowsOpponentWin(boardState, opponent, aiDifficulty = 'hard1') {


    // 1. Immediate win check
    const immediateWin = canWinInOneMove(boardState, opponent);
    if (immediateWin.canWin) {

        return true;
    }

    // For 'easy' mode, if the opponent cannot win immediately, the move is considered safe.
    // Deeper checks are skipped for 'easy' difficulty.
    if (aiDifficulty === 'easy') {
        return false;
    }

    // For hard1/hard_ai: check for forced loss (all responses unsafe)
    if (aiDifficulty === 'hard1' || aiDifficulty === 'hard_ai') {
        const currentPlayer = opponent === PLAYER_A ? PLAYER_B : PLAYER_A;
        // For each opponent move, see if all responses by current player are unsafe
        for (let r = 0; r < ROWS; r++) {
            for (let c = 0; c < COLS; c++) {
                if (boardState[r][c] && boardState[r][c].player === opponent) {
                    const moves = calculateLegalMovesForState(boardState, r, c);
                    for (const move of moves) {
                        const tempBoard = cloneBoard(boardState);
                        const movingPiece = tempBoard[r][c];
                        const targetPiece = tempBoard[move.row][move.col];
                        if (!targetPiece) {
                            tempBoard[move.row][move.col] = { ...movingPiece };
                            tempBoard[r][c] = null;
                            unmarkSwapped(tempBoard);
                        } else {
                            tempBoard[move.row][move.col] = { ...movingPiece, state: SWAPPED };
                            tempBoard[r][c] = { ...targetPiece, state: SWAPPED };
                        }
                        if (allCurrentPlayerMovesUnsafe(tempBoard, currentPlayer, opponent)) {
                            console.log("Testing Board State:");
                            logBoardForDebug(boardState);
                            console.log(`[allowsOpponentWin] Opponent can create a forced move (all responses unsafe) after: (${r},${c}) -> (${move.row},${move.col})`);
                            logBoardForDebug(tempBoard);

                            return true;
                        }
                    }
                }
            }
        }
    }

    return false;
}

// Helper: are all current player's moves unsafe (allow immediate or next-move win by opponent)?
function allCurrentPlayerMovesUnsafe(boardState, currentPlayer, opponent) {

    let foundSafe = false;
    for (let r = 0; r < ROWS; r++) {
        for (let c = 0; c < COLS; c++) {
            if (boardState[r][c] && boardState[r][c].player === currentPlayer) {
                const moves = calculateLegalMovesForState(boardState, r, c);
                for (const move of moves) {
                    const tempBoard = cloneBoard(boardState);
                    const movingPiece = tempBoard[r][c];
                    const targetPiece = tempBoard[move.row][move.col];
                    if (!targetPiece) {
                        tempBoard[move.row][move.col] = { ...movingPiece };
                        tempBoard[r][c] = null;
                        unmarkSwapped(tempBoard);
                    } else {
                        tempBoard[move.row][move.col] = { ...movingPiece, state: SWAPPED };
                        tempBoard[r][c] = { ...targetPiece, state: SWAPPED };
                    }
                    if (!allowsOpponentWin(tempBoard, opponent, 'easy')) {
                        foundSafe = true;
                        break;
                    }
                }
                if (foundSafe) break;
            }
        }
        if (foundSafe) break;
    }
    return !foundSafe;
}

// Helper: log board for debug
export function logBoardForDebug(board) {
    let out = '';
    for (let r = 0; r < ROWS; r++) {
        let rowStr = '';
        for (let c = 0; c < COLS; c++) {
            const piece = board[r][c];
            if (!piece) {
                rowStr += '.';
            } else {
                const char = piece.player === PLAYER_A ? 'A' : 'B';
                rowStr += piece.state === SWAPPED ? char.toLowerCase() : char;
            }
        }
        out += rowStr + '\n';
    }
    console.log(out);
}

// Helper function to evaluate board state heuristically
export function evaluateBoardState(boardState, player, lastMove = null) {
    let score = 0;
    let occupiedRows = new Map();
    // Basic positional scoring: advancing pieces get higher scores
    for (let r = 0; r < ROWS; r++) {
        for (let c = 0; c < COLS; c++) {
            const piece = boardState[r][c];
            if (piece && piece.player === player) {
                if (r >= 1 && r <= 6) {
                    occupiedRows.set(r, true); // Track occupied rows by this player
                }
                
                // Bonus for pieces in center columns some of the time
                if (Math.random() < 0.5) {
                    const centerDistance = Math.abs(c - Math.floor(COLS / 2));
                    score += Math.max(0, 3 - centerDistance);
                }
                
                // Count friendly neighbors for formation bonus
                score += countFriendlyNeighbors(boardState, r, c, player);
            }
        }
    }
    // Bonus for occupying more rows
    const occupiedRowCount = occupiedRows.size;
    score += occupiedRowCount * 10; // 10 points per occupied row
    
    return score;
}

// Helper function to count friendly neighbors
export function countFriendlyNeighbors(boardState, row, col, player) {
    let count = 0;
    for (const direction of JS_DIRECTIONS) {
        const newRow = row + direction.dr;
        const newCol = col + direction.dc;
        if (newRow >= 0 && newRow < ROWS && newCol >= 0 && newCol < COLS) {
            const neighbor = boardState[newRow][newCol];
            if (neighbor && neighbor.player === player) {
                count++;
            }
        }
    }
    return Math.min(count,2); // Limit to 2 for balance
}

// Helper function to check win condition for a given board state
export function checkWinConditionForState(boardState, player) {
    return checkWinCondition(boardState, player);
}

// Helper function to check if a player has valid defensive responses
function hasValidDefensiveResponse(boardState, currentPlayer, opponent, depth) {
    if (depth <= 0) return true; // Assume defensive move exists at max depth

    // Get all possible moves for current player
    for (let r = 0; r < ROWS; r++) {
        for (let c = 0; c < COLS; c++) {
            if (boardState[r][c] && boardState[r][c].player === currentPlayer) {
                const moves = calculateLegalMovesForState(boardState, r, c);
                for (const move of moves) {
                    const tempBoard = cloneBoard(boardState);
                    const movingPiece = tempBoard[r][c];
                    const targetPiece = tempBoard[move.row][move.col];
                    
                    if (!targetPiece) {
                        tempBoard[move.row][move.col] = { ...movingPiece };
                        tempBoard[r][c] = null;
                        unmarkSwapped(tempBoard);
                    } else {
                        tempBoard[move.row][move.col] = { ...movingPiece, state: SWAPPED };
                        tempBoard[r][c] = { ...targetPiece, state: SWAPPED };
                    }
                    
                    // Check if this defensive move prevents the opponent from winning
                    if (!allowsOpponentWin(tempBoard, opponent, depth - 1)) {
                        return true; // Found a valid defensive move
                    }
                }
            }
        }
    }
    return false; // No valid defensive moves found
}

// Helper to build board history array for NN input (oldest first, newest last)
function buildHistoryArray(moveHistory, currentBoard) {
    // moveHistory: array of { boardAfter: ... } objects
    // Returns up to 3 boards: [oldest, ..., current]
    const arr = [];
    if (moveHistory && moveHistory.length >= 2) arr.push(moveHistory[moveHistory.length-2].boardAfter);
    if (moveHistory && moveHistory.length >= 1) arr.push(moveHistory[moveHistory.length-1].boardAfter);
    while (arr.length < 2) arr.unshift(null);
    arr.push(currentBoard);
    return arr;
}

// Main sophisticated AI function
export async function findBestAIMove(boardState, player = PLAYER_B, aiDifficulty = 'hard_ai', aiDepth = 1, analysisMode = false, mctsEnabled = false, mctsSimulations = 50, tfModel = null, mctsSearch = null, gameLogic = null, moveHistory = null) {
    let possibleMoves = [];
    for (let r = 0; r < ROWS; r++) {
        for (let c = 0; c < COLS; c++) {
            if (boardState[r][c] && boardState[r][c].player === player) {
                const moves = calculateLegalMovesForState(boardState, r, c);
                moves.forEach(move => {
                    possibleMoves.push({ start: { row: r, col: c }, end: { row: move.row, col: move.col }, isSwap: move.isSwap });
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
            unmarkSwapped(tempBoard);
        } else {
            tempBoard[end.row][end.col] = { ...movingPiece, state: SWAPPED };
            tempBoard[start.row][start.col] = { ...targetPiece, state: SWAPPED };
        }

        if (checkWinConditionForState(tempBoard, player).win) {
            if (analysisMode) {
                console.log(`Found immediate winning move: ${start.row},${start.col} to ${end.row},${end.col}`);
            }
            return move;
        }
    }

    // Build history array for NN input
    const nnHistory = buildHistoryArray(moveHistory, boardState);

    // If using neural network model
    if (aiDifficulty === 'hard_ai') {
        if (!tfModel) {
            console.warn("Neural network model not available. Falling back to heuristic AI (hard1).\n");
            // Recursively call with hard1 difficulty as fallback
            return await findBestAIMove(boardState, player, 'hard1', aiDepth, analysisMode, mctsEnabled, mctsSimulations, tfModel, mctsSearch, gameLogic, moveHistory);
        }
        // Use MCTS if enabled and available
        if (mctsEnabled && mctsSearch && gameLogic) {
            const neuralNetworkPredictWithModel = async (nnInput) => {
                return await neuralNetworkPredict(tfModel, nnInput);
            };
            // Pass nnHistory to MCTS search
            const actionProbs = await mctsSearch.search(boardState, player, neuralNetworkPredictWithModel, gameLogic, nnHistory);
            // Convert action probabilities to move
            const legalMoves = [];
            for (let r = 0; r < ROWS; r++) {
                for (let c = 0; c < COLS; c++) {
                    if (boardState[r][c] && boardState[r][c].player === player) {
                        const moves = calculateLegalMovesForState(boardState, r, c);
                        moves.forEach(move => {
                            legalMoves.push({ start: { row: r, col: c }, end: { row: move.row, col: move.col }, isSwap: move.isSwap });
                        });
                    }
                }
            }
            if (legalMoves.length > 0) {
                // Filter out moves that allow opponent to win immediately or force a loss
                const opponent = player === PLAYER_A ? PLAYER_B : PLAYER_A;
                const safeMoves = legalMoves.filter(move => {
                    const tempBoard = cloneBoard(boardState);
                    const { start, end } = move;
                    const movingPiece = tempBoard[start.row][start.col];
                    const targetPiece = tempBoard[end.row][end.col];
                    if (!targetPiece) {
                        tempBoard[end.row][end.col] = { ...movingPiece };
                        tempBoard[start.row][start.col] = null;
                        unmarkSwapped(tempBoard);
                    } else {
                        tempBoard[end.row][end.col] = { ...movingPiece, state: SWAPPED };
                        tempBoard[start.row][start.col] = { ...targetPiece, state: SWAPPED };
                    }
                    return !allowsOpponentWin(tempBoard, opponent, aiDifficulty);
                });
                // Select move based on MCTS action probabilities, but only among safe moves if any
                let bestMove = null;
                let bestProb = -1;
                const movesToConsider = safeMoves.length > 0 ? safeMoves : legalMoves;
                for (const move of movesToConsider) {
                    const actionIdx = moveToActionIndex(move);
                    if (actionIdx !== null && actionProbs[actionIdx] > bestProb) {
                        bestProb = actionProbs[actionIdx];
                        bestMove = move;
                    }
                }
                if (bestMove && analysisMode) {
                    console.log("Best move from MCTS (analysis):", bestMove, "prob:", bestProb);
                }
                if (bestMove) {
                    return bestMove;
                }
            }
        }
        // Fallback to neural network policy only (no MCTS)
        const nnInput = boardToNNInput(nnHistory, player);
        const { policy } = await neuralNetworkPredict(tfModel, nnInput);
        // Pick the legal move with highest policy probability
        let bestMove = null;
        let bestProb = -1;
        for (const move of possibleMoves) {
            const actionIdx = moveToActionIndex(move);
            if (actionIdx !== null && policy[actionIdx] > bestProb) {
                bestProb = policy[actionIdx];
                bestMove = move;
            }
        }
        if (bestMove) return bestMove;
        // Fallback to heuristic if all else fails
    }

    // Heuristic AI (hard1) - evaluate all possible moves and select the best one
    let bestMove = null;
    let bestScore = -Infinity;
    const opponent = player === PLAYER_A ? PLAYER_B : PLAYER_A;
    for (const move of possibleMoves) {
        const tempBoard = cloneBoard(boardState);
        const { start, end } = move;
        const movingPiece = tempBoard[start.row][start.col];
        const targetPiece = tempBoard[end.row][end.col];
        if (!targetPiece) {
            tempBoard[end.row][end.col] = { ...movingPiece };
            tempBoard[start.row][start.col] = null;
            unmarkSwapped(tempBoard);
        } else {
            tempBoard[end.row][end.col] = { ...movingPiece, state: SWAPPED };
            tempBoard[start.row][start.col] = { ...targetPiece, state: SWAPPED };
        }
        // Evaluate the board state after the hypothetical move
        let score = evaluateBoardState(tempBoard, player);
        // Check if this move is safe (doesn't allow opponent to win)
        const isSafe = !allowsOpponentWin(tempBoard, opponent, aiDifficulty);
        if (isSafe) {
            // Give higher preference to moves that are safe
            score += 1000;
        }
        if (score > bestScore) {
            bestScore = score;
            bestMove = move;
        }
    }
    return bestMove;
}
