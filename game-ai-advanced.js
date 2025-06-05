// Advanced AI module: Contains sophisticated findBestAIMove function and helper functions
// Extracted from script.js and adapted for modular system

import {
    ROWS, COLS, PLAYER_A, PLAYER_B, NORMAL, SWAPPED, NUM_DIRECTIONS, JS_DIRECTIONS
} from './game-constants.js';
import {
    cloneBoard
} from './game-board.js';
import {
    calculateLegalMoves, unmarkPlayerSwapped, checkWinCondition
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
    // Check all possible moves for the player
    for (let r = 0; r < ROWS; r++) {
        for (let c = 0; c < COLS; c++) {
            if (boardState[r][c] && boardState[r][c].player === player) {
                const moves = calculateLegalMovesForState(boardState, r, c);
                for (const move of moves) {
                    const tempBoard = cloneBoard(boardState);
                    const movingPiece = tempBoard[r][c];
                    const targetPiece = tempBoard[move.row][move.col];
                    
                    if (!targetPiece) {
                        // Move to empty cell
                        tempBoard[move.row][move.col] = { ...movingPiece };
                        tempBoard[r][c] = null;
                        unmarkPlayerSwapped(player, tempBoard);
                    } else {
                        // Swap move
                        tempBoard[move.row][move.col] = { ...movingPiece, state: SWAPPED };
                        tempBoard[r][c] = { ...targetPiece, state: SWAPPED };
                    }
                    
                    // Check if this move results in a win
                    const winResult = checkWinCondition(tempBoard, player);
                    if (winResult.win) {
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

// Helper function to check if a move allows opponent to win
export function allowsOpponentWin(boardState, opponent, depth = 1, aiDifficulty = 'hard1') {
    if (depth <= 0) return false;

    // Use the new canWinInOneMove function for immediate win check
    const immediateWin = canWinInOneMove(boardState, opponent);
    if (immediateWin.canWin) {
        return true;
    }

    // For easy mode: only check immediate wins and direct one-move threats
    if (aiDifficulty === 'easy') {
        // Easy mode only checks:
        // 1. If opponent can win immediately (already checked above)
        // 2. If opponent can win on their next move after our move
        
        if (depth > 1) {
            // Check if opponent can make a move that wins immediately on their next turn
            const currentPlayer = opponent === PLAYER_A ? PLAYER_B : PLAYER_A;
            
            // Get opponent's possible moves
            let opponentMoves = [];
            for (let r = 0; r < ROWS; r++) {
                for (let c = 0; c < COLS; c++) {
                    if (boardState[r][c] && boardState[r][c].player === opponent) {
                        const moves = calculateLegalMovesForState(boardState, r, c);
                        moves.forEach(move => {
                            opponentMoves.push({
                                start: { row: r, col: c },
                                end: { row: move.row, col: move.col },
                                isSwap: !!move.isSwap
                            });
                        });
                    }
                }
            }

            // Check if any opponent move leads to an immediate win
            for (const opponentMove of opponentMoves) {
                const tempBoard = cloneBoard(boardState);
                const { start, end } = opponentMove;
                const movingPiece = tempBoard[start.row][start.col];
                const targetPiece = tempBoard[end.row][end.col];

                // Apply opponent's move
                if (!targetPiece) {
                    tempBoard[end.row][end.col] = { ...movingPiece };
                    tempBoard[start.row][start.col] = null;
                    unmarkPlayerSwapped(opponent, tempBoard);
                } else {
                    tempBoard[end.row][end.col] = { ...movingPiece, state: SWAPPED };
                    tempBoard[start.row][start.col] = { ...targetPiece, state: SWAPPED };
                }

                // Check if this move wins immediately for opponent
                const opponentWinsNow = checkWinCondition(tempBoard, opponent);
                if (opponentWinsNow.win) {
                    return true;
                }
            }
        }
        
        return false; // Easy mode stops here - no forced move detection
    }

    // Enhanced forced loss detection for hard1 and hard_ai modes
    if (depth > 1 && (aiDifficulty === 'hard1' || aiDifficulty === 'hard_ai')) {
        const currentPlayer = opponent === PLAYER_A ? PLAYER_B : PLAYER_A;
        
        // Get all possible moves for the opponent
        let opponentMoves = [];
        for (let r = 0; r < ROWS; r++) {
            for (let c = 0; c < COLS; c++) {
                if (boardState[r][c] && boardState[r][c].player === opponent) {
                    const moves = calculateLegalMovesForState(boardState, r, c);
                    moves.forEach(move => {
                        opponentMoves.push({
                            start: { row: r, col: c },
                            end: { row: move.row, col: move.col },
                            isSwap: !!move.isSwap
                        });
                    });
                }
            }
        }

        // Check if opponent has any move that forces the current player into a losing position
        for (const opponentMove of opponentMoves) {
            const tempBoard = cloneBoard(boardState);
            const { start, end } = opponentMove;
            const movingPiece = tempBoard[start.row][start.col];
            const targetPiece = tempBoard[end.row][end.col];

            // Apply opponent's move
            if (!targetPiece) {
                tempBoard[end.row][end.col] = { ...movingPiece };
                tempBoard[start.row][start.col] = null;
                unmarkPlayerSwapped(opponent, tempBoard);
            } else {
                tempBoard[end.row][end.col] = { ...movingPiece, state: SWAPPED };
                tempBoard[start.row][start.col] = { ...targetPiece, state: SWAPPED };
            }

            // Check if this move wins immediately for opponent
            const opponentWinsNow = checkWinCondition(tempBoard, opponent);
            if (opponentWinsNow.win) {
                return true;
            }

            // Check if this move leaves current player with no safe responses (forced loss)
            if (allCurrentPlayerMovesAllowOpponentWin(tempBoard, currentPlayer, opponent, depth - 1, aiDifficulty)) {
                return true;
            }
        }
    }
    
    return false;
}

// Helper function to check if all moves by current player allow opponent to win
function allCurrentPlayerMovesAllowOpponentWin(boardState, currentPlayer, opponent, depth, aiDifficulty = 'hard1') {
    if (depth <= 0) return false;

    // Get all possible moves for current player
    let currentPlayerMoves = [];
    for (let r = 0; r < ROWS; r++) {
        for (let c = 0; c < COLS; c++) {
            if (boardState[r][c] && boardState[r][c].player === currentPlayer) {
                const moves = calculateLegalMovesForState(boardState, r, c);
                moves.forEach(move => {
                    currentPlayerMoves.push({
                        start: { row: r, col: c },
                        end: { row: move.row, col: move.col },
                        isSwap: !!move.isSwap
                    });
                });
            }
        }
    }

    // If current player has no moves, they lose
    if (currentPlayerMoves.length === 0) {
        return true;
    }

    // Check if ALL moves by current player allow opponent to win
    for (const playerMove of currentPlayerMoves) {
        const tempBoard = cloneBoard(boardState);
        const { start, end } = playerMove;
        const movingPiece = tempBoard[start.row][start.col];
        const targetPiece = tempBoard[end.row][end.col];

        // Apply current player's move
        if (!targetPiece) {
            tempBoard[end.row][end.col] = { ...movingPiece };
            tempBoard[start.row][start.col] = null;
            unmarkPlayerSwapped(currentPlayer, tempBoard);
        } else {
            tempBoard[end.row][end.col] = { ...movingPiece, state: SWAPPED };
            tempBoard[start.row][start.col] = { ...targetPiece, state: SWAPPED };
        }

        // Check if opponent can win after this current player move
        // Use recursion to check if opponent can force a win from this position
        if (!allowsOpponentWin(tempBoard, opponent, depth, aiDifficulty)) {
            // Found at least one safe move for current player
            return false;
        }
    }

    // All moves by current player allow opponent to win
    return true;
}

// Helper function to evaluate board state heuristically
export function evaluateBoardState(boardState, player, lastMove = null) {
    let score = 0;
    
    // Basic positional scoring: advancing pieces get higher scores
    for (let r = 0; r < ROWS; r++) {
        for (let c = 0; c < COLS; c++) {
            const piece = boardState[r][c];
            if (piece && piece.player === player) {
                if (player === PLAYER_B) {
                    // Player B wants to advance (lower row numbers)
                    score += (ROWS - r) * 10;
                } else {
                    // Player A wants to advance (higher row numbers)
                    score += r * 10;
                }
                
                // Bonus for pieces in center columns
                const centerDistance = Math.abs(c - Math.floor(COLS / 2));
                score += Math.max(0, 3 - centerDistance) * 5;
                
                // Count friendly neighbors for formation bonus
                score += countFriendlyNeighbors(boardState, r, c, player) * 3;
            }
        }
    }
    
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
    return count;
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
                        unmarkPlayerSwapped(currentPlayer, tempBoard);
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

// Main sophisticated AI function
export async function findBestAIMove(boardState, player = PLAYER_B, aiDifficulty = 'hard_ai', aiDepth = 1, analysisMode = false, mctsEnabled = false, mctsSimulations = 50, tfModel = null, mctsSearch = null, gameLogic = null) {
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
            unmarkPlayerSwapped(player, tempBoard);
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

    // If using neural network model
    if (aiDifficulty === 'hard_ai') {

        
        if (!tfModel) {
            console.warn("Neural network model not available. Falling back to heuristic AI (hard1).");
            // Recursively call with hard1 difficulty as fallback
            return await findBestAIMove(boardState, player, 'hard1', aiDepth, analysisMode, mctsEnabled, mctsSimulations, tfModel, mctsSearch, gameLogic);
        }
        
        console.log("Using neural network for move selection");

        // Use MCTS if enabled and available
        if (mctsEnabled && mctsSearch && gameLogic) {
            console.log(`Using MCTS with ${mctsSimulations} simulations`);
            try {
                // Create a wrapper function that includes tfModel
                const neuralNetworkPredictWithModel = async (nnInput) => {
                    return await neuralNetworkPredict(tfModel, nnInput);
                };
                
                const actionProbs = await mctsSearch.search(boardState, player, neuralNetworkPredictWithModel, gameLogic);

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
                            unmarkPlayerSwapped(player, tempBoard);
                        } else {
                            tempBoard[end.row][end.col] = { ...movingPiece, state: SWAPPED };
                            tempBoard[start.row][start.col] = { ...targetPiece, state: SWAPPED };
                        }
                        return !allowsOpponentWin(tempBoard, opponent, aiDepth, aiDifficulty);
                    });

                    // Select move based on MCTS action probabilities, but only among safe moves if any
                    let bestMove = null;
                    let bestProb = -1;
                    const movesToConsider = safeMoves.length > 0 ? safeMoves : legalMoves;
                    for (const move of movesToConsider) {
                        const actionIndex = moveToActionIndex(move);
                        if (actionIndex !== null && actionIndex >= 0 && actionIndex < actionProbs.length) {
                            const prob = actionProbs[actionIndex];
                            if (prob > bestProb) {
                                bestProb = prob;
                                bestMove = move;
                            }
                        }
                    }

                    if (bestMove && analysisMode) {
                        console.log(`MCTS selected move: ${bestMove.start.row},${bestMove.start.col} to ${bestMove.end.row},${bestMove.end.col} (prob: ${bestProb.toFixed(4)})`);
                        if (safeMoves.length === 0) {
                            console.warn('All MCTS moves allow opponent win; forced to pick among losing moves.');
                        }
                    }

                    if (bestMove) {
                        return bestMove;
                    }
                }
            } catch (error) {
                console.error("Error during MCTS move selection:", error);
            }
        }

        // Fallback to heuristic evaluation for move selection
        let bestMove = null;
        let bestScore = -Infinity;
        for (const move of possibleMoves) {
            const tempBoard = cloneBoard(boardState);
            const { start, end } = move;
            const movingPiece = tempBoard[start.row][start.col];
            const targetPiece = tempBoard[end.row][end.col];

            if (!targetPiece) {
                tempBoard[end.row][end.col] = { ...movingPiece };
                tempBoard[start.row][start.col] = null;
                unmarkPlayerSwapped(player, tempBoard);
            } else {
                tempBoard[end.row][end.col] = { ...movingPiece, state: SWAPPED };
                tempBoard[start.row][start.col] = { ...targetPiece, state: SWAPPED };
            }

            // Evaluate the board state after the hypothetical move
            const score = evaluateBoardState(tempBoard, player);
            if (score > bestScore) {
                bestScore = score;
                bestMove = move;
            }
        }

        if (bestMove && analysisMode) {
            console.log(`Heuristic selected move: ${bestMove.start.row},${bestMove.start.col} to ${bestMove.end.row},${bestMove.end.col} (score: ${bestScore})`);
        }

        return bestMove;
    }

    // For hard1 difficulty: use heuristic evaluation with forced loss detection
    if (aiDifficulty === 'hard1') {
        let bestMove = null;
        let bestScore = -Infinity;
        for (const move of possibleMoves) {
            const tempBoard = cloneBoard(boardState);
            const { start, end } = move;
            const movingPiece = tempBoard[start.row][start.col];
            const targetPiece = tempBoard[end.row][end.col];

            if (!targetPiece) {
                tempBoard[end.row][end.col] = { ...movingPiece };
                tempBoard[start.row][start.col] = null;
                unmarkPlayerSwapped(player, tempBoard);
            } else {
                tempBoard[end.row][end.col] = { ...movingPiece, state: SWAPPED };
                tempBoard[start.row][start.col] = { ...targetPiece, state: SWAPPED };
            }

            // Evaluate the board state after the hypothetical move
            const score = evaluateBoardState(tempBoard, player);

            // Penalize moves that allow opponent to win immediately
            if (allowsOpponentWin(tempBoard, player === PLAYER_A ? PLAYER_B : PLAYER_A, 1, 'easy')) {
                continue; // Skip this move, it's not safe
            }

            if (score > bestScore) {
                bestScore = score;
                bestMove = move;
            }
        }

        if (bestMove && analysisMode) {
            console.log(`Hard1 heuristic selected move: ${bestMove.start.row},${bestMove.start.col} to ${bestMove.end.row},${bestMove.end.col} (score: ${bestScore})`);
        }

        return bestMove;
    }

    return null;
}
