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

// Helper function to check if a move allows opponent to win
export function allowsOpponentWin(boardState, opponent, depth = 1) {
    if (depth <= 0) return false;

    // Check all possible opponent moves
    for (let r = 0; r < ROWS; r++) {
        for (let c = 0; c < COLS; c++) {
            if (boardState[r][c] && boardState[r][c].player === opponent) {
                const oppMoves = calculateLegalMovesForState(boardState, r, c);
                for (const oppMove of oppMoves) {
                    const tempBoard = cloneBoard(boardState);
                    const movingPiece = tempBoard[r][c];
                    const targetPiece = tempBoard[oppMove.row][oppMove.col];
                    
                    if (!targetPiece) {
                        tempBoard[oppMove.row][oppMove.col] = { ...movingPiece };
                        tempBoard[r][c] = null;
                        unmarkPlayerSwapped(opponent, tempBoard);
                    } else {
                        tempBoard[oppMove.row][oppMove.col] = { ...movingPiece, state: SWAPPED };
                        tempBoard[r][c] = { ...targetPiece, state: SWAPPED };
                    }
                    
                    const win = checkWinCondition(tempBoard, opponent);
                    if (win.win) {
                        return true;
                    }
                    
                    // Recursive check for deeper analysis
                    if (depth > 1) {
                        const currentPlayer = opponent === PLAYER_A ? PLAYER_B : PLAYER_A;
                        if (allowsOpponentWin(tempBoard, currentPlayer, depth - 1)) {
                            return true;
                        }
                    }
                }
            }
        }
    }
    return false;
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
    if (aiDifficulty === 'hard_ai' && tfModel) {
        console.log("Using neural network for move selection");

        // Use MCTS if enabled and available
        if (mctsEnabled && mctsSearch && gameLogic) {
            console.log(`Using MCTS with ${mctsSimulations} simulations`);
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

                    if (bestMove && analysisMode) {
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
        let topMoves = [];

        // Get the neural network input from the CURRENT board state
        const currentNNInput = boardToNNInput(boardState, player);

        if (!currentNNInput || currentNNInput.length !== 192) {
            console.error("Failed to generate valid NN input from current board state. Expected 192 elements but got:", currentNNInput?.length, "Falling back.");
            // Fallback to heuristic AI if NN input generation fails
            const fallbackMove = await findBestAIMove(boardState, player, 'hard1', aiDepth, analysisMode);
            return fallbackMove;
        }

        let inputTensor;
        let outputTensors;
        let policyQValues;

        try {
            // Create a single tensor for the current board state input
            inputTensor = tf.tensor2d([currentNNInput]);

            if (analysisMode) console.log("NN Input Tensor Shape:", inputTensor.shape);

            // Use executeAsync for GraphModel prediction
            const inputNodeName = tfModel.inputs[0].name;
            outputTensors = await tfModel.executeAsync({ [inputNodeName]: inputTensor });

            if (analysisMode) {
                console.log("--- Model Output Inspection (Current State Prediction) ---");
                console.log("tfModel.outputs:", tfModel.outputs);
                if (tfModel.outputs && tfModel.outputs.length > 0) {
                    tfModel.outputs.forEach((output, index) => {
                        console.log(`Output ${index}:`, output.name, output.shape);
                    });
                }
            }

            // Extract policy and value from output
            let policyOutput, valueOutput;
            if (Array.isArray(outputTensors)) {
                // Multiple outputs: policy and value
                policyOutput = outputTensors[0];
                valueOutput = outputTensors.length > 1 ? outputTensors[1] : null;
            } else {
                // Single output: assume it's policy
                policyOutput = outputTensors;
            }

            if (analysisMode) {
                console.log("Policy Output Shape:", policyOutput.shape);
                if (valueOutput) console.log("Value Output Shape:", valueOutput.shape);
            }

            policyQValues = await policyOutput.data();

            // Dispose tensors
            if (inputTensor && !inputTensor.isDisposed) inputTensor.dispose();
            outputTensors.forEach(t => { if (t && !t.isDisposed) t.dispose(); });

            // Process Scores for all possible moves from the CURRENT state
            for (const move of possibleMoves) {
                const { start, end } = move;

                // Create a temporary board to check if the move allows an opponent win
                const tempBoard = cloneBoard(boardState);
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

                // Check if this move allows opponent to win
                const opponent = player === PLAYER_A ? PLAYER_B : PLAYER_A;
                if (allowsOpponentWin(tempBoard, opponent, aiDepth)) {
                    if (analysisMode) {
                        console.log(`NN: Skipping move ${start.row},${start.col} to ${end.row},${end.col}: allows opponent win`);
                    }
                    continue; // Skip moves that allow opponent to win
                }

                const actionIndex = moveToActionIndex(move);

                if (actionIndex === null || actionIndex < 0 || actionIndex >= policyQValues.length) {
                    if (analysisMode) console.warn(`NN: Skipping move due to invalid actionIndex (${actionIndex}) for move:`, move);
                    continue;
                }

                const score = policyQValues[actionIndex];

                if (typeof score === 'number' && !isNaN(score)) {
                    scoredMoves.push({ move, score });
                    if (score > bestScore) {
                        bestScore = score;
                    }
                    if (analysisMode) {
                        console.log(`NN Move ${start.row},${start.col} to ${end.row},${end.col} (Action ${actionIndex}): Q-value ${score.toFixed(4)}`);
                    }
                } else {
                    if (analysisMode) {
                        console.warn(`NN Move ${start.row},${start.col} to ${end.row},${end.col} (Action ${actionIndex}): Q-value is invalid (Score: ${score}, Type: ${typeof score})`);
                    }
                }
            }

            if (scoredMoves.length > 0 && bestScore > -Infinity) {
                // Use a threshold relative to the best score
                let dynamicScoreThreshold;
                if (bestScore >= 0) {
                    dynamicScoreThreshold = bestScore * 0.8;
                } else {
                    dynamicScoreThreshold = bestScore * 1.2;
                }
                const scoreTolerance = 0.05;
                dynamicScoreThreshold = bestScore - Math.abs(bestScore * 0.20) - scoreTolerance;

                if (analysisMode) console.log(`NN: Final bestScore: ${bestScore.toFixed(4)}, dynamicScoreThreshold: ${dynamicScoreThreshold.toFixed(4)}`);

                topMoves = scoredMoves
                    .filter(sm => typeof sm.score === 'number' && !isNaN(sm.score) && sm.score >= dynamicScoreThreshold)
                    .map(({ move }) => move);

                if (analysisMode && topMoves.length === 0 && scoredMoves.length > 0) {
                    console.log("NN: No moves met the dynamic threshold, but scored moves exist. Considering all scored moves as top moves.");
                    topMoves = scoredMoves.filter(sm => sm.score === bestScore).map(({ move }) => move);
                }

            } else {
                if (analysisMode) console.log("NN: No moves scored or bestScore remained -Infinity.");
            }

            if (topMoves.length > 0) {
                const randomIndex = Math.floor(Math.random() * topMoves.length);
                if (analysisMode) console.log(`NN chose from ${topMoves.length} top moves. Best Q-value: ${bestScore.toFixed(4)}`);
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

    // Fallback Heuristic AI Logic (easy, hard1, or if NN fails)
    console.log(`Falling back to heuristic AI: ${aiDifficulty}`);

    if (aiDifficulty === 'easy') {
        // Filter out moves that allow immediate opponent win
        let safeMoves = [];
        for (const move of possibleMoves) {
            // Test each move on a cloned board
            const tempBoard = cloneBoard(boardState);
            const { start, end } = move;

            // Apply move to temporary board
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

            // Check if this move allows opponent to win on their next turn(s)
            const opponent = player === PLAYER_A ? PLAYER_B : PLAYER_A;
            const isSafe = !allowsOpponentWin(tempBoard, opponent, aiDepth);
            if (analysisMode) {
                console.log(`Evaluating move: ${start.row},${start.col} to ${end.row},${end.col} - Safe: ${isSafe}`);
            }
            if (isSafe) {
                safeMoves.push(move);
            }
        }
        if (analysisMode) {
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
        const tempBoard = boardState.map(row =>
            row.map(cell => cell ? { ...cell } : null)
        );

        const movingPiece = tempBoard[move.start.row][move.start.col];
        const targetPiece = tempBoard[move.end.row][move.end.col];

        if (targetPiece === null) {
            // Empty cell move
            tempBoard[move.end.row][move.end.col] = { ...movingPiece };
            tempBoard[move.start.row][move.start.col] = null;
            unmarkPlayerSwapped(player, tempBoard);
        } else {
            // Swap move
            tempBoard[move.end.row][move.end.col] = { ...movingPiece, state: SWAPPED };
            tempBoard[move.start.row][move.start.col] = { ...targetPiece, state: SWAPPED };
        }

        const score = evaluateBoardState(tempBoard, player, move);
        // Check if this move allows opponent to win
        const opponent = player === PLAYER_A ? PLAYER_B : PLAYER_A;
        if (!allowsOpponentWin(tempBoard, opponent, aiDepth)) {
            if (score > bestScore) {
                bestScore = score;
                bestMoves = [move];
            } else if (score === bestScore) {
                bestMoves.push(move);
            }
        } else {
            console.log(`AI Avoids move: ${move.start.row},${move.start.col} -> ${move.end.row},${move.end.col} (allows Player A win)`);
            // If all moves lead to a loss, give these moves a massive penalty instead of excluding them
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
    const lossPenalty = -100000;
    if (bestMoves.length > 0 && bestScore === lossPenalty) {
        console.log("AI: All moves lead to a loss. Forced move.");
    }

    // If bestMoves is empty (shouldn't happen), pick any move
    if (bestMoves.length === 0) {
        console.warn("AI couldn't find a non-losing move, or evaluation error. Picking random move.");
        if (possibleMoves.length > 0) {
            return possibleMoves[Math.floor(Math.random() * possibleMoves.length)];
        } else {
            return null; // Truly no moves available
        }
    }

    // Choose randomly among the best moves
    return bestMoves[Math.floor(Math.random() * bestMoves.length)];
}

// Analyze historical move function
export async function analyzeHistoricalMove(moveHistory, currentHistoryIndex, tfModel = null, aiDifficulty = 'hard_ai', aiDepth = 1, mctsEnabled = false, mctsSimulations = 50, mctsSearch = null, gameLogic = null) {
    if (currentHistoryIndex !== undefined && currentHistoryIndex > 0) {
        const historicalBoard = moveHistory[currentHistoryIndex - 1].boardAfter;

        // Determine whose turn it is for this historical board state
        // The boardAfter represents the state after a move was made
        // So the next player to move is the opposite of who made that move
        const playerWhoMadePreviousMove = moveHistory[currentHistoryIndex - 1].player;
        const playerToMoveNext = playerWhoMadePreviousMove === PLAYER_A ? PLAYER_B : PLAYER_A;

        const analysisMode = true;
        const bestMove = await findBestAIMove(
            historicalBoard, 
            playerToMoveNext, 
            aiDifficulty, 
            aiDepth, 
            analysisMode, 
            mctsEnabled, 
            mctsSimulations, 
            tfModel, 
            mctsSearch, 
            gameLogic
        );
        console.log(`Best move for this board state (Player ${playerToMoveNext}'s turn):`, bestMove);
        return bestMove;
    } else {
        console.log('No historical move selected.');
        return null;
    }
}
