// Test case for the specific board position where hard_ai makes incorrect move
// This file creates a test scenario where the AI should NOT move (1,1) to (1,2)
// as it allows Player A to win immediately

import {
    ROWS, COLS, PLAYER_A, PLAYER_B, NORMAL, SWAPPED
} from './game-constants.js';
import {
    parseStartingPosition, cloneBoard
} from './game-board.js';
import {
    checkWinCondition
} from './game-logic.js';
import {
    findBestAIMove, canWinInOneMove
} from './game-ai-advanced.js';
import {
    calculateLegalMovesForState
} from './game-ai-advanced.js';

// Test board position string (uppercase = normal, lowercase = swapped)
const TEST_POSITION = `
A..B
...B
AAB.
A..B
..BB
A.BB
AAA.
....`.trim();

const TEST_MOVE = { start: { row: 2, col: 1 }, end: { row: 2, col: 2 } };

const CURRENT_PLAYER = PLAYER_A; // Player B is the one making the move

export async function runBoardPositionTest() {
    console.log("=== Running Board Position Test ===");
    
    // Wait for model to be loaded if we're in a browser environment
    const isBrowser = typeof window !== 'undefined' && typeof tf !== 'undefined';
    if (isBrowser) {
        console.log("Browser environment detected, checking model status...");
        
        // Wait for model to be loaded (max 10 seconds)
        let modelCheckAttempts = 0;
        while (!window.tfModel && modelCheckAttempts < 100) {
            console.log(`Waiting for model to load... (attempt ${modelCheckAttempts + 1})`);
            await new Promise(resolve => setTimeout(resolve, 100));
            modelCheckAttempts++;
        }
        
        if (window.tfModel) {
            console.log("✅ Model loaded successfully!");
        } else {
            console.log("⚠️  Model not available after waiting");
        }
    }
    
    // Parse the test position
    const testBoard = parseStartingPosition(TEST_POSITION);
    console.log("Test board setup:");
    logBoardState(testBoard);
    
    // Verify initial state - neither player should have won yet
    const playerAWin = checkWinCondition(testBoard, PLAYER_A);
    const playerBWin = checkWinCondition(testBoard, PLAYER_B);
    
    console.log("Initial win conditions:");
    console.log(`Player A win: ${playerAWin.win}`, playerAWin.path);
    console.log(`Player B win: ${playerBWin.win}`, playerBWin.path);
    
    if (playerAWin.win || playerBWin.win) {
        console.error("ERROR: Game already over in initial position!");
        return false;
    }
    
    // Test the problematic move
    console.log(`\n=== Testing Problematic Move: ${TEST_MOVE} ===`);
    const problematicBoard = testProblematicMove(testBoard);
    
    // Test the better moves
    // Let me check what moves are actually available
    console.log("\n=== Testing Better Alternative Moves ===");
    await testBetterMoves(testBoard);
    
    // Run AI to see what move it actually chooses
    console.log("\n=== Testing AI Move Selection ===");
    await testAIMoveSelection(testBoard);
    
    return true;
}

function testProblematicMove(originalBoard) {
    const testBoard = cloneBoard(originalBoard);
    
    // Check if the move  is even legal
    const piece = testBoard[TEST_MOVE.start.row][TEST_MOVE.start.col];
    if (!piece) {
        console.log("No piece at start");
        return null;
    }

    if (piece.player !== CURRENT_PLAYER) {
        console.log(`Piece at start belongs to Player ${piece.player}, not current player ${CURRENT_PLAYER}`);
        return null;
    }
    if (TEST_MOVE.end.row < 0 || TEST_MOVE.end.row >= ROWS || TEST_MOVE.end.col < 0 || TEST_MOVE.end.col >= COLS) {
        console.log("Target position is out of bounds");
        return null;
    }
    if (testBoard[TEST_MOVE.end.row][TEST_MOVE.end.col] && testBoard[TEST_MOVE.end.row][TEST_MOVE.end.col].player === piece.player) {
        console.log("Target position already occupied by own piece");
        return null;
    }
    
    console.log(`Piece at start: Player ${piece.player}, State: ${piece.state}`);
    
    // Check target position
    const targetPiece = testBoard[TEST_MOVE.end.row][TEST_MOVE.end.col];
    console.log(`Target at end: ${targetPiece ? `Player ${targetPiece.player}, State: ${targetPiece.state}` : 'Empty'}`);
    
    // Apply the move 
    if (!targetPiece) {
        // Move to empty cell
        testBoard[TEST_MOVE.end.row][TEST_MOVE.end.col] = { ...piece };
        testBoard[TEST_MOVE.start.row][TEST_MOVE.start.col] = null;
        // Unmark all swapped pieces (simplified)
        for (let r = 0; r < ROWS; r++) {
            for (let c = 0; c < COLS; c++) {
                if (testBoard[r][c] && testBoard[r][c].state === SWAPPED) {
                    testBoard[r][c].state = NORMAL;
                }
            }
        }
    } else {
        // Swap move
        testBoard[TEST_MOVE.end.row][TEST_MOVE.end.col] = { ...piece, state: SWAPPED };
        testBoard[TEST_MOVE.start.row][TEST_MOVE.start.col] = { ...targetPiece, state: SWAPPED };
    }
    
    console.log(`Board after problematic move ${TEST_MOVE}:`);
    logBoardState(testBoard);
    
    // Check if opponent can now win in one move
    const opponent = CURRENT_PLAYER === PLAYER_A ? PLAYER_B : PLAYER_A;
    const opponentCanWin = canWinInOneMove(testBoard, opponent);
    console.log(`After move - Player ${opponent} can win in one move: ${opponentCanWin.canWin}`);
    if (opponentCanWin.canWin) {
        console.log("Opponent winning move:", opponentCanWin.winningMove);
        console.log("Win condition details:", opponentCanWin.winResult);
        console.log("*** CONFIRMED: This move allows opponent to win in one move! ***");
        return testBoard;
    }
    
   
    return testBoard;
}



async function testBetterMoves(originalBoard) {
    // Let's find all legal moves for Player B and test a few
    const possibleMoves = [];
    
    for (let r = 0; r < ROWS; r++) {
        for (let c = 0; c < COLS; c++) {
            const piece = originalBoard[r][c];
            if (piece && piece.player === PLAYER_B) {
                // Check all 8 directions for legal moves
                for (let dr = -1; dr <= 1; dr++) {
                    for (let dc = -1; dc <= 1; dc++) {
                        if (dr === 0 && dc === 0) continue;
                        
                        const nr = r + dr;
                        const nc = c + dc;
                        
                        if (nr >= 0 && nr < ROWS && nc >= 0 && nc < COLS) {
                            const targetCell = originalBoard[nr][nc];
                            
                            if (!targetCell) {
                                // Empty cell move
                                possibleMoves.push({
                                    start: { row: r, col: c },
                                    end: { row: nr, col: nc },
                                    type: 'empty'
                                });
                            } else if (targetCell.player === PLAYER_A && targetCell.state === NORMAL) {
                                // Swap move
                                possibleMoves.push({
                                    start: { row: r, col: c },
                                    end: { row: nr, col: nc },
                                    type: 'swap'
                                });
                            }
                        }
                    }
                }
            }
        }
    }
    
    console.log(`Found ${possibleMoves.length} possible moves for Player B:`);
    possibleMoves.slice(0, 10).forEach((move, i) => {
        console.log(`${i + 1}. (${move.start.row},${move.start.col}) to (${move.end.row},${move.end.col}) [${move.type}]`);
    });
    
    // Test a few moves to see which ones are safe
    let safeMoves = 0;
    let dangerousMoves = 0;
    
    for (let i = 0; i < Math.min(possibleMoves.length, 15); i++) {
        const move = possibleMoves[i];
        const testBoard = cloneBoard(originalBoard);
        
        // Apply the move
        const piece = testBoard[move.start.row][move.start.col];
        const targetPiece = testBoard[move.end.row][move.end.col];
        
        if (move.type === 'empty') {
            testBoard[move.end.row][move.end.col] = { ...piece };
            testBoard[move.start.row][move.start.col] = null;
            // Unmark swapped pieces
            for (let r = 0; r < ROWS; r++) {
                for (let c = 0; c < COLS; c++) {
                    if (testBoard[r][c] && testBoard[r][c].state === SWAPPED) {
                        testBoard[r][c].state = NORMAL;
                    }
                }
            }
        } else {
            testBoard[move.end.row][move.end.col] = { ...piece, state: SWAPPED };
            testBoard[move.start.row][move.start.col] = { ...targetPiece, state: SWAPPED };
        }
        
        // Check if Player A can win after this move
        const playerACanWin = canWinInOneMove(testBoard, PLAYER_A);
        if (playerACanWin.canWin) {
            dangerousMoves++;
            if (move.start.row === 4 && move.start.col === 3 && move.end.row === 5 && move.end.col === 3) {
                console.log(`*** CONFIRMED: Move (4,3) to (5,3) allows Player A to win in one move! ***`);
                console.log("Player A winning move would be:", playerACanWin.winningMove);
            }
        } else {
            safeMoves++;
            console.log(`Safe move: (${move.start.row},${move.start.col}) to (${move.end.row},${move.end.col})`);
        }
    }
    
    console.log(`\nMove analysis: ${safeMoves} safe moves, ${dangerousMoves} dangerous moves`);
}

async function testAIMoveSelection(originalBoard) {
    try {
        // Check if we're running in a browser environment (where TensorFlow.js is available)
        const isBrowser = typeof window !== 'undefined' && typeof tf !== 'undefined';
        
        // Test with different AI difficulties
        const difficulties = isBrowser ? ['easy', 'hard1', 'hard_ai'] : ['easy', 'hard1'];
        
        if (!isBrowser) {
            console.log("⚠️  Running in Node.js environment - TensorFlow.js not available");
            console.log("⚠️  Skipping hard_ai test (requires browser environment)");
            console.log("✅ Testing heuristic AI difficulties only...\n");
        }
        
        for (const difficulty of difficulties) {
            console.log(`\n--- Testing AI difficulty: ${difficulty} ---`);
            
            // Debug: Check if model is available
            if (difficulty === 'hard_ai') {
                console.log("Checking model availability:");
                console.log("window.tfModel:", window.tfModel);
                console.log("typeof window.tfModel:", typeof window.tfModel);
                console.log("window.tfModel is null:", window.tfModel === null);
                console.log("window.tfModel is undefined:", window.tfModel === undefined);
                if (window.tfModel) {
                    console.log("Model inputs:", window.tfModel.inputs);
                    console.log("Model outputs:", window.tfModel.outputs);
                }
            }
            
            const aiMove = await findBestAIMove(
                originalBoard, 
                PLAYER_B, 
                difficulty, 
                2, // depth
                true, // analysis mode
                false, // mcts disabled for now
                50, // mcts simulations
                isBrowser ? window.tfModel : null, // tfModel only available in browser
                null, // mctsSearch
                null  // gameLogic
            );
            
            if (aiMove) {
                const moveDesc = `(${aiMove.start.row},${aiMove.start.col}) to (${aiMove.end.row},${aiMove.end.col})`;
                console.log(`${difficulty} AI selected: ${moveDesc}`);
                
                // Test if this move allows Player A to win
                const testBoard = cloneBoard(originalBoard);
                const piece = testBoard[aiMove.start.row][aiMove.start.col];
                const targetPiece = testBoard[aiMove.end.row][aiMove.end.col];
                
                if (!targetPiece) {
                    testBoard[aiMove.end.row][aiMove.end.col] = { ...piece };
                    testBoard[aiMove.start.row][aiMove.start.col] = null;
                    // Unmark swapped pieces
                    for (let r = 0; r < ROWS; r++) {
                        for (let c = 0; c < COLS; c++) {
                            if (testBoard[r][c] && testBoard[r][c].state === SWAPPED) {
                                testBoard[r][c].state = NORMAL;
                            }
                        }
                    }
                } else {
                    testBoard[aiMove.end.row][aiMove.end.col] = { ...piece, state: SWAPPED };
                    testBoard[aiMove.start.row][aiMove.start.col] = { ...targetPiece, state: SWAPPED };
                }
                
                const playerACanWin = canWinInOneMove(testBoard, PLAYER_A);
                if (playerACanWin.canWin) {
                    console.log(`*** ERROR: ${difficulty} AI chose move that allows Player A to win in one move! ***`);
                    console.log("Player A winning move would be:", playerACanWin.winningMove);
                } else {
                    console.log(`✅ ${difficulty} AI chose a safe move`);
                }
                
                // Check if this is the problematic move
                if (aiMove.start.row === 4 && aiMove.start.col === 3 && aiMove.end.row === 5 && aiMove.end.col === 3) {
                    console.log(`*** CONFIRMED: ${difficulty} AI chose the problematic move (4,3) to (5,3)! ***`);
                }
            } else {
                console.log(`${difficulty} AI returned no move`);
            }
        }
        
        if (!isBrowser) {
            console.log("\n🔧 To test hard_ai (neural network):");
            console.log("   1. Open http://localhost:8000 in a web browser");
            console.log("   2. Open browser developer console (F12)");
            console.log("   3. Click the 'Test AI Fix' button");
            console.log("   4. Or run: await runBoardPositionTest()");
        }
    } catch (error) {
        console.error("Error testing AI move selection:", error);
    }
}

function logBoardState(board) {
    console.log("Board state (row indices on left):");
    for (let r = 0; r < ROWS; r++) {
        let rowStr = `${r}: `;
        for (let c = 0; c < COLS; c++) {
            const piece = board[r][c];
            if (!piece) {
                rowStr += '. ';
            } else {
                const playerChar = piece.player === PLAYER_A ? 'A' : 'B';
                const stateChar = piece.state === NORMAL ? playerChar : playerChar.toLowerCase();
                rowStr += stateChar + ' ';
            }
        }
        console.log(rowStr);
    }
}
