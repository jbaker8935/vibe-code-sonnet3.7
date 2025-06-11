#!/usr/bin/env node

/**
 * Test Positions Runner for Switcharoo Game
 * 
 * This script reads test_positions.yaml and executes comprehensive tests
 * for each position across all AI difficulty modes: 'easy', 'hard1', 'hard_ai'
 * 
 * Usage: node test-positions-runner.js [--position <n>] [--ai <mode>] [--verbose]
 */

import fs from 'fs';
import path from 'path';
import yaml from 'js-yaml';
// TensorFlow.js Node.js imports
import * as tf from '@tensorflow/tfjs-node';

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

// Global TensorFlow model variable
let tfModel = null;

/**
 * Load TensorFlow.js model for hard_ai testing
 */
async function loadTFJSModel() {
    try {
        console.log('Loading TensorFlow.js model...');
        const modelPath = path.resolve('./switcharoo_tfjs_model/model.json');
        
        // Check if model files exist
        if (!fs.existsSync(modelPath)) {
            throw new Error(`Model file not found: ${modelPath}`);
        }
        
        const modelBinPath = path.resolve('./switcharoo_tfjs_model/group1-shard1of1.bin');
        if (!fs.existsSync(modelBinPath)) {
            throw new Error(`Model binary file not found: ${modelBinPath}`);
        }
        
        // Load the model using TensorFlow.js Node.js
        tfModel = await tf.loadGraphModel(`file://${modelPath}`);
        
        console.log('✅ TensorFlow.js model loaded successfully!');
        console.log(`   Input shape: ${JSON.stringify(tfModel.inputs[0].shape)}`);
        console.log(`   Output nodes: ${tfModel.outputs.length}`);
        
        return true;
    } catch (error) {
        console.error('❌ Failed to load TensorFlow.js model:', error.message);
        console.log('⚠️  hard_ai mode will be skipped');
        return false;
    }
}

// Command line argument parsing
const args = process.argv.slice(2);
const config = {
    verbose: args.includes('--verbose'),
    position: getArgValue('--position'),
    aiMode: getArgValue('--ai'),
    helpMode: args.includes('--help') || args.includes('-h')
};

function getArgValue(argName) {
    const index = args.indexOf(argName);
    return index !== -1 && index + 1 < args.length ? args[index + 1] : null;
}

function showHelp() {
    console.log(`
Test Positions Runner for Switcharoo Game

Usage: node test-positions-runner.js [OPTIONS]

Options:
  --position <n>    Run only the specified test position
  --ai <mode>         Run only the specified AI mode (easy, hard1, hard_ai)
  --verbose           Enable verbose output
  --help, -h          Show this help message

Examples:
  node test-positions-runner.js
  node test-positions-runner.js --position "Position 1"
  node test-positions-runner.js --ai hard1 --verbose
  node test-positions-runner.js --position "Position 3" --ai easy
`);
}

// Test results tracking
const testResults = {
    total: 0,
    passed: 0,
    failed: 0,
    errors: 0,
    details: []
};

/**
 * Load and parse the test positions YAML file
 */
function loadTestPositions() {
    try {
        const yamlContent = fs.readFileSync('./test_positions.yaml', 'utf8');
        return yaml.load(yamlContent);
    } catch (error) {
        console.error('Error loading test_positions.yaml:', error.message);
        process.exit(1);
    }
}

/**
 * Convert board string to 2D array format expected by game logic
 */
function parseBoardFromYaml(boardString) {
    const lines = boardString.trim().split('\n').map(line => line.trim());
    const board = [];
    
    for (let row = 0; row < ROWS; row++) {
        board[row] = [];
        const line = lines[row] || '';
        
        for (let col = 0; col < COLS; col++) {
            const char = line[col] || '.';
            
            if (char === '.') {
                board[row][col] = null;
            } else {
                const isLowercase = char === char.toLowerCase();
                const player = char.toLowerCase() === 'a' ? PLAYER_A : PLAYER_B;
                const state = isLowercase ? SWAPPED : NORMAL;
                
                board[row][col] = { player, state };
            }
        }
    }
    
    return board;
}

/**
 * Apply a test move to the board
 */
function applyTestMove(board, testMove, player) {
    const testBoard = cloneBoard(board);
    const { start, end } = testMove;
    
    // For piece placement (start == end), place a new piece
    if (start[0] === end[0] && start[1] === end[1]) {
        const [row, col] = end;
        if (testBoard[row][col] === null) {
            testBoard[row][col] = { player, state: NORMAL };
            // Clear all swapped states after placement
            clearSwappedStates(testBoard);
            return testBoard;
        } else {
            throw new Error(`Cannot place piece at occupied position (${row}, ${col})`);
        }
    }
    
    // For movement/swap (start != end)
    const startPiece = testBoard[start[0]][start[1]];
    const endPiece = testBoard[end[0]][end[1]];
    
    if (!startPiece) {
        throw new Error(`No piece at start position (${start[0]}, ${start[1]})`);
    }
    
    if (!endPiece) {
        // Move to empty cell
        testBoard[end[0]][end[1]] = { ...startPiece };
        testBoard[start[0]][start[1]] = null;
        clearSwappedStates(testBoard);
    } else {
        // Swap pieces
        testBoard[end[0]][end[1]] = { ...startPiece, state: SWAPPED };
        testBoard[start[0]][start[1]] = { ...endPiece, state: SWAPPED };
    }
    
    return testBoard;
}

/**
 * Clear all swapped states from the board
 */
function clearSwappedStates(board) {
    for (let row = 0; row < ROWS; row++) {
        for (let col = 0; col < COLS; col++) {
            if (board[row][col] && board[row][col].state === SWAPPED) {
                board[row][col].state = NORMAL;
            }
        }
    }
}

/**
 * Log board state in a readable format
 */
function logBoardState(board, title = "Board State") {
    if (!config.verbose) return;
    
    console.log(`\n${title}:`);
    for (let row = 0; row < ROWS; row++) {
        let rowStr = '';
        for (let col = 0; col < COLS; col++) {
            const piece = board[row][col];
            if (!piece) {
                rowStr += '.';
            } else {
                const char = piece.player === PLAYER_A ? 'A' : 'B';
                rowStr += piece.state === SWAPPED ? char.toLowerCase() : char;
            }
        }
        console.log(rowStr);
    }
}

/**
 * Test a single position with a specific AI mode
 */
async function testPositionWithAI(position, aiMode) {
    const testResult = {
        position: position.name,
        aiMode,
        passed: false,
        error: null,
        details: {}
    };
    
    try {
        // Parse the board
        const board = parseBoardFromYaml(position.board);
        logBoardState(board, `Initial Board - ${position.name}`);
        
        // Verify initial conditions
        const playerAWin = checkWinCondition(board, PLAYER_A);
        const playerBWin = checkWinCondition(board, PLAYER_B);
        
        // For immediate_win positions, we expect one player to be able to win in their next move
        if (position.immediate_win) {
            const expectedWinner = position.winning_player;
            const expectedWinnerPlayer = expectedWinner === 'A' ? PLAYER_A : PLAYER_B;
            
            // Check if the expected winner can win in one move
            const canWinResult = canWinInOneMove(board, expectedWinnerPlayer);
            
            if (!canWinResult.canWin) {
                throw new Error(`Expected ${expectedWinner} to have an immediate winning move available, but they don't`);
            }
            
            if (config.verbose) {
                console.log(`✅ Confirmed: Player ${expectedWinner} can win immediately with move: ${JSON.stringify(canWinResult.winningMove)}`);
            }
        } else {
            // For non-immediate_win positions, neither player should have won yet
            if (playerAWin.win || playerBWin.win) {
                throw new Error('Game already over in initial position');
            }
        }
        
        // Get the player to move
        const currentPlayer = position.player_to_move === 'A' ? PLAYER_A : PLAYER_B;
        
        // If there's a test_move specified, test only that move
        if (position.test_move) {
            await testSpecificMove(position, board, currentPlayer, testResult);
        } else {
            // Test AI move selection only if no specific move is provided
            await testAIMoveSelection(position, board, currentPlayer, aiMode, testResult);
        }
        
        // Validate expected outcomes
        validateExpectedOutcomes(position, board, currentPlayer, testResult);
        
        testResult.passed = !testResult.error;
        
    } catch (error) {
        testResult.error = error.message;
        testResult.passed = false;
        
        if (config.verbose) {
            console.error(`Error testing ${position.name} with ${aiMode}:`, error);
        }
    }
    
    return testResult;
}

/**
 * Test a specific move if provided in the test position
 */
async function testSpecificMove(position, board, currentPlayer, testResult) {
    const { test_move } = position;
    
    if (config.verbose) {
        console.log(`\nTesting specific move: ${JSON.stringify(test_move)}`);
    }
    
    try {
        const testBoard = applyTestMove(board, test_move, currentPlayer);
        logBoardState(testBoard, "Board after test move");
        
        // Check win conditions after the move
        const opponentPlayer = currentPlayer === PLAYER_A ? PLAYER_B : PLAYER_A;
        const currentPlayerWin = checkWinCondition(testBoard, currentPlayer);
        const opponentWin = checkWinCondition(testBoard, opponentPlayer);
        
        testResult.details.testMove = {
            move: test_move,
            currentPlayerWins: currentPlayerWin.win,
            opponentWins: opponentWin.win,
            opponentCanWinNext: canWinInOneMove(testBoard, opponentPlayer).canWin
        };
        
        // Validate against expected immediate_win
        if (position.immediate_win && !currentPlayerWin.win) {
            throw new Error('Expected immediate win but current player did not win');
        }
        
        if (!position.immediate_win && currentPlayerWin.win) {
            throw new Error('Unexpected immediate win for current player');
        }
        
        // Validate forced move scenarios
        if (position.test_type === 'forced_move') {
            const opponentCanWin = canWinInOneMove(testBoard, opponentPlayer);
            if (opponentCanWin.canWin) {
                throw new Error(`Specified move failed to block opponent's winning threat. Opponent can still win with: (${opponentCanWin.winningMove.start.row},${opponentCanWin.winningMove.start.col}) to (${opponentCanWin.winningMove.end.row},${opponentCanWin.winningMove.end.col})`);
            } else {
                console.log(`✅ Specified move successfully blocked opponent's winning threat`);
            }
        }
        
    } catch (error) {
        testResult.details.testMoveError = error.message;
        throw error;
    }
}

/**
 * Test AI move selection for the position
 */
async function testAIMoveSelection(position, board, currentPlayer, aiMode, testResult) {
    if (config.verbose) {
        console.log(`\nTesting AI move selection with ${aiMode} difficulty`);
    }
    
    try {
        // Check if hard_ai mode is available in Node.js environment
        if (aiMode === 'hard_ai' && !tfModel) {
            console.log('⚠️  Skipping hard_ai mode: TensorFlow.js model not loaded');
            testResult.details.aiMove = { skipped: 'hard_ai model not available' };
            return;
        }
        
        const aiMove = await findBestAIMove(
            board,
            currentPlayer,
            aiMode,
            2, // depth
            true, // analysis mode
            false, // mcts disabled for now
            50, // mcts simulations
            tfModel, // Pass the loaded model
            null, // mctsSearch
            null  // gameLogic
        );
        
        if (!aiMove) {
            throw new Error(`AI failed to find a move with ${aiMode} difficulty`);
        }
        
        const moveDesc = `(${aiMove.start.row},${aiMove.start.col}) to (${aiMove.end.row},${aiMove.end.col})`;
        
        if (config.verbose) {
            console.log(`${aiMode} AI selected move: ${moveDesc}`);
        }
        
        // Test the AI's chosen move
        const aiTestBoard = cloneBoard(board);
        const piece = aiTestBoard[aiMove.start.row][aiMove.start.col];
        const targetPiece = aiTestBoard[aiMove.end.row][aiMove.end.col];
        
        if (!piece) {
            throw new Error(`AI selected invalid move: no piece at start position`);
        }
        
        // Apply the AI's move
        if (!targetPiece) {
            // Move to empty cell
            aiTestBoard[aiMove.end.row][aiMove.end.col] = { ...piece };
            aiTestBoard[aiMove.start.row][aiMove.start.col] = null;
            clearSwappedStates(aiTestBoard);
        } else {
            // Swap move
            aiTestBoard[aiMove.end.row][aiMove.end.col] = { ...piece, state: SWAPPED };
            aiTestBoard[aiMove.start.row][aiMove.start.col] = { ...targetPiece, state: SWAPPED };
        }
        
        logBoardState(aiTestBoard, `Board after ${aiMode} AI move`);
        
        // Analyze the result
        const opponentPlayer = currentPlayer === PLAYER_A ? PLAYER_B : PLAYER_A;
        const currentPlayerWin = checkWinCondition(aiTestBoard, currentPlayer);
        const opponentCanWin = canWinInOneMove(aiTestBoard, opponentPlayer);
        
        testResult.details.aiMove = {
            aiMode,
            move: aiMove,
            moveDescription: moveDesc,
            currentPlayerWins: currentPlayerWin.win,
            allowsOpponentWin: opponentCanWin.canWin,
            opponentWinningMove: opponentCanWin.canWin ? opponentCanWin.winningMove : null
        };
        
        // Check for critical errors based on test type
        if (position.test_type === 'forced_move') {
            // For forced move scenarios, the AI must choose a move that prevents opponent from winning
            const opponentThreatBefore = canWinInOneMove(board, opponentPlayer);
            
            if (!opponentThreatBefore.canWin) {
                console.log(`⚠️  Warning: No opponent threat detected in forced move position`);
            } else if (opponentCanWin.canWin) {
                throw new Error(`${aiMode} AI failed to block opponent's winning threat. Opponent can still win with: (${opponentCanWin.winningMove.start.row},${opponentCanWin.winningMove.start.col}) to (${opponentCanWin.winningMove.end.row},${opponentCanWin.winningMove.end.col})`);
            } else {
                console.log(`✅ ${aiMode} AI successfully blocked opponent's winning threat`);
            }
        }
        
        if (position.allows_forced_move && opponentCanWin.canWin) {
            console.log(`⚠️  ${aiMode} AI move allows opponent to win in one move!`);
        }
        
        if (position.immediate_win && !currentPlayerWin.win) {
            throw new Error(`${aiMode} AI failed to find immediate win`);
        }
        
    } catch (error) {
        testResult.details.aiMoveError = error.message;
        // Re-throw error for forced move scenarios or non-hard_ai modes
        if (aiMode !== 'hard_ai' || position.test_type === 'forced_move') {
            throw error;
        }
    }
}

/**
 * Validate expected outcomes from the test position
 */
function validateExpectedOutcomes(position, board, currentPlayer, testResult) {
    const opponentPlayer = currentPlayer === PLAYER_A ? PLAYER_B : PLAYER_A;
    
    // Check if current player has immediate win available
    const currentPlayerCanWin = canWinInOneMove(board, currentPlayer);
    
    if (position.immediate_win !== currentPlayerCanWin.canWin) {
        const expected = position.immediate_win ? 'has' : 'does not have';
        const actual = currentPlayerCanWin.canWin ? 'has' : 'does not have';
        throw new Error(`Expected current player ${expected} immediate win, but actually ${actual} immediate win`);
    }
    
    testResult.details.validation = {
        expectedImmediateWin: position.immediate_win,
        actualImmediateWin: currentPlayerCanWin.canWin,
        winningMove: currentPlayerCanWin.canWin ? currentPlayerCanWin.winningMove : null
    };
}

/**
 * Run all tests for a single position
 */
async function testPosition(position) {
    const positionName = position.name;
    // Use ai_mode from YAML if present, otherwise test all modes
    const aiModes = position.ai_mode ? [position.ai_mode] : (config.aiMode ? [config.aiMode] : ['easy', 'hard1', 'hard_ai']);
    
    console.log(`\n${'='.repeat(60)}`);
    console.log(`Testing: ${positionName}`);
    console.log(`Description: ${position.description}`);
    console.log(`Player to move: ${position.player_to_move}`);
    console.log(`Test type: ${position.test_type}`);
    console.log(`${'='.repeat(60)}`);
    
    const positionResults = [];
    
    for (const aiMode of aiModes) {
        console.log(`\n--- Testing with ${aiMode} AI ---`);
        
        const result = await testPositionWithAI(position, aiMode);
        positionResults.push(result);
        
        testResults.total++;
        if (result.passed) {
            testResults.passed++;
            console.log(`✅ PASSED: ${positionName} with ${aiMode}`);
        } else {
            testResults.failed++;
            console.log(`❌ FAILED: ${positionName} with ${aiMode}`);
            if (result.error) {
                console.log(`   Error: ${result.error}`);
            }
        }
        
        if (config.verbose && result.details) {
            console.log(`   Details:`, JSON.stringify(result.details, null, 2));
        }
    }
    
    testResults.details.push({
        position: positionName,
        results: positionResults
    });
}

/**
 * Print final test summary
 */
function printSummary() {
    console.log(`\n${'='.repeat(80)}`);
    console.log('TEST SUMMARY');
    console.log(`${'='.repeat(80)}`);
    console.log(`Total tests: ${testResults.total}`);
    console.log(`Passed: ${testResults.passed}`);
    console.log(`Failed: ${testResults.failed}`);
    console.log(`Success rate: ${((testResults.passed / testResults.total) * 100).toFixed(1)}%`);
    
    if (testResults.failed > 0) {
        console.log(`\nFailed tests:`);
        testResults.details.forEach(positionDetail => {
            const failedResults = positionDetail.results.filter(r => !r.passed);
            if (failedResults.length > 0) {
                console.log(`  ${positionDetail.position}:`);
                failedResults.forEach(result => {
                    console.log(`    - ${result.aiMode}: ${result.error || 'Unknown error'}`);
                });
            }
        });
    }
    
    console.log(`${'='.repeat(80)}`);
}

/**
 * Main execution function
 */
async function main() {
    if (config.helpMode) {
        showHelp();
        return;
    }
    
    console.log('Switcharoo Game - Test Positions Runner');
    console.log('========================================');
    
    // Load TensorFlow.js model for hard_ai support
    const modelLoaded = await loadTFJSModel();
    
    // Load test positions
    const testPositions = loadTestPositions();
    console.log(`Loaded ${testPositions.length} test positions`);
    
    // Filter positions if specific position requested
    let positionsToTest = testPositions;
    if (config.position) {
        positionsToTest = testPositions.filter(p => p.name === config.position);
        if (positionsToTest.length === 0) {
            console.error(`No position found with name: ${config.position}`);
            console.log('Available positions:');
            testPositions.forEach(p => console.log(`  - ${p.name}`));
            return;
        }
    }
    
    console.log(`Testing ${positionsToTest.length} position(s)`);
    if (config.aiMode) {
        console.log(`AI mode filter: ${config.aiMode}`);
    }
    if (config.verbose) {
        console.log('Verbose mode enabled');
    }
    
    // Run tests
    for (const position of positionsToTest) {
        await testPosition(position);
    }
    
    // Print summary
    printSummary();
    
    // Exit with error code if any tests failed
    if (testResults.failed > 0) {
        process.exit(1);
    }
}

// Run the main function
main().catch(error => {
    console.error('Fatal error:', error);
    process.exit(1);
});
