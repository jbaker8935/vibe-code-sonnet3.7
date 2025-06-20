// AI logic, neural network, and heuristic move selection
import { ROWS, COLS, PLAYER_A, PLAYER_B, NORMAL, SWAPPED, JS_DIRECTIONS, NUM_DIRECTIONS } from './game-constants.js';
import { cloneBoard } from './game-board.js';

// --- AI Logic ---

// Helper to create an empty board (for padding history)
function createEmptyBoard() {
    const board = [];
    for (let r = 0; r < ROWS; r++) {
        const row = [];
        for (let c = 0; c < COLS; c++) {
            row.push(null);
        }
        board.push(row);
    }
    return board;
}

// Updated: Accepts an array of up to 3 boards (oldest first, newest last)
export function boardToNNInput(historyBoards, currentPlayer = PLAYER_B) {
    // historyBoards: array of up to 3 board states, oldest first, newest last
    // Output: [8, 4, 18] tensor (rows, cols, channels)
    // Channels: 0-5: Player A (normal, swapped, empty, ...), 6-11: Player B, 12-17: misc (see below)
    // We'll use:
    // 0: Player A, normal, board t-2
    // 1: Player A, swapped, board t-2
    // 2: Player B, normal, board t-2
    // 3: Player B, swapped, board t-2
    // 4: empty, board t-2
    // 5: player channel, board t-2
    // 6-11: same for t-1
    // 12-17: same for t (current)
    // But for simplicity, use 3 boards, 6 channels each: [A-normal, A-swapped, B-normal, B-swapped, empty, player]
    // So channel = h*6 + type
    const numBoards = 3;
    const channelsPerBoard = 6;
    const nnInput = [];
    for (let r = 0; r < ROWS; r++) {
        nnInput[r] = [];
        for (let c = 0; c < COLS; c++) {
            nnInput[r][c] = new Array(numBoards * channelsPerBoard).fill(0);
        }
    }
    for (let h = 0; h < numBoards; h++) {
        const board = historyBoards && historyBoards[h] ? historyBoards[h] : createEmptyBoard();
        for (let r = 0; r < ROWS; r++) {
            for (let c = 0; c < COLS; c++) {
                const pieceData = board[r][c];
                let base = h * channelsPerBoard;
                if (pieceData) {
                    if (pieceData.player === PLAYER_A) {
                        if (pieceData.state === NORMAL) {
                            nnInput[r][c][base + 0] = 1.0;
                        } else {
                            nnInput[r][c][base + 1] = 1.0;
                        }
                    } else if (pieceData.player === PLAYER_B) {
                        if (pieceData.state === NORMAL) {
                            nnInput[r][c][base + 2] = 1.0;
                        } else {
                            nnInput[r][c][base + 3] = 1.0;
                        }
                    }
                } else {
                    nnInput[r][c][base + 4] = 1.0;
                }
                // Player channel for current board only
                if (h === 2) {
                    nnInput[r][c][base + 5] = (currentPlayer === PLAYER_B) ? 1.0 : 0.0;
                }
            }
        }
    }
    // Output shape: [8, 4, 18]
    // Flatten last dimension if needed by tfjs
    return nnInput;
}

export async function neuralNetworkPredict(tfModel, nnInput) {
    if (!tfModel) throw new Error("TensorFlow model not loaded");
    // nnInput: [8,4,18] (rows, cols, channels)
    if (!nnInput || nnInput.length !== 8 || nnInput[0].length !== 4 || nnInput[0][0].length !== 18) throw new Error(`Invalid NN input: expected [8,4,18], got ${JSON.stringify([nnInput?.length, nnInput?.[0]?.length, nnInput?.[0]?.[0]?.length])}`);
    let inputTensor;
    let outputTensors;
    try {
        inputTensor = tf.tensor([nnInput], [1, 8, 4, 18]); // batch, rows, cols, channels
        // Model expects input shape [1,8,4,18]
        const outputs = tfModel.predict(inputTensor);
        // outputs: [policy, value] or {policy_output, value_output}
        let policy, value;
        if (Array.isArray(outputs)) {
            [policy, value] = outputs;
        } else if (outputs.policy_output && outputs.value_output) {
            policy = outputs.policy_output;
            value = outputs.value_output;
        } else {
            throw new Error("Unexpected model output format");
        }
        // policy: [1,256], value: [1,1]
        const policyData = (await policy.data())[0] ? Array.from(await policy.data()) : Array.from(await policy.arraySync())[0];
        const valueData = (await value.data())[0] ? Array.from(await value.data()) : Array.from(await value.arraySync())[0];
        inputTensor.dispose();
        if (policy.dispose) policy.dispose();
        if (value.dispose) value.dispose();
        return { value: valueData[0], policy: policyData };
    } catch (error) {
        if (inputTensor && !inputTensor.isDisposed) inputTensor.dispose();
        if (outputTensors) outputTensors.forEach(tensor => { if (tensor && !tensor.isDisposed) tensor.dispose(); });
        throw error;
    }
}

// Export all AI functions needed by main.js

// ...
