// AI logic, neural network, and heuristic move selection
import { ROWS, COLS, PLAYER_A, PLAYER_B, NORMAL, SWAPPED, JS_DIRECTIONS, NUM_DIRECTIONS } from './game-constants.js';
import { cloneBoard } from './game-board.js';

// --- AI Logic ---

export function boardToNNInput(boardState, currentPlayer = PLAYER_B) {
    const flatBoardSize = ROWS * COLS;
    const totalSize = 6 * flatBoardSize;
    const nnInput = new Float32Array(totalSize);
    for (let i = 0; i < totalSize; i++) nnInput[i] = 0.0;
    for (let r = 0; r < ROWS; r++) {
        for (let c = 0; c < COLS; c++) {
            const pos = r * COLS + c;
            const pieceData = boardState[r][c];
            if (pieceData) {
                if (pieceData.player === PLAYER_A) {
                    if (pieceData.state === NORMAL) {
                        nnInput[pos] = 1.0;
                    } else {
                        nnInput[flatBoardSize + pos] = 1.0;
                    }
                } else {
                    if (pieceData.state === NORMAL) {
                        nnInput[2 * flatBoardSize + pos] = 1.0;
                    } else {
                        nnInput[3 * flatBoardSize + pos] = 1.0;
                    }
                }
            } else {
                nnInput[4 * flatBoardSize + pos] = 1.0;
            }
        }
    }
    const playerValue = (currentPlayer === PLAYER_A) ? 0.0 : 1.0;
    for (let pos = 0; pos < flatBoardSize; pos++) {
        nnInput[5 * flatBoardSize + pos] = playerValue;
    }
    console.log("Converted board to NN input:", nnInput);
    return nnInput;
}

export async function neuralNetworkPredict(tfModel, nnInput) {
    if (!tfModel) throw new Error("TensorFlow model not loaded");
    if (!nnInput || nnInput.length !== 192) throw new Error(`Invalid NN input: expected 192 elements, got ${nnInput?.length}`);
    let inputTensor;
    let outputTensors;
    try {
        inputTensor = tf.tensor2d([nnInput]);
        const inputNodeName = tfModel.inputs[0].name;
        console.log("Running neural network prediction with input:", nnInput);
        outputTensors = await tfModel.executeAsync({[inputNodeName]: inputTensor});
        if (outputTensors.length < 2) throw new Error(`Expected 2 output tensors (value, policy), got ${outputTensors.length}`);
        const valueOutputTensor = outputTensors[0];
        const policyOutputTensor = outputTensors[1];
        const valueData = await valueOutputTensor.array();
        const policyData = await policyOutputTensor.array();
        const value = valueData[0][0];
        const policy = policyData[0];
        inputTensor.dispose();
        outputTensors.forEach(tensor => tensor.dispose());
        console.log("Neural network prediction result:", { value, policy });
        return { value, policy };
    } catch (error) {
        if (inputTensor && !inputTensor.isDisposed) inputTensor.dispose();
        if (outputTensors) outputTensors.forEach(tensor => { if (tensor && !tensor.isDisposed) tensor.dispose(); });
        throw error;
    }
}

// Export all AI functions needed by main.js

// ...
