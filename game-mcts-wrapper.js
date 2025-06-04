// MCTS integration and configuration (wrapper, not the actual MCTS implementation)
// This file wraps MCTS usage for the game, does not conflict with mcts.js
import { ROWS, COLS, PLAYER_A, PLAYER_B, NORMAL, SWAPPED, JS_DIRECTIONS, NUM_DIRECTIONS } from './game-constants.js';

export function initializeMCTS(MCTSSearch, SwitcharooGameLogic, config) {
    if (typeof MCTSSearch !== 'undefined' && typeof SwitcharooGameLogic !== 'undefined') {
        const gameLogic = new SwitcharooGameLogic({
            ROWS, COLS, PLAYER_A, PLAYER_B, NORMAL, SWAPPED, NUM_DIRECTIONS, JS_DIRECTIONS
        });
        const mctsSearch = new MCTSSearch({
            numSimulations: config.MCTS_SIMULATIONS,
            cPuct: config.MCTS_PUCT_CONSTANT,
            temperature: config.MCTS_TEMPERATURE,
            dirichletAlpha: config.MCTS_DIRICHLET_ALPHA,
            dirichletEpsilon: config.MCTS_DIRICHLET_EPSILON,
            enabled: config.MCTS_ENABLED,
            verbose: config.MCTS_VERBOSE,
            logSearchStats: true
        });
        return { mctsSearch, gameLogic };
    } else {
        return { mctsSearch: null, gameLogic: null };
    }
}

// Example: initializeMCTS, updateMCTSSettings, etc.
// (Implementations would be moved here from script.js)

// Export all MCTS wrapper functions needed by main.js

// ...
