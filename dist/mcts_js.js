/**
 * JavaScript MCTS Implementation for Switcharoo Game
 * Provides MCTS tree search functionality for the web interface
 * Compatible with TensorFlow.js models and existing game logic
 */

import { boardToNNInput } from './game-ai.js';

class MCTSNode {
    constructor(parent = null, priorP = 0.0, playerId = null) {
        this.parent = parent;
        this.children = new Map(); // action -> MCTSNode
        this.visitCount = 0;
        this.totalActionValue = new Map(); // action -> sum of values
        this.meanActionValue = new Map(); // action -> mean value Q(s,a)
        this.priorProbability = new Map(); // action -> prior P(s,a)
        this.isExpanded = false;
        this.playerIdAtNode = playerId;
    }

    expand(actionPriors, legalActions) {
        this.isExpanded = true;
        for (const action of legalActions) {
            const prior = actionPriors[action] || 0.0;
            this.priorProbability.set(action, prior);
            this.totalActionValue.set(action, 0.0);
            this.meanActionValue.set(action, 0.0);
            // Children will be created lazily when selected
        }
    }

    selectChildPUCT(cPuct = 1.0) {
        let bestScore = -Infinity;
        let bestAction = null;
        let bestChild = null;

        const sqrtTotalParentVisits = Math.sqrt(this.visitCount + 1);

        for (const [action, prior] of this.priorProbability) {
            const nSA = this.getActionVisitCount(action);
            const qSA = this.meanActionValue.get(action) || 0.0;
            
            const ucbScore = qSA + cPuct * prior * (sqrtTotalParentVisits / (1 + nSA));

            if (ucbScore > bestScore) {
                bestScore = ucbScore;
                bestAction = action;
                bestChild = this.children.get(action);
            }
        }

        // Create child if it doesn't exist
        if (bestAction !== null && !this.children.has(bestAction)) {
            this.children.set(bestAction, new MCTSNode(this, this.priorProbability.get(bestAction), null));
        }

        return { action: bestAction, child: this.children.get(bestAction) };
    }

    updateEdgeStats(action, value) {
        const currentVisits = this.getActionVisitCount(action);
        const currentTotal = this.totalActionValue.get(action) || 0.0;
        
        this.totalActionValue.set(action, currentTotal + value);
        this.meanActionValue.set(action, (currentTotal + value) / (currentVisits + 1));
    }

    updateNodeValue(value) {
        this.visitCount += 1;
    }

    getActionVisitCount(action) {
        if (!this.children.has(action)) return 0;
        return this.children.get(action).visitCount;
    }

    isLeaf() {
        return !this.isExpanded;
    }

    getActionPolicyTarget(temperature = 1.0) {
        const policyTarget = new Array(256).fill(0.0); // NUM_ACTIONS = 256
        
        if (this.visitCount === 0) return policyTarget;

        let totalVisitCounts = 0;
        const visitsRaised = new Map();

        // Apply temperature to visit counts
        for (const [action, child] of this.children) {
            const visits = child.visitCount;
            if (visits > 0) {
                const visitRaised = Math.pow(visits, 1.0 / temperature);
                visitsRaised.set(action, visitRaised);
                totalVisitCounts += visitRaised;
            }
        }

        // Normalize to probabilities
        if (totalVisitCounts > 0) {
            for (const [action, visitRaised] of visitsRaised) {
                policyTarget[action] = visitRaised / totalVisitCounts;
            }
        }

        return policyTarget;
    }
}

class MCTSSearch {
    constructor(config = {}) {
        this.numSimulations = config.numSimulations || 50;
        this.cPuct = config.cPuct || 1.0;
        this.temperature = config.temperature || 1.0;
        this.dirichletAlpha = config.dirichletAlpha || 0.3;
        this.dirichletEpsilon = config.dirichletEpsilon || 0.25;
        this.enabled = config.enabled !== false; // Default to enabled
        this.maxSimulationDepth = config.maxSimulationDepth || 100;
        
        // Debug options
        this.verbose = config.verbose || false; // Default to disabled for performance
        this.logSearchStats = config.logSearchStats !== false; // Default to enabled for testing
    }

    /**
     * Main MCTS search function
     * @param {Array} boardState - 2D board array
     * @param {string} currentPlayer - PLAYER_A or PLAYER_B
     * @param {Function} neuralNetworkPredict - Function that takes board state and returns {policy, value}
     * @param {Function} gameLogic - Object with game logic functions
     * @param {Array} boardHistory - Array of up to 3 board states (oldest first, newest last)
     * @returns {Array} - Action probabilities array
     */
    async search(boardState, currentPlayer, neuralNetworkPredict, gameLogic, boardHistory = null) {
        // Helper to get last 3 boards for history stacking
        function getHistoryArray(current, historyArr) {
            // historyArr: array of previous boards, oldest first, newest last (excluding current)
            // Returns array of 3 boards: [oldest, ..., current]
            const arr = [];
            if (historyArr && Array.isArray(historyArr)) {
                // Take up to last 2 from historyArr
                const len = historyArr.length;
                if (len >= 2) arr.push(historyArr[len-2]);
                if (len >= 1) arr.push(historyArr[len-1]);
            }
            // Pad with nulls if needed
            while (arr.length < 2) arr.unshift(null);
            arr.push(current); // Current board is always last
            return arr;
        }
        const historyArr = boardHistory || [];
        const nnHistory = getHistoryArray(boardState, historyArr);
        if (!this.enabled) {
            // Fall back to direct neural network policy
            const nnInput = boardToNNInput(nnHistory, currentPlayer);
            const prediction = await neuralNetworkPredict(nnInput);
            return this.applyTemperatureToPolicy(prediction.policy);
        }
        const rootNode = new MCTSNode(null, 0.0, currentPlayer);
        // Get initial neural network prediction for root
        const nnInput = boardToNNInput(nnHistory, currentPlayer);
        const rootPrediction = await neuralNetworkPredict(nnInput);
        // Get legal actions
        const legalActions = gameLogic.getLegalActions(boardState, currentPlayer);
        if (legalActions.length === 0) {
            if (this.verbose) console.log("MCTS: No legal actions available");
            return new Array(256).fill(0.0);
        }

        // Expand root node
        rootNode.expand(rootPrediction.policy, legalActions);
        
        // Apply Dirichlet noise to root node if configured
        if (this.dirichletAlpha > 0 && this.dirichletEpsilon > 0) {
            this.addDirichletNoise(rootNode, legalActions);
        }

        // Run simulations
        console.log(`MCTS: Starting ${this.numSimulations} simulations`);
        let completedSimulations = 0;
        let failedSimulations = 0;
        
        for (let sim = 0; sim < this.numSimulations; sim++) {
            try {
                if (this.verbose) console.log(`MCTS: Starting simulation ${sim + 1}/${this.numSimulations}`);
                await this.runSingleSimulation(rootNode, boardState, currentPlayer, neuralNetworkPredict, gameLogic);
                completedSimulations++;
                if (this.verbose) console.log(`MCTS: Completed simulation ${sim + 1}/${this.numSimulations}`);
            } catch (error) {
                failedSimulations++;
                console.error(`MCTS: Simulation ${sim + 1} failed:`, error);
            }
        }
        
        console.log(`MCTS: Completed ${completedSimulations}/${this.numSimulations} simulations (${failedSimulations} failed)`);
        console.log(`MCTS: Root node has ${rootNode.visitCount} visits after simulations`);

        // Get action probabilities from visit counts
        const actionProbs = rootNode.getActionPolicyTarget(this.temperature);
        
        // Log summary statistics
        const totalVisits = rootNode.visitCount;
        const nonZeroProbs = actionProbs.filter(p => p > 0).length;
        const maxProb = Math.max(...actionProbs);
        const entropySum = actionProbs.reduce((sum, p) => p > 0 ? sum - p * Math.log2(p) : sum, 0);
        
        console.log(`MCTS: Search completed - Total visits: ${totalVisits}, Non-zero actions: ${nonZeroProbs}, Max prob: ${maxProb.toFixed(3)}, Entropy: ${entropySum.toFixed(3)}`);
        
        if (this.logSearchStats) {
            this.logRootStats(rootNode, legalActions);
        }

        return actionProbs;
    }

    async runSingleSimulation(rootNode, initialBoardState, initialPlayer, neuralNetworkPredict, gameLogic) {
        const simulationStartTime = performance.now();
        
        // Create simulation environment
        let simBoard = gameLogic.cloneBoard(initialBoardState);
        let currentPlayer = initialPlayer;
        let currentNode = rootNode;
        const searchPath = [{ node: rootNode, action: null }];
        
        let depth = 0;
        let nodesTraversed = 1; // Count root node
        
        if (this.verbose) console.log(`MCTS: Simulation starting - Initial player: ${initialPlayer}`);
        
        // Selection phase: traverse tree until we reach a leaf
        while (!currentNode.isLeaf() && depth < this.maxSimulationDepth) {
            const { action, child } = currentNode.selectChildPUCT(this.cPuct);
            
            if (action === null || child === null) {
                if (this.verbose) console.log("MCTS: No valid action selected during tree traversal");
                break;
            }

            // Apply move to simulation board
            const moveResult = gameLogic.applyAction(simBoard, action, currentPlayer);
            if (!moveResult.success) {
                if (this.verbose) console.log("MCTS: Invalid action during simulation:", action);
                break;
            }
            
            simBoard = moveResult.newBoard;
            currentPlayer = gameLogic.getNextPlayer(currentPlayer);
            currentNode = child;
            currentNode.playerIdAtNode = currentPlayer;
            
            searchPath.push({ node: currentNode, action: action });
            depth++;
            nodesTraversed++;
            
            if (this.verbose) console.log(`MCTS: Traversed to depth ${depth}, action ${action}, player ${currentPlayer}`);
        }

        // Evaluation phase
        let value = 0.0;
        const playerAtLeaf = currentNode.playerIdAtNode || currentPlayer;

        // Check if terminal state
        const gameResult = gameLogic.checkGameEnd(simBoard);
        if (gameResult.isTerminal) {
            // Terminal state - use actual game outcome
            if (gameResult.winner === playerAtLeaf) {
                value = 1.0;
            } else if (gameResult.winner === null) {
                value = 0.0; // Draw
            } else {
                value = -1.0;
            }
            if (this.verbose) console.log(`MCTS: Terminal state reached - Winner: ${gameResult.winner}, Value: ${value}`);
        } else {
            // Non-terminal state - expand and evaluate with neural network
            if (currentNode.isLeaf()) {
                const legalActions = gameLogic.getLegalActions(simBoard, currentPlayer);
                
                if (legalActions.length > 0) {
                    // Get neural network prediction
                    const nnInput = boardToNNInput([simBoard], currentPlayer);
                    const prediction = await neuralNetworkPredict(nnInput);
                    // Expand the leaf node
                    currentNode.expand(prediction.policy, legalActions);
                    value = prediction.value;
                    if (this.verbose) console.log(`MCTS: Expanded leaf node - Legal actions: ${legalActions.length}, NN Value: ${value.toFixed(3)}`);
                } else {
                    // No legal moves - treat as terminal loss
                    value = -1.0;
                }
            } else {
                // Already expanded, use neural network value
                const nnInput = boardToNNInput(simBoard, currentPlayer);
                const prediction = await neuralNetworkPredict(nnInput);
                value = prediction.value;
            }
        }

        // Backpropagation phase
        this.backpropagate(searchPath, value, playerAtLeaf);
        
        const simulationEndTime = performance.now();
        const simulationDuration = simulationEndTime - simulationStartTime;
        
        if (this.verbose) {
            console.log(`MCTS: Simulation completed - Depth: ${depth}, Nodes: ${nodesTraversed}, Value: ${value.toFixed(3)}, Duration: ${simulationDuration.toFixed(2)}ms`);
        }
    }

    backpropagate(searchPath, leafValue, leafPlayer) {
        let value = leafValue;
        
        if (this.verbose) {
            console.log(`MCTS: Backpropagating value ${leafValue.toFixed(3)} for player ${leafPlayer} through path of length ${searchPath.length}`);
        }
        
        // Traverse path in reverse order
        for (let i = searchPath.length - 1; i >= 0; i--) {
            const { node, action } = searchPath[i];
            
            // Update node visit count
            node.updateNodeValue(value);
            
            // Update edge statistics if this isn't the root
            if (action !== null && node.parent) {
                node.parent.updateEdgeStats(action, value);
                if (this.verbose) {
                    console.log(`MCTS: Updated action ${action} with value ${value.toFixed(3)} (visits: ${node.visitCount})`);
                }
            }
            
            // Flip value for next level up (opponent's perspective)
            value = -value;
        }
    }

    addDirichletNoise(rootNode, legalActions) {
        // Generate Dirichlet noise
        const noise = this.generateDirichletNoise(legalActions.length, this.dirichletAlpha);
        
        // Apply noise to root priors
        let noiseIndex = 0;
        for (const action of legalActions) {
            const originalPrior = rootNode.priorProbability.get(action) || 0.0;
            const noisyPrior = (1 - this.dirichletEpsilon) * originalPrior + 
                              this.dirichletEpsilon * noise[noiseIndex];
            rootNode.priorProbability.set(action, noisyPrior);
            noiseIndex++;
        }
    }

    generateDirichletNoise(size, alpha) {
        // Simple Dirichlet noise generation using Gamma distribution
        const samples = [];
        let sum = 0;
        
        for (let i = 0; i < size; i++) {
            // Approximate Gamma(alpha, 1) using exponential distribution when alpha is small
            const sample = this.gammaRandom(alpha);
            samples.push(sample);
            sum += sample;
        }
        
        // Normalize to sum to 1
        return samples.map(s => s / sum);
    }

    gammaRandom(alpha) {
        // Simple approximation for Gamma distribution
        if (alpha < 1) {
            return Math.pow(Math.random(), 1 / alpha);
        } else {
            // Use approximation for alpha >= 1
            let sum = 0;
            for (let i = 0; i < Math.floor(alpha); i++) {
                sum += -Math.log(Math.random());
            }
            if (alpha % 1 !== 0) {
                sum += Math.pow(Math.random(), 1 / (alpha % 1));
            }
            return sum;
        }
    }

    applyTemperatureToPolicy(policy, temperature = null) {
        const temp = temperature !== null ? temperature : this.temperature;
        
        if (temp === 0) {
            // Greedy selection
            const result = new Array(policy.length).fill(0.0);
            const maxIndex = policy.indexOf(Math.max(...policy));
            result[maxIndex] = 1.0;
            return result;
        }
        
        // Apply temperature
        const logits = policy.map(p => Math.log(Math.max(p, 1e-10)) / temp);
        const maxLogit = Math.max(...logits);
        const expLogits = logits.map(l => Math.exp(l - maxLogit));
        const sumExp = expLogits.reduce((sum, val) => sum + val, 0);
        
        return expLogits.map(val => val / sumExp);
    }

    logRootStats(rootNode, legalActions) {
        console.log("MCTS Root Statistics:");
        console.log(`Total root visits: ${rootNode.visitCount}`);
        console.log(`Legal actions: ${legalActions.length}`);
        
        // Sort actions by visit count
        const actionStats = [];
        for (const action of legalActions) {
            const visits = rootNode.getActionVisitCount(action);
            const qValue = rootNode.meanActionValue.get(action) || 0.0;
            const prior = rootNode.priorProbability.get(action) || 0.0;
            
            if (visits > 0) {
                actionStats.push({ action, visits, qValue, prior });
            }
        }
        
        actionStats.sort((a, b) => b.visits - a.visits);
        
        console.log("Top actions by visit count:");
        for (let i = 0; i < Math.min(5, actionStats.length); i++) {
            const stat = actionStats[i];
            console.log(`  Action ${stat.action}: ${stat.visits} visits, Q=${stat.qValue.toFixed(3)}, P=${stat.prior.toFixed(3)}`);
        }
    }

    // Configuration methods
    setNumSimulations(num) {
        this.numSimulations = Math.max(1, Math.min(1000, num));
    }

    setTemperature(temp) {
        this.temperature = Math.max(0, Math.min(2.0, temp));
    }

    setEnabled(enabled) {
        this.enabled = enabled;
    }

    getConfig() {
        return {
            numSimulations: this.numSimulations,
            cPuct: this.cPuct,
            temperature: this.temperature,
            enabled: this.enabled,
            dirichletAlpha: this.dirichletAlpha,
            dirichletEpsilon: this.dirichletEpsilon
        };
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { MCTSNode, MCTSSearch };
}

export { MCTSNode, MCTSSearch };
