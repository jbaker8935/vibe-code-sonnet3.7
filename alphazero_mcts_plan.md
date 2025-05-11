# Step-by-Step Plan: Implementing AlphaZero-like MCTS Self-Play for Switcharoo

This document outlines the steps to implement a self-play training regime using Monte Carlo Tree Search (MCTS) guided by a neural network, similar to the AlphaZero approach, for the Switcharoo game.

## I. Foundational Setup & Neural Network Design

1.  **Project Structure & Configuration:**
    *   Create new Python files for MCTS logic (e.g., `mcts.py`), the new neural network model (e.g., `az_network.py`), and the main self-play training loop (e.g., `alphazero_trainer.py`).
    *   Update `config.py` to include new hyperparameters for MCTS (e.g., `NUM_SIMULATIONS_PER_MOVE`, `C_PUCT_CONSTANT`, `TEMPERATURE_SCHEDULE`), NN architecture, and the self-play loop (e.g., `GAMES_PER_ITERATION`, `MCTS_REPLAY_BUFFER_SIZE`, `EVALUATION_GAMES_COUNT`).

2.  **Neural Network (NN) - `az_network.py`:**
    *   **Input:** Define the input representation.
        *   Start with the existing binary board representation (`_get_state()` in `game_env.py`: 5 binary values + player indicator).
        *   *Consider experimenting with a 2D board representation (e.g., `(ROWS, COLS, num_piece_types_plus_player)`) if the 1D binary proves insufficient for convolutional layers.*
    *   **Architecture (TensorFlow/Keras):**
        *   **Common Body:**
            *   Input layer matching the chosen state representation.
            *   A series of convolutional layers (if using 2D input) or dense layers. Residual connections (ResNet blocks) are highly recommended.
            *   Example ResNet block: `Conv2D -> BatchNorm -> ReLU -> Conv2D -> BatchNorm -> Add (with input) -> ReLU`.
        *   **Policy Head:**
            *   Convolutional layer (e.g., 1x1 filter) or Dense layer reducing dimensionality.
            *   Flatten if necessary.
            *   Dense layer outputting `NUM_ACTIONS` logits.
            *   Softmax activation to get a probability distribution over moves.
        *   **Value Head:**
            *   Convolutional layer (e.g., 1x1 filter) or Dense layer reducing dimensionality.
            *   Flatten if necessary.
            *   Dense layer with a smaller number of units (e.g., 256).
            *   ReLU activation.
            *   Dense layer outputting a single scalar.
            *   `tanh` activation to constrain the value between -1 (loss) and +1 (win).
    *   **Compilation:**
        *   Use an optimizer like Adam or SGD.
        *   Define a combined loss function:
            *   Value loss: Mean Squared Error (MSE) between predicted value and game outcome.
            *   Policy loss: Categorical Cross-Entropy between predicted policy and MCTS policy target.
            *   Add L2 regularization to weights.
            *   `total_loss = value_loss_weight * mse_loss + policy_loss_weight * crossentropy_loss + l2_regularization_loss`.
    *   **Helper Functions:**
        *   `predict(state)`: Takes a game state, returns policy probabilities and value.
        *   `save_model(filepath)`, `load_model(filepath)`.

## II. Monte Carlo Tree Search (MCTS) Implementation - `mcts.py`

1.  **`MCTSNode` Class:**
    *   Attributes:
        *   `parent`: Parent node.
        *   `children`: Dictionary mapping action to child `MCTSNode`.
        *   `state`: The game state this node represents (can be stored implicitly or explicitly).
        *   `visit_count (N)`: How many times this node has been visited during MCTS.
        *   `total_action_value (W)`: Sum of values from simulations passing through this node for each action taken from it. (e.g., `W[action]`)
        *   `mean_action_value (Q)`: `W[action] / N[action]`.
        *   `prior_probability (P)`: Policy probability from the NN for actions from this state. (e.g., `P[action]`)
        *   `is_expanded`: Boolean, true if children have been created.
        *   `player_id_at_node`: Player whose turn it is at this node.

2.  **MCTS Algorithm Core Logic:**
    *   **`run_simulations(root_state, neural_network, num_simulations, game_env_instance)`:**
        *   Initialize `root_node` from `root_state`.
        *   Loop `num_simulations` times:
            *   `node_to_expand = root_node`
            *   `search_path = [node_to_expand]`
            *   **a. Selection:**
                *   While `node_to_expand` is expanded and not a terminal state:
                    *   Select action `a` from `node_to_expand` that maximizes PUCT:
                        `PUCT(s, a) = Q(s, a) + c_puct * P(s, a) * sqrt(sum_visits_of_parent) / (1 + N(s, a))`
                        (Ensure `sum_visits_of_parent` is `node_to_expand.visit_count`).
                    *   `node_to_expand = node_to_expand.children[a]`
                    *   `search_path.append(node_to_expand)`
            *   **b. Expansion & Evaluation:**
                *   `leaf_node = node_to_expand`
                *   Get the game state corresponding to `leaf_node`.
                *   Check if `leaf_node` represents a terminal game state using `game_env_instance.check_win_condition()`.
                    *   If terminal, `value = game_outcome` (e.g., +1 for win, -1 for loss, 0 for draw from the perspective of `leaf_node.player_id_at_node`).
                    *   If not terminal:
                        *   If `leaf_node` has not been visited before (i.e., not in NN cache or first time):
                            *   Call `neural_network.predict(leaf_node_state)` to get `policy_probs` and `value`.
                            *   Expand `leaf_node`: For each legal move from `leaf_node_state`, create a child node, store `policy_probs[action]` as its `P[action]`. Mark `leaf_node` as expanded.
                        *   Else (already visited, use cached value or re-evaluate if strategy demands):
                            *   `value` is the NN's value prediction for `leaf_node_state`.
            *   **c. Backpropagation:**
                *   Iterate `node_in_path`, `action_taken_to_reach_it` in reverse over `search_path`:
                    *   `node_in_path.visit_count += 1`
                    *   If `action_taken_to_reach_it` is not None (i.e., not the root of this particular simulation path if starting from an existing node):
                        *   `parent_of_node_in_path.N[action_taken_to_reach_it] += 1`
                        *   `parent_of_node_in_path.W[action_taken_to_reach_it] += value` (adjust `value` based on whose perspective it is: if current player in path is different from player who made the value assessment, flip sign of value).
                        *   `parent_of_node_in_path.Q[action_taken_to_reach_it] = parent_of_node_in_path.W[action_taken_to_reach_it] / parent_of_node_in_path.N[action_taken_to_reach_it]`
    *   **`get_action_probabilities(root_node, temperature)`:**
        *   After simulations, calculate move probabilities from `root_node`.
        *   `probs[action] = (root_node.N[action] ^ (1/temperature)) / sum(root_node.N[b] ^ (1/temperature) for b in legal_actions)`.
        *   Return a dictionary mapping action to probability.

## III. Self-Play and Training Loop - `alphazero_trainer.py`

1.  **Initialization:**
    *   Initialize the neural network (`az_network.py`). Load if a checkpoint exists.
    *   Initialize a replay buffer (e.g., a `deque` or a custom class) to store `(state, mcts_policy_target, outcome)` tuples.
    *   Initialize `game_env = SwitcharooEnv()`.

2.  **Main Training Loop (Iterations):**
    *   `for iteration in range(NUM_ITERATIONS):`
        *   **a. Self-Play Game Generation:**
            *   `for game_num in range(GAMES_PER_ITERATION):`
                *   `current_game_data = []`
                *   `state = game_env.reset()` (potentially with varied initial positions from `config.py`)
                *   `done = False`
                *   `while not done:`
                    *   `mcts_instance = MCTS(...)` (or reset an existing one)
                    *   Run MCTS: `mcts_policy_targets = mcts_instance.run_simulations(state, current_best_nn, NUM_SIMULATIONS_PER_MOVE, game_env)`
                    *   Determine `temperature` based on game progress/iteration.
                    *   `action_probs_from_mcts = mcts_instance.get_action_probabilities(mcts_root_node, temperature)`
                    *   Select `action` based on `action_probs_from_mcts` (e.g., `np.random.choice` with `p=action_probs_from_mcts`).
                    *   Store `(state, action_probs_from_mcts)` in `current_game_data`.
                    *   `next_state, reward, done, info = game_env.step(action)` (Note: `reward` here is the immediate step reward, not the final game outcome yet).
                    *   `state = next_state`
                *   `game_outcome = game_env.winner_id` (map to +1, -1, 0 from current player's perspective for each stored state).
                *   Iterate through `current_game_data`:
                    *   For each `(s, pi)`, determine the correct `outcome_for_s` based on `game_outcome` and player at state `s`.
                    *   Add `(s, pi, outcome_for_s)` to the main replay buffer.
        *   **b. Neural Network Training:**
            *   `for train_step in range(TRAINING_STEPS_PER_ITERATION):`
                *   Sample a mini-batch `(batch_states, batch_policies, batch_outcomes)` from the replay buffer.
                *   Train the NN: `loss = nn_model.train_on_batch(batch_states, {'policy_head_output': batch_policies, 'value_head_output': batch_outcomes})`.
                *   Log training loss (e.g., to WandB).
        *   **c. Model Evaluation & Update (Optional but Recommended):**
            *   Periodically (e.g., every N iterations):
                *   Save the current trained model as `candidate_model`.
                *   Load the `previous_best_model`.
                *   Play `EVALUATION_GAMES_COUNT` games between `candidate_model` and `previous_best_model` (using MCTS for both, but selecting moves greedily or with low temperature).
                *   If `candidate_model` wins by a significant margin (e.g., >55%), it becomes the `current_best_nn`. Save it as `best_model.weights.h5`.
                *   Otherwise, keep the `previous_best_model` as `current_best_nn`.
        *   Save NN checkpoints regularly.

## IV. Utilities and Integration

1.  **`game_env.py` & `env_util.py`:**
    *   Ensure `get_legal_moves_jit` is efficient.
    *   `_apply_move_jit` will be used by MCTS to explore states.
    *   `_check_win_condition_jit` is critical for MCTS to identify terminal nodes.
    *   The state representation from `_get_state()` will be the NN input.

2.  **`config.py`:**
    *   Add sections for MCTS parameters: `NUM_SIMULATIONS_PER_MOVE`, `C_PUCT`, `TEMPERATURE_START`, `TEMPERATURE_END`, `TEMPERATURE_ANNEAL_STEPS`, `DIRICHLET_ALPHA`, `DIRICHLET_EPSILON`.
    *   Add sections for AlphaZero training loop: `ITERATIONS`, `GAMES_PER_ITERATION`, `TRAINING_STEPS_PER_ITERATION`, `REPLAY_BUFFER_SIZE_MCTS`, `BATCH_SIZE_MCTS`, `EVALUATION_GAMES`, `MODEL_UPDATE_WIN_RATE_THRESHOLD`.

3.  **Logging & Monitoring:**
    *   Integrate WandB (as in existing scripts) to log:
        *   Training loss (total, policy, value).
        *   Game statistics (win/loss/draw rates during self-play).
        *   Evaluation results (win rate against previous best).
        *   MCTS statistics (e.g., average tree depth, branching factor if desired).
        *   Hyperparameters.

## V. Execution and Iteration

1.  **Initial Run & Debugging:**
    *   Start with very small `NUM_SIMULATIONS_PER_MOVE` (e.g., 10-50).
    *   Use a small `GAMES_PER_ITERATION` and `REPLAY_BUFFER_SIZE_MCTS`.
    *   Focus on getting the pipeline working end-to-end without errors.
    *   Verify MCTS logic: inspect tree statistics, ensure PUCT selection works, backpropagation updates values correctly.
    *   Verify NN training: check if loss decreases, ensure policy and value heads learn plausible outputs for simple scenarios.

2.  **Scaling Up:**
    *   Gradually increase `NUM_SIMULATIONS_PER_MOVE`. This is the most impactful parameter for strength but also the most computationally expensive.
    *   Increase `GAMES_PER_ITERATION` to gather more diverse data.
    *   Adjust learning rate, batch size, and other NN hyperparameters.
    *   Run for a large number of iterations. AlphaZero-style training often requires days or weeks of computation.

3.  **Hyperparameter Tuning:**
    *   Experiment with `C_PUCT`.
    *   Tune the temperature schedule for exploration vs. exploitation during self-play move selection.
    *   Adjust NN architecture (depth, width of ResNet blocks).

4.  **Analysis:**
    *   Regularly play against the trained agent to assess its strength and identify weaknesses or repetitive patterns.
    *   Analyze game data to understand emerging strategies.

This step-by-step plan should guide the development of your AlphaZero-like MCTS training system for Switcharoo. Remember that this is a complex undertaking, and iterative development with thorough testing at each stage is key.
