import numpy as np
import math

from config import C_PUCT_CONSTANT
from env_const import PLAYER_A_ID, PLAYER_B_ID, NUM_ACTIONS  # Assuming these are 1 and 2 respectively

class MCTSNode:
    def __init__(self, parent, prior_p, player_id):
        """
        Initializes a new MCTS Node.

        Args:
            parent: The parent node of this node. None for the root node.
            prior_p: The prior probability of selecting this node (i.e., taking the action that leads to this node).
                     For the root node, this can be considered 1 or an arbitrary value if not used.
            player_id: The ID of the player whose turn it is at this node's state.
        """
        self.parent = parent
        self.children = {}  # Dictionary mapping action to child MCTSNode
        self.visit_count = 0  # N(s) or N(s,a) if this node represents state *after* an action
        self.total_action_value = 0.0  # W(s) - total value of this state from simulations
        self.edge_visit_count = {} # N(s,a) - count of visits for each action 'a' taken from this state 's'
        self.edge_total_action_value = {} # W(s,a) - sum of values for each action 'a' from simulations passing through (s,a)
        self.edge_mean_action_value = {} # Q(s,a) - mean value of taking action 'a' from state 's'
        self.edge_prior_probability = {} # P(s,a) - prior probability from NN for taking action 'a' from state 's'
        self.is_expanded = False # True if children have been populated (i.e., NN has been queried for this state)
        self.player_id_at_node = player_id # Player whose turn it is at this state

    def expand(self, action_priors, legal_actions):
        """
        Expands the node by creating children for all legal actions.
        The NN provides the prior probabilities for these actions.

        Args:
            action_priors: A list or array of prior probabilities for all actions, from the NN policy head.
                         The order should correspond to the game's action indexing.
            legal_actions: A list of legal action indices from the current state (node).
        """
        self.is_expanded = True
        for action_idx in legal_actions:
            if action_idx not in self.children:
                child_player_id = PLAYER_B_ID if self.player_id_at_node == PLAYER_A_ID else PLAYER_A_ID
                self.children[action_idx] = MCTSNode(parent=self, prior_p=action_priors[action_idx], player_id=child_player_id)
                self.edge_prior_probability[action_idx] = action_priors[action_idx]
                self.edge_visit_count[action_idx] = 0
                self.edge_total_action_value[action_idx] = 0.0
                self.edge_mean_action_value[action_idx] = 0.0

    def select_child_puct(self):
        """
        Selects the child that maximizes the PUCT (Polynomial Upper Confidence Trees) score.
        PUCT(s, a) = Q(s, a) + c_puct * P(s, a) * sqrt(sum_visits_of_parent_edges) / (1 + N(s, a))
                   where sum_visits_of_parent_edges is self.visit_count (total visits to state s)

        Returns:
            A tuple (action, child_node) corresponding to the selected child.
            Returns None, None if no children or node not expanded.
        """
        if not self.is_expanded or not self.children:
            return None, None

        best_score = -float('inf')
        best_action = -1
        best_child = None

        sqrt_total_parent_visits = math.sqrt(self.visit_count)

        for action, child in self.children.items():
            q_s_a = self.edge_mean_action_value.get(action, 0.0)
            p_s_a = self.edge_prior_probability.get(action, 0.0)
            n_s_a = self.edge_visit_count.get(action, 0)
            
            ucb_score = q_s_a + C_PUCT_CONSTANT * p_s_a * (sqrt_total_parent_visits / (1 + n_s_a))

            if ucb_score > best_score:
                best_score = ucb_score
                best_action = action
                best_child = child
        
        return best_action, best_child

    def update_edge_stats(self, action, value):
        """
        Updates the statistics for a specific action (edge) taken from this node.
        This is called during backpropagation.
        The value is from the perspective of the current player at this node.
        """
        if action in self.edge_visit_count:
            self.edge_visit_count[action] += 1
            self.edge_total_action_value[action] += value
            self.edge_mean_action_value[action] = self.edge_total_action_value[action] / self.edge_visit_count[action]
        else:
            print(f"Warning: Attempting to update non-existent edge for action {action}")

    def update_node_value(self, value):
        """
        Updates the visit count and total action value of the node itself.
        This is also part of backpropagation.
        The value should be from the perspective of the player whose turn it is at this node.
        """
        self.visit_count += 1

    def is_leaf(self):
        """Checks if the node is a leaf node (i.e., not expanded)."""
        return not self.is_expanded

    def get_action_policy_target(self, temperature=1.0):
        """
        Calculates the policy target for training the NN, based on visit counts of child edges.
        pi(a|s) = N(s,a)^(1/temp) / sum_b (N(s,b)^(1/temp))

        Args:
            temperature: Controls the exploration/exploitation in policy. 
                         Higher temp -> more exploration.
        Returns:
            A numpy array representing the policy target vector (size NUM_ACTIONS).
        """
        if not self.children:
            return np.zeros(NUM_ACTIONS)

        policy_target = np.zeros(NUM_ACTIONS, dtype=np.float32)
        total_visit_counts_raised = 0.0
        
        visits_raised = {}
        for action_idx in self.children.keys():
            n_s_a = self.edge_visit_count.get(action_idx, 0)
            if n_s_a > 0:
                val = n_s_a**(1.0 / temperature)
                visits_raised[action_idx] = val
                total_visit_counts_raised += val

        if total_visit_counts_raised > 0:
            for action_idx, val_raised in visits_raised.items():
                policy_target[action_idx] = val_raised / total_visit_counts_raised

        return policy_target

class MCTS:
    def __init__(self, neural_network, game_env_class, config):
        """
        Initializes the Monte Carlo Tree Search algorithm.

        Args:
            neural_network: An instance of the AlphaZeroNetwork (or a compatible predictor)
                            that provides policy_probs, value = predict(state).
            game_env_class: The class of the game environment (e.g., SwitcharooEnv).
                            Used to create fresh environment instances for simulations.
            config: A configuration object or dictionary holding MCTS parameters.
                    Expected to have attributes like NUM_SIMULATIONS_PER_MOVE.
        """
        self.neural_network = neural_network
        self.game_env_class = game_env_class
        self.config = config
        self.num_simulations = self.config.NUM_SIMULATIONS_PER_MOVE

    def run_mcts_simulations(self, root_node, initial_board_array, initial_player_id):
        """
        Runs a number of MCTS simulations from the root_node.

        Args:
            root_node: The MCTSNode representing the current game state (root of the search).
            initial_board_array: The numpy array of the board state at the root.
            initial_player_id: The player ID whose turn it is at the root_node.
        """
        if root_node.player_id_at_node != initial_player_id:
            pass

        for _ in range(self.num_simulations):
            sim_env = self.game_env_class()
            sim_env.board = np.copy(initial_board_array)
            sim_env.current_player_id = root_node.player_id_at_node

            current_selection_node = root_node
            path_taken = [root_node]

            while not current_selection_node.is_leaf():
                action, next_node = current_selection_node.select_child_puct()
                
                if next_node is None:
                    break 
                
                _, _, done, _ = sim_env.step(action) 
                
                current_selection_node = next_node
                path_taken.append(current_selection_node)

                if done:
                    break
            
            value_for_backprop = 0.0
            
            player_at_eval_node = current_selection_node.player_id_at_node
            has_won_current, _ = sim_env.check_win_condition(player=sim_env.ID_PLAYER_MAP[player_at_eval_node])

            if has_won_current:
                value_for_backprop = 1.0
            else:
                opponent_id = PLAYER_B_ID if player_at_eval_node == PLAYER_A_ID else PLAYER_A_ID
                has_won_opponent, _ = sim_env.check_win_condition(player=sim_env.ID_PLAYER_MAP[opponent_id])
                if has_won_opponent:
                    value_for_backprop = -1.0
                elif not sim_env.get_legal_action_indices(player=sim_env.ID_PLAYER_MAP[player_at_eval_node]):
                    value_for_backprop = 0.0 
                else:
                    if current_selection_node.is_leaf():
                        current_game_state_repr = sim_env._get_state()
                        action_priors_nn, value_nn = self.neural_network.predict(current_game_state_repr)
                        
                        legal_actions = sim_env.get_legal_action_indices(player=sim_env.ID_PLAYER_MAP[player_at_eval_node])
                        if legal_actions: 
                            current_selection_node.expand(action_priors_nn, legal_actions)
                        
                        value_for_backprop = value_nn
                    else:
                        current_game_state_repr = sim_env._get_state()
                        _, value_nn = self.neural_network.predict(current_game_state_repr)
                        value_for_backprop = value_nn

            for node_in_path in reversed(path_taken):
                node_in_path.update_node_value(value_for_backprop)

                if node_in_path.parent is not None:
                    action_from_parent = None
                    for act, child_node in node_in_path.parent.children.items():
                        if child_node == node_in_path:
                            action_from_parent = act
                            break
                    if action_from_parent is not None:
                        node_in_path.parent.update_edge_stats(action_from_parent, -value_for_backprop)
                
                value_for_backprop = -value_for_backprop

    def get_action_probabilities(self, root_node, temperature):
        """
        Calculates the final action probabilities from the MCTS search.

        Args:
            root_node: The root node of the MCTS tree after simulations.
            temperature: Temperature parameter to control exploration/exploitation.
                         Typically 1 for early game, approaches 0 for late game.
        Returns:
            A policy vector (numpy array) of probabilities for each action.
        """
        return root_node.get_action_policy_target(temperature)

if __name__ == '__main__':
    print("\n--- MCTS Class Conceptual Test ---")
    from game_env import SwitcharooEnv, A_NORMAL, A_SWAPPED, B_NORMAL, B_SWAPPED
    from az_network import AlphaZeroNetwork

    class MockConfig:
        NUM_SIMULATIONS_PER_MOVE = 50
        C_PUCT_CONSTANT = 1.0

    mock_config_instance = MockConfig()

    class MockNN:
        def predict(self, state):
            policy = np.ones(NUM_ACTIONS, dtype=np.float32) / NUM_ACTIONS
            return policy, np.random.uniform(-1, 1)

    mock_nn_instance = MockNN()
    game_class_instance = SwitcharooEnv 

    mcts_instance = MCTS(neural_network=mock_nn_instance, game_env_class=game_class_instance, config=mock_config_instance)
    print(f"MCTS instance created with {mcts_instance.num_simulations} simulations per move.")

    test_game_env_instance = game_class_instance() 
    test_game_env_instance.reset()
    initial_board_state_arr = np.copy(test_game_env_instance.board) 
    initial_player_id_val = test_game_env_instance.current_player_id

    root = MCTSNode(parent=None, prior_p=0.0, player_id=initial_player_id_val) 
    
    print(f"Running MCTS simulations for root node (Player {initial_player_id_val})...")
    mcts_instance.run_mcts_simulations(root, initial_board_state_arr, initial_player_id_val)
    print("MCTS simulations completed.")

    temperature_val = 1.0
    action_probs_arr = mcts_instance.get_action_probabilities(root, temperature_val)
    print(f"Action probabilities (temp={temperature_val}, sum={np.sum(action_probs_arr):.4f}):")
    found_probs = False
    for i, p_val in enumerate(action_probs_arr):
        if p_val > 1e-5:
            print(f"  Action {i}: {p_val:.4f}")
            found_probs = True
    if not found_probs:
        print("  No actions with significant probability.")

    print(f"Root node visit count: {root.visit_count}")
    if root.children:
        print(f"Root node has {len(root.children)} children.")
        for i, (action, child) in enumerate(root.children.items()):
            if i < 5:
                print(f"  Child for action {action}: Visits N(s,a)={root.edge_visit_count.get(action,0)}, Q(s,a)={root.edge_mean_action_value.get(action,0):.2f}, P(s,a)={root.edge_prior_probability.get(action,0):.3f}")
    else:
        print("Root node has no children after simulation.")

    print("\nMCTS class structure and simulation logic conceptual test added and run.")
