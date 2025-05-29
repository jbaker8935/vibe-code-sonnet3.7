import math
import numpy as np
import numpy as np
import math
from numba import njit, int32, float32, void

from game_env import SwitcharooEnv # For type hinting and environment interaction
from az_network import AlphaZeroNetwork # For type hinting
from env_const import NUM_ACTIONS, PLAYER_A_ID, PLAYER_B_ID, ID_PLAYER_MAP

# --- MCTS Node and Edge Array Structure ---
# Node fields:
# 0: visit_count (N_s)
# 1: total_value (W_s_a) - sum of values from simulations passing through this node
# 2: player_id - player whose turn it is at this node
# 3: is_terminal - 0.0 for false, 1.0 for true
# 4: terminal_value - value if terminal (e.g. -1, 0, 1), 0 otherwise
NODE_FIELDS = 5
NODE_DTYPE = np.float32

# Edge fields:
# 0: visit_count (N_s_a)
# 1: total_value (W_s_a) - sum of values from simulations passing through this edge
# 2: mean_value (Q_s_a) - W_s_a / N_s_a
# 3: prior_prob (P_s_a) - from neural network
# 4: child_node_idx - index of the child node this edge leads to (-1 if not expanded)
EDGE_FIELDS = 5
EDGE_DTYPE = np.float32

MAX_NODES = 80000  # Max nodes in the tree (adjust based on memory/simulations)
MAX_EDGES_PER_NODE = NUM_ACTIONS # Max possible actions from any state

# --- Numba JIT Helper Functions ---

@njit((float32[:], float32[:,:,:], int32[:], float32, float32, float32), cache=True, fastmath=True)
def run_mcts_simulation_single(nodes, edges, node_counter, c_puct, dirichlet_alpha, dirichlet_epsilon):
    """
    Optimized single-pass MCTS simulation combining selection, expansion and backpropagation.
    Neural network prediction and environment interaction are handled by the caller.
    """
    path_nodes = np.zeros(200, dtype=np.int32)  # Pre-allocate path buffer
    path_actions = np.full(200, -1, dtype=np.int32) # Initialize with -1
    path_length = 0
    
    current_node_idx = 0
    # is_root is used by the original expansion check logic, effectively checking if current_node_idx == 0
    # and if its visit count is 0.

    MAX_PATH_DEPTH = 200 # Corresponds to the size of path_nodes and path_actions buffers

    while path_length < MAX_PATH_DEPTH: # Ensure loop terminates if path buffer is full
        path_nodes[path_length] = current_node_idx
        
        # Original logic for is_root, used for expansion check.
        # A bit convoluted, direct check of current_node_idx and its visit count is clearer.
        # For minimal change, keeping original is_root derivation for now.
        is_root = (path_length == 0) # Simpler: current_node_idx == 0 and this is the first node in this sim pass.
                                     # However, the MCTS tree root is always node 0.
                                     # is_root here means "is current_node_idx the root of this simulation's path traversal?"

        # Check if terminal
        if nodes[current_node_idx * NODE_FIELDS + 3] == 1.0: # is_terminal field
            value_for_backprop = nodes[current_node_idx * NODE_FIELDS + 4] # terminal_value field
            # Path has path_length+1 nodes, path_length actions
            return value_for_backprop, path_nodes[:path_length+1], path_actions[:path_length]

        # Node needs expansion if unvisited or is root with no visits
        # Note: visit_count is nodes[...NODE_FIELDS + 0]
        # An unvisited node (visit_count == 0) always needs expansion.
        # The root node (idx 0) is initialized with visit_count 0.
        # Other new nodes are also initialized with visit_count 0.
        if nodes[current_node_idx * NODE_FIELDS + 0] == 0.0:
            # Signal to caller that expansion is needed for current_node_idx
            return -2.0, path_nodes[:path_length+1], path_actions[:path_length]
        
        # Selection using PUCT
        best_score = -np.inf
        best_action = -1
        parent_visits = nodes[current_node_idx * NODE_FIELDS + 0] # N_s
        sqrt_parent_visits = math.sqrt(parent_visits + 1e-8) # Added 1e-8 to avoid issues if parent_visits is 0 (shouldn't be here)
        
        # Get valid actions from edges
        n_possible_actions = edges.shape[1] # MAX_EDGES_PER_NODE
        
        # Iterate over possible actions to find the best one via PUCT
        # This loop structure implicitly handles n_valid == 0 if no action scores > -inf
        for action_iter in range(n_possible_actions):
            # Check if this action corresponds to an existing edge leading to a child
            if edges[current_node_idx, action_iter, 4] >= 0.0: # child_node_idx is valid for this action
                q_val = edges[current_node_idx, action_iter, 2]  # Q_s_a
                p_val = edges[current_node_idx, action_iter, 3]  # P_s_a
                n_val = edges[current_node_idx, action_iter, 0]  # N_s_a
                
                puct_score = q_val + c_puct * p_val * (sqrt_parent_visits / (1.0 + n_val))
                if puct_score > best_score:
                    best_score = puct_score
                    best_action = action_iter
        
        if best_action == -1:
            # No valid action found (e.g. node is a leaf in MCTS tree but not game-terminal, or all PUCT scores were -inf)
            # This implies the node should be treated as terminal for this path, or an error.
            return -1.0, path_nodes[:path_length+1], path_actions[:path_length]
            
        # Record chosen action
        path_actions[path_length] = best_action # Action taken from path_nodes[path_length]
        
        # Get child and update current_node_idx
        child_idx_float = edges[current_node_idx, best_action, 4]
        # This should be a valid index as best_action was chosen from existing edges.
        # Adding a safeguard, though it indicates deeper issues if triggered.
        if child_idx_float < 0:
            return -1.0, path_nodes[:path_length+1], path_actions[:path_length] # Should not happen

        current_node_idx = int(child_idx_float)
        path_length += 1 # Increment path length after successfully taking a step

    # Loop finished because path_length == MAX_PATH_DEPTH
    # The path is full. The last node stored is path_nodes[MAX_PATH_DEPTH-1].
    # This node needs to be evaluated (expand).
    # The path taken to reach path_nodes[MAX_PATH_DEPTH-1] consists of MAX_PATH_DEPTH nodes
    # and MAX_PATH_DEPTH-1 actions.
    # path_actions[MAX_PATH_DEPTH-1] was the action from path_nodes[MAX_PATH_DEPTH-1] leading to the node that exceeded depth.
    # So, the path to return for expansion of path_nodes[MAX_PATH_DEPTH-1] is:
    # Nodes: path_nodes[:MAX_PATH_DEPTH] (contains path_nodes[0]...path_nodes[MAX_PATH_DEPTH-1])
    # Actions: path_actions[:MAX_PATH_DEPTH-1] (contains actions path_actions[0]...path_actions[MAX_PATH_DEPTH-2])
    # This means path_actions[MAX_PATH_DEPTH-1] is not part of the "actions taken to reach these nodes".
    return -2.0, path_nodes[:MAX_PATH_DEPTH], path_actions[:MAX_PATH_DEPTH-1]


@njit(cache=True)
def get_action_probabilities(node_idx, edges, legal_actions_np, temp=1.0): # Renamed legal_actions to legal_actions_np
    """
    Calculate action probabilities based on visit counts with temperature scaling.
    Uses log-sum-exp for numerical stability.
    `legal_actions_np` must be a NumPy array of int32.
    """
    # Ensure legal_actions_np is not empty, otherwise behavior is undefined or error-prone later.
    if len(legal_actions_np) == 0:
        return np.zeros(MAX_EDGES_PER_NODE, dtype=np.float32)

    visits = np.zeros(MAX_EDGES_PER_NODE, dtype=np.float32)
    total_visits = 0.0
    
    # Collect visit counts for legal actions
    for i in range(len(legal_actions_np)):
        action = legal_actions_np[i]
        # Boundary check for safety, though legal_actions_np should contain valid actions
        if action >= 0 and action < MAX_EDGES_PER_NODE:
            n_visits = edges[node_idx, action, 0]
            visits[action] = n_visits # Store in full 'visits' array at correct index
            total_visits += n_visits
    
    # Handle zero total visits by falling back to prior probabilities
    if total_visits < 1e-8:
        probs = np.zeros(MAX_EDGES_PER_NODE, dtype=np.float32)
        prob_sum = 0.0
        for i in range(len(legal_actions_np)):
            action = legal_actions_np[i]
            if action >= 0 and action < MAX_EDGES_PER_NODE:
                prior_prob = edges[node_idx, action, 3]
                probs[action] = prior_prob
                prob_sum += prior_prob
        
        if prob_sum > 1e-8:
            # Normalize priors over legal actions
            for i in range(len(legal_actions_np)):
                action = legal_actions_np[i]
                if action >= 0 and action < MAX_EDGES_PER_NODE:
                    if probs[action] > 0 : # Avoid division by zero if prior_prob was 0
                        probs[action] /= prob_sum
        else:
            # Uniform distribution over legal actions as last resort
            n_legal = len(legal_actions_np)
            if n_legal > 0: 
                uniform_prob = 1.0 / n_legal
                for i in range(len(legal_actions_np)):
                    action = legal_actions_np[i]
                    if action >= 0 and action < MAX_EDGES_PER_NODE:
                        probs[action] = uniform_prob
        return probs
    
    # Apply temperature
    if temp < 1e-3: # For very low temperature (deterministic), use argmax over legal actions
        best_action = -1
        max_visits_val = -1.0
        # Find best action among legal ones
        for i in range(len(legal_actions_np)):
            action = legal_actions_np[i]
            if action >= 0 and action < MAX_EDGES_PER_NODE:
                if visits[action] > max_visits_val:
                    max_visits_val = visits[action]
                    best_action = action
        
        probs = np.zeros(MAX_EDGES_PER_NODE, dtype=np.float32)
        if best_action != -1: 
            probs[best_action] = 1.0
        elif len(legal_actions_np) > 0: 
            # All legal actions had zero/negative visits, or no legal actions. Fallback to uniform.
            uniform_prob = 1.0 / len(legal_actions_np)
            for i in range(len(legal_actions_np)):
                action = legal_actions_np[i]
                if action >= 0 and action < MAX_EDGES_PER_NODE:
                    probs[action] = uniform_prob
        return probs

    # Regular temperature scaling using log-sum-exp
    # logits_i = (1/temp) * log(visits_i) for visits_i > 0 for legal actions
    # For non-legal actions or actions with 0 visits, their probability will be 0.

    logits = np.full(MAX_EDGES_PER_NODE, -np.inf, dtype=np.float32)
    max_logit = -np.inf
    found_positive_visit_for_logit = False

    for i in range(len(legal_actions_np)):
        action = legal_actions_np[i]
        if action >= 0 and action < MAX_EDGES_PER_NODE:
            v = visits[action]
            if v > 1e-9: # Consider only positive visits
                current_logit = (1.0 / temp) * math.log(v)
                logits[action] = current_logit
                if current_logit > max_logit:
                    max_logit = current_logit
                found_positive_visit_for_logit = True

    probs = np.zeros(MAX_EDGES_PER_NODE, dtype=np.float32)

    if not found_positive_visit_for_logit:
        # All legal actions had zero (or very near zero) visits.
        # This case should have been caught by `total_visits < 1e-8`.
        # If it's reached, it's a fallback. Uniform over legal actions.
        n_legal = len(legal_actions_np)
        if n_legal > 0:
            uniform_prob = 1.0 / n_legal
            for i in range(len(legal_actions_np)):
                action = legal_actions_np[i]
                if action >= 0 and action < MAX_EDGES_PER_NODE:
                    probs[action] = uniform_prob
        return probs

    # Calculate sum of exp(logit - max_logit) for legal actions with positive visits
    sum_exp_logits = 0.0
    for i in range(len(legal_actions_np)):
        action = legal_actions_np[i]
        if action >= 0 and action < MAX_EDGES_PER_NODE:
            if logits[action] > -np.inf + 1.0: # Check it's not the initial -np.inf
                sum_exp_logits += math.exp(logits[action] - max_logit)

    if sum_exp_logits > 1e-8:
        for i in range(len(legal_actions_np)):
            action = legal_actions_np[i]
            if action >= 0 and action < MAX_EDGES_PER_NODE:
                if logits[action] > -np.inf + 1.0:
                    probs[action] = math.exp(logits[action] - max_logit) / sum_exp_logits
    else:
        # All exp(logit - max_logit) were zero or sum was too small. Fallback to uniform.
        n_legal = len(legal_actions_np)
        if n_legal > 0:
            uniform_prob = 1.0 / n_legal
            for i in range(len(legal_actions_np)):
                action = legal_actions_np[i]
                if action >= 0 and action < MAX_EDGES_PER_NODE:
                    probs[action] = uniform_prob
    return probs


@njit(cache=True)
def _backpropagate_stats(nodes, edges, path_nodes, path_actions, value):
    """Helper function to update stats during backpropagation."""
    for i in range(len(path_nodes)-1, -1, -1):
        node_idx = path_nodes[i]
        action = path_actions[i]
        
        # Update node stats
        nodes[node_idx * NODE_FIELDS + 0] += 1.0
        nodes[node_idx * NODE_FIELDS + 1] += value
        
        # Update edge stats if not root
        if action != -1:
            edges[node_idx, action, 0] += 1.0
            edges[node_idx, action, 1] += value
            n_s_a = edges[node_idx, action, 0]
            if n_s_a > 0:
                edges[node_idx, action, 2] = edges[node_idx, action, 1] / n_s_a
        
        value = -value  # Flip for parent's perspective


@njit(cache=True)
def _fast_reset_tree(nodes, edges, node_counter, initial_player_id):
    """Fast tree reset that only clears used portions of arrays."""
    # Only clear the nodes that were actually used
    used_nodes = node_counter[0]
    if used_nodes > 0:
        # Clear only the used portion of nodes array
        for i in range(used_nodes * NODE_FIELDS):
            nodes[i] = 0.0
        # Clear only the used portion of edges array
        for i in range(used_nodes):
            for j in range(MAX_EDGES_PER_NODE):
                for k in range(EDGE_FIELDS):
                    edges[i,j,k] = 0.0
    
    # Initialize root node (node 0)
    nodes[0 * NODE_FIELDS + 2] = float32(initial_player_id)  # Only need to set player_id
    node_counter[0] = 1  # Root node is used


@njit(cache=True)
def _expand_node_fast(nodes, edges, node_counter, node_idx, action_priors, valid_actions_buffer, n_valid):
    """Fast node expansion that minimizes memory operations."""
    if node_counter[0] + n_valid <= MAX_NODES:
        child_start = node_counter[0]
        
        # Initialize edges and children in one pass
        for i in range(n_valid):
            action = valid_actions_buffer[i]
            child_idx = child_start + i
            
            # Edge initialization 
            edges[node_idx, action, 0] = 0.0  # N_s_a
            edges[node_idx, action, 1] = 0.0  # W_s_a
            edges[node_idx, action, 2] = 0.0  # Q_s_a
            edges[node_idx, action, 3] = action_priors[action]  # P_s_a
            edges[node_idx, action, 4] = float32(child_idx)  # child_node_idx
            
            # Child node initialization (minimal required fields)
            base_idx = child_idx * NODE_FIELDS
            nodes[base_idx:base_idx + NODE_FIELDS] = 0.0  # Fast zero initialization
            
        node_counter[0] = child_start + n_valid
        return True
    return False  # Expansion failed (out of nodes)


class MCTSNumba:
    def __init__(self, neural_network: AlphaZeroNetwork, game_env_class, config):
        self.neural_network = neural_network
        self.game_env_class = game_env_class
        self.config = config
        
        # Tree structure stored in flat NumPy arrays
        self.nodes = np.zeros(MAX_NODES * NODE_FIELDS, dtype=NODE_DTYPE)
        self.edges = np.zeros((MAX_NODES, MAX_EDGES_PER_NODE, EDGE_FIELDS), dtype=EDGE_DTYPE)
        self.node_counter = np.array([0], dtype=np.int32)
        
        # Pre-allocate reusable buffers
        self.sim_env_for_mcts = self.game_env_class()
        self._action_buffer = np.zeros(MAX_EDGES_PER_NODE, dtype=np.float32)
        self._valid_actions_buffer = np.zeros(MAX_EDGES_PER_NODE, dtype=np.int32)
        self._dirichlet_noise_buffer = np.zeros(MAX_EDGES_PER_NODE, dtype=np.float32)

    def _reset_tree_for_new_search(self, initial_player_id):
        """Resets the tree structure for a new MCTS search (e.g., for a new move)."""
        _fast_reset_tree(self.nodes, self.edges, self.node_counter, initial_player_id)

    def run_mcts_simulations(self, start_board_state_arr, start_player_id, nn_policy_values, nn_value_estimate, num_simulations):
        """
        Runs MCTS simulations from the given board state.
        This version correctly handles expansion and backpropagation.
        """
        _fast_reset_tree(self.nodes, self.edges, self.node_counter, float32(start_player_id)) # Changed initial_player_id to start_player_id

        for i_sim in range(num_simulations):
            # Each simulation starts from the true root state in a fresh environment copy
            sim_env_for_this_pass = self.game_env_class()
            sim_env_for_this_pass.reset(board_state_array=start_board_state_arr.copy(), specific_player_to_start=start_player_id)

            value_at_leaf, path_nodes, path_actions = run_mcts_simulation_single(
                self.nodes, self.edges, self.node_counter,
                self.config.C_PUCT_CONSTANT,
                self.config.DIRICHLET_ALPHA,
                self.config.DIRICHLET_EPSILON
            )

            # Advance sim_env_for_this_pass to the state of the leaf node
            # path_actions[k] is the action taken from path_nodes[k] to reach path_nodes[k+1]
            # We need to apply path_actions[0]...path_actions[len(path_nodes)-2]
            for k_step in range(len(path_nodes) - 1):
                action_in_path = int(path_actions[k_step])
                if action_in_path == -1: # Should ideally not happen for an intermediate step
                    # This might indicate an issue if not the last action placeholder
                    # For robustness, we might break or log, but assume valid path for now.
                    break 
                if not sim_env_for_this_pass.done:
                    sim_env_for_this_pass.step(action_in_path)
                else: # Game ended before MCTS leaf was reached according to tree traversal
                    # This implies an inconsistency or a terminal node was traversed.
                    # The value_at_leaf should reflect this if MCTS is consistent.
                    break 
            
            leaf_node_idx = path_nodes[-1]

            if value_at_leaf == -2.0:  # Node needs expansion
                player_at_leaf_node = self.nodes[leaf_node_idx * NODE_FIELDS + 2]

                if sim_env_for_this_pass.done:
                    # The state reached is terminal, but MCTS thought it needed expansion.
                    # Mark as terminal and backpropagate true outcome.
                    self.nodes[leaf_node_idx * NODE_FIELDS + 3] = 1.0  # is_terminal_flag
                    
                    outcome = 0.0
                    if sim_env_for_this_pass.winner_id == 0 or sim_env_for_this_pass.winner_id == 3: # Draw
                        outcome = 0.0
                    elif sim_env_for_this_pass.winner_id == int(player_at_leaf_node): # Player at leaf won
                        outcome = 1.0
                    else: # Player at leaf lost
                        outcome = -1.0
                    self.nodes[leaf_node_idx * NODE_FIELDS + 4] = outcome
                    _backpropagate_stats(self.nodes, self.edges, path_nodes, path_actions, outcome)
                else:
                    # Not terminal, proceed with NN evaluation and expansion
                    state_repr_leaf = sim_env_for_this_pass._get_state_for_nn()
                    priors_leaf_nn, val_nn_leaf = self.neural_network.predict(state_repr_leaf)
                    
                    # Ensure player ID at node is correctly set before expansion
                    self.nodes[leaf_node_idx * NODE_FIELDS + 2] = float32(sim_env_for_this_pass.current_player_id)

                    legal_actions_leaf = sim_env_for_this_pass.get_legal_action_indices()
                    n_valid_leaf = len(legal_actions_leaf)

                    for i, act in enumerate(legal_actions_leaf):
                        self._valid_actions_buffer[i] = act
                    
                    priors_for_expansion = priors_leaf_nn # Use original NN priors by default

                    # Apply Dirichlet noise if expanding the root node (path length 1)
                    if len(path_nodes) == 1 and self.config.DIRICHLET_ALPHA > 0 and n_valid_leaf > 0:
                        noise = np.random.dirichlet([self.config.DIRICHLET_ALPHA] * n_valid_leaf).astype(np.float32)
                        # Create a temporary priors array to modify with noise
                        priors_for_expansion = priors_leaf_nn.copy() # Ensure it's a copy
                        for i_noise, action_idx in enumerate(legal_actions_leaf):
                            if action_idx < len(priors_for_expansion): # Boundary check
                                priors_for_expansion[action_idx] = \
                                    (1.0 - self.config.DIRICHLET_EPSILON) * priors_for_expansion[action_idx] + \
                                    self.config.DIRICHLET_EPSILON * noise[i_noise]
                    
                    if n_valid_leaf > 0:
                        _expand_node_fast(self.nodes, self.edges, self.node_counter, leaf_node_idx, priors_for_expansion, self._valid_actions_buffer, n_valid_leaf)
                    else:
                        # No legal actions from this state, effectively terminal (loss for current player)
                        self.nodes[leaf_node_idx * NODE_FIELDS + 3] = 1.0  # is_terminal_flag
                        self.nodes[leaf_node_idx * NODE_FIELDS + 4] = -1.0 # Terminal value (loss)
                        val_nn_leaf = -1.0  # Override NN value with loss for backpropagation

                    _backpropagate_stats(self.nodes, self.edges, path_nodes, path_actions, val_nn_leaf)

            elif value_at_leaf == -1.0:
                # Error from run_mcts_simulation_single (e.g., no valid moves from a non-terminal node that wasn't marked for expansion)
                # This implies the node should be terminal (loss for the player whose turn it was).
                # The value -1.0 is from the perspective of the player at that node.
                _backpropagate_stats(self.nodes, self.edges, path_nodes, path_actions, -1.0)
            
            else: # Terminal state reached, value_at_leaf is the game outcome for the player at path_nodes[-1]
                # value_at_leaf is already the correct value to backpropagate.
                # It would have been set by run_mcts_simulation_single if nodes[leaf_node_idx * NODE_FIELDS + 3] == 1.0
                _backpropagate_stats(self.nodes, self.edges, path_nodes, path_actions, value_at_leaf)

        # Log the sum of visit counts (N) for all actions at the root node
        # print(f"[MCTS Root Stats] Total N at root: {total_root_visits:.1f} (Simulations: {num_simulations})")

    def get_action_probabilities(self, board_state_arr, player_id, temp=1.0, root_node_dummy=None):
        """
        Returns the MCTS policy (visit counts normalized) for the root state.
        `root_node_dummy` is ignored.
        Assumes run_mcts_simulations was called before this for the same state.
        """
        # action_probs_fallback = np.zeros(NUM_ACTIONS, dtype=np.float32) # Fallback if issues
        
        # Root node visit count
        root_visit_count = self.nodes[0 * NODE_FIELDS + 0]

        # Ensure sim_env_for_mcts is in the correct state to get legal_actions
        current_env_state_matches = False
        try:
            if np.array_equal(self.sim_env_for_mcts.board_state_array, board_state_arr) and \
               self.sim_env_for_mcts.current_player_id == player_id:
                current_env_state_matches = True
        except AttributeError: # If sim_env_for_mcts not fully initialized
            pass

        if not current_env_state_matches:
            self.sim_env_for_mcts.reset(board_state_array=board_state_arr.copy(), specific_player_to_start=player_id)
        
        legal_actions = self.sim_env_for_mcts.get_legal_action_indices()
        
        # If no legal actions, return zeros. The Numba helper also handles empty legal_actions_np.
        if len(legal_actions) == 0:
            return np.zeros(NUM_ACTIONS, dtype=np.float32)

        # Convert legal_actions to NumPy array for Numba helper
        np_legal_actions = np.array(legal_actions, dtype=np.int32)

        if root_visit_count == 0:  # No simulations run or tree reset without root expansion
            # Fallback: Use prior probabilities from the neural network if available at root
            # The Numba helper's first block (total_visits < 1e-8) handles this by using priors.
            # So, we can directly call the Numba helper.
            action_probs = get_action_probabilities(0, self.edges, np_legal_actions, temp)
            return action_probs

        # Use the Numba-compiled get_action_probabilities helper
        action_probs = get_action_probabilities(0, self.edges, np_legal_actions, temp)
        
        # The Numba helper get_action_probabilities returns an array of size MAX_EDGES_PER_NODE.
        # Since MAX_EDGES_PER_NODE is set to NUM_ACTIONS in this file, 
        # action_probs is already the correct size (NUM_ACTIONS).
        # The previous conditional slicing logic is not needed.

        return action_probs

    def get_root_node_stats(self, legal_actions):
        """
        Retrieves the visit count and value statistics for the root node (node 0).
        Returns a dictionary with action indices as keys and (visit_count, mean_value) tuples as values.
        """
        stats = {}
        
        # Root node visit count
        root_visit_count = self.nodes[0 * NODE_FIELDS + 0]

        if root_visit_count == 0:
            # No simulations have run, return empty stats
            return stats

        # Iterate over possible actions
        for action in legal_actions:
            if 0 <= action < MAX_EDGES_PER_NODE:
                n_s_a = self.edges[0, action, 0] # Visit count for the action
                q_s_a = self.edges[0, action, 2] # Mean value for the action (Q_s_a)
                
                stats[action] = (n_s_a, q_s_a)
        
        return stats
