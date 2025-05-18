import numpy as np
import numba
from numba import njit, int32, float32, void, typeof
import math
import random

from game_env import SwitcharooEnv # For type hinting and environment interaction
from az_network import AlphaZeroNetwork # For type hinting
from env_const import NUM_ACTIONS, PLAYER_A_ID, PLAYER_B_ID, ID_PLAYER_MAP

# --- MCTS Node and Edge Array Structure ---
# Node fields:
# 0: visit_count (N_s)
# 1: total_value (W_s) - sum of values from simulations passing through this node
# 2: player_id_at_node - whose turn it is at this node (1 for A, 2 for B)
# 3: is_terminal_flag (0 for non-terminal, 1 for terminal)
# 4: terminal_value (value if terminal, e.g., 1.0 for win, -1.0 for loss, 0.0 for draw)
NODE_FIELDS = 5
NODE_DTYPE = np.float32

# Edge fields (for each action from a node):
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
    path_actions = np.zeros(200, dtype=np.int32)
    path_length = 0
    
    current_node_idx = 0
    is_root = True
    value_for_backprop = 0.0

    while True:
        # Record path
        if path_length < 200:  # Safety check
            path_nodes[path_length] = current_node_idx
            path_actions[path_length] = -1 if is_root else path_actions[path_length]
            path_length += 1
        
        # Check if terminal
        if nodes[current_node_idx * NODE_FIELDS + 3] == 1.0:
            value_for_backprop = nodes[current_node_idx * NODE_FIELDS + 4]
            return value_for_backprop, path_nodes[:path_length], path_actions[:path_length]

        # Node needs expansion if unvisited or is root with no visits
        is_unvisited = (nodes[current_node_idx * NODE_FIELDS + 0] == 0.0 and not is_root)
        needs_root_expansion = is_root and nodes[current_node_idx * NODE_FIELDS + 0] == 0.0

        if is_unvisited or needs_root_expansion:
            # Signal to caller that expansion is needed
            return -2.0, path_nodes[:path_length], path_actions[:path_length]
        
        # Selection using PUCT
        best_score = -np.inf
        best_action = -1
        parent_visits = nodes[current_node_idx * NODE_FIELDS + 0]
        sqrt_parent_visits = math.sqrt(parent_visits + 1e-8)
        
        # Get valid actions from edges
        n_actions = edges.shape[1]
        valid_actions = np.zeros(n_actions, dtype=np.int32)
        n_valid = 0
        for action in range(n_actions):
            if edges[current_node_idx, action, 4] >= 0.0:  # Has valid child
                valid_actions[n_valid] = action
                n_valid += 1
        
        if n_valid == 0:
            # No valid moves, signal to caller
            return -1.0, path_nodes[:path_length], path_actions[:path_length]
            
        # Inline PUCT calculation
        for i in range(n_valid):
            action = valid_actions[i]
            q_val = edges[current_node_idx, action, 2]  # Q_s_a
            p_val = edges[current_node_idx, action, 3]  # P_s_a
            n_val = edges[current_node_idx, action, 0]  # N_s_a
            
            puct_score = q_val + c_puct * p_val * (sqrt_parent_visits / (1.0 + n_val))
            if puct_score > best_score:
                best_score = puct_score
                best_action = action
        
        if best_action == -1:
            # No valid action found, signal to caller
            return -1.0, path_nodes[:path_length], path_actions[:path_length]
            
        # Get child and update path
        child_idx_float = edges[current_node_idx, best_action, 4]
        if child_idx_float < 0:
            # Unexpanded child, signal to caller
            return -2.0, path_nodes[:path_length], path_actions[:path_length]
        
        # Continue to child node
        next_node_idx = int(child_idx_float)
        if path_length < 200:
            path_actions[path_length-1] = best_action
        current_node_idx = next_node_idx
        is_root = False


@njit(cache=True)
def get_action_probabilities(node_idx, edges, legal_actions, temp=1.0):
    """
    Calculate action probabilities based on visit counts with temperature scaling.
    Uses more numerically stable operations and handles zero visits.
    """
    visits = np.zeros(MAX_EDGES_PER_NODE, dtype=np.float32)
    total_visits = 0.0
    
    # Collect visit counts
    for action in legal_actions:
        if action >= 0 and action < MAX_EDGES_PER_NODE:
            n_visits = edges[node_idx, action, 0]
            visits[action] = n_visits
            total_visits += n_visits
    
    # Handle zero total visits by falling back to prior probabilities
    if total_visits < 1e-8:
        probs = np.zeros(MAX_EDGES_PER_NODE, dtype=np.float32)
        prob_sum = 0.0
        for action in legal_actions:
            if action >= 0 and action < MAX_EDGES_PER_NODE:
                prior_prob = edges[node_idx, action, 3]
                probs[action] = prior_prob
                prob_sum += prior_prob
        
        if prob_sum > 1e-8:
            probs /= prob_sum
        else:
            # Uniform distribution as last resort
            n_legal = len(legal_actions)
            if n_legal > 0:
                uniform_prob = 1.0 / n_legal
                for action in legal_actions:
                    if action >= 0 and action < MAX_EDGES_PER_NODE:
                        probs[action] = uniform_prob
        return probs
    
    # Apply temperature
    if temp < 1e-3:
        # For very low temperature, use argmax
        best_action = -1
        max_visits = -1.0
        for action in legal_actions:
            if action >= 0 and action < MAX_EDGES_PER_NODE:
                if visits[action] > max_visits:
                    max_visits = visits[action]
                    best_action = action
        
        probs = np.zeros(MAX_EDGES_PER_NODE, dtype=np.float32)
        if best_action >= 0:
            probs[best_action] = 1.0
        return probs
    
    # Regular temperature scaling
    scaled_visits = np.zeros(MAX_EDGES_PER_NODE, dtype=np.float32)
    total_scaled = 0.0
    
    for action in legal_actions:
        if action >= 0 and action < MAX_EDGES_PER_NODE:
            if visits[action] > 0:
                # Use power function instead of exp/log for better numerical stability
                scaled_val = math.pow(visits[action], 1.0 / temp)
                scaled_visits[action] = scaled_val
                total_scaled += scaled_val
    
    # Normalize
    probs = np.zeros(MAX_EDGES_PER_NODE, dtype=np.float32)
    if total_scaled > 1e-8:
        for action in range(MAX_EDGES_PER_NODE):
            if scaled_visits[action] > 0:
                probs[action] = scaled_visits[action] / total_scaled
    
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

    def run_mcts_simulations(self, num_simulations, initial_board_state_arr, initial_player_id, root_node_dummy=None):
        """
        Runs MCTS simulations from the given board state.
        Optimized version that reduces function calls and memory allocations.
        """
        # Fast reset of tree structure
        _fast_reset_tree(self.nodes, self.edges, self.node_counter, initial_player_id)
        
        # Initialize simulation environment
        self.sim_env_for_mcts.reset(board_state_array=initial_board_state_arr, specific_player_to_start=initial_player_id)
        
        # Pre-compute neural network evaluation for root
        root_state_repr = self.sim_env_for_mcts._get_state_for_nn()
        action_priors_nn, value_nn = self.neural_network.predict(root_state_repr)
        
        # Run optimized simulations
        for i_sim in range(num_simulations):
            run_mcts_simulation_single(
                self.nodes,
                self.edges,
                self.node_counter,
                self.config.C_PUCT_CONSTANT,
                self.config.DIRICHLET_ALPHA,
                self.config.DIRICHLET_EPSILON
            )

    def get_action_probabilities(self, board_state_arr, player_id, temp=1.0, root_node_dummy=None):
        """
        Returns the MCTS policy (visit counts normalized) for the root state.
        `root_node_dummy` is ignored.
        Assumes run_mcts_simulations was called before this for the same state.
        """
        action_probs = np.zeros(NUM_ACTIONS, dtype=np.float32)
        
        # Root node is always 0
        root_visit_count = self.nodes[0 * NODE_FIELDS + 0]

        if root_visit_count == 0:  # No simulations run
            # Get legal moves from root node's prior probabilities
            for action in range(NUM_ACTIONS):
                action_probs[action] = self.edges[0, action, 3]  # Use prior probabilities P_s_a
            
            # Normalize priors over legal actions
            temp_env = self.game_env_class()
            temp_env.reset(board_state_array=board_state_arr, specific_player_to_start=player_id)
            legal_actions = temp_env.get_legal_action_indices()
            if legal_actions:
                legal_mask = np.zeros(NUM_ACTIONS, dtype=np.float32)
                for action in legal_actions:
                    legal_mask[action] = 1.0
                action_probs *= legal_mask
                sum_probs = np.sum(action_probs)
                if sum_probs > 0:
                    action_probs /= sum_probs
                else:
                    # If no valid priors, use uniform distribution
                    prob = 1.0 / len(legal_actions)
                    for action in legal_actions:
                        action_probs[action] = prob
            return action_probs

        # Get visit counts for actions from root
        legal_actions = self.sim_env_for_mcts.get_legal_action_indices()
        action_probs = get_action_probabilities(0, self.edges, legal_actions, temp)

        return action_probs

    def search(self, state, n_simulations, temperature=1.0):
        """
        Conduct a MCTS search using optimized Numba functions.
        """
        # Reset node counter and initialize root
        self.node_counter[:] = 1  # Keep root node
        self._copy_state_to_sim_env(state)
        
        # Pre-compute neural network evaluation for root
        state_repr = self.sim_env._get_state_for_nn()
        action_priors, root_value = self.neural_network.predict(state_repr)
        
        # Initialize root edges with priors
        legal_actions = self.sim_env.get_legal_action_indices()
        for action in legal_actions:
            if action >= 0 and action < MAX_EDGES_PER_NODE:
                self.edges[0, action, 3] = action_priors[action]  # P_s_a
                self.edges[0, action, 4] = -1.0  # No child yet
        
        # Run optimized simulations
        for _ in range(n_simulations):
            self._copy_state_to_sim_env(state)
            sim_env_copy = self.sim_env.clone()  # Create a copy for this simulation
            
            while True:
                value, path_nodes, path_actions = run_mcts_simulation_single(
                    self.nodes, self.edges, self.node_counter,
                    self.config.C_PUCT_CONSTANT, 
                    self.config.DIRICHLET_ALPHA,
                    self.config.DIRICHLET_EPSILON
                )
                
                if value == -2.0:  # Node needs expansion
                    current_node = path_nodes[-1]
                    # Get neural network evaluation
                    state_repr = sim_env_copy._get_state_for_nn()
                    action_priors, value_nn = self.neural_network.predict(state_repr)
                    
                    if sim_env_copy.done:
                        # Handle terminal state
                        self.nodes[current_node * NODE_FIELDS + 3] = 1.0
                        player_at_node = self.nodes[current_node * NODE_FIELDS + 2]
                        
                        if sim_env_copy.winner_id == int(player_at_node):
                            self.nodes[current_node * NODE_FIELDS + 4] = 1.0
                        elif sim_env_copy.winner_id == 0 or sim_env_copy.winner_id == 3:
                            self.nodes[current_node * NODE_FIELDS + 4] = 0.0
                        else:
                            self.nodes[current_node * NODE_FIELDS + 4] = -1.0
                        
                        # Backpropagate
                        self._backpropagate(path_nodes, path_actions, self.nodes[current_node * NODE_FIELDS + 4])
                        break
                        
                    legal_actions = sim_env_copy.get_legal_action_indices()
                    if not legal_actions:
                        self.nodes[current_node * NODE_FIELDS + 3] = 1.0
                        self.nodes[current_node * NODE_FIELDS + 4] = -1.0
                        self._backpropagate(path_nodes, path_actions, -1.0)
                        break
                    
                    # Apply Dirichlet noise at root
                    if len(path_nodes) == 1 and self.config.DIRICHLET_ALPHA > 0:
                        noise = np.random.dirichlet([self.config.DIRICHLET_ALPHA] * len(legal_actions))
                        for i, action in enumerate(legal_actions):
                            action_priors[action] = (
                                (1 - self.config.DIRICHLET_EPSILON) * action_priors[action] +
                                self.config.DIRICHLET_EPSILON * noise[i]
                            )
                    
                    # Expand node
                    self._expand_node(current_node, legal_actions, action_priors)
                    self._backpropagate(path_nodes, path_actions, value_nn)
                    break
                    
                elif value == -1.0:  # No valid moves
                    self._backpropagate(path_nodes, path_actions, 0.0)
                    break
                    
                else:  # Terminal state reached
                    self._backpropagate(path_nodes, path_actions, value)
                    break
                
                # Take action in environment
                action = path_actions[-1]
                if action >= 0:
                    sim_env_copy.step(action)
        
        # Get action probabilities
        legal_actions = state.get_legal_action_indices()
        action_probs = get_action_probabilities(0, self.edges, legal_actions, temperature)
        return action_probs
