import numpy as np
import numba as nb
from numba.experimental import jitclass
from numba.typed import Dict

# Import JITted utility functions and constants
# Ensure all constants are Numba-compatible (e.g., DIRECTIONS_TUPLE)
from env_const import (
    ROWS, COLS, PLAYER_A_ID, PLAYER_B_ID, EMPTY_CELL,
    A_NORMAL, B_NORMAL, A_SWAPPED, B_SWAPPED,
    NUM_ACTIONS, NUM_DIRECTIONS, DIRECTIONS, DIRECTIONS_TUPLE
)
# from env_util import (
#     _get_legal_moves_jit, _check_win_condition_jit,
#     _find_winning_path_jit, _calculate_action_indices_jit,
#     _is_valid, _apply_move_jit, _evaluate_board_jit
# ) # These will be methods or called by methods of the jitclass

# Import necessary JIT utility functions
from env_util import (
    _evaluate_board_jit, _get_legal_moves_jit, 
    _apply_move_jit, _is_valid, _check_win_condition_jit, 
    _find_winning_path_jit, _calculate_action_indices_jit
)

# Define Numba types for dictionaries
# Player IDs are int8, counts/values are int32
int8_int_dict_type = nb.types.DictType(nb.int8, nb.int32) # For general integer values
# If specific float values are needed for other dicts, define their types too.

spec = [
    ('board', nb.int8[:,:]),
    ('current_player_id', nb.int8),
    ('winner_id', nb.int8),       # 0: None, 1: A, 2: B, 3: Draw
    ('step_count', nb.int32),
    # Dictionaries for game logic/heuristics
    ('_early_game_move_counts', int8_int_dict_type),
    ('_initial_starting_rows', int8_int_dict_type),
    ('_last_starting_row_counts', int8_int_dict_type),
    # Add any other simple numerical attributes if they were part of the original class
    # For example, if move_history was essential and could be made Numba-compatible:
    # ('move_history_a', nb.optional(nb.int32[:,:])), # Example if storing fixed-size history
    # ('move_history_b', nb.optional(nb.int32[:,:])), # Example
]

# Placeholder for the jitclass definition, to be filled in Phase 2
@jitclass(spec)
class SwitcharooEnvJit:
    def __init__(self, initial_board_state_flat=None, current_player_id_init=PLAYER_A_ID, step_count_init=0):
        # Initialize all attributes from spec
        # Ensure board is initialized with correct dimensions before _reset_internal might use it
        self.board = np.zeros((ROWS, COLS), dtype=np.int8) 
        self.current_player_id = PLAYER_A_ID # Default, will be set by reset
        self.winner_id = np.int8(0) # Explicitly cast to ensure type
        self.step_count = np.int32(0) # Explicitly cast

        # Initialize typed dictionaries
        self._early_game_move_counts = Dict.empty(nb.int8, nb.int32)
        self._initial_starting_rows = Dict.empty(nb.int8, nb.int32)
        self._last_starting_row_counts = Dict.empty(nb.int8, nb.int32)

        # Call reset logic, ensuring to pass only the expected arguments
        # step_count_init is not used by _reset_internal, it's set directly in _reset_internal
        self._reset_internal(initial_board_state_flat, current_player_id_init)

    def _reset_internal(self, board_state_array_reset, specific_player_to_start_reset):
        self.current_player_id = specific_player_to_start_reset
        self.winner_id = np.int8(0)
        self.step_count = np.int32(0) # step_count is reset here

        # Clear and initialize dictionaries
        self._early_game_move_counts.clear()
        self._initial_starting_rows.clear()
        self._last_starting_row_counts.clear()
        # Ensure keys exist for both players
        self._early_game_move_counts[PLAYER_A_ID] = np.int32(0)
        self._early_game_move_counts[PLAYER_B_ID] = np.int32(0)
        self._initial_starting_rows[PLAYER_A_ID] = np.int32(0) # Will be updated below
        self._initial_starting_rows[PLAYER_B_ID] = np.int32(0) # Will be updated below
        self._last_starting_row_counts[PLAYER_A_ID] = np.int32(0) # Will be updated below
        self._last_starting_row_counts[PLAYER_B_ID] = np.int32(0) # Will be updated below

        if board_state_array_reset is not None:
            # Numba requires explicit loops for array copy
            for r_idx in range(ROWS):
                for c_idx in range(COLS):
                    self.board[r_idx, c_idx] = np.int8(board_state_array_reset[r_idx, c_idx])
        else:
            # Default board initialization
            for r_fill in range(ROWS): # Iterate through all rows to ensure EMPTY_CELL is set
                for c_fill in range(COLS):
                    self.board[r_fill, c_fill] = np.int8(EMPTY_CELL)

            for r_init in range(ROWS):
                for c_init in range(COLS):
                    if r_init < 2:
                        self.board[r_init, c_init] = np.int8(B_NORMAL)
                    elif r_init > 5:
                        self.board[r_init, c_init] = np.int8(A_NORMAL)
        
        # Calculate and store initial starting row piece counts (Numba-compatible)
        count_a = 0
        for r_idx_a in range(6, 8): # Player A rows
            for c_idx_a in range(COLS):
                piece_a = self.board[r_idx_a, c_idx_a]
                if piece_a == A_NORMAL or piece_a == A_SWAPPED:
                    count_a += 1
        self._initial_starting_rows[PLAYER_A_ID] = np.int32(count_a)
        self._last_starting_row_counts[PLAYER_A_ID] = np.int32(count_a)

        count_b = 0
        for r_idx_b in range(0, 2): # Player B rows
            for c_idx_b in range(COLS):
                piece_b = self.board[r_idx_b, c_idx_b]
                if piece_b == B_NORMAL or piece_b == B_SWAPPED:
                    count_b += 1
        self._initial_starting_rows[PLAYER_B_ID] = np.int32(count_b)
        self._last_starting_row_counts[PLAYER_B_ID] = np.int32(count_b)
        # No return needed, modifies self

    def _get_state_internal(self):
        flat_board_size = ROWS * COLS
        # Create channels
        channel_a_normal = (self.board == A_NORMAL).astype(np.float32).flatten()
        channel_a_swapped = (self.board == A_SWAPPED).astype(np.float32).flatten()
        channel_b_normal = (self.board == B_NORMAL).astype(np.float32).flatten()
        channel_b_swapped = (self.board == B_SWAPPED).astype(np.float32).flatten()
        channel_empty = (self.board == EMPTY_CELL).astype(np.float32).flatten()

        player_channel = np.zeros(flat_board_size, dtype=np.float32)
        if self.current_player_id == PLAYER_B_ID:
            player_channel.fill(1.0)
        
        # Numba's np.concatenate requires a tuple of arrays
        # Ensure all arrays are 1D and of the same type (float32)
        return np.concatenate((channel_a_normal, channel_a_swapped, 
                               channel_b_normal, channel_b_swapped, 
                               channel_empty, player_channel))

    def _action_index_to_move_internal(self, action_index):
        if not (0 <= action_index < NUM_ACTIONS):
            return -1, -1, -1, -1 # Error/Invalid

        direction_idx = action_index % NUM_DIRECTIONS
        start_cell_idx = action_index // NUM_DIRECTIONS
        start_r = start_cell_idx // COLS
        start_c = start_cell_idx % COLS

        dr_dc_tuple = DIRECTIONS_TUPLE[direction_idx]
        dr = dr_dc_tuple[0]
        dc = dr_dc_tuple[1]
        
        end_r, end_c = start_r + dr, start_c + dc

        if not (_is_valid(start_r, start_c, ROWS, COLS) and _is_valid(end_r, end_c, ROWS, COLS)):
            return -1, -1, -1, -1 # Error/Invalid
        return start_r, start_c, end_r, end_c

    def step_internal(self, action_index): # Type hint for action_index removed for Numba
        # Returns: flat_board_state, reward, done, winner_id, current_player_id, step_count, move_type_code
        
        current_player_id_before_move = self.current_player_id
        current_score = _evaluate_board_jit(self.board, current_player_id_before_move)

        legal_action_indices = self._get_legal_action_indices_internal(current_player_id_before_move)
        is_legal = False
        for i in range(legal_action_indices.shape[0]):
            if legal_action_indices[i] == action_index:
                is_legal = True
                break
        
        if not is_legal:
            # Always return the 6-channel float32 state for type consistency
            return self._get_state_internal(), -10.0, False, np.int8(0), current_player_id_before_move, self.step_count, 0

        self.step_count += 1

        move = self._action_index_to_move_internal(action_index)
        start_r, start_c, end_r, end_c = move

        move_type_code_applied = _apply_move_jit(self.board, start_r, start_c, end_r, end_c) # Corrected function name
        reward = 0.0

        # REMOVED: Early game heuristic and starting row bonus

        has_won = _check_win_condition_jit(self.board, current_player_id_before_move)
        done_flag = False

        if has_won:
            self.winner_id = current_player_id_before_move
            reward = 15.0
            quick_win_bonus = max(0.0, 10.0 - float(self.step_count) / 10.0)
            reward += quick_win_bonus
            done_flag = True
        else:
            if current_player_id_before_move == PLAYER_A_ID:
                self.current_player_id = PLAYER_B_ID
            else:
                self.current_player_id = PLAYER_A_ID

            opponent_legal_moves = self._get_legal_action_indices_internal(self.current_player_id)
            if opponent_legal_moves.shape[0] == 0: # No legal moves for opponent
                self.winner_id = np.int8(3) # Draw
                reward = -10.0 # Penalty for stalemate or forcing opponent into no-move state if that's bad
                done_flag = True
            else:
                if move_type_code_applied == 2: # Swap
                    reward += 1.0
                new_score = _evaluate_board_jit(self.board, current_player_id_before_move)
                position_delta = new_score - current_score
                reward += position_delta * 0.5
                done_flag = False # Game continues

        # If game ended, and winner is not current player (e.g. opponent won or draw), adjust reward
        if done_flag and self.winner_id != 0 and self.winner_id != current_player_id_before_move:
            # If it's a draw (winner_id == 3), reward is already -10.0 from above.
            # If opponent won (winner_id is opponent's ID), then current player lost.
            if self.winner_id != np.int8(3): # Not a draw, means opponent won
                 reward = -15.0

        # Return the 6-channel state representation, not just the flattened board
        return self._get_state_internal(), reward, done_flag, self.winner_id, self.current_player_id, self.step_count, move_type_code_applied

    # --- Methods called by SwitcharooEnvJitWrapper or internally ---
    def _get_legal_action_indices_internal(self, player_id): # Type hint for player_id removed
        """Returns a Numba array of valid action indices for the given player."""
        legal_moves = _get_legal_moves_jit(self.board, player_id)
        action_indices = _calculate_action_indices_jit(legal_moves, DIRECTIONS) # Already a NumPy array
        return action_indices

    def check_win_internal(self, player_id): # Type hint for player_id removed
        """Checks if the given player has won."""
        return _check_win_condition_jit(self.board, player_id)

    def is_done_internal(self): # Return type hint removed
        """Checks if the game is done (win, draw)."""
        return self.winner_id != np.int8(0)

# --- Python Wrapper for SwitcharooEnvJit ---

# Import original parser and other Python-specific utilities/constants if needed by the wrapper
from game_env import parse_initial_position # For parsing string-based initial positions
from env_const import PLAYER_A_ID, ID_PLAYER_MAP, PIECE_MAP # For mapping IDs to strings, rendering

class SwitcharooEnvJitWrapper:
    def __init__(self):
        self._env_jit = None # This will hold the instance of SwitcharooEnvJit
        # For history tracking
        self._prev_board_1 = None
        self._prev_board_2 = None
        # Initial reset to instantiate the jitclass object
        # Pass None for board_state_array_init to trigger default setup in jitclass
        # Pass PLAYER_A_ID as default starting player
        self.reset(board_state_array=None, specific_player_to_start=PLAYER_A_ID)

    def reset(self, starting_position=None, board_state_array=None, specific_player_to_start=None):
        """
        Reset the environment to the initial state.
        Args:
            starting_position: Optional string describing the board.
            board_state_array: Optional numpy array with the board state to use.
            specific_player_to_start: Optional player ID to force as the starting player.
        Returns:
            The initial state representation.
        """
        parsed_board_for_jit = None
        if board_state_array is not None:
            # Ensure it's a copy with the correct dtype if coming from an external source
            if not (isinstance(board_state_array, np.ndarray) and board_state_array.dtype == np.int8):
                raise ValueError("board_state_array must be a numpy array with dtype np.int8 for JIT env.")
            parsed_board_for_jit = np.array(board_state_array, dtype=np.int8)
        elif starting_position:
            # Use the original Python parser from game_env.py
            parsed_board_for_jit = parse_initial_position(starting_position, A_NORMAL, A_SWAPPED, B_NORMAL, B_SWAPPED, EMPTY_CELL)
        
        # Determine the player to start
        player_to_start_id = PLAYER_A_ID if specific_player_to_start is None else specific_player_to_start

        # Instantiate or re-initialize the jitclass instance
        # For simplicity in this plan, we re-instantiate on each reset.
        # A more complex setup could reuse the instance and call its _reset_internal method,
        # but re-instantiation is cleaner for ensuring a fresh state.
        self._env_jit = SwitcharooEnvJit(parsed_board_for_jit, player_to_start_id)
        # Reset history: pad with zeros for first two states
        self._prev_board_1 = np.zeros_like(self._env_jit.board)
        self._prev_board_2 = np.zeros_like(self._env_jit.board)
        return self._get_state_for_nn()

    def step(self, action_index):
        """
        Performs a move based on the action index.
        Returns: (next_state, reward, done, info)
        """
        # Call the jitclass's internal step method
        # Corrected to unpack 7 values
        next_state, reward, done, winner_id, _, _, move_type_code = \
            self._env_jit.step_internal(action_index)

        # Update history: shift boards
        self._prev_board_2 = np.copy(self._prev_board_1)
        self._prev_board_1 = np.copy(self._env_jit.board)

        # next_state is not used directly; we always use _get_state_for_nn()
        info = {}
        # The outcome_code logic was based on a misunderstanding. 
        # step_internal now returns winner_id directly.
        # Illegal moves are handled within step_internal and return specific rewards/done states.
        # We can simplify the info dictionary construction.

        # Map move_type_code to string
        if move_type_code == 1: # empty (normal move)
            info['move_type'] = 'empty'
        elif move_type_code == 2: # swap
            info['move_type'] = 'swap'
        elif move_type_code == 0: # illegal move
            info['move_type'] = 'illegal' # Or 'error', or handle as per original env
            info['error'] = 'Illegal move'
        else:
            info['move_type'] = 'unknown' 
            
        # Map winner_id to string representation
        if winner_id == 3: # Draw
            info['winner'] = 'DRAW'
        elif winner_id != 0: # Player A or B won
            info['winner'] = ID_PLAYER_MAP.get(winner_id) # Uses dict from env_const
        else: # No winner yet or game ongoing
            info['winner'] = None
                
        return self._get_state_for_nn(), reward, done, info

    def get_legal_action_indices(self, player=None):
        """Returns a list of valid action indices for the current or specified player."""
        player_id_to_use = self._env_jit.current_player_id
        if player:  # If a specific player string ('A' or 'B') is provided
            found_player_id = -1  # Default to an invalid ID
            for pid_key, name_val in ID_PLAYER_MAP.items():
                if name_val == player:
                    found_player_id = pid_key
                    break
            if found_player_id != -1:
                player_id_to_use = found_player_id
        return self._env_jit._get_legal_action_indices_internal(player_id_to_use)

    def check_win_condition(self, player=None):
        """Checks if the given player has won and returns the winning path if any."""
        player_id_to_use = self._env_jit.current_player_id
        if player:
            found_player_id = -1
            for pid_key, name_val in ID_PLAYER_MAP.items():
                if name_val == player:
                    found_player_id = pid_key
                    break
            if found_player_id != -1:
                player_id_to_use = found_player_id
        has_won = self._env_jit.check_win_internal(player_id_to_use)
        win_path = []
        if has_won:
            win_path = _find_winning_path_jit(self._env_jit.board, player_id_to_use)
        return has_won, win_path

    def _get_state(self): # To match original API if called directly
        """Returns state representation for the neural network: current + 2 prior boards."""
        return self._get_state_for_nn()

    def _get_state_for_nn(self):
        """Returns state representation for the neural network: current + 2 prior boards, as (8, 4, 18) for tfjs."""
        current_player_id = self._env_jit.current_player_id
        opponent_player_id = PLAYER_B_ID if current_player_id == PLAYER_A_ID else PLAYER_A_ID
        def encode_board(board, player_id):
            # Returns (6, 8, 4): channels, rows, cols
            channel_a_normal = (board == A_NORMAL).astype(np.float32)
            channel_a_swapped = (board == A_SWAPPED).astype(np.float32)
            channel_b_normal = (board == B_NORMAL).astype(np.float32)
            channel_b_swapped = (board == B_SWAPPED).astype(np.float32)
            channel_empty = (board == EMPTY_CELL).astype(np.float32)
            player_channel = np.zeros((ROWS, COLS), dtype=np.float32)
            if player_id == PLAYER_B_ID:
                player_channel.fill(1.0)
            return np.stack([
                channel_a_normal, channel_a_swapped, channel_b_normal,
                channel_b_swapped, channel_empty, player_channel
            ], axis=0)
        current = encode_board(self._env_jit.board, current_player_id)  # (6, 8, 4)
        prev1 = encode_board(self._prev_board_1, opponent_player_id)  # (6, 8, 4)
        prev2 = encode_board(self._prev_board_2, current_player_id)  # (6, 8, 4)
        stacked = np.concatenate([current, prev1, prev2], axis=0)  # (18, 8, 4)
        state = np.transpose(stacked, (1, 2, 0))  # (8, 4, 18)
        return state
        # Properties to match the original SwitcharooEnv API
    @property
    def board(self):
        return self._env_jit.board
    @property
    def current_player_id(self):
        return self._env_jit.current_player_id
    @property
    def winner_id(self):
        return self._env_jit.winner_id
    @property
    def step_count(self):
        return self._env_jit.step_count
    
    @property
    def current_player(self): # Returns 'A' or 'B'
        return ID_PLAYER_MAP.get(self._env_jit.current_player_id)
    @property
    def winner(self): # Returns 'A', 'B', or 'DRAW'
        winner_id_val = self._env_jit.winner_id
        if winner_id_val == 3: 
            return 'DRAW'
        return ID_PLAYER_MAP.get(winner_id_val)
    @property
    def done(self):
        return self._env_jit.is_done_internal()

    def _action_index_to_move(self, action_index):
        """Converts an action index to a move tuple (start_r, start_c, end_r, end_c)."""
        return self._env_jit._action_index_to_move_internal(action_index)


    def render(self):
        """Prints the board to the console. This logic remains in Python."""
        print("-" * COLS)
        for r_idx in range(ROWS):
            row_str = ""
            for c_idx in range(COLS):
                piece_code = self._env_jit.board[r_idx, c_idx]
                # PIECE_MAP is imported from env_const
                piece_info = PIECE_MAP.get(piece_code)
                if piece_info and 'char' in piece_info: # Check if piece_info is not None and has 'char'
                    row_str += f"{piece_info['char']}"
                else: # Handle EMPTY_CELL or unexpected codes gracefully
                    row_str += "." # Adjusted spacing for empty
            print(row_str)
        print("-" * (COLS))
        
        current_player_str = self.current_player # Use property
        print(f"Current Player: {current_player_str} (ID: {self.current_player_id})")
        
        winner_str = self.winner # Use property
        if winner_str:
            print(f"Winner: {winner_str}")

if __name__ == "__main__":
    import numpy as np
    # Create a dummy board with some pieces for more realistic testing
    board = np.zeros((6, 7), dtype=np.int8)
    board[0, 0] = 1  # A_NORMAL
    board[5, 6] = 2  # B_NORMAL
    env = SwitcharooEnvJit(board, np.int8(1), 0)
    # Test _reset_internal with a different player
    env._reset_internal(board, np.int8(2))
    # Test step_internal with a dummy action
    dummy_action = 0
    env.step_internal(dummy_action)
    # Test _get_state_internal
    state = env._get_state_internal()
    # Print state shape to ensure it runs
    print("State shape:", state.shape)

