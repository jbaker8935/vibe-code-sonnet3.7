import numpy as np
import random
from collections import deque
import time # Added for testing speed
from env_const import *

from env_util import (_get_legal_moves_batch, _get_legal_moves_jit,
                      _check_win_condition_jit, _find_winning_path_jit,
                      _calculate_action_indices_jit, _is_valid,
                      _evaluate_board_jit, _apply_move_jit)



# --- Numba JIT Helper Functions ---

def parse_initial_position(position_str, a_normal, a_swapped, b_normal, b_swapped, empty_cell):
    """Parses the initial position string into a numerical board representation.
    Args:
        position_str: String representation of the board with A for player A, B for player B, and . for empty
        a_normal: Integer code for player A normal piece
        b_normal: Integer code for player B normal piece
        a_swapped: Integer code for player A swapped piece
        b_swapped: Integer code for player B swapped piece
        empty_cell: Integer code for empty cell
        
    Returns:
        8x4 numpy array with the parsed board
    """
    # Create empty board
    board = np.zeros((8, 4), dtype=np.int8)
    
    # Split the string into rows
    rows = []
    current_row = ""
    for char in position_str:
        if (char == '\n'):
            if current_row:  # Only add non-empty rows
                rows.append(current_row)
                current_row = ""
        elif char not in (' ', '\t', '\r'):  # Skip whitespace
            current_row += char
    
    # Add the last row if it exists
    if current_row:
        rows.append(current_row)
    
    # Parse each character in each row
    for r in range(min(len(rows), 8)):
        row = rows[r]
        for c in range(min(len(row), 4)):
            char = row[c]
            if char == 'A':
                board[r, c] = a_normal
            elif char == 'a':
                board[r, c] = a_swapped
            elif char == 'B':
                board[r, c] = b_normal
            elif char == 'b':
                board[r, c] = b_swapped
            else:  # Default to empty cell for any other character
                board[r, c] = empty_cell
                
    return board

# --- Environment Class ---

from binary_board import board_to_binary, binary_to_board

class SwitcharooEnv:
    def __init__(self):
        self.board = np.zeros((ROWS, COLS), dtype=np.int8)
        self.current_player_id = PLAYER_A_ID
        self.winner_id = 0  # 0: None, 1: A, 2: B, 3: Draw
        self.step_count = 0  # Add step counter
        self.move_history_a = None
        self.move_history_b = None
        self.reset()

    def reset(self, starting_position=None, board_state_array=None, specific_player_to_start=None):
        """
        Reset the environment to the initial state.

        Args:
            starting_position: Optional string describing the board (from config.initial_position).
            board_state_array: Optional numpy array with the board state to use.
            specific_player_to_start: Optional player ID to force as the starting player.
            
        Returns:
            The initial state representation.
        """
        self.board.fill(EMPTY_CELL)
        self.current_player_id = PLAYER_A_ID if specific_player_to_start is None else specific_player_to_start
        self.winner_id = 0
        self.step_count = 0  # Reset step counter

        # Support both string position and direct board array initialization
        if board_state_array is not None:
            # Direct copy of provided board state
            self.board = np.copy(board_state_array)
        elif starting_position:
            # Parse string representation
            self.board = parse_initial_position(starting_position, A_NORMAL, A_SWAPPED, B_NORMAL, B_SWAPPED, EMPTY_CELL)  
        else:
            # Default initialization
            for r in range(ROWS):
                for c in range(COLS):
                    if (r < 2):
                        self.board[r, c] = B_NORMAL
                    elif (r > 5):
                        self.board[r, c] = A_NORMAL
                        
        # Store initial starting row piece counts
        self._initial_starting_rows = {
            PLAYER_A_ID: np.sum((self.board[6:8, :] == A_NORMAL) | (self.board[6:8, :] == A_SWAPPED)),
            PLAYER_B_ID: np.sum((self.board[0:2, :] == B_NORMAL) | (self.board[0:2, :] == B_SWAPPED)),
        }

        return self._get_state()

    def _get_state(self):
        """
        Returns a 6-channel binary representation of the board state and current player.
        Channel 0: Player A Normal pieces (A_NORMAL) (1.0 if present, 0.0 otherwise)
        Channel 1: Player A Swapped pieces (A_SWAPPED)
        Channel 2: Player B Normal pieces (B_NORMAL)
        Channel 3: Player B Swapped pieces (B_SWAPPED)
        Channel 4: Empty cells (EMPTY_CELL)
        Channel 5: Current player (all 0.0 for Player A, all 1.0 for Player B)

        Each channel is a flattened ROWS x COLS board.
        Total state size: 6 * ROWS * COLS.
        """
        flat_board_size = ROWS * COLS

        # Board state channels
        channel_a_normal = (self.board == A_NORMAL).astype(np.float32).flatten()
        channel_a_swapped = (self.board == A_SWAPPED).astype(np.float32).flatten()
        channel_b_normal = (self.board == B_NORMAL).astype(np.float32).flatten()
        channel_b_swapped = (self.board == B_SWAPPED).astype(np.float32).flatten()
        channel_empty = (self.board == EMPTY_CELL).astype(np.float32).flatten()

        # Player channel
        player_channel = np.zeros(flat_board_size, dtype=np.float32)
        if self.current_player_id == PLAYER_B_ID:
            player_channel.fill(1.0)

        # Concatenate all channels
        state = np.concatenate([
            channel_a_normal,
            channel_a_swapped,
            channel_b_normal,
            channel_b_swapped,
            channel_empty,
            player_channel
        ])
        return state

    def _get_state_for_nn(self):
        """Returns state representation in the format expected by the neural network.
        This is an alias for _get_state() to maintain compatibility with both MCTS implementations."""
        return self._get_state()

    # _is_valid is now a JIT function

    # _unmark_all_swapped is now a JIT function (operates on board passed to it)

    def get_legal_moves(self, player=None):
        """Calculates all legal moves for the given player (A or B)."""
        player_id = PLAYER_ID_MAP[player] if player else self.current_player_id
        # Call the JIT compiled function
        return _get_legal_moves_jit(self.board, player_id)

    def get_legal_action_indices(self, player=None):
        """Returns a list of valid action indices for the current player."""
        legal_moves = self.get_legal_moves(player)  # Uses JIT function internally
        return _calculate_action_indices_jit(legal_moves, DIRECTIONS)

    def _action_index_to_move(self, action_index):
        """Converts an action index back to (start_r, start_c, end_r, end_c)."""
        if not (0 <= action_index < NUM_ACTIONS):
            return None # Invalid index

        direction_index = action_index % NUM_DIRECTIONS
        start_cell_index = action_index // NUM_DIRECTIONS

        start_r = start_cell_index // COLS
        start_c = start_cell_index % COLS

        dr, dc = DIRECTIONS[direction_index]
        end_r, end_c = start_r + dr, start_c + dc

        # Basic validation (coordinate check)
        if not _is_valid(start_r, start_c) or not _is_valid(end_r, end_c):
             # print(f"Warning: Action index {action_index} results in invalid coords.")
             return None # Should ideally not happen if action index is from legal list

        return start_r, start_c, end_r, end_c

    def check_win_condition(self, player=None):
        """Checks if the given player (A or B) has won and returns the winning path if any."""
        player_id = PLAYER_ID_MAP[player] if player else self.current_player_id
        
        # First use the faster win check to see if there's a win at all
        if _check_win_condition_jit(self.board, player_id):
            # If a win is detected, call the path-finding function to get the actual path
            win_path = _find_winning_path_jit(self.board, player_id)
            return True, win_path
        
        return False, []

    def step(self, action_index):
        """
        Performs a move based on the action index.
        Returns: (next_state, reward, done, info)
        """

        reward = 0.0
        
        self.step_count += 1  # Increment step counter
        # move = self._action_index_to_move(action_index) # Deferred
        current_player_id = self.current_player_id # Store before potential switch

        # Optimized move legality check:
        # Use get_legal_action_indices() which should return a list/array of integers.
        # Checking for an integer in a list/array of integers is generally faster
        # than checking for a tuple in a list of tuples.
        legal_action_indices = self.get_legal_action_indices()
        is_legal = action_index in legal_action_indices

        if not is_legal:
            # print(f"Warning: Illegal move attempted with action index: {action_index} by Player {ID_PLAYER_MAP[current_player_id]}.")
            # print(f"Legal action indices: {legal_action_indices}")
            return self._get_state(), -10.0, False, {'error': 'Illegal move', 'winner': None}

        # If the action_index is legal, now convert it to move coordinates.
        # This work is now only done for legal moves.
        move = self._action_index_to_move(action_index)

        # Robustness check: If action_index was in legal_action_indices,
        # _action_index_to_move should always return a valid move.
        if move is None:
            print(f"CRITICAL INCONSISTENCY: Action index {action_index} was in legal_action_indices but _action_index_to_move returned None.")
            # This indicates a potential bug or inconsistency between get_legal_action_indices and _action_index_to_move.
            return self._get_state(), -100.0, True, {'error': 'Internal move consistency error', 'winner': None}

        current_score = _evaluate_board_jit(self.board, current_player_id)

        start_r, start_c, end_r, end_c = move

        # Apply the move using JIT function (modifies self.board)
        move_type_code = _apply_move_jit(self.board, start_r, start_c, end_r, end_c)
        move_type = 'empty' if move_type_code == 1 else 'swap'

        # Check for win condition for the current player (using JIT with path finding)
        has_won, win_path = self.check_win_condition(ID_PLAYER_MAP[current_player_id])
        if has_won:
            self.winner_id = current_player_id
            reward = 15.0 # Further reduced win reward for balance
            # Add quick win bonus
            quick_win_bonus = max(0, 10 - self.step_count / 10)
            reward += quick_win_bonus
            done = True
        else:
            # Switch player
            opponent_id = PLAYER_B_ID if current_player_id == PLAYER_A_ID else PLAYER_A_ID
            self.current_player_id = opponent_id

            # Check if the opponent now has any legal moves (stalemate check) (using JIT)
            opponent_legal_moves = _get_legal_moves_jit(self.board, opponent_id)
            if not opponent_legal_moves:
                # It's a draw/stalemate if the opponent has no moves
                self.winner_id = 3 # Draw
                reward = -10.0 # Penalize draws more heavily
                done = True  # <-- Ensure done is set to True for draw
            else:
                # Game continues - Modified reward structure
                reward = 0.0  # Base reward
                
                if move_type == 'swap':
                    reward += 1.0  # Reduced swap bonus
                
                # Position improvement reward
                position_delta = _evaluate_board_jit(self.board, current_player_id) - current_score
                reward += position_delta * 0.5 # Further reduced position delta multiplier for balance
                
                done = False

        next_state = self._get_state()
        winner_player = ID_PLAYER_MAP.get(self.winner_id, None if self.winner_id != 3 else 'DRAW')
        info = {'winner': winner_player, 'move_type': move_type}
        if not is_legal:
            info['error'] = 'Illegal move'
        return next_state, reward, done, info

    @property
    def current_player(self):
        """Returns the string representation of the current player."""
        return ID_PLAYER_MAP[self.current_player_id]

    @property
    def winner(self):
        """Returns the string representation of the winner."""
        if self.winner_id == 3:
            return 'DRAW'
        return ID_PLAYER_MAP.get(self.winner_id)

    @property
    def done(self):
        """Returns whether the game is finished (has a winner or is a draw)."""
        return self.winner_id != 0  # Any non-zero winner_id (1, 2, or 3) means game is done

    def render(self):
        """Prints the board to the console."""
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
        print(f"Current Player: {self.current_player} (ID: {self.current_player_id})")
        if self.winner:
            print(f"Winner: {self.winner}")

# --- Example Usage (Updated) ---
if __name__ == "__main__":
    # Test JIT compilation on load
    print("Testing JIT compilation...")
    test_board = np.random.randint(0, 5, size=(ROWS, COLS), dtype=np.int8)
    _ = _get_legal_moves_jit(test_board, PLAYER_A_ID)
    _ = _check_win_condition_jit(test_board, PLAYER_B_ID)
    print("JIT functions compiled (or cached).")

    env = SwitcharooEnv()
    print("\n--- Initial Board ---")
    env.render()
    state = env.reset()
    # print("Initial State:", state) # State is larger now

    done = False
    turn = 0
    max_turns = 200 # Limit game length for testing

    start_time = time.time()

    while not done and turn < max_turns:
        # print(f"\n--- Turn {turn + 1} ---")
        # env.render() # Optional: render every turn slows down testing
        legal_actions = env.get_legal_action_indices()
        # print(f"Player {env.current_player} legal action indices: {len(legal_actions)}")

        if not legal_actions:
            print(f"Player {env.current_player} has no legal moves! Stalemate?")
            env.winner_id = 3 # Manually set draw if no moves
            done = True
            break

        # Choose a random legal action
        action = random.choice(legal_actions)
        # move = env._action_index_to_move(action) # Optional debug
        # print(f"Player {env.current_player} choosing action index: {action} -> move: {move}")

        next_state, reward, done, info = env.step(action)

        # print(f"Reward: {reward}, Done: {done}, Info: {info}")
        state = next_state
        turn += 1

    end_time = time.time()

    print("\n--- Final Board ---")
    env.render()
    if turn == max_turns:
        print("Max turns reached.")
    print(f"Game finished in {turn} turns.")
    print(f"Simulation took {end_time - start_time:.4f} seconds.")