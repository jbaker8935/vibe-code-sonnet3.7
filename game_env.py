import numpy as np
import random
from collections import deque
import time # Added for testing speed
import numba
from numba import njit

# --- Constants ---
ROWS = 8
COLS = 4
MAX_STEPS_PER_EPISODE = 300  # Maximum steps before forcing a draw

# Numerical Player IDs (Used internally and in JIT functions)
PLAYER_A_ID = 1
PLAYER_B_ID = 2

# Player representations (for external interface/rendering)
PLAYER_A = 'A' # White
PLAYER_B = 'B' # Black

# Numerical Piece States (Used internally and in JIT functions)
STATE_NORMAL = 1
STATE_SWAPPED = 2

# State representation on the board (int8)
EMPTY_CELL = 0
A_NORMAL = 1  # PLAYER_A_ID, STATE_NORMAL
A_SWAPPED = 2 # PLAYER_A_ID, STATE_SWAPPED
B_NORMAL = 3  # PLAYER_B_ID, STATE_NORMAL
B_SWAPPED = 4 # PLAYER_B_ID, STATE_SWAPPED

# Mappings for convenience
PIECE_MAP = {
    A_NORMAL: {'player': PLAYER_A, 'player_id': PLAYER_A_ID, 'state': STATE_NORMAL, 'char': 'AN'},
    A_SWAPPED: {'player': PLAYER_A, 'player_id': PLAYER_A_ID, 'state': STATE_SWAPPED, 'char': 'AS'},
    B_NORMAL: {'player': PLAYER_B, 'player_id': PLAYER_B_ID, 'state': STATE_NORMAL, 'char': 'BN'},
    B_SWAPPED: {'player': PLAYER_B, 'player_id': PLAYER_B_ID, 'state': STATE_SWAPPED, 'char': 'BS'},
    EMPTY_CELL: None
}
PLAYER_ID_MAP = {PLAYER_A: PLAYER_A_ID, PLAYER_B: PLAYER_B_ID}
ID_PLAYER_MAP = {v: k for k, v in PLAYER_ID_MAP.items()}

# Action encoding: (start_cell_index * 8) + direction_index
# Directions: 0: (-1,-1), 1: (-1,0), 2: (-1,1), 3: (0,-1), 4: (0,1), 5: (1,-1), 6: (1,0), 7: (1,1)
DIRECTIONS = np.array([(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)], dtype=np.int8)
NUM_DIRECTIONS = len(DIRECTIONS)
NUM_CELLS = ROWS * COLS
NUM_ACTIONS = NUM_CELLS * NUM_DIRECTIONS # 32 * 8 = 256

# --- Numba JIT Helper Functions ---

@njit(cache=True)
def _is_valid(r, c, rows=ROWS, cols=COLS):
    """Checks if coordinates are within board bounds."""
    return 0 <= r < rows and 0 <= c < cols

@njit(cache=True)
def _get_piece_player_id(piece_code):
    """Gets the player ID from the numerical piece code."""
    if piece_code == A_NORMAL or piece_code == A_SWAPPED:
        return PLAYER_A_ID
    elif piece_code == B_NORMAL or piece_code == B_SWAPPED:
        return PLAYER_B_ID
    else:
        return 0 # No player

@njit(cache=True)
def _get_piece_state_id(piece_code):
    """Gets the state ID from the numerical piece code."""
    if piece_code == A_NORMAL or piece_code == B_NORMAL:
        return STATE_NORMAL
    elif piece_code == A_SWAPPED or piece_code == B_SWAPPED:
        return STATE_SWAPPED
    else:
        return 0 # No state

@njit(cache=True)
def _is_piece_normal(piece_code):
    """Checks if the piece state is NORMAL."""
    return piece_code == A_NORMAL or piece_code == B_NORMAL

@njit(cache=True)
def _unmark_all_swapped(board):
    """Resets all SWAPPED pieces to NORMAL on the board (numerical)."""
    rows, cols = board.shape
    for r in range(rows):
        for c in range(cols):
            piece = board[r, c]
            if piece == A_SWAPPED:
                board[r, c] = A_NORMAL
            elif piece == B_SWAPPED:
                board[r, c] = B_NORMAL
    # No return needed, modifies board in place

@njit(cache=True)
def _get_legal_moves_jit(board, player_id):
    """Calculates all legal moves for the given player_id. Returns list of (start_r, start_c, end_r, end_c)."""
    rows, cols = board.shape
    opponent_id = PLAYER_B_ID if player_id == PLAYER_A_ID else PLAYER_A_ID
    legal_moves = [] # Numba supports lists of simple types like tuples

    for r in range(rows):
        for c in range(cols):
            piece = board[r, c]
            if piece != EMPTY_CELL and _get_piece_player_id(piece) == player_id:
                for i in range(NUM_DIRECTIONS):
                    dr, dc = DIRECTIONS[i]
                    nr, nc = r + dr, c + dc
                    if _is_valid(nr, nc, rows, cols):
                        target_cell = board[nr, nc]
                        if target_cell == EMPTY_CELL:
                            # Move to empty cell
                            legal_moves.append((r, c, nr, nc))
                        elif _get_piece_player_id(target_cell) == opponent_id and _is_piece_normal(target_cell):
                            # Swap with opponent's NORMAL piece
                            legal_moves.append((r, c, nr, nc))
    return legal_moves

@njit(cache=True)
def _apply_move_jit(board, start_r, start_c, end_r, end_c):
    """Applies the move on the numerical board. Returns move_type_code (1=empty, 2=swap)."""
    moving_piece = board[start_r, start_c]
    target_piece = board[end_r, end_c]
    move_type_code = 0

    if target_piece == EMPTY_CELL:
        # Move to empty cell
        board[end_r, end_c] = moving_piece
        board[start_r, start_c] = EMPTY_CELL
        _unmark_all_swapped(board)
        move_type_code = 1 # 'empty'
    else:
        # Swap move (legality check happens before calling this)
        # Determine swapped state for moving piece
        swapped_moving_piece = A_SWAPPED if _get_piece_player_id(moving_piece) == PLAYER_A_ID else B_SWAPPED
        # Determine swapped state for target piece
        swapped_target_piece = A_SWAPPED if _get_piece_player_id(target_piece) == PLAYER_A_ID else B_SWAPPED

        board[end_r, end_c] = swapped_moving_piece
        board[start_r, start_c] = swapped_target_piece
        move_type_code = 2 # 'swap'

    return move_type_code

@njit(cache=True)
def _check_win_condition_jit(board, player_id):
    """Checks if the given player_id has won using BFS on the numerical board."""
    rows, cols = board.shape
    start_row = rows - 2 if player_id == PLAYER_A_ID else 1
    target_row = 1 if player_id == PLAYER_A_ID else rows - 2

    visited = np.zeros((rows, cols), dtype=numba.boolean)
    # Use a list as a queue for Numba compatibility in nopython mode
    queue = [] # List of (r, c) tuples

    # Initialize queue with starting pieces
    for c in range(cols):
        piece = board[start_row, c]
        if piece != EMPTY_CELL and _get_piece_player_id(piece) == player_id:
            queue.append((start_row, c))
            visited[start_row, c] = True

    head = 0 # Index for dequeuing from the list
    while head < len(queue):
        r, c = queue[head]
        head += 1

        if r == target_row:
            return True # Found a path

        for i in range(NUM_DIRECTIONS):
            dr, dc = DIRECTIONS[i]
            nr, nc = r + dr, c + dc
            if _is_valid(nr, nc, rows, cols) and not visited[nr, nc]:
                neighbor_piece = board[nr, nc]
                if neighbor_piece != EMPTY_CELL and _get_piece_player_id(neighbor_piece) == player_id:
                    visited[nr, nc] = True
                    queue.append((nr, nc))

    return False

@njit(cache=True)
def _evaluate_board_jit(board, player_id):
    rows, cols = board.shape
    start_row = rows - 2 if player_id == PLAYER_A_ID else 1
    target_row = 1 if player_id == PLAYER_A_ID else rows - 2
    
    # Add forward progress tracking
    forward_progress = 0
    piece_count = 0
    
    for r in range(rows):
        for c in range(cols):
            piece = board[r, c]
            if piece != EMPTY_CELL and _get_piece_player_id(piece) == player_id:
                piece_count += 1
                # Reward being closer to target row
                if player_id == PLAYER_A_ID:
                    forward_progress += (rows - r)  # Higher score for being closer to top
                else:
                    forward_progress += r  # Higher score for being closer to bottom

    # Normalize forward progress
    avg_progress = forward_progress / max(piece_count, 1)
    
    # Original evaluation components
    active_row_count = 0
    back_row_count = 0
    on_start = 0
    on_target = 0
    
    # ...existing counting code...
    
    # Modified scoring that heavily weights forward progress
    score = (
        (on_start + on_target) * 0.5 +  # Base position score
        2.0 * active_row_count +         # Active pieces
        -1.0 * back_row_count +          # Penalize back row pieces
        3.0 * (avg_progress / rows)      # Normalized forward progress bonus
    )
    
    return score


# --- Environment Class ---

class SwitcharooEnv:
    def __init__(self):
        self.board = np.zeros((ROWS, COLS), dtype=np.int8)
        self.current_player_id = PLAYER_A_ID
        self.winner_id = 0  # 0: None, 1: A, 2: B, 3: Draw
        self.step_count = 0  # Add step counter
        self.reset()

    def reset(self):
        """Resets the game to the initial state."""
        self.board.fill(EMPTY_CELL)
        self.current_player_id = PLAYER_A_ID
        self.winner_id = 0
        self.step_count = 0  # Reset step counter

        for r in range(ROWS):
            for c in range(COLS):
                if r < 2: # Player B (Black) at top
                    self.board[r, c] = B_NORMAL
                elif r >= ROWS - 2: # Player A (White) at bottom
                    self.board[r, c] = A_NORMAL
        return self._get_state()

    def _get_state(self):
        """Enhanced state representation with more meaningful features."""
        # Base state (current implementation)
        flat_board = self.board.flatten().astype(np.float32)
        
        # 1. Create normalized state representation
        state = np.zeros(NUM_CELLS + 1, dtype=np.float32)
        
        # 2. Normalize board values (0-4) to (0-1) range for better neural network processing
        state[:NUM_CELLS] = flat_board / 4.0
        
        # 3. Add current player info (0.0 for A, 1.0 for B)
        state[-1] = 0.0 if self.current_player_id == PLAYER_A_ID else 1.0
        
        return state

    # _is_valid is now a JIT function

    # _unmark_all_swapped is now a JIT function (operates on board passed to it)

    def get_legal_moves(self, player=None):
        """Calculates all legal moves for the given player (A or B)."""
        player_id = PLAYER_ID_MAP[player] if player else self.current_player_id
        # Call the JIT compiled function
        return _get_legal_moves_jit(self.board, player_id)

    def get_legal_action_indices(self, player=None):
        """Returns a list of valid action indices for the current player."""
        legal_moves = self.get_legal_moves(player) # Uses JIT function internally
        legal_indices = []
        for start_r, start_c, end_r, end_c in legal_moves:
            dr, dc = end_r - start_r, end_c - start_c
            # Find direction index (can be slow if done many times, but ok here)
            direction_index = -1
            for i in range(NUM_DIRECTIONS):
                if DIRECTIONS[i, 0] == dr and DIRECTIONS[i, 1] == dc:
                    direction_index = i
                    break

            if direction_index != -1:
                start_cell_index = start_r * COLS + start_c
                action_index = start_cell_index * NUM_DIRECTIONS + direction_index
                legal_indices.append(action_index)
            else:
                 # Should not happen if DIRECTIONS is correct
                 print(f"Error: Invalid direction ({dr}, {dc}) derived from legal move.")

        return legal_indices

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
        """Checks if the given player (A or B) has won."""
        player_id = PLAYER_ID_MAP[player] if player else self.current_player_id
        # Call the JIT compiled function
        return _check_win_condition_jit(self.board, player_id)

    def step(self, action_index):
        """
        Performs a move based on the action index.
        Returns: (next_state, reward, done, info)
        """
        self.step_count += 1  # Increment step counter
        move = self._action_index_to_move(action_index)
        current_player_id = self.current_player_id # Store before potential switch

        # Check move legality (using JIT function indirectly)
        # This check is somewhat redundant if agent always chooses from legal actions,
        # but good for safety. Comparing tuples can be slow.
        # A faster check might be to verify the action_index is in get_legal_action_indices()
        legal_moves_tuples = self.get_legal_moves() # List of tuples
        is_legal = move is not None and move in legal_moves_tuples

        if not is_legal:
            # print(f"Warning: Illegal move {move} attempted by Player {ID_PLAYER_MAP[current_player_id]}. Action index: {action_index}")
            # print(f"Legal moves: {legal_moves_tuples}")
            return self._get_state(), -10.0, False, {'error': 'Illegal move', 'winner': None}

        current_score = _evaluate_board_jit(self.board, current_player_id)

        start_r, start_c, end_r, end_c = move

        # Apply the move using JIT function (modifies self.board)
        move_type_code = _apply_move_jit(self.board, start_r, start_c, end_r, end_c)
        move_type = 'empty' if move_type_code == 1 else 'swap'

        # Check for win condition for the current player (using JIT)
        if _check_win_condition_jit(self.board, current_player_id):
            self.winner_id = current_player_id
            reward = 100.0 # Win reward
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
                reward = 0.0 # Draw reward
                done = True
            else:
                # Game continues
                turn_number_penalty = -0.2 * (1.0 + (self.step_count / MAX_STEPS_PER_EPISODE) ** 2)
                reward = turn_number_penalty

                if move_type == 'swap':
                    reward += 3.0  # Increased swap bonus to encourage aggressive play
                    
                # Modify the heuristic score to reward forward progress
                position_delta = _evaluate_board_jit(self.board, current_player_id) - current_score
                reward += position_delta * 3.0  # Increased weight of position improvements

                done = False

        next_state = self._get_state()
        winner_player = ID_PLAYER_MAP.get(self.winner_id, None if self.winner_id != 3 else 'DRAW')
        info = {'winner': winner_player, 'move_type': move_type}

        # If the game ended because the *other* player won (we lost)
        # This condition needs careful checking after player switch
        if done and self.winner_id != 0 and self.winner_id != 3 and self.winner_id != current_player_id:
             reward = -100.0 # Make sure loss reward is applied correctly

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


    def render(self):
        """Prints the board to the console."""
        print("-" * (COLS * 4 + 1))
        for r in range(ROWS):
            row_str = "|"
            for c in range(COLS):
                piece_code = self.board[r, c]
                piece_info = PIECE_MAP[piece_code]
                if piece_info:
                    row_str += f" {piece_info['char']} |" # Use pre-defined char
                else:
                    row_str += "    |"
            print(row_str)
            print("-" * (COLS * 4 + 1))
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