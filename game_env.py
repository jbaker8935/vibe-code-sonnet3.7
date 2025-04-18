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
HISTORY_LENGTH = 4  # Store 2 moves per player (reduced from 8)
EARLY_GAME_MOVES = 40  # First 20 moves by each player
MID_GAME_MOVES = 60   # Next 10 moves by each player

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
def parse_initial_position_jit(position_str, a_normal, b_normal, empty_cell):
    """JIT-compiled function to convert string representation of a board to a numerical board array.
    
    Args:
        position_str: String representation of the board with A for player A, B for player B, and . for empty
        a_normal: Integer code for player A normal piece
        b_normal: Integer code for player B normal piece
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
            elif char == 'B':
                board[r, c] = b_normal
            else:  # Default to empty cell for any other character
                board[r, c] = empty_cell
                
    return board

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
def _unmark_swapped_jit(board):
    """Resets SWAPPED pieces to NORMAL"""
    rows, cols = board.shape
    for r in range(rows):
        for c in range(cols):
            piece = board[r, c]
            if (piece == A_SWAPPED):
                board[r, c] = A_NORMAL
            elif (piece == B_SWAPPED):
                board[r, c] = B_NORMAL

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
    moving_player_id = _get_piece_player_id(moving_piece)

    if target_piece == EMPTY_CELL:
        # Move to empty cell
        board[end_r, end_c] = moving_piece
        board[start_r, start_c] = EMPTY_CELL
        # unmark swapped pieces
        _unmark_swapped_jit(board)
        move_type_code = 1 # 'empty'
    else:
        # Swap move (legality check happens before calling this)
        # Determine swapped state for moving piece
        swapped_moving_piece = A_SWAPPED if moving_player_id == PLAYER_A_ID else B_SWAPPED
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

    visited = np.zeros((rows, cols), dtype=np.bool_)
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

@njit(parallel=True, cache=True)
def _evaluate_board_batch(boards, player_id):
    """Evaluates multiple board states in parallel."""
    num_boards = boards.shape[0]
    scores = np.zeros(num_boards, dtype=np.float32)
    
    for i in numba.prange(num_boards):
        scores[i] = _evaluate_board_jit(boards[i], player_id)
    
    return scores

@njit(cache=True)
def _evaluate_board_jit(board, player_id):
    """Optimized board evaluation with opponent move checks and position scoring."""
    rows, cols = board.shape
    start_row = rows - 2 if player_id == PLAYER_A_ID else 1
    target_row = 1 if player_id == PLAYER_A_ID else rows - 2
    opponent_id = PLAYER_B_ID if player_id == PLAYER_A_ID else PLAYER_A_ID

    # Check if any opponent move results in a win
    opponent_moves = _get_legal_moves_jit(board, opponent_id)
    for move in opponent_moves:
        start_r, start_c, end_r, end_c = move
        board_copy = board.copy()
        _apply_move_jit(board_copy, start_r, start_c, end_r, end_c)
        if _check_win_condition_jit(board_copy, opponent_id):
            return -100.0  # Penalize if opponent can win in one move

    # Vectorized piece counting and progress calculation
    piece_positions = np.where((board == (A_NORMAL if player_id == PLAYER_A_ID else B_NORMAL)) | 
                              (board == (A_SWAPPED if player_id == PLAYER_A_ID else B_SWAPPED)))

    if len(piece_positions[0]) == 0:
        return -100.0  # Penalize states with no pieces

    # Calculate row occupancy score
    occupied_rows = np.unique(piece_positions[0])
    rows_in_range = np.sum((occupied_rows >= 1) & (occupied_rows <= 6))
    row_occupancy_score = rows_in_range * 1.5  # Weight for row occupancy

    # Check for pieces on start_row and target_row
    has_piece_on_start_row = start_row in piece_positions[0]
    has_piece_on_target_row = target_row in piece_positions[0]
    
    start_row_bonus = 1.0 if has_piece_on_start_row else 0.0
    target_row_bonus = 1.0 if has_piece_on_target_row else 0.0

    # Count swapped pieces for the current player
    swapped_pieces = np.sum(board == (A_SWAPPED if player_id == PLAYER_A_ID else B_SWAPPED))
    swapped_bonus = swapped_pieces * 2.0  # Bonus for swapped pieces

    # Penalize based on the number of legal moves for the opponent
    opponent_legal_moves_count = len(opponent_moves)
    opponent_moves_penalty = opponent_legal_moves_count * -0.5  # Penalty for opponent's legal moves

    return (row_occupancy_score +  # Add row occupancy score
            start_row_bonus +      # Add bonus for having a piece on start_row
            target_row_bonus +     # Add bonus for having a piece on target_row
            swapped_bonus +        # Add bonus for swapped pieces
            opponent_moves_penalty)  # Subtract penalty for opponent's legal moves

@njit(parallel=True, cache=True)
def _get_legal_moves_batch(boards, player_id):
    """Gets legal moves for multiple boards in parallel."""
    num_boards = boards.shape[0]
    all_moves = []
    
    for i in numba.prange(num_boards):
        moves = _get_legal_moves_jit(boards[i], player_id)
        all_moves.append(moves)
    
    return all_moves

@njit(cache=True)
def _is_move_legal_jit(board, start_r, start_c, end_r, end_c, player_id):
    """Quickly check if a specific move is legal without generating all possible moves.
    
    Args:
        board: 8x4 numpy array representing the board
        start_r, start_c: Starting position coordinates
        end_r, end_c: Ending position coordinates
        player_id: ID of the player making the move
        
    Returns:
        Boolean indicating whether the move is legal
    """
    # Bounds check
    if not (_is_valid(start_r, start_c) and _is_valid(end_r, end_c)):
        return False
    
    # Check if move is to an adjacent cell
    if abs(end_r - start_r) > 1 or abs(end_c - start_c) > 1:
        return False
    
    # Check if start position contains the player's piece
    start_piece = board[start_r, start_c]
    if start_piece == EMPTY_CELL or _get_piece_player_id(start_piece) != player_id:
        return False
    
    # Check target cell
    end_piece = board[end_r, end_c]
    opponent_id = PLAYER_B_ID if player_id == PLAYER_A_ID else PLAYER_A_ID
    
    if end_piece == EMPTY_CELL:
        # Empty cell - always legal
        return True
    elif _get_piece_player_id(end_piece) == opponent_id and _is_piece_normal(end_piece):
        # Can swap with opponent's normal piece
        return True
    
    # All other cases are illegal
    return False

@njit(parallel=True, cache=True)
def _simulate_moves_batch_jit(board, moves, player_id):
    """Simulate multiple moves in parallel and return resulting boards.
    
    Args:
        board: 8x4 numpy array representing the current board state
        moves: List of tuples (start_r, start_c, end_r, end_c) representing moves to simulate
        player_id: ID of the player making the moves
        
    Returns:
        Numpy array of shape (num_moves, 8, 4) containing all resulting board states
    """
    num_moves = len(moves)
    result_boards = np.zeros((num_moves, 8, 4), dtype=np.int8)
    
    for i in numba.prange(num_moves):
        result_boards[i] = board.copy()
        start_r, start_c, end_r, end_c = moves[i]
        _apply_move_jit(result_boards[i], start_r, start_c, end_r, end_c)
    
    return result_boards

@njit(cache=True)
def _board_in_history_jit(current_board, history_boards):
    """Check if current board state exists in history array.
    
    Args:
        current_board: 8x4 numpy array representing current board
        history_boards: Array of shape (history_length, 8, 4) with previous boards
        
    Returns:
        Boolean indicating if the current board matches any in history
    """
    for i in range(len(history_boards)):
        if np.array_equal(current_board, history_boards[i]):
            return True
    return False

@njit(cache=True)
def _scale_reward_jit(reward, episode_step, max_steps=300):
    """Scale reward based on episode progress to encourage faster wins.
    
    Args:
        reward: Base reward value
        episode_step: Current step in the episode
        max_steps: Maximum steps per episode
        
    Returns:
        Scaled reward value
    """
    # Incentivize faster wins/progress
    if reward > 0:
        # Reward earlier wins more
        return reward * (1.0 + (max_steps - episode_step) / max_steps)
    elif reward < 0:
        # Penalize later losses more
        return reward * (1.0 + episode_step / max_steps)
    return reward

@njit(cache=True)
def _hash_board_state_jit(board):
    """Create a fast 64-bit hash of the board state for transposition tables.
    
    Args:
        board: 8x4 numpy array representing the board
        
    Returns:
        64-bit integer hash of the board state
    """
    hash_value = np.uint64(0)
    
    # Use Zobrist hashing approach with pre-defined random values
    # For simplicity, we'll use a simpler method here
    for r in range(8):
        for c in range(4):
            piece = board[r, c]
            if piece != 0:
                # Combine position and piece type into the hash
                position_value = r * 4 + c
                hash_value ^= np.uint64(piece * 37 + position_value * 41)
                hash_value = np.uint64(hash_value * 17 + 23)  # Simple prime number mixing
    
    return hash_value

@njit(cache=True)
def _find_winning_path_jit(board, player_id):
    """Find a winning path using BFS, returns path if found.
    
    Args:
        board: 8x4 numpy array representing the board
        player_id: ID of the player to check for win
        
    Returns:
        List of (row, col) coordinates for the winning path, or empty list if no win
    """
    rows, cols = board.shape
    start_row = rows - 2 if player_id == PLAYER_A_ID else 1
    target_row = 1 if player_id == PLAYER_A_ID else rows - 2
    
    visited = np.zeros((rows, cols), dtype=np.bool_)
    # Parent array to reconstruct path (stores flattened indices)
    parent = np.full((rows, cols), -1, dtype=np.int32)
    
    # Use a list as a queue for Numba compatibility
    queue = []
    
    # Initialize queue with starting pieces
    for c in range(cols):
        piece = board[start_row, c]
        if piece != EMPTY_CELL and _get_piece_player_id(piece) == player_id:
            queue.append((start_row, c))
            visited[start_row, c] = True
            # Self is the parent of start positions
            parent[start_row, c] = start_row * cols + c
    
    # Run BFS
    head = 0
    target_found = False
    target_r, target_c = -1, -1
    
    while head < len(queue) and not target_found:
        r, c = queue[head]
        head += 1
        
        if r == target_row:
            target_found = True
            target_r, target_c = r, c
            break
        
        for i in range(NUM_DIRECTIONS):
            dr, dc = DIRECTIONS[i]
            nr, nc = r + dr, c + dc
            
            if _is_valid(nr, nc, rows, cols) and not visited[nr, nc]:
                neighbor_piece = board[nr, nc]
                if neighbor_piece != EMPTY_CELL and _get_piece_player_id(neighbor_piece) == player_id:
                    visited[nr, nc] = True
                    queue.append((nr, nc))
                    # Store parent position for path reconstruction
                    parent[nr, nc] = r * cols + c
    
    # Return empty list if no path found - FIX: Specify the return type as a list of tuples
    if not target_found:
        return [(0, 0) for _ in range(0)]  # Empty list with explicit tuple type
    
    # Reconstruct the path
    path = []
    current_r, current_c = target_r, target_c
    
    while True:
        path.append((current_r, current_c))
        parent_index = parent[current_r, current_c]
        
        # If we reached a start position (parent points to self)
        if parent_index == current_r * cols + current_c:
            break
            
        current_r = parent_index // cols
        current_c = parent_index % cols
    
    # Reverse path to get it from start to target
    path.reverse()
    return path

@njit(parallel=True, cache=True)
def board_to_binary_batch(boards):
    """Convert batch of boards to binary representation in parallel.
    
    Args:
        boards: Numpy array of shape (num_boards, 8, 4) containing board states
        
    Returns:
        Numpy array of shape (num_boards, 5) with binary board representations
    """
    from binary_board import board_to_binary
    
    num_boards = boards.shape[0]
    binary_boards = np.zeros((num_boards, 5), dtype=np.uint32)
    
    for i in numba.prange(num_boards):
        binary_boards[i] = board_to_binary(boards[i])
    
    return binary_boards

@njit(cache=True)
def _calculate_action_indices_jit(legal_moves, directions):
    """JIT-compiled function to convert moves to action indices."""
    legal_indices = []
    
    for start_r, start_c, end_r, end_c in legal_moves:
        dr, dc = end_r - start_r, end_c - start_c
        
        # Find direction index
        direction_index = -1
        for i in range(len(directions)):
            if directions[i, 0] == dr and directions[i, 1] == dc:
                direction_index = i
                break
                
        if direction_index != -1:
            start_cell_index = start_r * 4 + start_c  # Using COLS=4 directly for JIT
            action_index = start_cell_index * 8 + direction_index  # Using NUM_DIRECTIONS=8 directly
            legal_indices.append(action_index)
    
    return legal_indices

@njit(cache=True)
def _calculate_progress_reward(board, player_id):
    """Calculate a progress-based reward for the given player.
    
    Args:
        board: 8x4 numpy array representing the board.
        player_id: ID of the player (PLAYER_A_ID or PLAYER_B_ID).
    
    Returns:
        A float representing the progress reward.
    """
    rows, cols = board.shape
    progress = 0.0

    # Define progress direction based on player
    if player_id == PLAYER_A_ID:
        for r in range(rows):
            for c in range(cols):
                if _get_piece_player_id(board[r, c]) == PLAYER_A_ID:
                    progress += (rows - r - 1)  # Closer to the opponent's side
    elif player_id == PLAYER_B_ID:
        for r in range(rows):
            for c in range(cols):
                if _get_piece_player_id(board[r, c]) == PLAYER_B_ID:
                    progress += r  # Closer to the opponent's side

    # Normalize progress by the maximum possible value
    max_progress = rows * cols
    return progress / max_progress

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

    def reset(self, initial_state=None):
        """
        Resets the game to the initial state.
        If `initial_state` is provided, it should be a string describing the board.
        """
        self.board.fill(EMPTY_CELL)
        self.current_player_id = PLAYER_A_ID
        self.winner_id = 0
        self.step_count = 0  # Reset step counter
        self._early_game_move_counts = {PLAYER_A_ID: 0, PLAYER_B_ID: 0}
        # Only store initial starting row piece counts, not the whole board
        if initial_state:
            self.board = parse_initial_position_jit(initial_state, A_NORMAL, B_NORMAL, EMPTY_CELL)
        else:
            for r in range(ROWS):
                for c in range(COLS):
                    if (r < 2):
                        self.board[r, c] = B_NORMAL
                    elif (r > 5):
                        self.board[r, c] = A_NORMAL
        self._initial_starting_rows = {
            PLAYER_A_ID: np.sum((self.board[6:8, :] == A_NORMAL) | (self.board[6:8, :] == A_SWAPPED)),
            PLAYER_B_ID: np.sum((self.board[0:2, :] == B_NORMAL) | (self.board[0:2, :] == B_SWAPPED)),
        }
        return self._get_state()

    def _get_state(self):
        """Simplified state representation using binary board encoding."""
        # Convert current board to binary
        current_binary = board_to_binary(self.board)

        # Combine current state and player indicator
        state = np.zeros(5 + 1, dtype=np.float32)  # reduced size: 5 + 1 = 6
        state[:5] = current_binary
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

        # --- EARLY GAME HEURISTIC REWARD ---
        # Track early game move counts
        self._early_game_move_counts[current_player_id] = self._early_game_move_counts.get(current_player_id, 0) + 1
        # Only apply in first 12 moves by this player, and only if initial row had >=4 pieces
        early_game_moves = self._early_game_move_counts[current_player_id]
        if early_game_moves <= 12 and self._initial_starting_rows and self._initial_starting_rows[current_player_id] >= 4:
            # For Player A: rows 6,7; for Player B: rows 0,1
            if current_player_id == PLAYER_A_ID:
                rows = self.board[6:8, :]
                piece_types = (A_NORMAL, A_SWAPPED)
            else:
                rows = self.board[0:2, :]
                piece_types = (B_NORMAL, B_SWAPPED)
            # Count pieces in starting rows after move
            pieces_in_rows = np.sum((rows == piece_types[0]) | (rows == piece_types[1]))
            # Only apply if more than one piece remains in those rows
            if pieces_in_rows > 1:
                # Compare to previous count (before move)
                if not hasattr(self, '_last_starting_row_counts'):
                    self._last_starting_row_counts = {PLAYER_A_ID: self._initial_starting_rows[PLAYER_A_ID], PLAYER_B_ID: self._initial_starting_rows[PLAYER_B_ID]}
                prev_count = self._last_starting_row_counts[current_player_id]
                if pieces_in_rows < prev_count:
                    # Give a bonus for moving a piece out
                    reward += 10.0  # Heuristic bonus (tune as needed)
                self._last_starting_row_counts[current_player_id] = pieces_in_rows

        # Check for win condition for the current player (using JIT with path finding)
        has_won, win_path = self.check_win_condition(ID_PLAYER_MAP[current_player_id])
        if has_won:
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
                # Game continues - Modified reward structure
                reward = 0.0  # Base reward
                
                if move_type == 'swap':
                    reward += 3.0  # Keep swap bonus
                
                # Position improvement reward
                position_delta = _evaluate_board_jit(self.board, current_player_id) - current_score
                reward += position_delta * 3.0
                
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