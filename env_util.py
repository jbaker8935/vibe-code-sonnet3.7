from env_const import (ROWS, COLS, EMPTY_CELL, PLAYER_A_ID, PLAYER_B_ID, A_NORMAL,
                      A_SWAPPED, B_NORMAL, B_SWAPPED, STATE_NORMAL, STATE_SWAPPED,
                      NUM_DIRECTIONS, DIRECTIONS)
import numpy as np
import numba
from numba import njit
from numba.typed import List
from numba import types
LEGAL_MOVE_TYPE = types.UniTuple(types.int8, 4)



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
            piece = board[np.intp(r), np.intp(c)]
            if piece == A_SWAPPED:
                board[np.intp(r), np.intp(c)] = np.int8(A_NORMAL)
            elif piece == B_SWAPPED:
                board[np.intp(r), np.intp(c)] = np.int8(B_NORMAL)
    # No return needed, modifies board in place

@njit(cache=True)
def _unmark_swapped_jit(board):
    """Resets SWAPPED pieces to NORMAL"""
    rows, cols = board.shape
    for r in range(rows):
        for c in range(cols):
            piece = board[np.intp(r), np.intp(c)]
            if (piece == A_SWAPPED):
                board[np.intp(r), np.intp(c)] = np.int8(A_NORMAL)
            elif (piece == B_SWAPPED):
                board[np.intp(r), np.intp(c)] = np.int8(B_NORMAL)

@njit(cache=True)
def _get_legal_moves_jit(board, player_id):
    rows, cols = board.shape
    opponent_id = PLAYER_B_ID if player_id == PLAYER_A_ID else PLAYER_A_ID
    legal_moves = List.empty_list(LEGAL_MOVE_TYPE)
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
                            legal_moves.append((np.int8(r), np.int8(c), np.int8(nr), np.int8(nc)))
                        elif _get_piece_player_id(target_cell) == opponent_id and _is_piece_normal(target_cell):
                            legal_moves.append((np.int8(r), np.int8(c), np.int8(nr), np.int8(nc)))
    return legal_moves

@njit(cache=True)
def _apply_move_jit(board, start_r, start_c, end_r, end_c):
    """Applies the move on the numerical board. Returns move_type_code (1=empty, 2=swap)."""
    moving_piece = board[np.intp(start_r), np.intp(start_c)]
    target_piece = board[np.intp(end_r), np.intp(end_c)]
    move_type_code = 0
    moving_player_id = _get_piece_player_id(moving_piece)

    if target_piece == EMPTY_CELL:
        board[np.intp(end_r), np.intp(end_c)] = np.int8(moving_piece)
        board[np.intp(start_r), np.intp(start_c)] = np.int8(EMPTY_CELL)
        _unmark_swapped_jit(board)
        move_type_code = 1 # 'empty'
    else:
        swapped_moving_piece = np.int8(A_SWAPPED) if moving_player_id == PLAYER_A_ID else np.int8(B_SWAPPED)
        swapped_target_piece = np.int8(A_SWAPPED) if _get_piece_player_id(target_piece) == PLAYER_A_ID else np.int8(B_SWAPPED)
        board[np.intp(end_r), np.intp(end_c)] = swapped_moving_piece
        board[np.intp(start_r), np.intp(start_c)] = swapped_target_piece
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

    num_moves = len(moves)
    result_boards = np.zeros((num_moves, 8, 4), dtype=np.int8)
    
    for i in numba.prange(num_moves):
        result_boards[i] = board.copy()
        start_r, start_c, end_r, end_c = moves[i]
        _apply_move_jit(result_boards[i], start_r, start_c, end_r, end_c)
    
    return result_boards

@njit(cache=True)
def _board_in_history_jit(current_board, history_boards):

    for i in range(len(history_boards)):
        if np.array_equal(current_board, history_boards[i]):
            return True
    return False

@njit(cache=True)
def _scale_reward_jit(reward, episode_step, max_steps=300):

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
        dr = np.int32(end_r - start_r)
        dc = np.int32(end_c - start_c)
        # Find direction index
        direction_index = -1
        for i in range(len(directions)):
            if directions[i, 0] == dr and directions[i, 1] == dc:
                direction_index = np.int32(i)
                break
        if direction_index != -1:
            start_cell_index = np.int32(start_r) * 4 + np.int32(start_c)  # Using COLS=4 directly for JIT
            action_index = start_cell_index * 8 + direction_index  # Using NUM_DIRECTIONS=8 directly
            legal_indices.append(np.int32(action_index))
    # Return as np.array with explicit dtype to avoid unsafe cast warnings
    return np.array(legal_indices, dtype=np.int32)

@njit(cache=True)
def _calculate_progress_reward(board, player_id):

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


