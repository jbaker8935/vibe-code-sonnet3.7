import numpy as np
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

# Numba-compatible version of DIRECTIONS
DIRECTIONS_TUPLE = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))

NUM_CELLS = ROWS * COLS
NUM_ACTIONS = NUM_CELLS * NUM_DIRECTIONS # 32 * 8 = 256

# Constants for AlphaZero
NUM_PIECES = 8  # Each player has 2 rows * 4 cols = 8 pieces
MAX_MOVES_PER_PIECE = NUM_DIRECTIONS # Each piece can move in 8 directions