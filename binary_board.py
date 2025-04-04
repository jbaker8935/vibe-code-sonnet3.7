import numpy as np
from numba import njit

# Board state representation using 5 32-bit integers:
# - Empty cells mask
# - Player A normal pieces mask
# - Player A swapped pieces mask 
# - Player B normal pieces mask
# - Player B swapped pieces mask

@njit(cache=True)
def board_to_binary(board):
    """Convert 8x4 board array to 5 32-bit binary masks"""
    binary_board = np.zeros(5, dtype=np.uint32)
    
    # Each position gets a bit in the 32-bit masks
    for r in range(8):
        for c in range(4):
            pos = r * 4 + c
            piece = board[r, c]
            
            if piece == 0:  # Empty
                binary_board[0] |= (1 << pos)
            elif piece == 1:  # Player A normal
                binary_board[1] |= (1 << pos)
            elif piece == 2:  # Player A swapped
                binary_board[2] |= (1 << pos)
            elif piece == 3:  # Player B normal
                binary_board[3] |= (1 << pos)
            elif piece == 4:  # Player B swapped
                binary_board[4] |= (1 << pos)
                
    return binary_board

@njit(cache=True)
def binary_to_board(binary_board):
    """Convert binary masks back to 8x4 board array"""
    board = np.zeros((8, 4), dtype=np.int8)
    
    for r in range(8):
        for c in range(4):
            pos = r * 4 + c
            
            if binary_board[0] & (1 << pos):  # Empty
                board[r, c] = 0
            elif binary_board[1] & (1 << pos):  # Player A normal
                board[r, c] = 1
            elif binary_board[2] & (1 << pos):  # Player A swapped
                board[r, c] = 2
            elif binary_board[3] & (1 << pos):  # Player B normal
                board[r, c] = 3
            elif binary_board[4] & (1 << pos):  # Player B swapped
                board[r, c] = 4
                
    return board