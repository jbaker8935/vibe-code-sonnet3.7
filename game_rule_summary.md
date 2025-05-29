you are a game developer that is prototyping a two player game for a vibe coding jam. the game rules are as follows:
1. the game board is 8 rows and 4 columns.
2. player A starts with 8 pieces in the cells in the 2 rows at the bottom of the game board
3. player B starts with 8 pieces in the cells in the 2 rows at the top of the game board
4. players alternate moves with player A moving first.
5. a player may move a piece to any adjacent cell in all 8 directions: orthogonally or diagonally.
6. a legal move can be made to an adjacent empty cell Or to an adjacent cell occupied by the opponent's normal piece.
7. if a target cell is occupied by an opponent's piece then a player's piece may swap positions with the adjacent opponent's piece.  both pieces are marked to indicate they have been swapped.
8. while a piece is marked swapped it cannot be the target of a swap moves.  however, a swapped piece can swap positions with an opponent's adjacent normal piece.
9. when either player moves a piece to an empty cell, then all swapped pieces are unmarked and can be a target of a swap on a subsequent move.
10. the game ends when either player has created a connected path of pieces between rows 1 and 6 (note: 0 based row numbering). 