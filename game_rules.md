you are a game developer that is prototyping a two player game for a vibe coding jam. the game rules are as follows:
1. the game board is 8 rows and 4 columns.
2. player A starts with 8 pieces in the cells in rows 1 and 2
3. player B starts with 8 pieces in the cells in rows 7 and 8
4. players alternate moves with player A moving first.
5. a player may move a piece to any adjacent cell in all 8 directions: orthogonally or diagonally.
6. a legal move can be made to an adjacent empty cell Or to an adjacent cell occupied by the opponent's normal piece.
7. if a target cell is occupied by an opponent's piece then a player's piece may swap positions with the adjacent opponent's piece.  both pieces are marked to indicate they have been swapped.
8. while a piece is marked swapped it cannot be the target of a swap moves.  however, a swapped piece can swap positions with an opponent's adjacent normal piece.
9. when either player moves a piece to an empty cell, then all swapped pieces are unmarked and can be a target of a swap on a subsequent move.
10. the game ends when either player has created a connected path of pieces between rows 2 and 7. 

design requirements:

1. create a prototype user interface for the game so that a user can select pieces for player A and move them subject to the rules of the game.  Use html, js and css.
2. write a heuristic algorithm to automatically execute player B's moves subject to the rules of the game with the goal of reaching a winning position.  encourage piece advancement, creating connected paths and blocking the human player's connected paths.  the AI should try to maximize the number occupied rows 1 through 7 on the board, so that a winning connection can be made
3. the game should be playable on mobile web 
4. Display the game board as a checkerboard with alternating white and buff colors. The  gameboard container has a black outline.  each cell within the game board should have a fine black outline.  Orient the board so Player A is at the bottom.
5. Pieces should look like checker pieces having a normal side and a 'swapped' side that has an inset crown graphic. player A normal piece is images/white_normal.png, player A swapped piece is images/white_swapped.png,  player B normal piece is images/black_normal.png, player B swapped piece is images/black_swapped.png.  Scale the piece images so that they fit inside a game board cell.
6. Under the game board container add a text display area for messages like current player and win status.
7. Under the text area add a reset button to allow the game to be returned to the initialized state.
8. The human player can select a piece using a mouse.  When selected, the cells with legal moves should be highlighted.  When a target cell is selected, apply the move.   If a selected piece is clicked again, it becomes deselected and another piece may be selected by the player.
9. When the game ends, highlight the cells in the winning path.
10. as the game progresses store a list of game action history showing: the move number, the player, the starting and ending cell.  when displayed, moves should be shown in descending order.
11. create a game action history container to display the move history.
12. If the display is wide enough display the game action history container to the right of the game board.
13. if the display is not wide enough provide a button to overlay the game history container on the display so that the user can select rows from the game history for display.  if the game history is overlaying the game board hide the game history so the selected game state can be shown.
14. once the game has ended, allow the user to select a row in the game action history and display the game board state at that point in the game.   the highlighted winning path should only be shown for the final move in the game.