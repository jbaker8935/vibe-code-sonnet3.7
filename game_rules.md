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

1. create a prototype user interface for the game so that a user can select pieces for player A and move them subject to the rules of the game.
- prototype is an in-browser game which does not require a server to run
- use any preferred framework for code or themes
2. write a heuristic algorithm to automatically execute player B's moves subject to the rules of the game with the goal of reaching a winning position.
- avoid moves that allow the human player to make a winning move on their next turn.
- encourage piece advancement
- discourage having 4 player pieces in a row unless required for other tactics
- encourage moving pieces off of the back row
- position pieces to allow creation of connected paths
- use swapping moves to limit the move options of the opponent
3. the game must have responsive design and be playable on mobile web 
4. Display the game board as a checkerboard with alternating colors. Orient the game board so Player A is at the bottom.
5. Pieces should look like checker pieces having a normal side and a 'swapped' side that has an inset crown graphic. 
6. The following images will be used for the pieces:
- player A normal piece is images/white_normal.png
- player A swapped piece is images/white_swapped.png
- player B normal piece is images/black_normal.png
- player B swapped piece is images/black_swapped.png.
- Scale the piece images so that they fit inside a game board cell.
7. Display a control container with:
- a reset button that initializes the game state.
- reset button will display the icon in images/refresh-circle-outline.svg
- an info button that displays an overlay with a description of the game rules.
- info button will display the icon in images/information-circle-outline.svg
- a history button that displays an overlay listing the move history with the most recent move at the top.  each row shows: the player, the start cell, the end cell
- the history button is disabled by default.   the history button is enabled when the game ends. 
- history button will display the icon in images/list-circle-outline.svg
- when the history overlay is displayed and a user selects a row in the history table, then the overlay is closed and the game board displays the game state after that selected move.
- the info and history overlays should have a close button.   the overlays can also be dismissed if a user clicks outside the overlay boundry.
- close button will display the icon in images/close-circle-outline.svg
- in mobile portrait mode display the control container under the game board with buttons displayed horizontally
- in mobile landscape or wide screen displays display the control container to the right of the game board, with buttons displayed vertically.
8. The human player can select a piece using a mouse or touch on mobile.  When selected, the cells with legal moves should be highlighted.  When a target cell is selected, apply the move.   If a selected piece is clicked again, it becomes deselected and another piece may be selected by the player.
9. When the game ends:
- display an overlay with text describing the winning player.  the overlay should auto close after 5 seconds Or can be dismissed by clicking outside the overlay boundry
- highlight the cells in the winning path from rows 1 to 7.
10. Use a modern polished GUI theme

