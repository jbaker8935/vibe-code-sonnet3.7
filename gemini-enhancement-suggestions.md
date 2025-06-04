# Game Rule Enhancement Suggestions

Based on the core rules of your two-player connection game, here are some suggestions to potentially enhance challenge, strategic depth, and replayability.

## 1. Refining the Swap & Mark Mechanic

The current swap/mark system is a good foundation. Here are ways to add more nuance:

### 1.1. Targeted Unmarking / Swap Interactions

*   **Idea:** Allow a "swapped" piece to initiate a swap with an opponent's "swapped" piece.
*   **Mechanic:** If your swapped piece targets an opponent's swapped piece:
    *   Both pieces swap positions.
    *   *Both* pieces become unmarked.
    *   Other swapped pieces on the board remain marked.
*   **Impact:** This introduces a way to selectively unmark key pieces without resetting the entire board. It creates more tactical decisions about which "marked" standoffs to break. It also gives swapped pieces a new offensive/utility role.

### 1.2. Self-Swap for State Change (Your Idea)

*   **Idea:** Allow a player to target one of their *own* adjacent pieces for a "swap."
*   **Mechanic:**
    *   Option A (Positional Swap): The two friendly pieces swap positions. The piece *initiating* the self-swap could become "focused" or "energized" (a new temporary state, see below) or perhaps both become marked as if they were swapped with an opponent (less useful, perhaps).
    *   Option B (State Change, No Position Swap): The targeted friendly piece changes state without moving (e.g., becomes "Phased" or "Anchored" - see Piece States below). The active piece does not move. This costs a turn.
*   **Impact:** Adds another layer of strategic positioning or piece preparation. Option A is more about board control, Option B about specific piece utility.

### 1.3. Diminishing Global Unmark / Localized Unmark

*   **Idea:** Make the global unmark less frequent or more costly.
*   **Mechanic Options:**
    *   **Empty Cell Cooldown:** Moving to an empty cell only unmarks all pieces if, say, at least 2-3 turns have passed since the last global unmark.
    *   **Localized Unmark:** Moving to an empty cell only unmarks pieces within a certain radius (e.g., 1 or 2 cells) of the piece that moved.
    *   **Selective Unmark:** When moving to an empty cell, the player *chooses* to unmark either all *their own* swapped pieces, or all *opponent's* swapped pieces, but not both.
*   **Impact:** Makes the decision to move to an empty cell more tactical and less of an automatic "reset button." Players would have to weigh the benefits of board reset more carefully.

## 2. Piece States & Special Abilities

Beyond "normal" and "marked swapped," new states could emerge.

### 2.1. Promoted Pieces

*   **Idea:** Pieces that reach the opponent's starting rows gain a new ability or state.
*   **Mechanic:** If Player A moves a piece into row 0 or 1, or Player B moves a piece into row 6 or 7, that piece becomes "Promoted."
*   **Promoted Abilities (choose one or combine carefully):**
    *   Can move 2 cells in one orthogonal direction (if the intermediate cell is empty).
    *   Can initiate a swap with an opponent's "marked swapped" piece (unmarking both, as per 1.1, even if that rule isn't generally adopted).
    *   Cannot be targeted by a swap itself (unless by another promoted piece).
    *   Counts as two pieces for connection paths.
*   **Impact:** Creates a new objective beyond just making the connection. Promoted pieces become high-value targets and powerful tools.

### 2.2. "Anchored" State

*   **Idea:** A piece can temporarily become immune to being swapped.
*   **Mechanic:** As a full turn action, a player can "anchor" one of their normal (unmarked) pieces. An anchored piece cannot be the target of an opponent's swap. It remains anchored for X turns (e.g., 2 turns) or until it moves. It can still initiate swaps.
*   **Impact:** Allows for defensive plays, securing key points in a potential connection path, but at the cost of tempo.

## 3. Alternative Win Conditions / Game Modes

### 3.1. Target Cell Connection (Your Idea)

*   **Idea:** Instead of connecting rows, players must connect specific, pre-defined (or randomly chosen at game start) "target cells" on opposite sides of the board.
*   **Mechanic:** For example, Player A must connect cell (7,0) to (0,3) and Player B must connect (0,0) to (7,3).
*   **Impact:** Makes the connection path more specific and potentially more challenging to block/achieve. Could be part of a "challenge mode."

### 3.2. "King of the Hill" / Central Control

*   **Idea:** Control a specific central cell or small group of cells for a certain number of cumulative turns.
*   **Mechanic:** Designate a 2x2 area in the center. The first player to have their pieces occupy 3 out of 4 of these cells for a total of, say, 5 of their own turns, wins. Swapped pieces still count.
*   **Impact:** Shifts focus from pure connection to area control, creating a different strategic dynamic.

### 3.3. "Eradication" (Piece Capture Variant)

*   **Idea:** Win by reducing the opponent to a small number of pieces.
*   **Mechanic:** This would require a way to *remove* pieces, not just swap. Perhaps:
    *   If a piece is swapped twice *without moving to an empty cell or being unmarked by other means*, it's removed from the game. (This would need careful thought about the marking/unmarking rules).
    *   Alternatively, if you surround an opponent's piece (e.g., all 8 adjacent cells are occupied by your pieces or the board edge), it's captured. (This is a common mechanic in other games).
*   **Impact:** Adds a more aggressive, confrontational win condition. Might be a separate game mode.

## 4. Board Variations & Special Cells

### 4.1. Obstacle Cells

*   **Idea:** Some cells are impassable.
*   **Mechanic:** 2-3 cells on the board are marked as "blocked" and cannot be entered or occupied. Their positions could be fixed or randomized per game.
*   **Impact:** Creates chokepoints and forces players to navigate differently, making board layout more significant.

### 4.2. "Sanctuary" Cells

*   **Idea:** Cells where pieces are safe from swaps.
*   **Mechanic:** 1-2 designated cells. If a piece is on a sanctuary cell, it cannot be targeted by an opponent's swap. It can still initiate swaps.
*   **Impact:** Creates safe havens for regrouping or protecting key pieces.

### 4.3. "Flux" Cells

*   **Idea:** Cells that modify the swap/mark rules.
*   **Mechanic:** Moving onto a "Flux" cell:
    *   Option A: Immediately unmarks the piece that moved onto it, even if it was swapped.
    *   Option B: Allows the piece to perform a swap on the *next* turn even if the target is already "marked swapped."
*   **Impact:** Adds specific locations that players might compete for to gain a temporary rule-bending advantage.

## 5. Meta-Game Structure

### 5.1. Series of Challenges (Your Idea)

*   **Idea:** A single-player mode with pre-set scenarios.
*   **Mechanic:**
    *   Challenge 1: Start with fewer pieces but need to connect.
    *   Challenge 2: Opponent starts with a piece already "promoted."
    *   Challenge 3: Board has specific obstacle cells.
    *   Challenge 4: Win by "King of the Hill" on a specific map.
*   **Impact:** Great for teaching mechanics incrementally and providing replayable content for solo players. AI difficulty can scale with challenges.

### 5.2. Asymmetric Starting Positions/Objectives

*   **Idea:** Instead of symmetrical setups and goals, give players slightly different starting piece counts, positions, or even slightly different win conditions for a variant mode.
*   **Example:** Player A needs to connect rows 0-7 as normal. Player B only needs to capture 3 of Player A's pieces (if a capture mechanic is added) OR Player B wins if Player A cannot connect within X turns.
*   **Impact:** Can create very different strategic feels for each side, increasing replayability as players master both roles. This needs careful balancing.

---

When considering these, think about:
*   **Complexity Creep:** How many new rules can be added before the game loses its elegant simplicity?
*   **Balance:** How does each change affect the offensive/defensive balance?
*   **Fun Factor:** Does it make the game more engaging and offer more meaningful decisions?
*   **Potential for Deadlock/Stalemate:** Do any rules make it too easy for the game to get stuck?

I'd recommend playtesting one or two simpler changes first (like 1.1 Targeted Unmarking or 2.1 Promoted Pieces) to see how they affect the game flow before trying more complex overhauls. Good luck with your vibe coding jam!