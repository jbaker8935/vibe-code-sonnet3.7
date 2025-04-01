## Vibe Coded Game
- Draft game coded with Sonnet 3.7.  Code has been replaced in fully by a Gemini 2.5 pro version.
- Gemini 2.5 pro built the framework using the game_rules.md.
- Framework was generally working out of the gate.
- Copilot and Sonnet 3.5 was used to refine the GUI, update the AI and implement responsive design.
- Nearly 100% coded by prompts.
- Player B AI logic:
  - Includes board eval heuristics to encourage movement across the board, creating chains of pieces, and restricting opponent play.
  - Heuristic weights have not been tuned beyond the initial guess for values. 
  - Easy mode has a depth 1 look-ahead to avoid making moves that will lead to a winning move by the opponent.
  - Hard mode level 1 has a depth 3 look-ahead which avoids setting up a forcing move by the opponent.
  - Hard mode level 2 has a depth 5 look-ahead. *disabled* 

