"""Configuration settings for the Switcharoo game training."""
import os

# Training Phases Configuration
PHASE1_EPISODES = 30000      # Keep same episode count for Phase 1
PHASE2_EPISODES = 25000      # Increase Phase 2 episodes for better learning
MAX_STEPS_PER_EPISODE = 200  # Increased to allow more decisive play
REPLAY_FREQUENCY = 4         # Keep same replay frequency

# Tournament Configuration
TOURNAMENT_FREQ = 1000       # More frequent tournaments
NUM_VARIANTS = 6             # Increase number of agent variants for diversity
NOISE_SCALE = 0.08           # Increased noise for more exploration
TOURNAMENT_MATCHES = 30      # More matches for better evaluation
SAVE_FREQ = 2000             # Keep same save frequency

# File paths for saving models
BASE_MODEL_FILE = "switcharoo_dqn_curriculum_phase1.weights.h5"
TOURNAMENT_MODEL_FILE = "switcharoo_dqn_tournament_best.weights.h5"
CHECKPOINT_FILE = "switcharoo_dqn_checkpoint_e{}.weights.h5"
BEST_MODEL_FILE = "switcharoo_dqn_best_winrate.weights.h5"  # For best win rate model
TFJS_MODEL_DIR = "./switcharoo_tfjs_model"

# Rewards configuration - NEW!
WIN_REWARD = 350.0           # Increased win reward
DRAW_PENALTY = -250.0        # Stronger penalty for draws
PROGRESS_REWARD = 0.75       # Small reward for making progress
MOVE_TIMEOUT = 3             # Maximum repeats of similar board positions (reduced)
STEP_PENALTY = -0.4          # Penalty for each step taken (increased)

# Weights & Biases configuration
WANDB_PROJECT = "switcharoo-dqn"
WANDB_ENTITY = "farmerjohn1958-self"

# Keep existing initial positions but add more aggressive starting positions
initial_position_base = [
    """\
BBBB
BBBB
....
....
....
....
AAAA
AAAA""",
    """\
....
....
BBBB
BBBB
AAAA
AAAA
....
....""",
    """\
BB..
BB..
BB..
BB..
..AA
..AA
..AA
..AA""",
    """\
..BB
..BB
BBBB
....
....
AAAA
AA..
AA..""",
    """\
B...
BB..
BB..
BBB.
.AAA
..AA
..AA
...A""",
    """\
....
....
BABA
ABAB
BABA
ABAB
....
....""",
    """\
B..B
.BB.
.BB.
B..B
A..A
.AA.
.AA.
A..A""",
    """\
B..B
.AA.
.AA.
B..B
A..A
.BB.
.BB.
A..A"""
]

# Add new positions that encourage more aggressive, direct play
initial_position_aggressive = [
    """\
B.B.
B.B.
....
....
....
A.A.
A.A.
....""",
    """\
BB..
....
....
....
....
....
....
AA..""",  
    """\
....
.BB.
.BB.
....
....
.AA.
.AA.
....""",
    """\
B...
.B..
..B.
...B
A...
.A..
..A.
...A"""
]

# Use a mix of standard and aggressive positions with more weight on aggressive ones
initial_position = initial_position_base + initial_position_aggressive * 2