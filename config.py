"""Configuration settings for the Switcharoo game training."""
import os

# Training Phases Configuration
PHASE1_EPISODES = 30000      # Reduced total episodes for Phase 1
PHASE2_EPISODES = 20000     # Episodes for Phase 2 (self-play)
MAX_STEPS_PER_EPISODE = 200  # Reduced maximum steps per episode 
REPLAY_FREQUENCY = 4         # Less frequent updates for stability

# Tournament Configuration
TOURNAMENT_FREQ = 2000       # How often to run tournaments
NUM_VARIANTS = 4             # Number of agent variants for tournament
NOISE_SCALE = 0.05          # Scale of Gaussian noise to apply to weights
TOURNAMENT_MATCHES = 20      # Matches per pair in tournament
SAVE_FREQ = 2000            # More frequent model saving

# File paths for saving models
BASE_MODEL_FILE = "switcharoo_dqn_curriculum_phase1.weights.h5"
TOURNAMENT_MODEL_FILE = "switcharoo_dqn_tournament_best.weights.h5"
CHECKPOINT_FILE = "switcharoo_dqn_checkpoint_e{}.weights.h5"
BEST_MODEL_FILE = "switcharoo_dqn_best_winrate.weights.h5"  # New file for best win rate model
TFJS_MODEL_DIR = "./switcharoo_tfjs_model"

# Weights & Biases configuration
WANDB_PROJECT = "switcharoo-dqn"
WANDB_ENTITY = "farmerjohn1958-self"

# Initial board positions
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

initial_position = initial_position_base