"""Configuration settings for the Switcharoo game training."""
import os

# Training Phases Configuration
PHASE1_EPISODES = 50000      # Episodes for Phase 1 (random opponent)
PHASE2_EPISODES = 50000     # Episodes for Phase 2 (self-play)
MAX_STEPS_PER_EPISODE = 1000  # Maximum steps per episode
REPLAY_FREQUENCY = 2         # Frequency of replay buffer sampling

# Learning rate decay and opponent epsilon cap configuration
EPISODE_THRESHOLD = 28000  # Episode threshold for learning rate decay and opponent epsilon cap
NEW_LR = 5e-7  # New learning rate after threshold
OPPONENT_EPSILON_CAP = 0.4  # Cap for opponent epsilon after threshold

# Tournament Configuration
TOURNAMENT_FREQ = 2000       # How often to run tournaments
NUM_VARIANTS = 4             # Number of agent variants for tournament
NOISE_SCALE = 0.05          # Scale of Gaussian noise to apply to weights
TOURNAMENT_MATCHES = 100      # Matches per pair in tournament
SAVE_FREQ = 5000            # Save model weights every N episodes

# File paths for saving models
BASE_MODEL_FILE = "switcharoo_dqn_curriculum_phase1.weights.h5"
TOURNAMENT_MODEL_FILE = "switcharoo_dqn_tournament_best.weights.h5"
CHECKPOINT_FILE = "switcharoo_dqn_checkpoint_e{}.weights.h5"
TFJS_MODEL_DIR = "./switcharoo_tfjs_model"

# Weights & Biases configuration
WANDB_PROJECT = "switcharoo"
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

# --- AlphaZero MCTS Configuration ---
# MCTS Parameters
NUM_SIMULATIONS_PER_MOVE = 300  # Increased from 200 for deeper search
C_PUCT_CONSTANT = 1.5           # Balanced value for exploration vs exploitation
TEMPERATURE_START = 1.1         # Moderately high for adequate exploration
TEMPERATURE_END = 0.05          # Low for deterministic play in later stages
TEMPERATURE_ANNEAL_STEPS = 200000 # Increased from 100000 for slower annealing with longer training
DIRICHLET_ALPHA = 0.3           # Moderate noise at root node for diversity
DIRICHLET_EPSILON = 0.25        # Moderate noise application for balance

# AlphaZero Training Loop Parameters
AZ_ITERATIONS = 500                # Increased from 180 to 500 for 12-hour training
AZ_GAMES_PER_ITERATION = 80        # Increased from 60 to 80 for more experience collection
AZ_TRAINING_STEPS_PER_ITERATION = 1500 # Increased from 800 to 1500 for deeper learning with smaller batch
AZ_REPLAY_BUFFER_SIZE = 500000     # Doubled from 250000 to maintain more diverse experiences
AZ_BATCH_SIZE = 384                # Reduced from 512 to 384 to avoid register spilling
AZ_EVALUATION_GAMES_COUNT = 60     # Increased from 40 to 60 for more statistically significant evaluation
AZ_MODEL_UPDATE_WIN_RATE = 0.52    # Slightly increased from 0.48 for more selective model updates

# Neural Network Architecture (AlphaZero)
AZ_NN_INPUT_DEPTH = 6 # Based on current binary board (5) + player (1)
AZ_NN_RESIDUAL_BLOCKS = 9          # Increased from 7 to 9 for better pattern recognition
AZ_NN_FILTERS = 128                # Increased from 96 to 128 for more feature extraction capacity
AZ_NN_POLICY_HEAD_UNITS = 256      # Increased from 192 to 256 for better policy representation
AZ_NN_VALUE_HEAD_UNITS = 128       # Increased from 96 to 128 for better value estimation
AZ_LEARNING_RATE = 0.0005          # Slightly decreased from 0.0006 for more stable learning
AZ_L2_REGULARIZATION = 0.001       # Slightly increased from 0.0008 for better generalization
AZ_VALUE_LOSS_WEIGHT = 1.0         # Kept balanced weighting
AZ_POLICY_LOSS_WEIGHT = 2.5        # Maintained emphasis on policy learning

# File paths for AlphaZero models
AZ_BEST_MODEL_FILE = "switcharoo_az_best.weights.h5"
AZ_CANDIDATE_MODEL_FILE = "switcharoo_az_candidate.weights.h5"
AZ_CHECKPOINT_FILE_PATTERN = "switcharoo_az_checkpoint_iter{}.weights.h5"

# MCTS Implementation Configuration
USE_NUMBA = True  # Set to True to use Numba MCTS, False for Python MCTS