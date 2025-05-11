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
NUM_SIMULATIONS_PER_MOVE = 25  # Number of MCTS simulations to run for each move (Reduced from 100)
C_PUCT_CONSTANT = 1.0           # Exploration constant in PUCT formula
TEMPERATURE_START = 1.0         # Initial temperature for move selection in self-play (higher for exploration)
TEMPERATURE_END = 0.1           # Final temperature for move selection (lower for exploitation)
TEMPERATURE_ANNEAL_STEPS = 1000 # Number of steps over which to anneal temperature (Reduced from 500000)
DIRICHLET_ALPHA = 0.3           # Alpha parameter for Dirichlet noise added to root priors
DIRICHLET_EPSILON = 0.25        # Epsilon for Dirichlet noise (fraction of noise to apply)

# AlphaZero Training Loop Parameters
AZ_ITERATIONS = 20                  # Total number of training iterations (Reduced from 1000)
AZ_GAMES_PER_ITERATION = 20         # Number of self-play games to generate per iteration (Reduced from 50)
AZ_TRAINING_STEPS_PER_ITERATION = 20 # Number of training steps (batches) per iteration (Reduced from 100)
AZ_REPLAY_BUFFER_SIZE = 20000       # Maximum number of game states to store in the replay buffer
AZ_BATCH_SIZE = 64                  # Batch size for training the neural network
AZ_EVALUATION_GAMES_COUNT = 4      # Number of games to play for evaluating a new model (Reduced from 20)
AZ_MODEL_UPDATE_WIN_RATE = 0.55     # Minimum win rate for a new model to replace the current best

# Neural Network Architecture (AlphaZero)
AZ_NN_INPUT_DEPTH = 6 # Based on current binary board (5) + player (1)
AZ_NN_RESIDUAL_BLOCKS = 5 # Number of residual blocks in the NN body
AZ_NN_FILTERS = 64        # Number of filters in convolutional layers / units in dense layers
AZ_NN_POLICY_HEAD_UNITS = 128 # Units in the dense layer before policy output
AZ_NN_VALUE_HEAD_UNITS = 64   # Units in the dense layer before value output
AZ_LEARNING_RATE = 0.001
AZ_L2_REGULARIZATION = 0.0001
AZ_VALUE_LOSS_WEIGHT = 1.0
AZ_POLICY_LOSS_WEIGHT = 1.0

# File paths for AlphaZero models
AZ_BEST_MODEL_FILE = "switcharoo_az_best.weights.h5"
AZ_CANDIDATE_MODEL_FILE = "switcharoo_az_candidate.weights.h5"
AZ_CHECKPOINT_FILE_PATTERN = "switcharoo_az_checkpoint_iter{}.weights.h5"