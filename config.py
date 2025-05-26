"""Configuration settings for the Switcharoo game training."""
import os

# Training Phases Configuration
PHASE1_EPISODES = 50000      # Episodes for Phase 1 (random opponent)
PHASE2_EPISODES = 50000     # Episodes for Phase 2 (self-play)
MAX_STEPS_PER_EPISODE = 250  # Maximum steps per episode (Reduced from 1000)
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
....
....
BBBB
BBBB
AAAA
AAAA
....
...."""
]

initial_position = initial_position_base

# --- AlphaZero MCTS Configuration ---
# MCTS Parameters - BALANCED FOR LONG TRAINING
NUM_SIMULATIONS_PER_MOVE = 200   # Increased from 40. Slightly reduced for speed over 12 hours
C_PUCT_CONSTANT = 0.25           # Balanced exploration/exploitation for long runs

# Temperature schedule - PROGRESSIVE LEARNING FOR 12-HOUR TRAINING
TEMPERATURE_START = 1.0         # Higher initial temp for exploration in long training
TEMPERATURE_END = 0.05          # Very low final temp for decisive moves
TEMPERATURE_ANNEAL_STEPS = 25   # Longer annealing for gradual improvement

DIRICHLET_ALPHA = 0.5          # Higher alpha for more uniform exploration
DIRICHLET_EPSILON = 0.1        # Lower noise for more consistent play

# AlphaZero Training Loop Parameters - OPTIMIZED FOR 12-HOUR TRAINING
AZ_ITERATIONS = 3 # Reduced for gradient checking and overfit test
AZ_GAMES_PER_ITERATION = 15    # More games for better data quality over longer run
AZ_TRAINING_STEPS_PER_ITERATION = 150   # Balanced training steps for sustained learning
AZ_REPLAY_BUFFER_SIZE = 2000   # Larger buffer for extended training
AZ_BATCH_SIZE = 64             # Larger batches for efficiency in long training
AZ_EVALUATION_GAMES_COUNT = 6   # Slightly reduced for speed but sufficient confidence
AZ_MODEL_UPDATE_WIN_RATE = 0.55 # Slightly lower threshold for more frequent updates

# Neural Network Architecture - OPTIMIZED FOR EXTENDED LEARNING
AZ_NN_INPUT_DEPTH = 192  # Number of features in the input layer (e.g., 6 channels for 8x4 board = 6 * 32 = 192)
AZ_NN_RESIDUAL_BLOCKS = 1      # Simplified: Was 4
AZ_NN_FILTERS = 32             # Simplified: Was 96
AZ_NN_POLICY_HEAD_UNITS = 64  # Simplified: Was 192
AZ_NN_VALUE_HEAD_UNITS = 32    # Simplified: Was 96
AZ_LEARNING_RATE = 3e-4        # Step 1.1: Fixed higher LR. Was 1e-4
AZ_L2_REGULARIZATION = 0.0    # Step 1.1: Disable L2. Was 1e-5
AZ_VALUE_LOSS_WEIGHT = 1.0     # Step 1.1: Baseline. Was 1.0
AZ_POLICY_LOSS_WEIGHT = 1.0    # Step 1.1: Baseline. Was 1.5
AZ_POLICY_HEAD_TYPE = "default"  # Added: Specifies the type of policy head
AZ_VALUE_HEAD_TYPE = "default"   # Added: Specifies the type of value head

# Advanced Training Parameters for Emergent Strategy Development
# AZ_CURRICULUM_LEARNING = True      # Enable progressive difficulty (UNUSED)
# AZ_DYNAMIC_TEMPERATURE = True      # Adaptive temperature based on position complexity (RELATED TO CURRICULUM)
AZ_STRATEGIC_DIVERSITY_BONUS = 0.1 # Bonus for exploring novel move sequences
AZ_LONG_GAME_EMPHASIS = 1.2        # Weight factor for learning from longer games
AZ_MODEL_ENSEMBLE_SIZE = 3          # Number of recent models to ensemble for evaluation

# Learning Rate Schedule - OPTIMIZED FOR 12-HOUR SUSTAINED LEARNING
AZ_LR_DECAY_SCHEDULE = {
    'type': 'cosine',               # Cosine annealing for smooth convergence
    'initial_lr': 3e-4,             # Step 1.1: Align with AZ_LEARNING_RATE. Was 1e-4
    'min_lr': 1e-5,                 # Lower minimum for fine-tuning at the end
    'decay_steps': 3000             # Longer decay cycles for 12-hour training
}

# Temperature Schedule Enhancements for 12-Hour Training
# AZ_ADAPTIVE_TEMP_THRESHOLD = 300    # Reduced threshold for faster adaptation (was 500) (RELATED TO DYNAMIC TEMP)
# AZ_COMPLEX_POSITION_TEMP_BOOST = 1.3  # Moderate boost for efficiency (was 1.5) (RELATED TO DYNAMIC TEMP)

# File paths for AlphaZero models
AZ_BEST_MODEL_FILE = "switcharoo_az_best.weights.h5"
AZ_CANDIDATE_MODEL_FILE = "switcharoo_az_candidate.weights.h5"
AZ_CHECKPOINT_FILE_PATTERN = "switcharoo_az_checkpoint_iter{}.weights.h5"

# MCTS Implementation Configuration
USE_NUMBA = True  # Set to True to use Numba MCTS, False for Python MCTS

# Training Safety Parameters - OPTIMIZED FOR 12-HOUR SUSTAINED TRAINING
AZ_MAX_MOVE_TIME = 2.0           # Longer move time for quality decisions in extended training
AZ_MEMORY_LIMIT_GB = 6.0         # Higher memory limit for longer training sessions
AZ_CHECKPOINT_FREQUENCY = 5      # More frequent checkpoints for safety
AZ_EVALUATION_TIMEOUT = 120      # Longer eval timeout for thorough assessment
AZ_GRADIENT_CLIP_NORM = 1.0      # Standard clipping for stability
AZ_INFERENCE_BATCH_SIZE = 64     # Larger batch size for efficiency
AZ_EARLY_STOP_SLOW_EVAL = False  # Allow slower games for quality learning
AZ_MAX_GAME_LENGTH = 150         # Moderate game length for comprehensive learning

# TRAINING MODE FLAGS - CONTINUATION MODE
AZ_RECOVERY_MODE = False         # Normal training mode
AZ_RESET_OPTIMIZER = True       # Keep optimizer state for momentum
AZ_CLEAR_REPLAY_BUFFER = False   # Preserve learned experiences
AZ_DISABLE_ATTENTION = True      # Keep attention disabled for efficiency
AZ_OVERFIT_TEST_REQUIRED = True  # Continue monitoring learning capability
AZ_LOG_ACTIVATION_STATS = True   # Log activation statistics

# --- Static Dataset Training (Step 1.3) ---
AZ_GENERATE_STATIC_DATASET = False # If True, will run self-play for one iteration and save experiences
AZ_TRAIN_ON_STATIC_DATASET = False # If True, will load and train on the static dataset
AZ_STATIC_DATASET_PATH = "static_dataset.pkl"
AZ_STATIC_TRAINING_EPOCHS = 100 # Number of epochs to train on the static dataset
AZ_STATIC_DATASET_GAMES = 50 # Number of games to generate for the static dataset (if AZ_GENERATE_STATIC_DATASET is True)


# LEARNING BREAKTHROUGH PARAMETERS - ADDRESS SPECIFIC ISSUES
AZ_FORCE_SHORTER_GAMES = True    # Implement game length penalties
AZ_IMPROVED_VALUE_TARGETS = True # Use better value target computation
AZ_POLICY_SMOOTHING = 0.05        # Add policy target smoothing
AZ_VALUE_CLIPPING = True         # Clip value predictions to [-1, 1]
AZ_GRADIENT_ACCUMULATION = 2     # Accumulate gradients over 2 steps
AZ_WARMUP_ITERATIONS = 5         # Use learning rate warmup for first 5 iterations

# GAME LENGTH CONTROL - BALANCED FOR 12-HOUR TRAINING
AZ_GAME_LENGTH_PENALTY = 0.005   # Lighter penalty to allow strategic development
AZ_DRAW_PENALTY = -0.05          # Reduced penalty for natural game flow
AZ_MOVE_TIME_PRESSURE = False    # Disable time pressure for quality learning
AZ_TERMINAL_STATE_BONUS = 0.1    # Moderate bonus for completion
AZ_STALEMATE_DETECTION = True    # Keep stalemate detection active

# DEBUGGING AND MONITORING ENHANCEMENTS FOR 12-HOUR TRAINING
AZ_DETAILED_LOGGING = True       # Enable detailed loss component logging
AZ_TRACK_PREDICTION_QUALITY = True  # Monitor prediction improvement
AZ_EARLY_CONVERGENCE_CHECK = False   # Disable for full 12-hour exploration
AZ_MIN_IMPROVEMENT_THRESHOLD = 0.01 # Lower threshold for continued training

# 12-HOUR TRAINING OPTIMIZATION PARAMETERS
AZ_TARGET_TRAINING_HOURS = 12.0    # Target training duration in hours
AZ_ITERATION_TIME_BUDGET = 600     # Target 10 minutes per iteration for 12-hour run
AZ_AGGRESSIVE_EARLY_STOPPING = False  # Disable for longer comprehensive training
AZ_FAST_EVALUATION_MODE = False    # Use thorough evaluation for quality assurance
AZ_PRIORITIZE_SPEED = False        # Prioritize quality over speed for 12-hour training
AZ_AUTO_ADJUST_PARAMETERS = True   # Keep parameter adjustment based on timing

# Performance monitoring for 12-hour constraint
AZ_LOG_TIMING_EVERY_N_ITERATIONS = 1  # Log timing every iteration
AZ_TIME_REMAINING_THRESHOLD = 0.95 # Stop training when 95% of time is used
AZ_ADAPTIVE_BATCH_SIZE = True      # Allow batch size adjustment based on memory/time
# AZ_PROGRESSIVE_COMPLEXITY = True   # Gradually increase model complexity over time (RELATED TO CURRICULUM)

# ADVANCED 12-HOUR TRAINING OPTIMIZATIONS
AZ_MEMORY_EFFICIENT_TRAINING = True    # Enable memory optimization techniques
AZ_DYNAMIC_EVALUATION_FREQUENCY = True # Adjust evaluation frequency based on progress
AZ_SAVE_BEST_N_MODELS = 5             # Keep top 5 models during training
# AZ_CURRICULUM_DIFFICULTY_SCALING = True # Scale difficulty over 12-hour period (RELATED TO CURRICULUM)
AZ_LEARNING_RATE_SCHEDULING = True     # Enable sophisticated LR scheduling

# Time-based parameter adjustments for 12-hour runs
# AZ_PHASE_1_HOURS = 4.0            # First 4 hours: exploration and foundation
# AZ_PHASE_2_HOURS = 6.0            # Next 6 hours: intensive learning
# AZ_PHASE_3_HOURS = 2.0            # Final 2 hours: refinement and convergence

# Phase-specific parameter overrides
# AZ_PHASE_1_SIMULATIONS = 35       # Fewer simulations for faster exploration
# AZ_PHASE_2_SIMULATIONS = 45       # Standard simulations for quality learning
# AZ_PHASE_3_SIMULATIONS = 55       # More simulations for refined play

# AZ_PHASE_1_TEMPERATURE = 1.2      # Higher temperature for exploration
# AZ_PHASE_2_TEMPERATURE = 0.8      # Moderate temperature for learning
# AZ_PHASE_3_TEMPERATURE = 0.3      # Lower temperature for refinement