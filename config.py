"""Configuration settings for the Switcharoo game training."""
import os

# Training Phases Configuration
PHASE1_EPISODES = 50000      # Episodes for Phase 1 (random opponent)
PHASE2_EPISODES = 50000     # Episodes for Phase 2 (self-play)
MAX_STEPS_PER_EPISODE = 175  # Slight increase from 150 for strategic depth in extended training
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

# Initial board positions - PROGRESSIVE CURRICULUM LEARNING
# Define named positions for curriculum learning
POSITION_ADVANCED_CENTER = """\
....
....
BBBB
BBBB
AAAA
AAAA
....
...."""

POSITION_STANDARD = """\
BBBB
BBBB
....
....
....
....
AAAA
AAAA"""

POSITION_INTERMEDIATE_1 = """\
....
BBBB
BBBB
....
....
AAAA
AAAA
...."""

POSITION_INTERMEDIATE_2 = """\
BBBB
....
BBBB
....
....
AAAA
....
AAAA"""

# All available positions for reference
initial_position_base = [
    POSITION_ADVANCED_CENTER,
    POSITION_STANDARD,
    POSITION_INTERMEDIATE_1,
    POSITION_INTERMEDIATE_2
]

# Default position set (will be overridden by curriculum)
initial_position = initial_position_base

# --- Progressive Curriculum Configuration ---
AZ_PROGRESSIVE_CURRICULUM = True

# Curriculum schedule definition
AZ_CURRICULUM_SCHEDULE = {
    # Phase 1: Extended foundation mastery (was 1-40, now 1-80)
    'phase_1': {
        'iterations': (1, 80),           # AGGRESSIVE: Further extended for better foundation (was 60)
        'positions': [POSITION_ADVANCED_CENTER],
        'target_policy_accuracy': 0.70,  # AGGRESSIVE: Higher target (was 0.65)
        'description': 'Extended foundation mastery'
    },
    # Phase 2: Gradual dual position learning (was 41-80, now 81-140) 
    'phase_2': {
        'iterations': (81, 140),         # FIXED: Sequential start after phase 1 ends
        'positions': [POSITION_ADVANCED_CENTER, POSITION_INTERMEDIATE_1],
        'target_policy_accuracy': 0.55,  # ADJUSTED: Reduced from 0.60 for realistic dual-position target
        'description': 'Gradual dual position learning'
    },
    # Phase 3: Controlled three-position mastery (was 81-140, now 141-200)
    'phase_3': {
        'iterations': (141, 260),        # EXTENDED: More time for three-position mastery
        'positions': [POSITION_ADVANCED_CENTER, POSITION_STANDARD, 
                     POSITION_INTERMEDIATE_1],  # IMPROVED: Only 3 positions
        'target_policy_accuracy': 0.50,  # REALISTIC: Lowered target for complex learning
        'description': 'Extended three-position mastery with stability focus'
    },
    # Phase 4: Full complexity with realistic targets (was 141-180, now 261-320)
    'phase_4': {
        'iterations': (261, 320),        # ADJUSTED: Sequential start after phase 3 ends
        'positions': [POSITION_ADVANCED_CENTER, POSITION_STANDARD, 
                     POSITION_INTERMEDIATE_1, POSITION_INTERMEDIATE_2],
        'target_policy_accuracy': 0.48,  # REALISTIC: Lowered for four-position complexity
        'description': 'Full strategic mastery with stability focus'
    },
    # Phase 5: Ultra-mastery and fine-tuning (321-400)
    'phase_5': {
        'iterations': (321, 400),        # ADVANCED: Building on Phase 4's ELO 1774 success
        'positions': [POSITION_ADVANCED_CENTER, POSITION_STANDARD, 
                     POSITION_INTERMEDIATE_1, POSITION_INTERMEDIATE_2],
        'target_policy_accuracy': 0.52,  # AMBITIOUS: Higher target based on Phase 4's 87.4% success
        'description': 'Ultra-precision mastery and tournament-level refinement'
    }
}

# Position-specific weights for balanced training
AZ_POSITION_WEIGHTS = {
    POSITION_ADVANCED_CENTER: 0.5,    # IMPROVED: Emphasize best-performing position
    POSITION_STANDARD: 0.3,           # Stable weight
    POSITION_INTERMEDIATE_1: 0.1,     # IMPROVED: Reduced complexity weight
    POSITION_INTERMEDIATE_2: 0.1      # IMPROVED: Reduced complexity weight
}

# Curriculum monitoring
AZ_CURRICULUM_LOGGING = True          # Enable detailed curriculum logging
AZ_POSITION_SPECIFIC_METRICS = True   # Track metrics per starting position

# --- AlphaZero MCTS Configuration ---
# MCTS Parameters - PHASE 4 OPTIMIZATION FOR 4-POSITION MASTERY
# Enhanced for complex multi-position strategic learning
NUM_SIMULATIONS_PER_MOVE = 60    # INCREASED: More simulations for 4-position complexity evaluation
C_PUCT_CONSTANT = 1.2           # INCREASED: Enhanced exploration for position variety

# Temperature schedule - PHASE 4 ENHANCED FOR STRATEGIC DIVERSITY
TEMPERATURE_START = 1.6         # INCREASED: Higher exploration for new position types
TEMPERATURE_END = 0.4           # MAINTAINED: Quality decision making
TEMPERATURE_ANNEAL_STEPS = 3500 # EXTENDED: Longer annealing for 4-position learning

DIRICHLET_ALPHA = 0.25          # OPTIMIZED: Balanced noise for position diversity
DIRICHLET_EPSILON = 0.18       # INCREASED: More noise for exploration across positions

# AlphaZero Training Loop Parameters - PHASE 4 FULL MASTERY (JIT-OPTIMIZED)
# Building on Phase 3's excellent results: 94.9% policy accuracy, 23/39 model updates
AZ_ITERATIONS = 400             # PHASE 5: Ultra-mastery and fine-tuning (321-400)
AZ_GAMES_PER_ITERATION = 25      # INCREASED: More experience per iteration for 4-position complexity
AZ_TRAINING_STEPS_PER_ITERATION = 1200   # INCREASED: Deeper learning for complex positions
AZ_REPLAY_BUFFER_SIZE = 20000    # EXPANDED: Larger buffer for 4-position diversity
AZ_BATCH_SIZE = 144              # OPTIMIZED: Memory-efficient batch size
AZ_EVALUATION_GAMES_COUNT = 18   # INCREASED: More robust evaluation for complex strategies
AZ_MODEL_UPDATE_WIN_RATE = 0.58  # INCREASED: Higher bar based on Phase 3 performance

# PHASE 2A CONFIGURATION (Iterations 61-80) - OPTIMIZER ENHANCED
# Applied recommendations from training_optimizer.py analysis:
# - Increased policy loss weight: 5.0 → 8.0
# - Reduced value loss weight: 1.0 → 0.5 
# - Increased learning rate: 1e-5 → 2e-5
# - Optimized gradient accumulation: 4 → 6
# - Target: Policy accuracy > 65%, Policy loss < 1.5

# Neural Network Architecture - PHASE 4 OPTIMIZATION
AZ_NN_INPUT_DEPTH = 192  # Number of features in the input layer (e.g., 6 channels for 8x4 board = 6 * 32 = 192)
AZ_NN_RESIDUAL_BLOCKS = 4      # MAINTAINED: Proven effective architecture
AZ_NN_FILTERS = 64            # MAINTAINED: Stable complexity
AZ_NN_POLICY_HEAD_UNITS = 192  # MAINTAINED: Effective policy head size
AZ_NN_VALUE_HEAD_UNITS = 96    # MAINTAINED: Effective value head size
AZ_LEARNING_RATE = 8e-6  # REDUCED: Finer learning for 4-position mastery
AZ_L2_REGULARIZATION = 1.5e-5    # INCREASED: Better generalization across positions
AZ_VALUE_LOSS_WEIGHT = 0.25     # FURTHER REDUCED: Focus on policy learning for complex positions
AZ_POLICY_LOSS_WEIGHT = 6.0    # INCREASED: Stronger policy learning emphasis
AZ_POLICY_HEAD_TYPE = "default"  # Added: Specifies the type of policy head
AZ_VALUE_HEAD_TYPE = "default"   # Added: Specifies the type of value head

# Advanced Training Parameters for Emergent Strategy Development
# AZ_CURRICULUM_LEARNING = True      # Enable progressive difficulty (UNUSED)
# AZ_DYNAMIC_TEMPERATURE = True      # Adaptive temperature based on position complexity (RELATED TO CURRICULUM)
AZ_STRATEGIC_DIVERSITY_BONUS = 0.15 # Increased for novel strategy exploration in extended training
AZ_LONG_GAME_EMPHASIS = 1.4        # Increased weight for learning from strategic games
AZ_MODEL_ENSEMBLE_SIZE = 5          # Larger ensemble for robust evaluation in extended training

# Learning Rate Schedule - STABILITY OPTIMIZED TO PREVENT NaN
AZ_LR_DECAY_SCHEDULE = {
    'type': 'cosine',               # Cosine annealing for smooth convergence
    'initial_lr': 1e-5,             # STABILIZED: Reduced to prevent instability
    'min_lr': 1e-6,                 # CONSERVATIVE: Lower minimum for stable learning
    'decay_steps': 30000            # EXTENDED: Slower decay for stability
}

# Temperature Schedule Enhancements for 12-Hour Training
# AZ_ADAPTIVE_TEMP_THRESHOLD = 300    # Reduced threshold for faster adaptation (was 500) (RELATED TO DYNAMIC TEMP)
# AZ_COMPLEX_POSITION_TEMP_BOOST = 1.3  # Moderate boost for efficiency (was 1.5) (RELATED TO DYNAMIC TEMP)


# EMERGENCY POLICY LEARNING BOOST - STABILIZED
AZ_POLICY_LEARNING_BOOST = True     # Enable aggressive policy learning
AZ_POLICY_LOSS_MULTIPLIER = 2.0     # REDUCED: Prevent numerical instability
AZ_EXPLORATION_BONUS = 0.1          # REDUCED: Conservative exploration to prevent NaN
AZ_POLICY_TEMPERATURE_BOOST = 1.1   # REDUCED: Moderate temperature boost

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
AZ_GRADIENT_CLIP_NORM = 0.3      # TIGHTENED: Prevent gradient explosion causing NaN
AZ_INFERENCE_BATCH_SIZE = 64     # Larger batch size for efficiency
AZ_EARLY_STOP_SLOW_EVAL = False  # Allow slower games for quality learning
AZ_MAX_GAME_LENGTH = 125         # Slight increase from 100 for strategic depth in extended training

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


# NUMERICAL STABILITY SAFEGUARDS - CRITICAL FOR NaN PREVENTION
AZ_LOSS_CLIPPING = True              # Clip loss values to prevent overflow
AZ_MAX_LOSS_VALUE = 50.0            # Maximum allowed loss value
AZ_POLICY_EPSILON = 1e-8            # Small epsilon for policy log calculations
AZ_VALUE_CLIPPING_RANGE = (-2.0, 2.0)  # Strict value prediction clipping
AZ_GRADIENT_ACCUMULATION_CLIPPING = True  # Clip accumulated gradients
AZ_NAN_CHECK_FREQUENCY = 1          # Check for NaN every iteration
AZ_EMERGENCY_STOP_ON_NAN = True     # Stop training if NaN detected

# GAME LENGTH CONTROL - BALANCED FOR 12-HOUR TRAINING
AZ_GAME_LENGTH_PENALTY = 0.003   # Reduced penalty to allow strategic game development
AZ_DRAW_PENALTY = -0.05          # Maintained for natural game flow
AZ_MOVE_TIME_PRESSURE = False    # Disabled for quality strategic learning
AZ_TERMINAL_STATE_BONUS = 0.1    # Maintained moderate bonus for completion
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