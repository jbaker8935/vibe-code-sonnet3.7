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
    # Phase 1: Master core position (iterations 1-40)
    'phase_1': {
        'iterations': (1, 40),
        'positions': [POSITION_ADVANCED_CENTER],
        'target_policy_accuracy': 0.60,
        'description': 'Advanced center position mastery'
    },
    # Phase 2: Add standard position (iterations 41-80) 
    'phase_2': {
        'iterations': (41, 80),
        'positions': [POSITION_ADVANCED_CENTER, POSITION_STANDARD],
        'target_policy_accuracy': 0.50,
        'description': 'Dual position learning'
    },
    # Phase 3: Full curriculum (iterations 81-140)
    'phase_3': {
        'iterations': (81, 140),
        'positions': [POSITION_ADVANCED_CENTER, POSITION_STANDARD, 
                     POSITION_INTERMEDIATE_1, POSITION_INTERMEDIATE_2],
        'target_policy_accuracy': 0.45,
        'description': 'Complete strategic mastery'
    },
    # Phase 4: Advanced strategic refinement (iterations 141-180)
    'phase_4': {
        'iterations': (141, 180),
        'positions': [POSITION_ADVANCED_CENTER, POSITION_STANDARD, 
                     POSITION_INTERMEDIATE_1, POSITION_INTERMEDIATE_2],
        'target_policy_accuracy': 0.50,
        'description': 'Advanced strategic refinement and mastery'
    },
    # Phase 5: Ultra-precision mastery (iterations 181-220)
    'phase_5': {
        'iterations': (181, 220),
        'positions': [POSITION_ADVANCED_CENTER, POSITION_STANDARD, 
                     POSITION_INTERMEDIATE_1, POSITION_INTERMEDIATE_2],
        'target_policy_accuracy': 0.55,
        'description': 'Ultra-precision mastery with minimal exploration'
    }
}

# Position-specific weights for balanced training
AZ_POSITION_WEIGHTS = {
    POSITION_ADVANCED_CENTER: 0.4,    # Emphasize well-learned position
    POSITION_STANDARD: 0.3,           # Standard weight
    POSITION_INTERMEDIATE_1: 0.15,    # Lower weight for complex positions
    POSITION_INTERMEDIATE_2: 0.15
}

# Curriculum monitoring
AZ_CURRICULUM_LOGGING = True          # Enable detailed curriculum logging
AZ_POSITION_SPECIFIC_METRICS = True   # Track metrics per starting position

# --- AlphaZero MCTS Configuration ---
# MCTS Parameters - JIT-OPTIMIZED BASED ON PERFORMANCE ANALYSIS (8.3x SPEEDUP)
# Performance Analysis Results (JIT Environment):
# • Current (400 sims): 0.591s/move, 1.5ms/sim, entropy: 2.23
# • Optimal (50 sims): 0.072s/move, 8.3x speedup, entropy: 2.22 (minimal quality loss)
NUM_SIMULATIONS_PER_MOVE = 50    # OPTIMIZED: 8.3x speedup with minimal quality loss
C_PUCT_CONSTANT = 0.1           # OPTIMIZED: Lower exploration for faster convergence

# Temperature schedule - PROGRESSIVE STRATEGIC MASTERY
TEMPERATURE_START = 1.4         # Higher for creative strategic exploration
TEMPERATURE_END = 0.02          # Lower for precise endgame execution  
TEMPERATURE_ANNEAL_STEPS = 5000 # Extended for gradual mastery development over many iterations

DIRICHLET_ALPHA = 0.7          # Increased for maximum opening diversity
DIRICHLET_EPSILON = 0.20        # Enhanced exploration for breakthrough strategies

# AlphaZero Training Loop Parameters - EXTENDED STRATEGIC MASTERY (JIT-OPTIMIZED)
# Extended for deeper strategic learning and mastery refinement
AZ_ITERATIONS = 220              # EXTENDED: Additional 40 iterations for Phase 5 ultra-mastery  
AZ_GAMES_PER_ITERATION = 30      # INCREASED: Higher game count with maintained speed
AZ_TRAINING_STEPS_PER_ITERATION = 300   # INCREASED: More training steps for comprehensive learning
AZ_REPLAY_BUFFER_SIZE = 10000    # EXPANDED: Larger buffer for richer experience diversity
AZ_BATCH_SIZE = 128              # MAINTAINED: Optimal batch size for stability
AZ_EVALUATION_GAMES_COUNT = 15   # INCREASED: More robust evaluation with faster MCTS
AZ_MODEL_UPDATE_WIN_RATE = 0.60  # MAINTAINED: Proven threshold for strategic advancement

# Neural Network Architecture - OPTIMIZED FOR EXTENDED LEARNING
AZ_NN_INPUT_DEPTH = 192  # Number of features in the input layer (e.g., 6 channels for 8x4 board = 6 * 32 = 192)
AZ_NN_RESIDUAL_BLOCKS = 3      # Increased from 1 to address overfit test failure
AZ_NN_FILTERS = 64             # Increased from 32 to improve model capacity
AZ_NN_POLICY_HEAD_UNITS = 64  # Simplified: Was 192
AZ_NN_VALUE_HEAD_UNITS = 32    # Simplified: Was 96
AZ_LEARNING_RATE = 2e-4        # Slightly reduced from 3e-4 for stability in extended training
AZ_L2_REGULARIZATION = 1e-6    # Minimal regularization for extended training (was 0.0)
AZ_VALUE_LOSS_WEIGHT = 1.0     # Maintained - value learning is working excellently
AZ_POLICY_LOSS_WEIGHT = 3.0    # Increased from 2.5 for continued strategic policy refinement
AZ_POLICY_HEAD_TYPE = "default"  # Added: Specifies the type of policy head
AZ_VALUE_HEAD_TYPE = "default"   # Added: Specifies the type of value head

# Advanced Training Parameters for Emergent Strategy Development
# AZ_CURRICULUM_LEARNING = True      # Enable progressive difficulty (UNUSED)
# AZ_DYNAMIC_TEMPERATURE = True      # Adaptive temperature based on position complexity (RELATED TO CURRICULUM)
AZ_STRATEGIC_DIVERSITY_BONUS = 0.15 # Increased for novel strategy exploration in extended training
AZ_LONG_GAME_EMPHASIS = 1.4        # Increased weight for learning from strategic games
AZ_MODEL_ENSEMBLE_SIZE = 5          # Larger ensemble for robust evaluation in extended training

# Learning Rate Schedule - COSINE ANNEALING FOR EXTENDED TRAINING
AZ_LR_DECAY_SCHEDULE = {
    'type': 'cosine',               # Cosine annealing for smooth convergence
    'initial_lr': 2e-4,             # Aligned with AZ_LEARNING_RATE
    'min_lr': 5e-6,                 # Lower minimum for fine-tuning in extended training
    'decay_steps': 8000             # Longer decay cycles for extended strategic training
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


# LEARNING BREAKTHROUGH PARAMETERS - ADDRESS SPECIFIC ISSUES
AZ_FORCE_SHORTER_GAMES = True    # Implement game length penalties
AZ_IMPROVED_VALUE_TARGETS = True # Use better value target computation
AZ_POLICY_SMOOTHING = 0.08        # Increased policy target smoothing for better generalization
AZ_VALUE_CLIPPING = True         # Clip value predictions to [-1, 1]
AZ_GRADIENT_ACCUMULATION = 3     # Increased accumulation for stable large-batch training
AZ_WARMUP_ITERATIONS = 5         # Use learning rate warmup for first 5 iterations

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