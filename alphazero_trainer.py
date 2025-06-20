import tensorflow as tf
import numpy as np
import collections
import random
import os
import time
from tqdm import tqdm
import wandb # For experiment tracking
import pickle # Added for static dataset saving/loading
import argparse # Added: For command-line arguments
import psutil   # Added: For memory monitoring
import gc       # Added: For garbage collection

from game_env import SwitcharooEnv, PLAYER_A_ID, PLAYER_B_ID # Keep game_env specific imports
from game_env_jit import SwitcharooEnvJitWrapper
from env_const import ROWS, COLS, NUM_PIECES, MAX_MOVES_PER_PIECE, NUM_ACTIONS # Import constants from env_const
from az_network import AlphaZeroNetwork
from mcts import MCTS, MCTSNode  # Python MCTS
from mcts_numba import MCTSNumba, NODE_FIELDS # Numba MCTS
from config import (
    AZ_ITERATIONS, AZ_GAMES_PER_ITERATION, AZ_TRAINING_STEPS_PER_ITERATION,
    AZ_REPLAY_BUFFER_SIZE, AZ_BATCH_SIZE, AZ_EVALUATION_GAMES_COUNT,
    AZ_MODEL_UPDATE_WIN_RATE, AZ_BEST_MODEL_FILE, AZ_CANDIDATE_MODEL_FILE,
    AZ_CHECKPOINT_FILE_PATTERN, NUM_SIMULATIONS_PER_MOVE, C_PUCT_CONSTANT,
    DIRICHLET_ALPHA, DIRICHLET_EPSILON, TEMPERATURE_START, TEMPERATURE_END,
    TEMPERATURE_ANNEAL_STEPS, AZ_LEARNING_RATE, AZ_LR_DECAY_SCHEDULE, WANDB_PROJECT, WANDB_ENTITY,
    USE_NUMBA, MAX_STEPS_PER_EPISODE, initial_position,
    AZ_RECOVERY_MODE, AZ_RESET_OPTIMIZER, AZ_CLEAR_REPLAY_BUFFER, AZ_DISABLE_ATTENTION,
    AZ_OVERFIT_TEST_REQUIRED,
    AZ_LOG_ACTIVATION_STATS, # Added for Step 1.4
    # Step 1.3: Static Dataset Config
    AZ_GENERATE_STATIC_DATASET, AZ_TRAIN_ON_STATIC_DATASET, AZ_STATIC_DATASET_PATH,
    AZ_STATIC_TRAINING_EPOCHS, AZ_STATIC_DATASET_GAMES,
    # Progressive Curriculum Config
    AZ_PROGRESSIVE_CURRICULUM, AZ_CURRICULUM_SCHEDULE, AZ_POSITION_WEIGHTS,
    AZ_CURRICULUM_LOGGING, AZ_POSITION_SPECIFIC_METRICS
)
# Helper function for temperature decay
def get_temperature(total_game_steps):
    """
    Calculate temperature for MCTS action selection based on total game steps.
    
    Temperature controls exploration vs exploitation:
    - High temperature (TEMPERATURE_START) = more exploration, stochastic moves
    - Low temperature (TEMPERATURE_END) = more exploitation, deterministic moves
    
    Args:
        total_game_steps: Total number of individual game moves across all iterations
        
    Returns:
        float: Temperature value between TEMPERATURE_START and TEMPERATURE_END
    """
    if total_game_steps < TEMPERATURE_ANNEAL_STEPS:
        return TEMPERATURE_START * (1 - total_game_steps / TEMPERATURE_ANNEAL_STEPS) + \
               TEMPERATURE_END * (total_game_steps / TEMPERATURE_ANNEAL_STEPS)
    else:
        return TEMPERATURE_END

def get_curriculum_aware_temperature(total_game_steps, iteration):
    """
    Calculate temperature with curriculum phase awareness.
    
    This function maintains higher exploration when new curriculum phases
    introduce additional starting positions, then gradually anneals within
    each phase. This addresses the critical issue where temperature reaches
    minimum (0.02) by iteration 8, leaving phases 2 and 3 with no exploration.
    
    Args:
        total_game_steps: Total number of individual game moves across all iterations
        iteration: Current training iteration (1-based)
        
    Returns:
        float: Temperature value optimized for current curriculum phase
    """
    base_temp = get_temperature(total_game_steps)
    if not AZ_PROGRESSIVE_CURRICULUM:
        return base_temp

    current_phase_name = None
    current_phase_config = None
    for phase_name, phase_config in AZ_CURRICULUM_SCHEDULE.items():
        phase_start, phase_end = phase_config['iterations']
        if phase_start <= iteration <= phase_end:
            current_phase_name = phase_name
            current_phase_config = phase_config
            break

    if not current_phase_config:
        return base_temp

    phase_start, phase_end = current_phase_config['iterations']
    phase_length = phase_end - phase_start + 1
    phase_progress = (iteration - phase_start) / phase_length  # 0.0 to 1.0

    # Phase-specific temperature annealing
    if current_phase_name == 'phase_1':
        phase_min_temp = 0.1
        phase_max_temp = 1.0
    elif current_phase_name == 'phase_2':
        phase_min_temp = 0.05
        phase_max_temp = 0.7
    elif current_phase_name == 'phase_3':
        phase_min_temp = 0.02
        phase_max_temp = 0.5
    elif current_phase_name == 'phase_4':
        phase_min_temp = 0.01
        phase_max_temp = 0.25
    elif current_phase_name == 'phase_5':
        phase_min_temp = 0.01
        phase_max_temp = 0.15
    elif current_phase_name == 'phase_6':
        phase_min_temp = 0.01
        phase_max_temp = 0.1
    else:
        phase_min_temp = TEMPERATURE_END
        phase_max_temp = TEMPERATURE_START

    # Anneal temperature within phase, with last 10% at min temp
    anneal_portion = 0.85  # 85% of phase for annealing, last 15% at min temp
    if phase_progress < anneal_portion:
        local_progress = phase_progress / anneal_portion
        phase_temp = phase_max_temp * (1 - local_progress) + phase_min_temp * local_progress
    else:
        phase_temp = phase_min_temp
    return phase_temp

# Progressive Curriculum Helper Functions
def get_current_curriculum_phase(iteration):
    """Determine which curriculum phase we're in based on iteration number."""
    if not AZ_PROGRESSIVE_CURRICULUM:
        return None
    
    for phase_name, phase_config in AZ_CURRICULUM_SCHEDULE.items():
        start_iter, end_iter = phase_config['iterations']
        if start_iter <= iteration <= end_iter:
            return phase_name, phase_config
    
    # If beyond all phases, use the last phase
    last_phase = list(AZ_CURRICULUM_SCHEDULE.items())[-1]
    return last_phase[0], last_phase[1]

def get_curriculum_positions(iteration):
    """Get the list of starting positions for the current iteration."""
    if not AZ_PROGRESSIVE_CURRICULUM:
        return initial_position  # Fall back to original behavior
    
    phase_name, phase_config = get_current_curriculum_phase(iteration)
    if phase_config is None:
        return initial_position
    
    return phase_config['positions']

def select_weighted_position(positions, iteration=None):
    """Select a position using curriculum weights if available."""
    if not AZ_PROGRESSIVE_CURRICULUM or not AZ_POSITION_WEIGHTS:
        return random.choice(positions)
    
    # Get weights for the available positions
    weights = []
    for pos in positions:
        weight = AZ_POSITION_WEIGHTS.get(pos, 1.0)  # Default weight 1.0
        weights.append(weight)
    
    # Normalize weights
    total_weight = sum(weights)
    if total_weight <= 0:
        return random.choice(positions)
    
    probabilities = [w / total_weight for w in weights]
    
    # Select position based on weights
    return np.random.choice(positions, p=probabilities)

class AlphaZeroTrainer:
    def __init__(self, use_wandb=True, use_jit_env=False):  # Add use_jit_env parameter
        print("Initializing AlphaZero Trainer...")
        
        # Configure GPU memory growth to prevent full VRAM allocation
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"🔧 GPU Memory Growth enabled for {len(gpus)} GPU(s)")
            except RuntimeError as e:
                print(f"⚠️  GPU Memory Growth setting failed: {e}")
        
        if use_jit_env:
            self.game_env_class = SwitcharooEnvJitWrapper
            self.game_env = self.game_env_class()
        else:
            self.game_env_class = SwitcharooEnv
            self.game_env = self.game_env_class()

        # --- Step 1.3: Static Dataset Handling ---
        if AZ_GENERATE_STATIC_DATASET:
            print(f"--- STATIC DATASET GENERATION MODE ---")
            print(f"Will generate {AZ_STATIC_DATASET_GAMES} games and save to {AZ_STATIC_DATASET_PATH}")
        elif AZ_TRAIN_ON_STATIC_DATASET:
            print(f"--- STATIC DATASET TRAINING MODE ---")
            print(f"Will load data from {AZ_STATIC_DATASET_PATH} and train for {AZ_STATIC_TRAINING_EPOCHS} epochs.")
        # --- End Step 1.3 ---

        # Cosine decay scheduler with warmup for smoother learning rate transitions
        # First create warmup schedule
        warmup_steps = 5000
        warmup_lr = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=AZ_LR_DECAY_SCHEDULE['initial_lr'] * 0.1,
            decay_steps=warmup_steps,
            end_learning_rate=AZ_LR_DECAY_SCHEDULE['initial_lr'],
            power=1.0  # Linear warmup
        )
        
        # Then create main cosine decay schedule
        cosine_decay_lr = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=AZ_LR_DECAY_SCHEDULE['initial_lr'],
            decay_steps=AZ_LR_DECAY_SCHEDULE['decay_steps'],
            alpha=AZ_LR_DECAY_SCHEDULE['min_lr']/AZ_LR_DECAY_SCHEDULE['initial_lr']  # Minimum learning rate factor
        )
        
        # Combine schedules using a proper LearningRateSchedule class
        class CustomLearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
            def __init__(self, warmup_schedule, main_schedule, warmup_steps):
                super(CustomLearningRateSchedule, self).__init__()
                self.warmup_schedule = warmup_schedule
                self.main_schedule = main_schedule
                self.warmup_steps = warmup_steps
            
            def __call__(self, step):
                step_float = tf.cast(step, tf.float32)
                # Use tf.cond for TensorFlow-compatible conditional logic
                return tf.cond(
                    tf.less(step_float, tf.cast(self.warmup_steps, tf.float32)),
                    lambda: self.warmup_schedule(step_float),
                    lambda: self.main_schedule(step_float - tf.cast(self.warmup_steps, tf.float32))
                )
            
            def get_config(self):
                return {
                    "warmup_schedule": self.warmup_schedule,
                    "main_schedule": self.main_schedule,
                    "warmup_steps": self.warmup_steps
                }
        
        self.lr_scheduler = CustomLearningRateSchedule(warmup_lr, cosine_decay_lr, warmup_steps)
        
        # Initialize Neural Networks
        self.current_nn = AlphaZeroNetwork(learning_rate_schedule=self.lr_scheduler) 
        self.candidate_nn = AlphaZeroNetwork(learning_rate_schedule=self.lr_scheduler)

        # Handle recovery mode and model loading
        if AZ_RECOVERY_MODE:
            print("=== RECOVERY MODE ACTIVATED ===")
            print("Using simplified architecture and conservative parameters")
            
            # Clear replay buffer if requested
            if AZ_CLEAR_REPLAY_BUFFER:
                print("Clearing replay buffer for fresh start")
                # Buffer will be initialized empty anyway
                
            # Reset optimizer if requested
            if AZ_RESET_OPTIMIZER:
                print("Optimizer will be reset with new learning rate and clipping")

        if os.path.exists(AZ_BEST_MODEL_FILE):
            print(f"Found existing model at {AZ_BEST_MODEL_FILE}, attempting to load...")
            try:
                # Re-initialize with scheduler before loading weights
                self.current_nn = AlphaZeroNetwork(learning_rate_schedule=self.lr_scheduler)
                self.current_nn.load_model(AZ_BEST_MODEL_FILE)
                self.candidate_nn = AlphaZeroNetwork(learning_rate_schedule=self.lr_scheduler)
                self.candidate_nn.load_model(AZ_BEST_MODEL_FILE)
                
                # In recovery mode, force optimizer reset
                if AZ_RECOVERY_MODE and AZ_RESET_OPTIMIZER:
                    print("Recovery mode: Rebuilding models with fresh optimizers")
                    self.current_nn = AlphaZeroNetwork(learning_rate_schedule=self.lr_scheduler)
                    self.current_nn.load_model(AZ_BEST_MODEL_FILE)
                    self.candidate_nn = AlphaZeroNetwork(learning_rate_schedule=self.lr_scheduler)
                    self.candidate_nn.load_model(AZ_BEST_MODEL_FILE)
                    
                print("Successfully loaded existing model weights.")
            except ValueError as e:
                print(f"Error loading model weights: {e}")
                print("\nModel architecture has changed - starting with fresh models.")
                # Reinitialize from scratch
                self.current_nn = AlphaZeroNetwork(learning_rate_schedule=self.lr_scheduler) 
                self.candidate_nn = AlphaZeroNetwork(learning_rate_schedule=self.lr_scheduler)
                # Rename the old model file to prevent future attempts to load it
                backup_file = f"{AZ_BEST_MODEL_FILE}.incompatible_arch.bak"
                print(f"Backing up incompatible model to {backup_file}")
                os.rename(AZ_BEST_MODEL_FILE, backup_file)
        else:
            print(f"No best model found at {AZ_BEST_MODEL_FILE}. Starting with fresh models (scheduler will be used).")

        # DEBUG: Test prediction
        print("DEBUG: Performing a test prediction with the initialized network...")
        try:
            dummy_state = self.game_env.reset()
            dummy_repr = self.game_env._get_state()
            start_time = time.time()
            policy, value = self.current_nn.predict(dummy_repr)
            end_time = time.time()
            print(f"DEBUG: Test prediction successful. Policy shape: {policy.shape}, Value: {value}, Time: {end_time - start_time:.4f}s")
        except Exception as e:
            print(f"DEBUG: Test prediction failed: {e}")
        # END DEBUG

        self.replay_buffer = collections.deque(maxlen=AZ_REPLAY_BUFFER_SIZE)
        
        class MCTSConfig:
            NUM_SIMULATIONS_PER_MOVE = NUM_SIMULATIONS_PER_MOVE
            C_PUCT_CONSTANT = C_PUCT_CONSTANT
            DIRICHLET_ALPHA = DIRICHLET_ALPHA
            DIRICHLET_EPSILON = DIRICHLET_EPSILON
        
        self.mcts_config = MCTSConfig()
        self.total_game_steps_for_temp = 0
        
        # Track current iteration for curriculum learning
        self.current_iteration = 0

        # Initialize WandB
        self.wandb_enabled = False  # Default to False
        if use_wandb:  # Check the flag
            try:
                # Get current learning rate from scheduler for logging
                # For ExponentialDecay, direct value access is tricky without a step; log initial or optimizer's current LR
                # Safely get learning rate - handle custom function-based scheduler
                try:
                    # Get current learning rate using the scheduler
                    step = tf.constant(0, dtype=tf.float32)
                    current_lr_for_log = self.lr_scheduler(step).numpy()
                except Exception as e:
                    current_lr_for_log = AZ_LEARNING_RATE
                    print(f"Could not get current learning rate: {e}, using config value instead.")

                wandb.init(
                    project=WANDB_PROJECT if WANDB_PROJECT else "switcharoo-alphazero", 
                    entity=WANDB_ENTITY, 
                    config={
                        "framework": "AlphaZero",
                        "iterations": AZ_ITERATIONS,
                        "games_per_iteration": AZ_GAMES_PER_ITERATION,
                        "training_steps_per_iteration": AZ_TRAINING_STEPS_PER_ITERATION,
                        "replay_buffer_size": AZ_REPLAY_BUFFER_SIZE,
                        "batch_size": AZ_BATCH_SIZE,
                        "num_simulations_per_move": NUM_SIMULATIONS_PER_MOVE,
                        "c_puct": self.mcts_config.C_PUCT_CONSTANT,
                        "dirichlet_alpha": self.mcts_config.DIRICHLET_ALPHA,
                        "dirichlet_epsilon": self.mcts_config.DIRICHLET_EPSILON,
                        "temperature_start": TEMPERATURE_START,
                        "temperature_end": TEMPERATURE_END,
                        "temperature_anneal_steps": TEMPERATURE_ANNEAL_STEPS,
                        "az_initial_learning_rate": AZ_LEARNING_RATE, # Log initial LR
                        "az_model_update_win_rate": AZ_MODEL_UPDATE_WIN_RATE,
                        "az_evaluation_games": AZ_EVALUATION_GAMES_COUNT,
                    }
                )
                print("Weights & Biases initialized.")
                self.wandb_enabled = True
            except Exception as e:
                print(f"Could not initialize Weights & Biases: {e}. Will run without WandB logging.")
        else:
            print("Weights & Biases logging disabled by command-line argument.")
        
        print(f"[DEBUG] Using env class: {type(self.game_env)}; _get_state shape: {self.game_env._get_state().shape}")    

    def print_sample_predictions(self, n=3):
        print("\n[Diagnostics] Sample predictions from candidate network:")
        samples = random.sample(list(self.replay_buffer), min(n, len(self.replay_buffer)))
        for i, (state, policy_target, value_target) in enumerate(samples):
            policy_pred, value_pred = self.candidate_nn.predict(state)
            value_pred_scalar = np.squeeze(value_pred)  # Fix: ensure scalar
            print(f"Sample {i+1}:")
            print(f"  True value: {value_target:.2f}, Predicted value: {value_pred_scalar:.2f}")
            print(f"  True policy (argmax): {np.argmax(policy_target)}, Predicted policy (argmax): {np.argmax(policy_pred)}")
            print(f"  Predicted policy (top 3): {np.argsort(policy_pred)[-3:][::-1]}")
            print(f"  Policy entropy: {-(policy_pred * np.log(policy_pred + 1e-8)).sum():.2f}")
            print(f"  Policy target entropy: {-(policy_target * np.log(policy_target + 1e-8)).sum():.2f}")

    def print_value_head_stats(self, n=100):
        # Print stats for value head predictions on a sample of the buffer
        if len(self.replay_buffer) == 0:
            print("[Diagnostics] Replay buffer empty, skipping value head stats.")
            return
        sample = random.sample(list(self.replay_buffer), min(n, len(self.replay_buffer)))
        states = np.array([s for s, _, _ in sample])
        targets = np.array([v for _, _, v in sample])
        preds = np.array([np.squeeze(self.candidate_nn.predict(s)[1]) for s in states])  # Fix: ensure scalar
        print(f"[Diagnostics] Value head stats on {len(sample)} samples:")
        print(f"  Target mean: {targets.mean():.3f}, std: {targets.std():.3f}, min: {targets.min():.2f}, max: {targets.max():.2f}")
        print(f"  Pred mean:   {preds.mean():.3f}, std: {preds.std():.3f}, min: {preds.min():.2f}, max: {preds.max():.2f}")
        # Optional: confusion matrix for sign of value
        try:
            from sklearn.metrics import confusion_matrix
            t_sign = np.sign(targets).astype(int)
            p_sign = np.sign(preds).astype(int)
            cm = confusion_matrix(t_sign, p_sign, labels=[-1,0,1])
            print("[Diagnostics] Value head sign confusion matrix (rows=true, cols=pred):")
            print(cm)
        except Exception as e:
            print(f"[Diagnostics] Could not compute confusion matrix: {e}")

    def print_policy_entropy_stats(self, n=100):
        if len(self.replay_buffer) == 0:
            print("[Diagnostics] Replay buffer empty, skipping policy entropy stats.")
            return
        sample = random.sample(list(self.replay_buffer), min(n, len(self.replay_buffer)))
        states = np.array([s for s, _, _ in sample])
        entropies = []
        for s in states:
            policy_pred, _ = self.candidate_nn.predict(s)
            entropy = -(policy_pred * np.log(policy_pred + 1e-8)).sum()
            entropies.append(entropy)
        print(f"[Diagnostics] Policy entropy: mean={np.mean(entropies):.2f}, std={np.std(entropies):.2f}, min={np.min(entropies):.2f}, max={np.max(entropies):.2f}")

    def print_replay_buffer_targets(self, n=5):
        print("\n[Diagnostics] Sample replay buffer targets:")
        samples = random.sample(list(self.replay_buffer), min(n, len(self.replay_buffer)))
        for i, (state, policy_target, value_target) in enumerate(samples):
            print(f"Sample {i+1}: value_target={value_target}, policy_target_sum={np.sum(policy_target):.3f}, policy_target_argmax={np.argmax(policy_target)}, policy_target_nonzero={np.count_nonzero(policy_target)}")

    def overfit_on_small_batch(self, steps=500):
        print("\n[Overfit Test] Training on a single minibatch...")
        minibatch = random.sample(list(self.replay_buffer), AZ_BATCH_SIZE)
        batch_states = np.array([data[0] for data in minibatch])
        batch_policy_targets = np.array([data[1] for data in minibatch])
        batch_value_targets = np.array([data[2] for data in minibatch]).reshape(-1, 1)

        # --- BEGIN: Step 1.2 Data Inspection ---
        print("\n[Overfit Test - Data Inspection] First 3 samples from minibatch:")
        for i in range(min(3, AZ_BATCH_SIZE)):
            print(f"  Sample {i+1}:")
            print(f"    State shape: {batch_states[i].shape}, State (first 10 flat): {batch_states[i].flatten()[:10]}")
            print(f"    Policy Target shape: {batch_policy_targets[i].shape}, Sum: {np.sum(batch_policy_targets[i]):.4f}, Max: {np.max(batch_policy_targets[i]):.4f}, Argmax: {np.argmax(batch_policy_targets[i])}")
            print(f"    Value Target: {batch_value_targets[i][0]:.4f}")
        # --- END: Step 1.2 Data Inspection ---

        for i in range(steps):
            loss_info = self.candidate_nn.model.train_on_batch(
                batch_states, 
                {'policy_output': batch_policy_targets, 'value_output': batch_value_targets},
                return_dict=True
            )
            if i % 50 == 0 or i == steps - 1:
                print(f"Step {i}: loss={loss_info['loss']:.4f}, policy_loss={loss_info['policy_output_loss']:.4f}, value_loss={loss_info['value_output_loss']:.4f}")
        print("[Overfit Test] After training:")
        for j in range(3):
            policy_pred, value_pred = self.candidate_nn.predict(batch_states[j])
            print(f"Sample {j+1}: True value={batch_value_targets[j][0]}, Pred value={np.squeeze(value_pred):.3f}, True policy argmax={np.argmax(batch_policy_targets[j]),}, Pred policy argmax={np.argmax(policy_pred)}")

    def _run_self_play_games(self, num_games_to_generate=None): # Added num_games_to_generate
        num_games = num_games_to_generate if num_games_to_generate is not None else AZ_GAMES_PER_ITERATION
        print(f"\\n--- Generating {num_games} Self-Play Games ---")
        games_played_this_iteration = 0
        new_experiences_count = 0
        iteration_game_outcomes = {"win": 0, "loss": 0, "draw": 0}
        iteration_game_steps = 0  # Track steps for this iteration only

        # Progressive curriculum logging
        if AZ_PROGRESSIVE_CURRICULUM and AZ_CURRICULUM_LOGGING:
            phase_name, phase_config = get_current_curriculum_phase(self.current_iteration)
            if phase_config:
                print(f"Curriculum Phase: {phase_name} - {phase_config['description']}")
                print(f"Available positions: {len(phase_config['positions'])}")

        for game_num in tqdm(range(num_games),desc="Self-Play Games"):  # Added tqdm for progress bar
            current_game_experiences = [] 
            
            # Use curriculum-based position selection
            if AZ_PROGRESSIVE_CURRICULUM:
                curriculum_positions = get_curriculum_positions(self.current_iteration)
                starting_position_str = select_weighted_position(curriculum_positions, self.current_iteration)
                
                if AZ_CURRICULUM_LOGGING and game_num < 3:  # Log first few games
                    print(f"  Game {game_num+1}: Selected position from curriculum (Phase: {get_current_curriculum_phase(self.current_iteration)[0]})")
            else:
                # Original random selection
                starting_position_str = random.choice(initial_position)
            
            current_board_state_arr = self.game_env.reset(starting_position=starting_position_str)
            done = False
            game_step_count = 0
            first_player_of_game = self.game_env.current_player_id

            # Log temperature at the start of the game
            current_temp_at_game_start = get_curriculum_aware_temperature(self.total_game_steps_for_temp, self.current_iteration)

            # Removed tqdm wrapper from here
            while not done:
                current_player_id = self.game_env.current_player_id
                if USE_NUMBA:
                    # print("Using Numba-optimized MCTS implementation.")
                    mcts_handler = MCTSNumba(neural_network=self.current_nn, 
                                             game_env_class=self.game_env_class, 
                                             config=self.mcts_config)
                    # Ensure board is copied and C-contiguous for Numba
                    board_copy_for_mcts = np.ascontiguousarray(self.game_env.board)
                    
                    # Get current state representation for NN prediction
                    current_state_repr = self.game_env._get_state() # (8, 4, 18) for NN input
                    # Do NOT flatten; pass as (8, 4, 18)
                    nn_policy, nn_value = self.current_nn.predict(current_state_repr)
                    
                    mcts_handler.run_mcts_simulations(
                        board_copy_for_mcts, # start_board_state_arr
                        current_player_id,   # start_player_id
                        nn_policy,           # nn_policy_values
                        nn_value,            # nn_value_estimate
                        NUM_SIMULATIONS_PER_MOVE # num_simulations
                    )
                    action_probs_from_mcts = mcts_handler.get_action_probabilities(
                        board_copy_for_mcts, # Pass the C-contiguous copy
                        current_player_id,
                        get_curriculum_aware_temperature(self.total_game_steps_for_temp, self.current_iteration)
                    )
                else:
                    # print("Using Python MCTS implementation.")
                    root_node = MCTSNode(parent=None, prior_p=0.0, player_id=current_player_id)
                    mcts_handler = MCTS(neural_network=self.current_nn, 
                                        game_env_class=self.game_env_class, 
                                        config=self.mcts_config)
                    mcts_handler.run_mcts_simulations(
                        root_node,
                        np.copy(self.game_env.board),
                        current_player_id
                    )
                    action_probs_from_mcts = mcts_handler.get_action_probabilities(
                        root_node,
                        get_curriculum_aware_temperature(self.total_game_steps_for_temp, self.current_iteration)
                    )
                temp = get_curriculum_aware_temperature(self.total_game_steps_for_temp, self.current_iteration)
                nn_input_state = self.game_env._get_state() 
                current_game_experiences.append({'state': nn_input_state, 
                                                 'policy_target': action_probs_from_mcts, 
                                                 'player_at_state': current_player_id})

                # --- BEGIN: MCTS Target Logging ---
                if game_step_count < 5 and game_num < 2: # Log for first 5 steps of first 2 games
                    print(f"  [MCTS Target Log] Game: {game_num+1}, Step: {game_step_count+1}, Player: {current_player_id}, Temp: {temp:.3f}")
                    print(f"    Policy Target (sum={np.sum(action_probs_from_mcts):.3f}, max={np.max(action_probs_from_mcts):.3f}, argmax={np.argmax(action_probs_from_mcts)}, non-zero={np.count_nonzero(action_probs_from_mcts)}):")
                    top_k_indices = np.argsort(action_probs_from_mcts)[-5:][::-1] # Top 5 actions
                    top_k_probs = action_probs_from_mcts[top_k_indices]
                    for k_idx, k_prob in zip(top_k_indices, top_k_probs):
                        if k_prob > 0: # Only print non-zero probabilities
                            print(f"      Action {k_idx}: {k_prob:.4f}")
                    # Log raw visit counts if available (Python MCTS)
                    if not USE_NUMBA and hasattr(root_node, 'children') and root_node.children:
                        print(f"    MCTS Visit Counts (Top 5):")
                        child_visits = {action: child.N for action, child in root_node.children.items()}
                        sorted_visits = sorted(child_visits.items(), key=lambda item: item[1], reverse=True)[:5]
                        for action_idx, visit_count in sorted_visits:
                             print(f"      Action {action_idx}: {visit_count} visits")
                # --- END: MCTS Target Logging ---

                if np.sum(action_probs_from_mcts) == 0:
                    print("Warning: MCTS returned all zero probabilities. Choosing random legal action.")
                    legal_actions = self.game_env.get_legal_action_indices()
                    if not legal_actions: 
                        print("Error: No legal actions and MCTS failed to identify terminal state.")
                        break 
                    action = random.choice(legal_actions)
                else:
                    action_probs_normalized = action_probs_from_mcts / np.sum(action_probs_from_mcts)
                    action = np.random.choice(NUM_ACTIONS, p=action_probs_normalized)
                
                next_state_repr, reward, done, info = self.game_env.step(action)
                game_step_count += 1
                self.total_game_steps_for_temp += 1
                iteration_game_steps += 1  # Track steps for this iteration
                # Removed move_bar.update(1)

                # Ensure game terminates if it hits max steps
                if game_step_count >= MAX_STEPS_PER_EPISODE:
                    if not done:  # Only print warning if game wasn't already done for other reasons
                        print(f"Warning: Game {game_num+1} reached MAX_STEPS_PER_EPISODE ({MAX_STEPS_PER_EPISODE}). Ending game.")
                    done = True 
                
                if done:
                    # Removed move_bar.n and move_bar.refresh()
                    # Add detailed logging for game end reason
                    winner_id_for_log = self.game_env.winner_id
                    reason_for_done_log = "Unknown"
                    if game_step_count >= MAX_STEPS_PER_EPISODE:
                        reason_for_done_log = "Max Steps Reached"
                    elif winner_id_for_log == PLAYER_A_ID or winner_id_for_log == PLAYER_B_ID:
                        reason_for_done_log = f"Player {winner_id_for_log} won"
                    elif winner_id_for_log == 3: # Draw
                        reason_for_done_log = "Draw (Stalemate)"
                    else: # If done is true but no specific win/draw/max_steps, could be an issue
                        reason_for_done_log = "Done (Other/Unexpected)"
                    break
            
            winner = self.game_env.winner_id # Get final winner status
            if winner == first_player_of_game:
                iteration_game_outcomes["win"] += 1
            elif winner == 0 or winner == 3: # 0 for no winner yet (should not happen if done), 3 for draw
                iteration_game_outcomes["draw"] += 1
            else:
                iteration_game_outcomes["loss"] += 1
            
            # --- BEGIN: Value Target Logging ---
            if game_num < 2: # Log for the first 2 games
                print(f"  [Value Target Log] Game: {game_num+1} ended. Winner: {winner}. First player was: {first_player_of_game}")
            # --- END: Value Target Logging ---

            for exp_idx, exp in enumerate(current_game_experiences):
                player_at_state = exp['player_at_state']
                value_target = 0.0
                if winner == player_at_state:
                    value_target = 1.0
                elif winner == 0 or winner == 3: # Draw or ongoing (should be resolved by now)
                    value_target = 0.0 
                else: 
                    value_target = -1.0
                
                # --- BEGIN: Value Target Assignment Logging ---
                if game_num < 2 and exp_idx < 5 : # Log for first 5 experiences of first 2 games
                    print(f"    [Value Target Assign Log] Game: {game_num+1}, Exp: {exp_idx}, Player at state: {player_at_state}, Assigned Value: {value_target:.1f}")
                # --- END: Value Target Assignment Logging ---
                
                self.replay_buffer.append((exp['state'], exp['policy_target'], value_target))
                new_experiences_count +=1

            games_played_this_iteration += 1

        print(f"Self-play finished. Added {new_experiences_count} experiences to replay buffer. Buffer size: {len(self.replay_buffer)}")  # Added buffer size here
        if self.wandb_enabled:
            wandb.log({
                "self_play/total_games_generated": games_played_this_iteration,
                "self_play/new_experiences": new_experiences_count,
                "self_play/buffer_size": len(self.replay_buffer),
                "self_play/wins_first_player": iteration_game_outcomes["win"],
                "self_play/losses_first_player": iteration_game_outcomes["loss"],
                "self_play/draws": iteration_game_outcomes["draw"],
                "self_play/avg_game_steps": iteration_game_steps / games_played_this_iteration if games_played_this_iteration > 0 else 0,
                "self_play/current_temperature": get_curriculum_aware_temperature(self.total_game_steps_for_temp, self.current_iteration),
                "self_play/base_temperature": get_temperature(self.total_game_steps_for_temp),
                "curriculum/current_phase": get_current_curriculum_phase(self.current_iteration)[0] if get_current_curriculum_phase(self.current_iteration)[0] else "none"
            })

    @tf.function
    def get_gradients_and_norm(self, state_batch, policy_targets, value_targets):
        with tf.GradientTape() as tape:
            # Use the model directly for the forward pass
            policy_output, value_output = self.candidate_nn.model(state_batch, training=True)
            
            # Ensure outputs are float32 for loss calculation if mixed precision is used
            policy_output = tf.cast(policy_output, tf.float32)
            value_output = tf.cast(value_output, tf.float32)
            policy_targets = tf.cast(policy_targets, tf.float32) # These are already float32 from caller
            value_targets = tf.cast(value_targets, tf.float32)   # These are already float32 from caller

            policy_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(policy_targets, policy_output))
            value_loss = tf.reduce_mean(tf.square(value_targets - value_output))
            total_loss = policy_loss + value_loss
            
            # Apply L2 regularization
            l2_loss = sum(self.candidate_nn.model.losses) # Access losses from the model
            total_loss += l2_loss

        gradients = tape.gradient(total_loss, self.candidate_nn.model.trainable_variables) # Get gradients from the model
        grad_norm = tf.linalg.global_norm(gradients)
        return gradients, grad_norm, total_loss, policy_loss, value_loss

    def _train_network(self):
        if AZ_TRAIN_ON_STATIC_DATASET: # Step 1.3: Load static dataset if enabled
            if not os.path.exists(AZ_STATIC_DATASET_PATH):
                print(f"ERROR: Static dataset not found at {AZ_STATIC_DATASET_PATH}. Cannot train.")
                print("Please generate it first by setting AZ_GENERATE_STATIC_DATASET=True in config.py")
                return
            print(f"Loading static dataset from {AZ_STATIC_DATASET_PATH}...")
            with open(AZ_STATIC_DATASET_PATH, 'rb') as f:
                static_dataset = pickle.load(f)
            
            print(f"Loaded {len(static_dataset)} samples. Training for {AZ_STATIC_TRAINING_EPOCHS} epochs.")
            
            # Shuffle the loaded dataset
            random.shuffle(static_dataset)

            all_states = np.array([data[0] for data in static_dataset])
            all_policy_targets = np.array([data[1] for data in static_dataset])
            all_value_targets = np.array([data[2] for data in static_dataset]).reshape(-1, 1)

            # --- BEGIN: Step 1.2 Data Inspection (for static data) ---
            print("\\n[Static Dataset - Data Inspection] First 3 samples from loaded dataset:")
            for i in range(min(3, len(static_dataset))):
                print(f"  Sample {i+1}:")
                print(f"    State shape: {all_states[i].shape}, State (first 10 flat): {all_states[i].flatten()[:10]}")
                print(f"    Policy Target shape: {all_policy_targets[i].shape}, Sum: {np.sum(all_policy_targets[i]):.4f}, Max: {np.max(all_policy_targets[i]):.4f}, Argmax: {np.argmax(all_policy_targets[i])}")
                print(f"    Value Target: {all_value_targets[i][0]:.4f}")
            # --- END: Step 1.2 Data Inspection ---

            for epoch in range(AZ_STATIC_TRAINING_EPOCHS):
                history = self.candidate_nn.model.fit(
                    all_states,
                    {'policy_output': all_policy_targets, 'value_output': all_value_targets},
                    batch_size=AZ_BATCH_SIZE,
                    epochs=1, # Train one epoch at a time to log
                    shuffle=True, # Shuffle data at each epoch
                    verbose=0 # Suppress Keras's own progress bar
                )
                loss_info = history.history
                avg_loss = np.mean(loss_info['loss'])
                avg_policy_loss = np.mean(loss_info['policy_output_loss'])
                avg_value_loss = np.mean(loss_info['value_output_loss'])
                print(f"Epoch {epoch+1}/{AZ_STATIC_TRAINING_EPOCHS}: loss={avg_loss:.4f}, policy_loss={avg_policy_loss:.4f}, value_loss={avg_value_loss:.4f}")
                if self.wandb_enabled:
                    wandb.log({
                        "static_train/epoch": epoch + 1,
                        "static_train/total_loss": avg_loss,
                        "static_train/policy_loss": avg_policy_loss,
                        "static_train/value_loss": avg_value_loss,
                        # Correctly access the learning rate
                        "static_train/learning_rate": self.candidate_nn.model.optimizer.learning_rate.numpy() if hasattr(self.candidate_nn.model.optimizer.learning_rate, 'numpy') else self.candidate_nn.model.optimizer.learning_rate
                    })

            print("Static dataset training finished.")
            # Save the model trained on static data
            static_trained_model_path = "switcharoo_az_static_trained.weights.h5"
            self.candidate_nn.save_model(static_trained_model_path)
            print(f"Model trained on static data saved to {static_trained_model_path}")
            return # End training after static training

        if len(self.replay_buffer) < AZ_BATCH_SIZE:
            print("Replay buffer too small to train. Skipping training phase.")
            if self.wandb_enabled:
                # Log current learning rate even if skipping
                try:
                    # Get current learning rate using the scheduler with current optimizer step
                    current_step = tf.cast(self.candidate_nn.model.optimizer.iterations, tf.float32)
                    current_lr = self.lr_scheduler(current_step).numpy()
                except Exception as e:
                    current_lr = AZ_LEARNING_RATE
                    print(f"Could not get current learning rate: {e}, using config value instead.")
                wandb.log({"training/skipped": True, "training/total_loss": 0, "training/policy_loss":0, "training/value_loss":0, "training/learning_rate": current_lr})
            return

        print(f"\n--- Training Neural Network for {AZ_TRAINING_STEPS_PER_ITERATION} Steps ---")
        total_loss_this_iteration = 0
        policy_loss_this_iteration = 0
        value_loss_this_iteration = 0
        policy_accuracy_this_iteration = 0
        value_mae_this_iteration = 0
        
        # Track gradient norms for debugging
        gradient_norms = []
        nan_loss_count = 0
        # Step 1.4: Track activation statistics
        activation_stats = {layer_name: [] for layer_name in self.candidate_nn.get_activation_layer_names()}


        # Diagnostics: before training
        print("\n[Diagnostics] Before training:")
        self.print_sample_predictions(n=2)
        self.print_value_head_stats(n=20)
        self.print_policy_entropy_stats(n=20)
        self.print_replay_buffer_targets(n=5)
        
        # CRUCIAL: Test if model can overfit on a single batch
        if AZ_OVERFIT_TEST_REQUIRED: # Check flag before running
            print("\n[OVERFIT TEST] Testing model's ability to learn...")
            self.overfit_on_small_batch(steps=100) # Reduced steps for quicker check
        else:
            print("\n[OVERFIT TEST] Skipped as per AZ_OVERFIT_TEST_REQUIRED=False.")

        for train_step in tqdm(range(AZ_TRAINING_STEPS_PER_ITERATION), desc="Training Steps"):
            if len(self.replay_buffer) < AZ_BATCH_SIZE:
                print("\nBuffer too small, breaking training early.")  # Added a print for early break
                break 
            minibatch = random.sample(list(self.replay_buffer), AZ_BATCH_SIZE)
            batch_states = np.array([data[0] for data in minibatch])
            batch_policy_targets = np.array([data[1] for data in minibatch])
            batch_value_targets = np.array([data[2] for data in minibatch]).reshape(-1, 1)
            
            # Check for NaN or Inf values in input data
            has_invalid_data = (
                np.any(np.isnan(batch_states)) or 
                np.any(np.isnan(batch_policy_targets)) or 
                np.any(np.isnan(batch_value_targets)) or
                np.any(np.isinf(batch_states)) or 
                np.any(np.isinf(batch_policy_targets)) or 
                np.any(np.isinf(batch_value_targets))
            )
            
            if has_invalid_data:
                print(f"Warning: Invalid data detected in batch {train_step}. Skipping this batch.")
                continue

            # Normalize policy targets to ensure they sum to 1.0
            # Sometimes numerical precision can cause softmax outputs to not sum exactly to 1
            sum_policy = np.sum(batch_policy_targets, axis=1, keepdims=True)
            valid_policy_sums = sum_policy > 1e-10  # Filter out all-zero policies
            batch_policy_targets = np.where(
                valid_policy_sums, 
                batch_policy_targets / np.maximum(sum_policy, 1e-10),  # Normalize to sum to 1
                batch_policy_targets
            )
            
            # Monitor gradients every 100 steps (to avoid slowing down training)
            if train_step % 100 == 0:
                try:
                    # Convert numpy arrays to tensors
                    state_tensor = tf.convert_to_tensor(batch_states, dtype=tf.float32)
                    policy_tensor = tf.convert_to_tensor(batch_policy_targets, dtype=tf.float32)
                    value_tensor = tf.convert_to_tensor(batch_value_targets, dtype=tf.float32)
                    
                    # Get gradients and norm
                    gradients, grad_norm, _, _, _ = self.get_gradients_and_norm(state_tensor, policy_tensor, value_tensor)
                    gradient_norms.append(grad_norm.numpy())

                    # Step 1.4: Log activation statistics
                    if AZ_LOG_ACTIVATION_STATS:
                        activations = self.candidate_nn.get_activations(state_tensor)
                        for layer_name, activation_value in activations.items():
                            activation_stats[layer_name].append({
                                'mean': tf.reduce_mean(activation_value).numpy(),
                                'std': tf.math.reduce_std(activation_value).numpy(),
                                'min': tf.reduce_min(activation_value).numpy(),
                                'max': tf.reduce_max(activation_value).numpy(),
                                'abs_mean': tf.reduce_mean(tf.abs(activation_value)).numpy()
                            })
                except TypeError as te:
                    print(f"Error calculating gradient norm: {te}")
                    # Optionally, log the traceback for more details
                    # traceback.print_exc()
                except Exception as e:
                    print(f"An unexpected error occurred during gradient norm calculation: {e}")
                    # traceback.print_exc()
            
            # Perform the training step
            # The train_step method is part of the Keras Model API, not custom
            loss_info = self.candidate_nn.model.train_on_batch(
                batch_states, 
                {'policy_output': batch_policy_targets, 'value_output': batch_value_targets},
                return_dict=True
            )
            # Keras train_on_batch returns a dict with loss and metrics if return_dict=True
            # The keys depend on what was compiled. Default is 'loss', 'policy_output_loss', 'value_output_loss'
            # and metrics like 'policy_output_accuracy', 'value_output_mae'
            total_loss = loss_info.get('loss', 0) # Default to 'loss' for total loss
            policy_loss = loss_info.get('policy_output_loss', 0) # Or the specific name of policy loss
            value_loss = loss_info.get('value_output_loss', 0)   # Or the specific name of value loss
            policy_accuracy = loss_info.get('policy_output_accuracy', 0) # Or specific name
            value_mae = loss_info.get('value_output_mae', 0) # Or specific name

            # Check for NaN in loss metrics
            if np.isnan(total_loss):
                print(f"Warning: NaN loss detected at step {train_step}. Model may be diverging.")
                nan_loss_count += 1
                
                # If we get too many NaN losses, stop training
                if nan_loss_count > 10:
                    print("Too many NaN losses detected. Stopping training.")
                    break
                continue  # Skip this batch for metrics accumulation
            
            total_loss_this_iteration += total_loss
            policy_loss_this_iteration += policy_loss
            value_loss_this_iteration += value_loss
            policy_accuracy_this_iteration += policy_accuracy
            value_mae_this_iteration += value_mae

        # Calculate effective number of steps for averaging
        # train_step is the index of the last step run or attempted in the loop.
        # If loop completed, train_step = AZ_TRAINING_STEPS_PER_ITERATION - 1.
        # If loop broke early, train_step is the index of the step it broke on.
        # Number of iterations for averaging = train_step + 1, if loop ran at least once.
        num_steps_processed = 0
        if 'train_step' in locals() and train_step >= 0 : # Check if loop ran at least one iteration
            num_steps_processed = train_step + 1
            if nan_loss_count >= 10: # If training stopped due to NaNs
                 # Only count steps before NaN limit was hit.
                 # train_step is the step where the 10th NaN occurred or loop ended.
                 # The actual number of steps that contributed to sums is train_step (0-indexed)
                 # if the break was due to NaNs.
                 # However, the current logic accumulates losses even for NaN steps before breaking.
                 # For simplicity, we use train_step + 1, assuming it reflects iterations attempted.
                 pass # num_steps_processed is already train_step + 1

        avg_total_loss = total_loss_this_iteration / num_steps_processed if num_steps_processed > 0 else 0
        avg_policy_loss = policy_loss_this_iteration / num_steps_processed if num_steps_processed > 0 else 0
        avg_value_loss = value_loss_this_iteration / num_steps_processed if num_steps_processed > 0 else 0
        avg_policy_accuracy = policy_accuracy_this_iteration / num_steps_processed if num_steps_processed > 0 else 0
        avg_value_mae = value_mae_this_iteration / num_steps_processed if num_steps_processed > 0 else 0
        
        current_lr_val = 0.0
        # Robustly get the current learning rate for logging
        try:
            # Try if learning_rate is a callable (schedule)
            lr_obj = self.candidate_nn.model.optimizer.learning_rate
            step = self.candidate_nn.model.optimizer.iterations
            if callable(lr_obj):
                current_lr_val = lr_obj(step).numpy()
            elif hasattr(lr_obj, 'numpy'):
                current_lr_val = lr_obj.numpy()
            else:
                current_lr_val = float(lr_obj)
        except Exception as e:
            try:
                # Try to get from the scheduler directly
                step = tf.cast(self.candidate_nn.model.optimizer.iterations, tf.float32)
                current_lr_val = self.lr_scheduler(step).numpy()
            except Exception as e2:
                current_lr_val = AZ_LEARNING_RATE
        
        print(f"Network training finished. Avg Total Loss: {avg_total_loss:.4f} (P: {avg_policy_loss:.4f}, V: {avg_value_loss:.4f}, P_Acc: {avg_policy_accuracy:.4f}, V_MAE: {avg_value_mae:.4f}), LR: {current_lr_val:.2e}")

        # Diagnostics: after training
        print("\n[Diagnostics] After training:")
        self.print_sample_predictions(n=2)
        self.print_value_head_stats(n=20)
        self.print_policy_entropy_stats(n=20)

        if self.wandb_enabled:
            wandb.log({
                "train/total_loss": total_loss_this_iteration / AZ_TRAINING_STEPS_PER_ITERATION,
                "train/policy_loss": policy_loss_this_iteration / AZ_TRAINING_STEPS_PER_ITERATION,
                "train/value_loss": value_loss_this_iteration / AZ_TRAINING_STEPS_PER_ITERATION,
                "train/policy_accuracy": policy_accuracy_this_iteration / AZ_TRAINING_STEPS_PER_ITERATION,
                "train/value_mae": value_mae_this_iteration / AZ_TRAINING_STEPS_PER_ITERATION,
                "train/learning_rate": current_lr_val, # Log current LR
                "train/gradient_norm_avg": np.mean(gradient_norms) if gradient_norms else 0,
                "train/gradient_norm_max": np.max(gradient_norms) if gradient_norms else 0,
                "train/gradient_norm_min": np.min(gradient_norms) if gradient_norms else 0,
                "train/nan_loss_count": nan_loss_count
            })
            # Step 1.4: Log aggregated activation statistics to WandB
            if AZ_LOG_ACTIVATION_STATS:
                for layer_name, stats_list in activation_stats.items():
                    if stats_list: # Only log if we collected stats
                        wandb.log({
                            f"activations/{layer_name}/mean_avg": np.mean([s['mean'] for s in stats_list]),
                            f"activations/{layer_name}/std_avg": np.mean([s['std'] for s in stats_list]),
                            f"activations/{layer_name}/min_avg": np.mean([s['min'] for s in stats_list]),
                            f"activations/{layer_name}/max_avg": np.mean([s['max'] for s in stats_list]),
                            f"activations/{layer_name}/abs_mean_avg": np.mean([s['abs_mean'] for s in stats_list]),
                        })

    def _evaluate_and_update_model(self, iteration_num):
        if iteration_num == 0: 
             print("\n--- Skipping Model Evaluation (First Iteration) ---")
             if not os.path.exists(AZ_BEST_MODEL_FILE):
                print(f"No best model exists. Saving current candidate_nn to {AZ_BEST_MODEL_FILE}")
                self.candidate_nn.save_model(AZ_BEST_MODEL_FILE)
                self.current_nn = AlphaZeroNetwork(learning_rate_schedule=self.lr_scheduler) # Ensure new current_nn also has scheduler
                self.current_nn.load_model(AZ_BEST_MODEL_FILE)
             if self.wandb_enabled:
                wandb.log({"evaluation/skipped": True, "evaluation/candidate_win_rate": 0})
             return

        print(f"\n--- Evaluating Candidate Model (Iteration {iteration_num}) ---")
        self.candidate_nn.save_model(AZ_CANDIDATE_MODEL_FILE)
        
        # --- Add RAM usage logging before evaluation ---
        process = psutil.Process(os.getpid())
        print(f"[Eval] RAM before evaluation: {process.memory_info().rss / (1024**3):.2f} GB")

        eval_candidate_nn = AlphaZeroNetwork(learning_rate_schedule=self.lr_scheduler) # Pass scheduler
        eval_candidate_nn.load_model(AZ_CANDIDATE_MODEL_FILE)
        
        eval_best_nn = AlphaZeroNetwork(learning_rate_schedule=self.lr_scheduler) # Pass scheduler
        if os.path.exists(AZ_BEST_MODEL_FILE):
            try:
                eval_best_nn.load_model(AZ_BEST_MODEL_FILE)
            except ValueError as e:
                print(f"Error loading best model for evaluation: {e}")
                print(f"Model architecture incompatible. Candidate becomes best without evaluation.")
                self.candidate_nn.save_model(AZ_BEST_MODEL_FILE)
                self.current_nn = AlphaZeroNetwork(learning_rate_schedule=self.lr_scheduler) # Pass scheduler
                self.current_nn.load_model(AZ_BEST_MODEL_FILE)
                if self.wandb_enabled:
                    wandb.log({"evaluation/skipped": True, "evaluation/candidate_win_rate": 1.0, "evaluation/best_model_updated": True})
                return
        else:
            print(f"No existing best model found at {AZ_BEST_MODEL_FILE} for evaluation. Candidate becomes best.")
            self.candidate_nn.save_model(AZ_BEST_MODEL_FILE)
            self.current_nn = AlphaZeroNetwork(learning_rate_schedule=self.lr_scheduler) # Pass scheduler
            self.current_nn.load_model(AZ_BEST_MODEL_FILE)
            if self.wandb_enabled:
                wandb.log({"evaluation/skipped": True, "evaluation/candidate_win_rate": 1.0, "evaluation/best_model_updated": True})
            return

        candidate_wins = 0
        best_wins = 0
        draws = 0
        eval_temperature = 0.01  # Further reduced for even more deterministic evaluation play

        for game_idx in tqdm(range(AZ_EVALUATION_GAMES_COUNT), desc="Evaluation Games"):  # Added tqdm
            eval_env = self.game_env_class()
            
            # Use curriculum-based position selection for evaluation
            if AZ_PROGRESSIVE_CURRICULUM:
                curriculum_positions = get_curriculum_positions(self.current_iteration)
                starting_position_str = select_weighted_position(curriculum_positions, self.current_iteration)
            else:
                # Original random selection
                starting_position_str = random.choice(initial_position)
            # print(f"Starting Evaluation Game {game_idx+1}/{AZ_EVALUATION_GAMES_COUNT} with starting position:\n{starting_position_str}")
            eval_env.reset(starting_position=starting_position_str)
            done = False
            game_step_count_eval = 0  # Initialize step counter for evaluation game
            if game_idx % 2 == 0:
                player_A_model, player_B_model = eval_candidate_nn, eval_best_nn
                player_A_name, player_B_name = "Candidate", "Best"
            else:
                player_A_model, player_B_model = eval_best_nn, eval_candidate_nn
                player_A_name, player_B_name = "Best", "Candidate"
            current_models = {PLAYER_A_ID: player_A_model, PLAYER_B_ID: player_B_model}
            current_names = {PLAYER_A_ID: player_A_name, PLAYER_B_ID: player_B_name}

            while not done:
                current_player_id_eval = eval_env.current_player_id
                active_model = current_models[current_player_id_eval]
                if USE_NUMBA:
                    # print("Using Numba-optimized MCTS implementation.")
                    mcts_eval = MCTSNumba(
                        neural_network=active_model,
                        game_env_class=self.game_env_class,
                        config=self.mcts_config
                    )
                    # Get current state representation for NN prediction
                    current_state_repr_eval = eval_env._get_state() # (8, 4, 18) for NN input
                    # Do NOT flatten; pass as (8, 4, 18)
                    nn_policy_eval, nn_value_eval = active_model.predict(current_state_repr_eval)
                    
                    mcts_eval.run_mcts_simulations(
                        np.ascontiguousarray(eval_env.board), # start_board_state_arr
                        current_player_id_eval,   # start_player_id
                        nn_policy_eval,           # nn_policy_values
                        nn_value_eval,            # nn_value_estimate
                        NUM_SIMULATIONS_PER_MOVE # num_simulations
                    )
                    action_probs_eval = mcts_eval.get_action_probabilities(
                        np.ascontiguousarray(eval_env.board),
                        current_player_id_eval,
                        eval_temperature
                    )
                else:
                    # print("Using Python MCTS implementation.")
                    root_eval = MCTSNode(parent=None, prior_p=0.0, player_id=current_player_id_eval)
                    mcts_eval = MCTS(
                        neural_network=active_model,
                        game_env_class=self.game_env_class,
                        config=self.mcts_config
                    )
                    mcts_eval.run_mcts_simulations(
                        root_eval,
                        np.copy(eval_env.board),
                        current_player_id_eval
                    )
                    action_probs_eval = mcts_eval.get_action_probabilities(
                        root_eval,
                        eval_temperature
                    )                    
                if np.sum(action_probs_eval) == 0:
                    legal_actions_eval = eval_env.get_legal_action_indices()
                    if not legal_actions_eval: 
                        break 
                    action_eval = random.choice(legal_actions_eval)
                else:
                    action_probs_eval_norm = action_probs_eval / np.sum(action_probs_eval)
                    action_eval = np.random.choice(NUM_ACTIONS, p=action_probs_eval_norm)
                
                _, _, done, _ = eval_env.step(action_eval)
                game_step_count_eval += 1

                if game_step_count_eval >= MAX_STEPS_PER_EPISODE:
                    if not done:
                        print(f"Warning: Eval Game {game_idx+1} reached MAX_STEPS_PER_EPISODE ({MAX_STEPS_PER_EPISODE}). Ending game.")
                    done = True
                if game_step_count_eval < args.log_first_n:
                    start_r, start_c, end_r, end_c = eval_env._action_index_to_move(action_eval)
                    print(f"Player {current_player_id_eval} ({current_names[current_player_id_eval]}) took action ({start_r}, {start_c}) -> ({end_r}, {end_c})")
                    eval_env.render()
            
            game_winner_id = eval_env.winner_id
            # --- Draw logic fix ---
            is_draw = False
            if game_winner_id == 3:
                is_draw = True
            elif game_winner_id is None or game_winner_id == 0:
                is_draw = True
            elif game_step_count_eval >= MAX_STEPS_PER_EPISODE:
                is_draw = True
            if is_draw:
                draws += 1
            elif (player_A_model == eval_candidate_nn and game_winner_id == PLAYER_A_ID) or \
                 (player_B_model == eval_candidate_nn and game_winner_id == PLAYER_B_ID):
                candidate_wins += 1
            else:
                best_wins += 1
            # --- End draw logic fix ---
            print(f"  Eval Game {game_idx+1}/{AZ_EVALUATION_GAMES_COUNT}: Candidate ({player_A_name if player_A_model==eval_candidate_nn else player_B_name}) vs Best ({player_A_name if player_A_model==eval_best_nn else player_B_name}). Winner: {eval_env.winner}")

        # Print/check sum of outcomes
        total_outcomes = candidate_wins + best_wins + draws

        win_rate = 0
        if (AZ_EVALUATION_GAMES_COUNT - draws) > 0:
            win_rate = candidate_wins / (AZ_EVALUATION_GAMES_COUNT - draws)

        print(f"Evaluation Results: Candidate Wins: {candidate_wins}, Best Wins: {best_wins}, Draws: {draws}")
        print(f"Candidate Win Rate (vs Best, excluding draws): {win_rate:.2%}")
        
        # --- Add RAM usage logging after evaluation ---
        print(f"[Eval] RAM after evaluation: {process.memory_info().rss / (1024**3):.2f} GB")

        # --- Explicitly delete evaluation models and run garbage collection ---
        del eval_candidate_nn
        del eval_best_nn
        gc.collect()
        tf.keras.backend.clear_session()
        print("[Eval] Cleared eval models and ran garbage collection.")

        best_model_updated = False
        if win_rate > AZ_MODEL_UPDATE_WIN_RATE:
            print(f"Candidate model is better (Win Rate {win_rate:.2%} > {AZ_MODEL_UPDATE_WIN_RATE:.2%}). Updating best model.")
            self.candidate_nn.save_model(AZ_BEST_MODEL_FILE)
            self.current_nn = AlphaZeroNetwork(learning_rate_schedule=self.lr_scheduler) # Pass scheduler
            self.current_nn.load_model(AZ_BEST_MODEL_FILE)
            best_model_updated = True
        else:
            print("Candidate model is not significantly better. Keeping previous best model.")
            best_model_updated = False
        
        if self.wandb_enabled:
            wandb.log({
                "evaluation/candidate_wins": candidate_wins,
                "evaluation/best_wins": best_wins,
                "evaluation/draws": draws,
                "evaluation/candidate_win_rate": win_rate,
                "evaluation/best_model_updated": best_model_updated,
                "evaluation/skipped": False
            })

    def train(self, start_iteration=1):
        print("Starting AlphaZero Training Process...")
        
        if start_iteration > 1:
            print(f"🔄 RESUMING training from iteration {start_iteration}")
        else:
            print("🚀 STARTING training from iteration 1")

        # --- Step 1.3: Static Dataset Generation / Training ---
        if AZ_GENERATE_STATIC_DATASET:
            print(f"Generating static dataset with {AZ_STATIC_DATASET_GAMES} games...")
            self._run_self_play_games(num_games_to_generate=AZ_STATIC_DATASET_GAMES) 
            
            static_dataset_to_save = list(self.replay_buffer) 
            with open(AZ_STATIC_DATASET_PATH, 'wb') as f:
                pickle.dump(static_dataset_to_save, f)
            print(f"Static dataset with {len(static_dataset_to_save)} samples saved to {AZ_STATIC_DATASET_PATH}")
            print("Exiting after dataset generation.")
            return 
        
        if AZ_TRAIN_ON_STATIC_DATASET:
            self._train_network() 
            print("Exiting after training on static dataset.")
            return
        # --- End Step 1.3 ---

        for iteration in range(start_iteration, AZ_ITERATIONS + 1):
            # Update current iteration for curriculum tracking
            self.current_iteration = iteration
            
            print(f"\n===== Iteration {iteration}/{AZ_ITERATIONS} =====")
            
            # Log curriculum phase information
            if AZ_PROGRESSIVE_CURRICULUM and AZ_CURRICULUM_LOGGING:
                phase_name, phase_config = get_current_curriculum_phase(iteration)
                if phase_config:
                    print(f"🎯 Curriculum Phase: {phase_name}")
                    print(f"   Description: {phase_config['description']}")
                    print(f"   Target Policy Accuracy: {phase_config['target_policy_accuracy']:.1%}")
                    print(f"   Available Positions: {len(phase_config['positions'])}")
            
            start_time_iter = time.time()
            # Log GPU memory usage if available
            try:
                gpu_devices = tf.config.list_physical_devices('GPU')
                if gpu_devices:
                    memory_info = {}
                    for device in gpu_devices:
                        device_idx = device.name.split(':')[-1]
                        memory_stats = tf.config.experimental.get_memory_info(f'/device:GPU:{device_idx}')
                        if memory_stats:
                            memory_info[f'gpu_{device_idx}_memory_used_gb'] = memory_stats['current'] / (1024**3)
                            memory_info[f'gpu_{device_idx}_memory_limit_gb'] = memory_stats['peak'] / (1024**3)
                    if memory_info and self.wandb_enabled:
                        wandb.log(memory_info)
                    print(f"GPU Memory Usage: {memory_info}")
            except Exception as e:
                print(f"Note: Could not log GPU memory usage: {e}")
            
            self._run_self_play_games()
            self._train_network()
            self._evaluate_and_update_model(iteration)
            # Enhanced selective checkpoint saving
            # Save checkpoints at more frequent intervals early in training
            if (iteration < 20 and iteration % 5 == 0) or \
               (iteration >= 20 and iteration < 50 and iteration % 10 == 0) or \
               (iteration >= 50 and iteration % 20 == 0) or \
               (iteration > AZ_ITERATIONS - 10) or \
               (iteration == 0):
                checkpoint_path = AZ_CHECKPOINT_FILE_PATTERN.format(iteration + 1)
                self.candidate_nn.save_model(checkpoint_path)
                print(f"Saved candidate_nn checkpoint to {checkpoint_path}")
            else:
                print(f"Skipping checkpoint save for iteration {iteration + 1} (using progressive saving strategy)")
            end_time_iter = time.time()
            iter_duration = end_time_iter - start_time_iter
            print(f"Iteration {iteration} completed in {iter_duration:.2f} seconds.")
            if self.wandb_enabled:
                wandb.log({"iteration": iteration, "iteration_duration_sec": iter_duration})

        print("\nAlphaZero Training Finished.")
        if self.wandb_enabled:
            wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run AlphaZero Trainer for Switcharoo.")
    parser.add_argument('--no-wandb', action='store_true', help="Disable Weights & Biases logging.")
    parser.add_argument('--debug-run', action='store_true', help="Run a short debug training session")
    parser.add_argument('--reduce-lr', type=float, help="Override learning rate with a lower value")
    parser.add_argument('--disable-xla', action='store_true', help="Disable XLA JIT compilation")
    parser.add_argument('--diagnostics', action='store_true', help="Enable extra diagnostics and debugging output.")
    parser.add_argument('--start-iteration', type=int, default=1, help="Starting iteration number for training (default: 1)")
    parser.add_argument('--log-first-n', type=int, default=0, help="Log the first N items for debugging/inspection (default: 0)")
    
    args = parser.parse_args()

    # Enable TensorFlow debug logging for numerical issues
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Show warnings

    # --- Force XLA off via environment variable as well as tf.config ---
    os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
    tf.config.optimizer.set_jit(False)
    print("XLA JIT compilation explicitly disabled to prevent slow compilation and potential OOM issues.")
    
    # Set shorter training parameters for debug runs
    if args.debug_run:
        AZ_ITERATIONS = 4
        AZ_GAMES_PER_ITERATION = 10
        AZ_TRAINING_STEPS_PER_ITERATION = 10
        AZ_EVALUATION_GAMES_COUNT = 10
        AZ_LEARNING_RATE = 1e-3  # <--- Add this line for debug runs
        print(f"Running in debug mode with reduced parameters:"
              f"\n- AZ_ITERATIONS: {AZ_ITERATIONS}"
              f"\n- AZ_GAMES_PER_ITERATION: {AZ_GAMES_PER_ITERATION}"
              f"\n- AZ_TRAINING_STEPS_PER_ITERATION: {AZ_TRAINING_STEPS_PER_ITERATION}"
              f"\n- AZ_EVALUATION_GAMES_COUNT: {AZ_EVALUATION_GAMES_COUNT}"
              f"\n- AZ_LEARNING_RATE: {AZ_LEARNING_RATE}")
    
    # Override learning rate if specified (useful for handling NaN losses)
    if args.reduce_lr is not None and args.reduce_lr > 0:
        original_lr = AZ_LEARNING_RATE
        AZ_LEARNING_RATE = args.reduce_lr
        print(f"Overriding learning rate: {original_lr} → {AZ_LEARNING_RATE}")
    
    # Enable mixed precision training for better GPU utilization and reduced register usage
    # try:
    #     policy = tf.keras.mixed_precision.Policy('mixed_float16')
    #     tf.keras.mixed_precision.set_global_policy(policy)
    #     print("Mixed precision training enabled (float16/float32)")
    # except Exception as e:
    #     print(f"Could not enable mixed precision training: {e}")
    print("Mixed precision training *disabled* for debugging NaN issues.")

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Running on GPU: {gpus}")
        except RuntimeError as e:
            print(e)
    else:
        print("Running on CPU")

    trainer = AlphaZeroTrainer(use_wandb=not args.no_wandb, use_jit_env=True)  # Pass the flag to the constructor

    # Fill replay buffer before overfit test, then proceed to main training loop
    if args.debug_run and args.diagnostics:
        import config as az_config  # Import config to access and modify AZ_BATCH_SIZE and AZ_LEARNING_RATE
        orig_batch_size = az_config.AZ_BATCH_SIZE
        orig_lr = az_config.AZ_LEARNING_RATE
        az_config.AZ_BATCH_SIZE = 4
        az_config.AZ_LEARNING_RATE = 0.01
        print(f"[Overfit Test] Temporarily set AZ_BATCH_SIZE=4 and AZ_LEARNING_RATE=0.01 for overfit test.")
        trainer._run_self_play_games()
        if len(trainer.replay_buffer) >= az_config.AZ_BATCH_SIZE:
            trainer.overfit_on_small_batch(steps=500)
        else:
            print("[Overfit Test] Not enough samples in replay buffer to run overfit test.")
        az_config.AZ_BATCH_SIZE = orig_batch_size
        az_config.AZ_LEARNING_RATE = orig_lr
        print(f"[Overfit Test] Restored AZ_BATCH_SIZE and AZ_LEARNING_RATE to original values.")

    # Now run the main training loop (which will fill the buffer again as needed)
    # Use command line argument for start iteration (default 1)
    
    trainer.train(start_iteration=args.start_iteration)
