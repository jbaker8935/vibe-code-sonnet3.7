import numpy as np
import tensorflow as tf
import collections
import time
import os
import random
import wandb
import argparse
import traceback
from tqdm import tqdm

from game_env import SwitcharooEnv
from az_network import AlphaZeroNetwork, build_alpha_zero_network
from mcts import MCTS, MCTSNode  # Your Python MCTS
from mcts_numba import MCTSNumba  # Your Numba MCTS
from config import (
    AZ_ITERATIONS, AZ_GAMES_PER_ITERATION, AZ_TRAINING_STEPS_PER_ITERATION,
    AZ_REPLAY_BUFFER_SIZE, AZ_BATCH_SIZE, AZ_EVALUATION_GAMES_COUNT,
    AZ_MODEL_UPDATE_WIN_RATE, AZ_BEST_MODEL_FILE, AZ_CANDIDATE_MODEL_FILE,
    AZ_CHECKPOINT_FILE_PATTERN, NUM_SIMULATIONS_PER_MOVE,
    TEMPERATURE_START, TEMPERATURE_END, TEMPERATURE_ANNEAL_STEPS,
    DIRICHLET_ALPHA, DIRICHLET_EPSILON, C_PUCT_CONSTANT,
    WANDB_PROJECT, WANDB_ENTITY, MAX_STEPS_PER_EPISODE, AZ_LEARNING_RATE,  # Added AZ_LEARNING_RATE
    initial_position,  # Added initial_position
    USE_NUMBA,  # Added USE_NUMBA
    AZ_POLICY_LOSS_WEIGHT, AZ_VALUE_LOSS_WEIGHT  # Added loss weights
)
from env_const import PLAYER_A_ID, PLAYER_B_ID, NUM_ACTIONS

# Helper for temperature annealing
def get_temperature(game_step, total_anneal_steps=TEMPERATURE_ANNEAL_STEPS, start_temp=TEMPERATURE_START, end_temp=TEMPERATURE_END):
    if total_anneal_steps == 0: # Avoid division by zero if annealing is instant
        return end_temp
    if game_step >= total_anneal_steps:
        return end_temp
    return start_temp - (start_temp - end_temp) * (game_step / total_anneal_steps)

class AlphaZeroTrainer:
    def __init__(self, use_wandb=True):  # Add use_wandb parameter
        print("Initializing AlphaZero Trainer...")
        self.game_env = SwitcharooEnv()

        # Cosine decay scheduler with warmup for smoother learning rate transitions
        # First create warmup schedule
        warmup_steps = 5000
        warmup_lr = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=AZ_LEARNING_RATE * 0.1,
            decay_steps=warmup_steps,
            end_learning_rate=AZ_LEARNING_RATE,
            power=1.0  # Linear warmup
        )
        
        # Then create main cosine decay schedule
        cosine_decay_lr = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=AZ_LEARNING_RATE,
            decay_steps=150000,
            alpha=0.05  # Minimum learning rate factor
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

        if os.path.exists(AZ_BEST_MODEL_FILE):
            print(f"Found existing model at {AZ_BEST_MODEL_FILE}, attempting to load...")
            try:
                # Re-initialize with scheduler before loading weights
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

    def _run_self_play_games(self):
        print(f"\n--- Generating {AZ_GAMES_PER_ITERATION} Self-Play Games ---")
        games_played_this_iteration = 0
        new_experiences_count = 0
        iteration_game_outcomes = {"win": 0, "loss": 0, "draw": 0}

        for game_num in range(AZ_GAMES_PER_ITERATION):
            current_game_experiences = [] 
            # Randomly select a starting position from config.initial_position
            starting_position_str = random.choice(initial_position)
            current_board_state_arr = self.game_env.reset(starting_position=starting_position_str)
            done = False
            game_step_count = 0
            first_player_of_game = self.game_env.current_player_id

            # Log temperature at the start of the game
            current_temp_at_game_start = get_temperature(self.total_game_steps_for_temp)

            with tqdm(total=MAX_STEPS_PER_EPISODE, desc=f"Game {game_num+1}/{AZ_GAMES_PER_ITERATION} Moves", position=1, leave=False) as move_bar:  # New inner tqdm
                while not done:
                    current_player_id = self.game_env.current_player_id
                    if USE_NUMBA:
                        # print("Using Numba-optimized MCTS implementation.")
                        mcts_handler = MCTSNumba(neural_network=self.current_nn, 
                                                 game_env_class=SwitcharooEnv, 
                                                 config=self.mcts_config)
                        mcts_handler.run_mcts_simulations(
                            NUM_SIMULATIONS_PER_MOVE,
                            np.copy(self.game_env.board),
                            current_player_id,
                            root_node_dummy=None
                        )
                        action_probs_from_mcts = mcts_handler.get_action_probabilities(
                            np.copy(self.game_env.board),
                            current_player_id,
                            get_temperature(self.total_game_steps_for_temp),
                            root_node_dummy=None
                        )
                    else:
                        # print("Using Python MCTS implementation.")
                        root_node = MCTSNode(parent=None, prior_p=0.0, player_id=current_player_id)
                        mcts_handler = MCTS(neural_network=self.current_nn, 
                                            game_env_class=SwitcharooEnv, 
                                            config=self.mcts_config)
                        mcts_handler.run_mcts_simulations(
                            root_node,
                            np.copy(self.game_env.board),
                            current_player_id
                        )
                        action_probs_from_mcts = mcts_handler.get_action_probabilities(
                            root_node,
                            get_temperature(self.total_game_steps_for_temp)
                        )
                    temp = get_temperature(self.total_game_steps_for_temp)
                    nn_input_state = self.game_env._get_state() 
                    current_game_experiences.append({'state': nn_input_state, 
                                                     'policy_target': action_probs_from_mcts, 
                                                     'player_at_state': current_player_id})
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
                    move_bar.update(1)  # Update inner progress bar

                    # Ensure game terminates if it hits max steps
                    if game_step_count >= MAX_STEPS_PER_EPISODE:
                        if not done:  # Only print warning if game wasn't already done for other reasons
                            print(f"Warning: Game {game_num+1} reached MAX_STEPS_PER_EPISODE ({MAX_STEPS_PER_EPISODE}). Ending game.")
                        done = True 
                    
                    if done:
                        move_bar.n = game_step_count  # Set final count for the progress bar to actual steps
                        move_bar.refresh()  # Refresh to show final count correctly
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
            else: # Loss for the first player
                iteration_game_outcomes["loss"] += 1
            
            for exp in current_game_experiences:
                player_at_state = exp['player_at_state']
                value_target = 0.0
                if winner == player_at_state:
                    value_target = 1.0
                elif winner == 0 or winner == 3:
                    value_target = 0.0
                else: 
                    value_target = -1.0
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
                "self_play/avg_game_steps": self.total_game_steps_for_temp / games_played_this_iteration if games_played_this_iteration > 0 else 0,
                "self_play/current_temperature": get_temperature(self.total_game_steps_for_temp) 
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

        for train_step in tqdm(range(AZ_TRAINING_STEPS_PER_ITERATION), desc="Training Steps"):  # Added tqdm
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
        try:
            # Try to get LR from the optimizer directly
            current_lr_val = self.candidate_nn.model.optimizer.learning_rate(self.candidate_nn.model.optimizer.iterations).numpy()
        except AttributeError: # Optimizer might not be Keras Adam or iterations not tracked that way
            try:
                current_lr_val = self.lr_scheduler(tf.cast(self.candidate_nn.model.optimizer.iterations, tf.float32)).numpy()
            except Exception: # Fallback if everything else fails
                 current_lr_val = AZ_LEARNING_RATE # Default to initial config if dynamic LR cannot be fetched
        except Exception: # Catch any other errors during LR fetching
            current_lr_val = AZ_LEARNING_RATE


        print(f"Network training finished. Avg Total Loss: {avg_total_loss:.4f} (P: {avg_policy_loss:.4f}, V: {avg_value_loss:.4f}, P_Acc: {avg_policy_accuracy:.4f}, V_MAE: {avg_value_mae:.4f}), LR: {current_lr_val:.2e}")

        if self.wandb_enabled:
            wandb_logs = {
                "training/total_loss": avg_total_loss,
                "training/policy_loss": avg_policy_loss,
                "training/value_loss": avg_value_loss,
                "training/policy_accuracy": avg_policy_accuracy,
                "training/value_mae": avg_value_mae,
                "training/learning_rate": current_lr_val,
                "training/skipped": False, # Reaching here means training wasn't fully skipped initially
                "training/nan_loss_count": nan_loss_count
            }
            if gradient_norms: 
                wandb_logs["training/avg_gradient_norm"] = np.mean(gradient_norms)
            wandb.log(wandb_logs)

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
            eval_env = SwitcharooEnv()
            # Randomly select a starting position from config.initial_position
            starting_position_str = random.choice(initial_position)
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
                        game_env_class=SwitcharooEnv,
                        config=self.mcts_config
                    )
                    mcts_eval.run_mcts_simulations(
                        NUM_SIMULATIONS_PER_MOVE,
                        np.copy(eval_env.board),
                        current_player_id_eval,
                        root_node_dummy=None
                    )
                    action_probs_eval = mcts_eval.get_action_probabilities(
                        np.copy(eval_env.board),
                        current_player_id_eval,
                        eval_temperature,
                        root_node_dummy=None
                    )
                else:
                    # print("Using Python MCTS implementation.")
                    root_eval = MCTSNode(parent=None, prior_p=0.0, player_id=current_player_id_eval)
                    mcts_eval = MCTS(
                        neural_network=active_model,
                        game_env_class=SwitcharooEnv,
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

    def train(self):
        print("Starting AlphaZero Training Process...")
        if self.wandb_enabled:
            pass

        for iteration in range(AZ_ITERATIONS):
            print(f"\n===== Iteration {iteration+1}/{AZ_ITERATIONS} =====")
            start_time_iter = time.time()
            # Log GPU memory usage if available
            try:
                if gpus:
                    memory_info = {}
                    for device in tf.config.list_physical_devices('GPU'):
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
            print(f"Iteration {iteration+1} completed in {iter_duration:.2f} seconds.")
            if self.wandb_enabled:
                wandb.log({"iteration": iteration + 1, "iteration_duration_sec": iter_duration})

        print("\nAlphaZero Training Finished.")
        if self.wandb_enabled:
            wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run AlphaZero Trainer for Switcharoo.")
    parser.add_argument('--no-wandb', action='store_true', help="Disable Weights & Biases logging.")
    parser.add_argument('--debug-run', action='store_true', help="Run a short debug training session")
    parser.add_argument('--reduce-lr', type=float, help="Override learning rate with a lower value")
    parser.add_argument('--disable-xla', action='store_true', help="Disable XLA JIT compilation")
    args = parser.parse_args()

    # Enable TensorFlow debug logging for numerical issues
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Show warnings

    # Disable XLA JIT compilation for now due to slow compilation and potential memory issues
    # if not args.disable_xla:
    #     tf.config.optimizer.set_jit(True)
    # else:
    #     tf.config.optimizer.set_jit(False) # Ensure it's off if arg is present
    tf.config.optimizer.set_jit(False)
    print("XLA JIT compilation explicitly disabled to prevent slow compilation and potential OOM issues.")
    
    # Set shorter training parameters for debug runs
    if args.debug_run:
        AZ_ITERATIONS = 2
        AZ_GAMES_PER_ITERATION = 5
        AZ_TRAINING_STEPS_PER_ITERATION = 50
        print(f"Running in debug mode with reduced parameters:"
              f"\n- AZ_ITERATIONS: {AZ_ITERATIONS}"
              f"\n- AZ_GAMES_PER_ITERATION: {AZ_GAMES_PER_ITERATION}"
              f"\n- AZ_TRAINING_STEPS_PER_ITERATION: {AZ_TRAINING_STEPS_PER_ITERATION}")
    
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

    trainer = AlphaZeroTrainer(use_wandb=not args.no_wandb)  # Pass the flag to the constructor
    trainer.train()
