import numpy as np
import tensorflow as tf
import collections
import time
import os
import random
import wandb
from tqdm import tqdm  # Added for progress bars
import argparse  # Add argparse import

from game_env import SwitcharooEnv
from az_network import AlphaZeroNetwork, build_alpha_zero_network
from mcts import MCTS, MCTSNode
from config import (
    AZ_ITERATIONS, AZ_GAMES_PER_ITERATION, AZ_TRAINING_STEPS_PER_ITERATION,
    AZ_REPLAY_BUFFER_SIZE, AZ_BATCH_SIZE, AZ_EVALUATION_GAMES_COUNT,
    AZ_MODEL_UPDATE_WIN_RATE, AZ_BEST_MODEL_FILE, AZ_CANDIDATE_MODEL_FILE,
    AZ_CHECKPOINT_FILE_PATTERN, NUM_SIMULATIONS_PER_MOVE,
    TEMPERATURE_START, TEMPERATURE_END, TEMPERATURE_ANNEAL_STEPS,
    DIRICHLET_ALPHA, DIRICHLET_EPSILON, C_PUCT_CONSTANT,
    WANDB_PROJECT, WANDB_ENTITY, MAX_STEPS_PER_EPISODE  # Added MAX_STEPS_PER_EPISODE
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
        
        # Initialize Neural Networks
        self.current_nn = AlphaZeroNetwork() 
        self.candidate_nn = AlphaZeroNetwork()

        if os.path.exists(AZ_BEST_MODEL_FILE):
            print(f"Loading best model from {AZ_BEST_MODEL_FILE} into current_nn and candidate_nn.")
            self.current_nn.load_model(AZ_BEST_MODEL_FILE)
            self.candidate_nn.load_model(AZ_BEST_MODEL_FILE)
        else:
            print(f"No best model found at {AZ_BEST_MODEL_FILE}. Starting with fresh models.")

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
                        "az_learning_rate": self.candidate_nn.model.optimizer.learning_rate.numpy() if hasattr(self.candidate_nn.model.optimizer, 'learning_rate') else 'N/A',
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

        for game_num in tqdm(range(AZ_GAMES_PER_ITERATION), desc="Self-Play Games", position=0, leave=True):  # Modified outer tqdm
            current_game_experiences = [] 
            current_board_state_arr = self.game_env.reset()
            done = False
            game_step_count = 0
            first_player_of_game = self.game_env.current_player_id

            with tqdm(total=MAX_STEPS_PER_EPISODE, desc=f"Game {game_num+1}/{AZ_GAMES_PER_ITERATION} Moves", position=1, leave=False) as move_bar:  # New inner tqdm
                while not done:
                    current_player_id = self.game_env.current_player_id
                    mcts_handler = MCTS(neural_network=self.current_nn, 
                                        game_env_class=SwitcharooEnv, 
                                        config=self.mcts_config)
                    root_node = MCTSNode(parent=None, prior_p=0.0, player_id=current_player_id)
                    mcts_handler.run_mcts_simulations(root_node, 
                                                      np.copy(self.game_env.board), 
                                                      current_player_id)
                    temp = get_temperature(self.total_game_steps_for_temp)
                    action_probs_from_mcts = mcts_handler.get_action_probabilities(root_node, temp)
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
                        break
            
            winner = self.game_env.winner_id
            if winner == first_player_of_game:
                iteration_game_outcomes["win"] += 1
            elif winner == 0 or winner == 3:
                iteration_game_outcomes["draw"] += 1
            else:
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

    def _train_network(self):
        if len(self.replay_buffer) < AZ_BATCH_SIZE:
            print("Replay buffer too small to train. Skipping training phase.")
            if self.wandb_enabled:
                wandb.log({"training/skipped": True, "training/total_loss": 0, "training/policy_loss":0, "training/value_loss":0})
            return

        print(f"\n--- Training Neural Network for {AZ_TRAINING_STEPS_PER_ITERATION} Steps ---")
        total_loss_this_iteration = 0
        policy_loss_this_iteration = 0
        value_loss_this_iteration = 0
        policy_accuracy_this_iteration = 0
        value_mae_this_iteration = 0

        for train_step in tqdm(range(AZ_TRAINING_STEPS_PER_ITERATION), desc="Training Steps"):  # Added tqdm
            if len(self.replay_buffer) < AZ_BATCH_SIZE:
                print("\nBuffer too small, breaking training early.")  # Added a print for early break
                break 
            minibatch = random.sample(list(self.replay_buffer), AZ_BATCH_SIZE)
            batch_states = np.array([data[0] for data in minibatch])
            batch_policy_targets = np.array([data[1] for data in minibatch])
            batch_value_targets = np.array([data[2] for data in minibatch]).reshape(-1, 1)

            loss_metrics = self.candidate_nn.model.train_on_batch(
                batch_states,
                {'policy_output': batch_policy_targets, 'value_output': batch_value_targets}
            )
            
            if isinstance(loss_metrics, list) and len(loss_metrics) >= 5:
                 total_loss_this_iteration += loss_metrics[0]
                 policy_loss_this_iteration += loss_metrics[1] 
                 value_loss_this_iteration += loss_metrics[2]
                 policy_accuracy_this_iteration += loss_metrics[3]
                 value_mae_this_iteration += loss_metrics[4]
            elif isinstance(loss_metrics, list) and len(loss_metrics) >=3:
                 total_loss_this_iteration += loss_metrics[0]
                 policy_loss_this_iteration += loss_metrics[1]
                 value_loss_this_iteration += loss_metrics[2]
            else: 
                total_loss_this_iteration += loss_metrics
        
        num_actual_train_steps = AZ_TRAINING_STEPS_PER_ITERATION if AZ_TRAINING_STEPS_PER_ITERATION > 0 else 1
        if train_step + 1 < AZ_TRAINING_STEPS_PER_ITERATION and train_step + 1 > 0:  # check if training broke early
             num_actual_train_steps = train_step + 1

        avg_total_loss = total_loss_this_iteration / num_actual_train_steps
        avg_policy_loss = policy_loss_this_iteration / num_actual_train_steps
        avg_value_loss = value_loss_this_iteration / num_actual_train_steps
        avg_policy_acc = policy_accuracy_this_iteration / num_actual_train_steps
        avg_value_mae = value_mae_this_iteration / num_actual_train_steps

        print(f"Network training finished. Avg Total Loss: {avg_total_loss:.4f} (P: {avg_policy_loss:.4f}, V: {avg_value_loss:.4f}, P_Acc: {avg_policy_acc:.4f}, V_MAE: {avg_value_mae:.4f})")
        if self.wandb_enabled:
            wandb.log({
                "training/total_loss": avg_total_loss,
                "training/policy_loss": avg_policy_loss,
                "training/value_loss": avg_value_loss,
                "training/policy_accuracy": avg_policy_acc,
                "training/value_mae": avg_value_mae,
                "training/skipped": False
            })

    def _evaluate_and_update_model(self, iteration_num):
        if iteration_num == 0: 
             print("\n--- Skipping Model Evaluation (First Iteration) ---")
             if not os.path.exists(AZ_BEST_MODEL_FILE):
                print(f"No best model exists. Saving current candidate_nn to {AZ_BEST_MODEL_FILE}")
                self.candidate_nn.save_model(AZ_BEST_MODEL_FILE)
                self.current_nn.load_model(AZ_BEST_MODEL_FILE)
             if self.wandb_enabled:
                wandb.log({"evaluation/skipped": True, "evaluation/candidate_win_rate": 0})
             return

        print(f"\n--- Evaluating Candidate Model (Iteration {iteration_num}) ---")
        self.candidate_nn.save_model(AZ_CANDIDATE_MODEL_FILE)
        eval_candidate_nn = AlphaZeroNetwork()
        eval_candidate_nn.load_model(AZ_CANDIDATE_MODEL_FILE)
        eval_best_nn = AlphaZeroNetwork()
        if os.path.exists(AZ_BEST_MODEL_FILE):
            eval_best_nn.load_model(AZ_BEST_MODEL_FILE)
        else:
            print(f"No existing best model found at {AZ_BEST_MODEL_FILE} for evaluation. Candidate becomes best.")
            self.candidate_nn.save_model(AZ_BEST_MODEL_FILE)
            self.current_nn.load_model(AZ_BEST_MODEL_FILE)
            if self.wandb_enabled:
                wandb.log({"evaluation/skipped": True, "evaluation/candidate_win_rate": 1.0, "evaluation/best_model_updated": True})
            return

        candidate_wins = 0
        best_wins = 0
        draws = 0
        eval_temperature = 0.1 

        for game_idx in tqdm(range(AZ_EVALUATION_GAMES_COUNT), desc="Evaluation Games"):  # Added tqdm
            eval_env = SwitcharooEnv()
            eval_env.reset()
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

            with tqdm(total=MAX_STEPS_PER_EPISODE, desc=f"Eval Game {game_idx+1}/{AZ_EVALUATION_GAMES_COUNT} Moves", position=1, leave=False) as eval_move_bar:
                while not done:
                    current_player_id_eval = eval_env.current_player_id
                    active_model = current_models[current_player_id_eval]
                    mcts_eval = MCTS(neural_network=active_model, 
                                     game_env_class=SwitcharooEnv, 
                                     config=self.mcts_config)
                    root_eval = MCTSNode(parent=None, prior_p=0.0, player_id=current_player_id_eval)
                    mcts_eval.run_mcts_simulations(root_eval, np.copy(eval_env.board), current_player_id_eval)
                    action_probs_eval = mcts_eval.get_action_probabilities(root_eval, eval_temperature)
                    
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
                    eval_move_bar.update(1)

                    if game_step_count_eval >= MAX_STEPS_PER_EPISODE:
                        if not done:
                            print(f"Warning: Eval Game {game_idx+1} reached MAX_STEPS_PER_EPISODE ({MAX_STEPS_PER_EPISODE}). Ending game.")
                        done = True
                    
                    if done:
                        eval_move_bar.n = game_step_count_eval
                        eval_move_bar.refresh()
                        break
            
            game_winner_id = eval_env.winner_id
            if game_winner_id == 3: 
                draws += 1
            elif game_winner_id != 0:
                if (player_A_model == eval_candidate_nn and game_winner_id == PLAYER_A_ID) or \
                   (player_B_model == eval_candidate_nn and game_winner_id == PLAYER_B_ID):
                    candidate_wins +=1
                else:
                    best_wins +=1
            
            print(f"  Eval Game {game_idx+1}/{AZ_EVALUATION_GAMES_COUNT}: Candidate ({player_A_name if player_A_model==eval_candidate_nn else player_B_name}) vs Best ({player_A_name if player_A_model==eval_best_nn else player_B_name}). Winner: {eval_env.winner}")

        win_rate = 0
        if (AZ_EVALUATION_GAMES_COUNT - draws) > 0:
            win_rate = candidate_wins / (AZ_EVALUATION_GAMES_COUNT - draws)

        print(f"Evaluation Results: Candidate Wins: {candidate_wins}, Best Wins: {best_wins}, Draws: {draws}")
        print(f"Candidate Win Rate (vs Best, excluding draws): {win_rate:.2%}")
        
        best_model_updated = False
        if win_rate > AZ_MODEL_UPDATE_WIN_RATE:
            print(f"Candidate model is better (Win Rate {win_rate:.2%} > {AZ_MODEL_UPDATE_WIN_RATE:.2%}). Updating best model.")
            self.candidate_nn.save_model(AZ_BEST_MODEL_FILE)
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
            self._run_self_play_games()
            self._train_network()
            self._evaluate_and_update_model(iteration)
            checkpoint_path = AZ_CHECKPOINT_FILE_PATTERN.format(iteration + 1)
            self.candidate_nn.save_model(checkpoint_path)
            print(f"Saved candidate_nn checkpoint to {checkpoint_path}")
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
    args = parser.parse_args()

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
