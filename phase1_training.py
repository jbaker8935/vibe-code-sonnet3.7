"""Phase 1 training implementation: Training against random opponent."""
import time
import numpy as np
from collections import deque  # Keep deque for metrics, not replay
import wandb
from random import choice
import tensorflow as tf  # Import tensorflow

from env_const import (PLAYER_A, PLAYER_B, PLAYER_B_ID)
from env_util import (_evaluate_board_jit, _calculate_progress_reward, _apply_move_jit)

from game_env import (SwitcharooEnv)
# Removed safe_replay import, using dataset now
from train_dqn import save_checkpoint  # , safe_replay
from utils import _action_index_to_move, _validate_reward, init_wandb
from config import (PHASE1_EPISODES, MAX_STEPS_PER_EPISODE, REPLAY_FREQUENCY,
                   BASE_MODEL_FILE, SAVE_FREQ, initial_position)
from dqn_agent import DQNAgent

def get_opponent_action_fast(board_state, legal_actions, opponent_epsilon):
    """Ultra-fast opponent action selection."""
    if not legal_actions:  # Handle empty list
        return None
    if np.random.random() < opponent_epsilon:
        # Ensure index is valid
        return legal_actions[np.random.randint(0, len(legal_actions))]

    best_score = float('-inf')
    best_action = legal_actions[0]

    for action in legal_actions:
        board_copy = board_state.copy()
        move = _action_index_to_move(action)
        if move is None:
            continue

        start_r, start_c, end_r, end_c = move
        # Bounds check (redundant if legal_actions are truly legal, but safe)
        if 0 <= start_r < 8 and 0 <= start_c < 8 and 0 <= end_r < 8 and 0 <= end_c < 8:
            _apply_move_jit(board_copy, start_r, start_c, end_r, end_c)
            # Ensure PLAYER_B_ID is correct contextually
            score = _evaluate_board_jit(board_copy, PLAYER_B_ID)

            if score > best_score:
                best_score = score
                best_action = action

    return best_action

def get_opponent_action(env, opponent_epsilon=0.3):
    """Improved policy for the opponent using JIT-compiled evaluation."""
    legal_actions = env.get_legal_action_indices(player=PLAYER_A)
    if not legal_actions:
        return None

    if np.random.random() < opponent_epsilon:
        return np.random.choice(legal_actions)

    # Consider evaluating more actions if performance allows
    eval_actions = legal_actions  # Evaluate all legal actions

    # Pass current board state directly
    return get_opponent_action_fast(env.board, eval_actions, opponent_epsilon)

def phase1_training(agent, start_episode=1, episodes=PHASE1_EPISODES, enable_wandb=True, opponent_epsilon_start=None):
    """Phase 1: Train against a progressively stronger opponent using tf.data."""
    env = SwitcharooEnv()
    scores = deque(maxlen=100)
    wins = deque(maxlen=100)
    losses = deque(maxlen=100)
    draws = deque(maxlen=100)
    steps = deque(maxlen=100)
    move_times = deque(maxlen=100)
    train_times = deque(maxlen=100)
    losses_log = deque(maxlen=100)  # Log training loss
    best_avg_score = float('-inf')

    log_freq = 10
    wandb_enabled = init_wandb(enable_wandb, "phase1_training_tfdata", agent)  # Updated project name

    # Determine starting opponent epsilon (allow override)
    if opponent_epsilon_start is not None:
        start_eps = opponent_epsilon_start
    else:
        start_eps = 1.0
    opponent_epsilon_end = 0.01
    opponent_epsilon_decay = (opponent_epsilon_end / start_eps) ** (1 / (episodes * 1.5)) # Slower decay
    opponent_epsilon = start_eps

    # Adjust epsilon decay to reach min around 80% of episodes
    agent_epsilon_decay = (agent.epsilon_min / agent.epsilon) ** (1 / (episodes * 0.80)) # Slower decay

    # Create dataset iterator outside the episode loop (or recreate periodically)
    # Only create if buffer has enough samples initially
    dataset_iterator = None
    if len(agent) >= agent.batch_size:
        print("Creating initial dataset iterator...")
        dataset = agent.create_tf_dataset()
        dataset_iterator = iter(dataset)
        print("Dataset iterator created.")
    else:
        print(f"Waiting for buffer to reach batch size ({agent.batch_size}). Current size: {len(agent)}")

    total_agent_steps = 0  # Track total steps for training frequency

    for e in range(start_episode, episodes + 1):
        move_time = train_time = 0
        episode_start = time.time()
        state = env.reset(choice(initial_position))
        # Ensure state is float32 from the start
        state = state.astype(np.float32)
        episode_reward = 0
        episode_steps = 0  # Steps within the current episode
        episode_loss_sum = 0.0
        episode_train_count = 0

        for step in range(MAX_STEPS_PER_EPISODE):
            current_player = env.current_player
            step_start = time.time()

            action = None  # Initialize action
            if current_player == PLAYER_B:  # Agent's turn
                legal_actions = env.get_legal_action_indices(player=PLAYER_B)
                if not legal_actions:
                    break  # No moves for agent, end episode

                # Agent acts
                action = agent.act(state, legal_actions)
                next_state, reward, done, info = env.step(action)
                next_state = next_state.astype(np.float32)  # Ensure float32

                # --- Reward Shaping --- 
                if done:
                    # Base win/loss/draw rewards are now handled in the environment
                    pass # No explicit reward setting here needed
                else:
                    # Progress reward (consider if still needed/effective)
                    # Reduced multiplier from 0.5 to 0.2
                    progress_reward = _calculate_progress_reward(env.board, PLAYER_B_ID) * 0.2 # Increased multiplier
                    reward += progress_reward

                reward = _validate_reward(reward)
                agent.remember(state, action, reward, next_state, done)
                episode_reward += reward
                total_agent_steps += 1
                episode_steps += 1

            else:  # Opponent's turn
                action = get_opponent_action(env, opponent_epsilon=opponent_epsilon)
                if action is None:
                    break  # No moves for opponent, end episode

                move_time += time.time() - step_start  # Log opponent move time? Or just agent?
                next_state, _, done, info = env.step(action)
                next_state = next_state.astype(np.float32)  # Ensure float32

            state = next_state

            # --- Training Step using tf.data ---
            # Check if buffer is large enough and if it's time to train
            if len(agent) >= agent.batch_size and total_agent_steps % REPLAY_FREQUENCY == 0:
                # Create iterator if it doesn't exist yet
                if dataset_iterator is None:
                    print("Buffer reached batch size. Creating dataset iterator...")
                    dataset = agent.create_tf_dataset()
                    dataset_iterator = iter(dataset)
                    print("Dataset iterator created.")

                # Perform one training step
                train_start = time.time()
                try:
                    # Get next batch from the dataset
                    batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones, batch_weights, batch_indices = next(dataset_iterator)

                    # Perform the training step
                    loss, td_errors = agent._train_step(
                        batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones, batch_weights
                    )

                    # Update priorities using the indices from the batch
                    agent.update_priorities(batch_indices.numpy(), td_errors.numpy())

                    # Log loss
                    current_loss = loss.numpy()
                    if not (np.isnan(current_loss) or np.isinf(current_loss)):
                        episode_loss_sum += current_loss
                        episode_train_count += 1

                    # Periodically update target network (moved inside training check)
                    agent.update_counter += 1
                    if agent.update_counter % agent.target_update_freq == 0:
                        agent.update_target_model()

                except StopIteration:
                    # Should not happen with infinite generator, but handle defensively
                    print("Warning: Dataset iterator exhausted unexpectedly. Recreating...")
                    dataset = agent.create_tf_dataset()
                    dataset_iterator = iter(dataset)
                except Exception as train_e:
                    print(f"\nError during training step: {train_e}")
                    # Optionally break or continue
                    import traceback
                    traceback.print_exc()

                train_time += time.time() - train_start

            if done:
                steps.append(episode_steps)  # Log agent steps per episode
                train_times.append(train_time)
                move_times.append(move_time)  # Check if move_time logging is correct
                break

        # --- End of Episode ---

        # Anneal agent epsilon
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent_epsilon_decay

        # Anneal opponent epsilon (Standard annealing)
        opponent_epsilon = max(opponent_epsilon_end, opponent_epsilon * opponent_epsilon_decay)
        # Removed the periodic random reset, standard annealing is preferred here.

        # Anneal PER beta
        agent.per_beta = min(1.0, agent.per_beta + agent.per_beta_increment)

        # Log metrics
        scores.append(episode_reward)
        avg_loss = (episode_loss_sum / episode_train_count) if episode_train_count > 0 else 0.0
        losses_log.append(avg_loss)  # Log average loss for the episode

        if info.get('winner') == PLAYER_B:
            wins.append(1); losses.append(0); draws.append(0)
        elif info.get('winner') == PLAYER_A:
            wins.append(0); losses.append(1); draws.append(0)
        else:  # Draw
            wins.append(0); losses.append(0); draws.append(1)

        # Calculate rolling metrics
        avg_score = np.mean(scores) if scores else 0.0
        win_rate = np.mean(wins) if wins else 0.0
        loss_rate = np.mean(losses) if losses else 0.0
        draw_rate = np.mean(draws) if draws else 0.0
        avg_steps = np.mean(steps) if steps else 0.0
        avg_move_time = np.mean(move_times) if move_times else 0.0
        avg_train_time = np.mean(train_times) if train_times else 0.0
        avg_rolling_loss = np.mean(losses_log) if losses_log else 0.0

        if wandb_enabled and e % log_freq == 0:
            try:
                wandb.log({
                    "episode": e,
                    "episode_reward": episode_reward,
                    "avg_score_100": avg_score,  # Rolling avg over 100 episodes
                    "win_rate_100": win_rate,
                    "loss_rate_100": loss_rate,
                    "draw_rate_100": draw_rate,
                    "avg_steps_100": avg_steps,
                    "avg_move_time_100": avg_move_time,  # Check if this metric is meaningful
                    "avg_train_time_100": avg_train_time,  # Avg train time per episode
                    "avg_loss_100": avg_rolling_loss,  # Rolling avg loss
                    "epsilon": agent.epsilon,
                    "opponent_epsilon": opponent_epsilon,
                    "per_beta": agent.per_beta,
                    "memory_size": len(agent),
                    "total_agent_steps": total_agent_steps,
                    "time_per_episode": time.time() - episode_start
                })
            except Exception as log_e:
                print(f"\nWarning: Failed to log to wandb: {log_e}")

        # Save best model based on rolling average score
        if avg_score > best_avg_score and len(scores) >= 50:  # Use rolling average
            best_avg_score = avg_score
            agent.save(BASE_MODEL_FILE)
            print(f"New best model saved with avg score (100 ep): {best_avg_score:.2f}")

        # Periodic checkpoint saving
        if e % SAVE_FREQ == 0:
            save_checkpoint(agent, e)

        # Console logging
        if e % 100 == 0:
            print(f"Phase 1 - Ep: {e}/{episodes} | "
                  f"Avg Score: {avg_score:.2f} | "
                  f"Avg Steps: {avg_steps:.2f} | "
                  f"Avg Train Time: {avg_train_time:.4f} | "
                  f"Avg Loss: {avg_rolling_loss:.4f} | "
                  f"Win Rate: {win_rate:.2f} | "
                  f"Epsilon: {agent.epsilon:.4f} | "
                  f"Opp Epsilon: {opponent_epsilon:.4f} | "
                  f"Buffer: {len(agent)}")

    # --- End of Training ---
    agent.save(BASE_MODEL_FILE)
    print(f"Phase 1 training completed. Final model saved to {BASE_MODEL_FILE}")

    if wandb_enabled:
        try:
            wandb.finish()
        except Exception as finish_e:
            print(f"Error finishing wandb run: {finish_e}")
            pass
    return agent

def run_sweep():
    sweep_config = {
        'method': 'bayes',  # Bayesian optimization
        'metric': {
            'name': 'avg_score_100',
            'goal': 'maximize'
        },
        'parameters': {
            'learning_rate': {
                'values': [5e-8, 1e-8, 1e-7, 2e-7]  # Updated learning rate range
            },
            'batch_size': {
                'values': [64, 128, 256]
            },
            'gamma': {
                'values': [0.95, 0.99, 0.999]
            },
            'epsilon_decay': {
                'values': [0.995, 0.999, 0.9995]
            },
            'per_alpha': {
                'values': [0.3, 0.5, 0.7]
            }
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="switcharoo-dqn")

    def train_with_sweep():
        with wandb.init() as run:
            config = run.config

            # Dynamically set parameters from the sweep
            agent = DQNAgent(
                learning_rate=config.learning_rate,
                batch_size=config.batch_size,
                gamma=config.gamma,
                epsilon_decay=config.epsilon_decay,
                per_alpha=config.per_alpha
            )

            # Limit episodes to 500 for the sweep
            phase1_training(agent, episodes=500, enable_wandb=True)

    wandb.agent(sweep_id, function=train_with_sweep)

if __name__ == "__main__":
    run_sweep()