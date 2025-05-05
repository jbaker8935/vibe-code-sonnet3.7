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
    
    # Training stability monitoring
    loss_trend = deque(maxlen=10)  # Track short-term loss trend
    consecutive_loss_increases = 0
    best_win_rate = 0.0
    no_improvement_count = 0
    initial_lr = agent.learning_rate

    log_freq = 10
    wandb_enabled = init_wandb(enable_wandb, "phase1_training_tfdata", agent)  # Updated project name

    # Determine starting opponent epsilon (allow override)
    if opponent_epsilon_start is not None:
        start_eps = opponent_epsilon_start
    else:
        start_eps = 1.0
    opponent_epsilon_end = 0.1  # Increased minimum epsilon for opponent
    # Slower opponent epsilon decay with square root curve
    opponent_epsilon_decay = (opponent_epsilon_end / start_eps) ** (1 / (episodes * 2.5))
    opponent_epsilon = start_eps

    # Adjust epsilon decay to reach min later in training
    # Agent should explore more consistently
    agent.epsilon_min = 0.15  # Increased minimum epsilon for better exploration
    agent_epsilon_decay = (agent.epsilon_min / agent.epsilon) ** (1 / (episodes * 0.85))

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
    buffer_refresh_scheduled = False

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

        # Refresh part of the buffer if scheduled
        if buffer_refresh_scheduled and e % 500 == 0:
            if len(agent) >= agent.replay_buffer_size * 0.9:  # Buffer almost full
                reset_size = int(agent.replay_buffer_size * 0.2)  # Reset 20% of buffer
                start_idx = np.random.randint(0, agent.replay_buffer_size - reset_size)
                agent.priorities[start_idx:start_idx+reset_size] = 1.0  # Reset priorities
                print(f"Refreshed {reset_size} experiences in buffer at episode {e}")
                buffer_refresh_scheduled = False

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

                reward = _validate_reward(reward) # Validate final reward from env
                agent.remember(state, action, reward, next_state, done)
                episode_reward += reward
                total_agent_steps += 1
                episode_steps += 1

            else:  # Opponent's turn
                action = get_opponent_action(env, opponent_epsilon=opponent_epsilon)
                if action is None:
                    break  # No moves for opponent, end episode

                move_time += time.time() - step_start
                next_state, _, done, info = env.step(action)
                next_state = next_state.astype(np.float32)

            state = next_state

            # --- Training Step using tf.data ---
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

                    # Periodically update target network
                    agent.update_counter += 1
                    if agent.update_counter % agent.target_update_freq == 0:
                        agent.update_target_model()

                except StopIteration:
                    print("Warning: Dataset iterator exhausted unexpectedly. Recreating...")
                    dataset = agent.create_tf_dataset()
                    dataset_iterator = iter(dataset)
                except Exception as train_e:
                    print(f"\nError during training step: {train_e}")
                    import traceback
                    traceback.print_exc()

                train_time += time.time() - train_start

            if done:
                steps.append(episode_steps)
                train_times.append(train_time)
                move_times.append(move_time)
                break

        # --- End of Episode ---

        # Anneal agent epsilon with minimum safeguard
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent_epsilon_decay

        # Anneal opponent epsilon with min safeguard
        opponent_epsilon = max(opponent_epsilon_end, opponent_epsilon * opponent_epsilon_decay)

        # Anneal PER beta
        agent.per_beta = min(1.0, agent.per_beta + agent.per_beta_increment)

        # Log metrics
        scores.append(episode_reward)
        avg_loss = (episode_loss_sum / episode_train_count) if episode_train_count > 0 else 0.0
        losses_log.append(avg_loss)
        loss_trend.append(avg_loss)

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

        # Loss trend analysis and learning rate adjustment
        if len(loss_trend) >= 5:  # Need at least 5 data points
            recent_loss_avg = np.mean(list(loss_trend)[-3:])
            older_loss_avg = np.mean(list(loss_trend)[:-3])
            
            # Check for consistent loss increase (sign of instability)
            if recent_loss_avg > older_loss_avg * 1.5:  # Loss increased by 50%+
                consecutive_loss_increases += 1
                
                # If loss keeps increasing, reduce learning rate
                if consecutive_loss_increases >= 3:
                    old_lr = agent.learning_rate
                    agent.learning_rate *= 0.8  # Reduce by 20%
                    print(f"Episode {e}: Loss increasing, reducing learning rate: {old_lr:.2e} → {agent.learning_rate:.2e}")
                    consecutive_loss_increases = 0
                    
                    # Also schedule buffer refresh
                    buffer_refresh_scheduled = True
                    
                    # Recreate dataset with new learning rate
                    dataset = agent.create_tf_dataset()
                    dataset_iterator = iter(dataset)
            else:
                consecutive_loss_increases = 0

        # Implement early stopping if win rate is not improving
        if win_rate > best_win_rate:
            best_win_rate = win_rate
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            
        # If win rate stalls for a long time, reset learning rate
        if no_improvement_count >= 1000:
            agent.learning_rate = initial_lr  # Reset to initial learning rate
            print(f"Episode {e}: No improvement for 1000 episodes, resetting learning rate to {initial_lr:.2e}")
            no_improvement_count = 0
            dataset = agent.create_tf_dataset() 
            dataset_iterator = iter(dataset)

        if wandb_enabled and e % log_freq == 0:
            try:
                wandb.log({
                    "episode": e,
                    "episode_reward": episode_reward,
                    "avg_score_100": avg_score,
                    "win_rate_100": win_rate,
                    "loss_rate_100": loss_rate,
                    "draw_rate_100": draw_rate,
                    "avg_steps_100": avg_steps,
                    "avg_move_time_100": avg_move_time,
                    "avg_train_time_100": avg_train_time,
                    "avg_loss_100": avg_rolling_loss,
                    "epsilon": agent.epsilon,
                    "learning_rate": agent.learning_rate,
                    "opponent_epsilon": opponent_epsilon,
                    "per_beta": agent.per_beta,
                    "memory_size": len(agent),
                    "total_agent_steps": total_agent_steps,
                    "time_per_episode": time.time() - episode_start
                })
            except Exception as log_e:
                print(f"\nWarning: Failed to log to wandb: {log_e}")

        # Save best model based on rolling average score
        if avg_score > best_avg_score and len(scores) >= 50:
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