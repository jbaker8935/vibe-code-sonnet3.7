"""Phase 1 training implementation: Training against random opponent."""
import time
import numpy as np
from collections import deque  # Keep deque for metrics, not replay
import wandb
from random import choice
import tensorflow as tf  # Import tensorflow
import os  # For file path operations

from env_const import (PLAYER_A, PLAYER_B, PLAYER_A_ID, PLAYER_B_ID)
from env_util import (_evaluate_board_jit, _calculate_progress_reward, _apply_move_jit)

from game_env import (SwitcharooEnv)
from train_dqn import save_checkpoint
from utils import _action_index_to_move, _validate_reward, init_wandb
from config import (PHASE1_EPISODES, MAX_STEPS_PER_EPISODE, REPLAY_FREQUENCY,
                   BASE_MODEL_FILE, SAVE_FREQ, initial_position,
                   WIN_REWARD, DRAW_PENALTY, PROGRESS_REWARD, MOVE_TIMEOUT,
                   STEP_PENALTY)  # Added STEP_PENALTY
from dqn_agent import DQNAgent

def get_opponent_action_fast(board_state, legal_actions, opponent_epsilon):
    """Ultra-fast opponent action selection with a bias toward aggressive play."""
    if not legal_actions:  # Handle empty list
        return None
    if np.random.random() < opponent_epsilon:
        # Ensure index is valid
        return legal_actions[np.random.randint(0, len(legal_actions))]

    best_score = float('-inf')
    best_action = legal_actions[0]
    
    # Evaluate each action
    action_scores = {}
    for action in legal_actions:
        board_copy = board_state.copy()
        move = _action_index_to_move(action)
        if move is None:
            continue

        start_r, start_c, end_r, end_c = move
        # Bounds check
        if 0 <= start_r < 8 and 0 <= start_c < 4 and 0 <= end_r < 8 and 0 <= end_c < 4:
            _apply_move_jit(board_copy, start_r, start_c, end_r, end_c)
            
            # Basic board evaluation
            score = _evaluate_board_jit(board_copy, PLAYER_A_ID)
            
            # Favor forward movement (more aggressive play)
            if end_r > start_r:  # Moving toward opponent's side
                progress_boost = (end_r - start_r) * 1.5  # Bigger boost for forward movement
                score += progress_boost
            
            action_scores[action] = score
            
            if score > best_score:
                best_score = score
                best_action = action
    
    # If we have very similar top actions, sometimes pick the more aggressive one
    top_actions = [a for a, s in action_scores.items() 
                  if s >= best_score - 1.0]  # Actions within 1.0 of best
                  
    if len(top_actions) > 1 and np.random.random() < 0.4:  # 40% chance to pick aggressive
        # Find the action that moves furthest forward
        most_aggressive = best_action
        max_forward = -1
        
        for action in top_actions:
            move = _action_index_to_move(action)
            if move is None:
                continue
            start_r, start_c, end_r, end_c = move
            forward_move = end_r - start_r  # Positive means moving down (forward for Player A)
            if forward_move > max_forward:
                max_forward = forward_move
                most_aggressive = action
        
        return most_aggressive

    return best_action

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
    best_win_rate = 0.0
    
    # Track repeated positions to discourage stalemates
    position_history = {}
    
    # Training stability monitoring
    loss_trend = deque(maxlen=10)  # Track short-term loss trend
    consecutive_loss_increases = 0
    no_improvement_count = 0
    initial_lr = agent.learning_rate

    log_freq = 10
    wandb_enabled = init_wandb(enable_wandb, "phase1_training_tfdata", agent)  # Updated project name

    # Determine starting opponent epsilon (allow override)
    if opponent_epsilon_start is not None:
        start_eps = opponent_epsilon_start
    else:
        start_eps = 1.0
    # More aggressive epsilon decay - make opponent tougher faster
    opponent_epsilon_end = 0.05  # Lower minimum epsilon for tougher opponent (was 0.08)
    opponent_epsilon_decay = (opponent_epsilon_end / start_eps) ** (1 / (episodes * 1.5)) # Faster decay (was 2.0)
    opponent_epsilon = start_eps

    # Adjust epsilon decay to reach min later in training
    agent.epsilon_min = 0.05  # Reduced minimum epsilon for more exploitation (was 0.1)
    agent_epsilon_decay = (agent.epsilon_min / agent.epsilon) ** (1 / (episodes * 0.9)) # Slower decay (was 0.8)

    # Create dataset iterator outside the episode loop
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
        state = state.astype(np.float32)
        episode_reward = 0
        episode_steps = 0  # Steps within the current episode
        episode_loss_sum = 0.0
        episode_train_count = 0
        
        # Clear position history for new episode
        if e % 100 == 0:  # Periodically clear to prevent memory bloat
            position_history.clear()
        
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
            
            # Track position by converting to binary representation
            from binary_board import board_to_binary
            current_binary = tuple(map(int, board_to_binary(env.board)))  # Convert ndarray to tuple for hashing
            position_key = f"{current_player}_{current_binary}"
            
            # Count position occurrences
            if position_key in position_history:
                position_history[position_key] += 1
            else:
                position_history[position_key] = 1

            action = None  # Initialize action
            if current_player == PLAYER_B:  # Agent's turn
                legal_actions = env.get_legal_action_indices(player=PLAYER_B)
                if not legal_actions:
                    break  # No moves for agent, end episode
                
                # Score and filter actions to encourage progress
                filtered_actions = []
                action_scores = {}
                
                for action in legal_actions:
                    board_copy = env.board.copy()
                    move = _action_index_to_move(action)
                    if move is None:
                        continue
                        
                    start_r, start_c, end_r, end_c = move
                    if 0 <= start_r < 8 and 0 <= start_c < 4 and 0 <= end_r < 8 and 0 <= end_c < 4:
                        _apply_move_jit(board_copy, start_r, start_c, end_r, end_c)
                        
                        # Evaluate based on board position
                        score = _evaluate_board_jit(board_copy, PLAYER_B_ID)
                        
                        # Add progress incentive: reward moving toward opponent's side
                        if start_r > end_r:  # Moving upward (toward Player A's side)
                            progress = (start_r - end_r) * PROGRESS_REWARD
                            score += progress
                        
                        # Penalize repetitive positions
                        next_binary = tuple(map(int, board_to_binary(board_copy)))
                        next_key = f"{PLAYER_B}_{next_binary}"
                        if next_key in position_history and position_history[next_key] >= MOVE_TIMEOUT:
                            score -= 20  # Significant penalty for repeated positions
                        
                        action_scores[action] = score
                        
                        # Only include reasonably good moves
                        if score > -50:  # Filter threshold
                            filtered_actions.append(action)
                
                # Use filtered actions unless none are good enough
                action_pool = filtered_actions if filtered_actions else legal_actions
                
                # Agent acts
                action = agent.act(state, action_pool)
                next_state, reward, done, info = env.step(action)
                next_state = next_state.astype(np.float32)  # Ensure float32
                
                # Apply reward modifications
                original_reward = reward
                
                # Enhanced win reward scaled by speed
                if done and info.get('winner') == PLAYER_B:
                    # Scale win reward by how quickly the game was won
                    speed_factor = 1.0 - (step / MAX_STEPS_PER_EPISODE) * 0.5
                    reward = WIN_REWARD * speed_factor
                
                # Heavy penalty for draws/timeouts
                if done and (info.get('winner') is None or info.get('winner') == 'DRAW'):
                    reward = DRAW_PENALTY
                
                # Apply repetition penalty
                pos_key = f"{PLAYER_B}_{current_binary}"
                if pos_key in position_history and position_history[pos_key] > MOVE_TIMEOUT:
                    reward -= 5.0  # Penalty for repeated positions
                
                # Apply step penalty
                reward += STEP_PENALTY

                reward = _validate_reward(reward) # Validate final reward
                agent.remember(state, action, reward, next_state, done)
                episode_reward += reward
                total_agent_steps += 1
                episode_steps += 1

            else:  # Opponent's turn
                legal_actions = env.get_legal_action_indices(player=PLAYER_A)
                if not legal_actions:
                    break  # No moves for opponent, end episode
                
                # Filter actions for opponent to avoid repetitive play
                if step > 10:  # Only after early game
                    filtered_actions = []
                    for act in legal_actions:
                        board_copy = env.board.copy()
                        move = _action_index_to_move(act)
                        if move is None:
                            continue
                            
                        start_r, start_c, end_r, end_c = move
                        if 0 <= start_r < 8 and 0 <= start_c < 4 and 0 <= end_r < 8 and 0 <= end_c < 4:
                            _apply_move_jit(board_copy, start_r, start_c, end_r, end_c)
                            next_binary = tuple(map(int, board_to_binary(board_copy)))
                            next_key = f"{PLAYER_A}_{next_binary}"
                            
                            # Skip highly repetitive positions
                            if next_key in position_history and position_history[next_key] >= MOVE_TIMEOUT + 1:
                                continue
                                
                            filtered_actions.append(act)
                    
                    # Use filtered actions unless all are filtered out
                    action_pool = filtered_actions if filtered_actions else legal_actions
                else:
                    action_pool = legal_actions

                action = get_opponent_action_fast(env.board, action_pool, opponent_epsilon)
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
                
            # Force early end to prevent stalemates
            if step >= MAX_STEPS_PER_EPISODE - 1:
                done = True
                # reward = DRAW_PENALTY  # Heavy penalty for timeouts. This is already applied if winner is DRAW
                info['winner'] = 'DRAW'
                info['timeout'] = True
                
                # Give the agent a negative reward if it's about to time out
                if current_player == PLAYER_B:
                    # Apply step penalty before final reward adjustment
                    current_reward = STEP_PENALTY + DRAW_PENALTY 
                    current_reward = _validate_reward(current_reward)
                    agent.remember(state, action, current_reward, next_state, done)
                    episode_reward += current_reward
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
            if recent_loss_avg > older_loss_avg * 1.75:  # Loss increased by 75%+ (was 1.5)
                consecutive_loss_increases += 1
                
                # If loss keeps increasing, reduce learning rate
                if consecutive_loss_increases >= 5: # Increased threshold (was 3)
                    old_lr = agent.learning_rate
                    agent.learning_rate = max(old_lr * 0.9, 1e-8)  # Reduce by 10%, ensure it doesn't go to zero (was 0.8)
                    print(f"Episode {e}: Loss increasing, reducing learning rate: {old_lr:.2e} → {agent.learning_rate:.2e}")
                    consecutive_loss_increases = 0
                    
                    # Also schedule buffer refresh
                    buffer_refresh_scheduled = True
                    
                    # Recreate dataset with new learning rate
                    dataset = agent.create_tf_dataset()
                    dataset_iterator = iter(dataset)
            else:
                consecutive_loss_increases = 0

        # Track best win rate
        if win_rate > best_win_rate and len(wins) >= 30:
            best_win_rate = win_rate
            # Save model with best win rate
            win_rate_model_file = f"phase1_best_winrate_{int(win_rate*100)}.weights.h5"
            agent.save(win_rate_model_file)
            print(f"New best win rate model: {win_rate:.3f}")
            no_improvement_count = 0
        else:
            no_improvement_count += 1

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
                    "time_per_episode": time.time() - episode_start,
                    "best_win_rate": best_win_rate
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
                  f"Win Rate: {win_rate:.2f} | "
                  f"Loss Rate: {loss_rate:.2f} | "  # Added Loss Rate
                  f"Draw Rate: {draw_rate:.2f} | "
                  f"Avg Steps: {avg_steps:.2f} | "
                  f"Epsilon: {agent.epsilon:.4f} | "
                  f"Opp Epsilon: {opponent_epsilon:.4f}")

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