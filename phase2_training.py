"""Phase 2 training implementation: Tournament-based self-play training."""
import time
import copy
import numpy as np
import os
from collections import deque
import wandb
from random import choice

from dqn_agent import DQNAgent
from env_const import (PLAYER_A, PLAYER_B, PLAYER_A_ID, PLAYER_B_ID)
from env_util import (_evaluate_board_jit, _calculate_progress_reward, _apply_move_jit) 
from game_env import (SwitcharooEnv)
from train_dqn import save_checkpoint
from tournament import run_tournament
from utils import _action_index_to_move, _validate_reward, init_wandb
from config import (PHASE2_EPISODES, MAX_STEPS_PER_EPISODE, REPLAY_FREQUENCY,
                   TOURNAMENT_FREQ, TOURNAMENT_MODEL_FILE, SAVE_FREQ,
                   TFJS_MODEL_DIR, CHECKPOINT_FILE, initial_position,
                   WIN_REWARD, DRAW_PENALTY, PROGRESS_REWARD, MOVE_TIMEOUT,
                   BEST_MODEL_FILE, STEP_PENALTY)  # Added STEP_PENALTY
from binary_board import board_to_binary

# Add position tracking to detect repetitive play
def detect_repetitive_play(history, current_binary_board, threshold=MOVE_TIMEOUT):
    """Detect if the current board state is repeated too many times."""
    if history is None or len(history) == 0:
        return False
    
    # Fix: Properly handle array comparison with numpy
    count = 0
    for board in history:
        if np.array_equal(board, current_binary_board):
            count += 1
    
    return count >= threshold

def phase2_training(agent, start_episode=1, episodes=PHASE2_EPISODES, direct_phase2=False, enable_wandb=True):
    """Phase 2: Tournament self-play training."""
    wandb_enabled = init_wandb(enable_wandb, "phase2_training", agent)
    print("\n===== STARTING PHASE 2: TOURNAMENT SELF-PLAY =====")
    
    env = SwitcharooEnv()
    scores = deque(maxlen=100)
    wins = deque(maxlen=100)
    losses = deque(maxlen=100)
    draws = deque(maxlen=100)
    tournament_agent = copy.deepcopy(agent)
    steps = deque(maxlen=100)
    move_times = deque(maxlen=100)
    train_times = deque(maxlen=100)
    
    # Higher epsilon to encourage more exploration in self-play
    tournament_agent.epsilon = 0.15
    
    # Keep track of board positions to detect repetitive play
    position_history = deque(maxlen=20)

    # Track the best win rate and corresponding model
    best_win_rate = 0.0
    win_rate_no_improvement = 0
    
    # Track past game states to discourage repetitive play
    board_history = {}

    for e in range(start_episode, episodes + 1):
        move_time = train_time = 0
        episode_start = time.time()
        state = env.reset(choice(initial_position))
        episode_reward = 0
        agent_steps = 0
        step_rewards = []  # Track step-by-step rewards
        position_history.clear()  # Clear position history for new game
        
        # Initialize move history for the current game
        if e % 100 == 0:  # Periodically clear to prevent memory bloat
            board_history.clear()
        
        for step in range(MAX_STEPS_PER_EPISODE):
            step_start = time.time()
            current_player = env.current_player
            
            # Convert current board to binary representation for tracking
            current_binary = board_to_binary(env.board)
            position_history.append(current_binary)
            
            # Fix: Use tuple for stable hashing instead of string conversion
            binary_tuple = tuple(map(int, current_binary))
            player_key = f"{current_player}_{binary_tuple}"
            if player_key in board_history:
                board_history[player_key] += 1
            else:
                board_history[player_key] = 1
            
            if current_player == PLAYER_B:
                legal_actions = env.get_legal_action_indices(player=PLAYER_B)
                if not legal_actions:
                    break
                
                # Score actions based on progress toward winning
                filtered_actions = []
                action_scores = {}
                
                for action in legal_actions:
                    board_copy = env.board.copy()
                    move = _action_index_to_move(action)
                    if move is None:
                        continue

                    start_r, start_c, end_r, end_c = move
                    if 0 <= start_r < 8 and 0 <= start_c < 4 and 0 <= end_r < 8 and 0 <= end_c < 4:
                        # Apply the move to see the resulting board
                        _apply_move_jit(board_copy, start_r, start_c, end_r, end_c)
                        
                        # Score based on board evaluation
                        score = _evaluate_board_jit(board_copy, PLAYER_B_ID)
                        
                        # Penalize moves that lead to repetitive positions
                        next_binary = board_to_binary(board_copy)
                        if detect_repetitive_play(position_history, next_binary):
                            score -= 50  # Heavy penalty for repetitive play
                        
                        # Boost score for moves that advance pieces toward goal
                        if end_r < start_r:  # Moving toward opponent's side
                            progress_boost = (start_r - end_r) * PROGRESS_REWARD
                            score += progress_boost
                        
                        action_scores[action] = score
                        
                        if score > -90:  # Filter out truly bad moves
                            filtered_actions.append(action)
                
                # Use filtered actions or fallback to all legal actions
                action_pool = filtered_actions if filtered_actions else legal_actions
                
                # Choose action from the agent
                action = agent.act(state, action_pool)
                next_state, reward, done, info = env.step(action)
                
                # Apply custom reward adjustments
                original_reward = reward
                
                # Adjust rewards: bigger win reward, heavy draw penalty
                if done:
                    if info.get('winner') == PLAYER_B:
                        # Win reward scaled by how quickly the game was won
                        turns_factor = 1.0 - (step / MAX_STEPS_PER_EPISODE) * 0.5
                        reward = WIN_REWARD * turns_factor  # Faster wins get higher rewards
                    elif info.get('winner') is None or info.get('winner') == 'DRAW':
                        reward = DRAW_PENALTY  # Heavy penalty for draws
                
                # Add a small penalty for repetitive moves
                player_pos_key = f"{PLAYER_B}_{tuple(map(int, board_to_binary(env.board)))}"
                if player_pos_key in board_history and board_history[player_pos_key] > MOVE_TIMEOUT:
                    reward -= 10  # Penalty for repeated positions
                
                # Apply step penalty
                reward += STEP_PENALTY

                # Validate reward
                reward = _validate_reward(reward)
                step_rewards.append(reward)
                
                # Log the reward change if significant
                if abs(reward - original_reward) > 1.0 and e % 100 == 0:
                    print(f"Reward adjusted: {original_reward:.1f} -> {reward:.1f}")
                
                agent.remember(state, action, reward, next_state, done)
                episode_reward += reward
                agent_steps += 1

                train_start = time.time()
                # Training step can go here if needed
                train_time += time.time() - train_start
            else:
                # Player A's turn (tournament agent)
                legal_actions = env.get_legal_action_indices(player=PLAYER_A)
                if not legal_actions:
                    break
                
                # For opponent, also filter actions that seem repetitive
                filtered_actions = []
                for action in legal_actions:
                    board_copy = env.board.copy()
                    move = _action_index_to_move(action)
                    if move is None:
                        continue
                        
                    start_r, start_c, end_r, end_c = move
                    if 0 <= start_r < 8 and 0 <= start_c < 4 and 0 <= end_r < 8 and 0 <= end_c < 4:
                        _apply_move_jit(board_copy, start_r, start_c, end_r, end_c)
                        next_binary = board_to_binary(board_copy)
                        
                        # Skip moves that lead to repetitive positions
                        if detect_repetitive_play(position_history, next_binary):
                            continue
                            
                        filtered_actions.append(action)
                
                # If all actions seem repetitive, fall back to all legal moves
                action_pool = filtered_actions if filtered_actions else legal_actions
                
                # Higher epsilon for tournament agent to encourage exploration
                old_epsilon = tournament_agent.epsilon
                tournament_agent.epsilon = max(0.1, old_epsilon) # Ensure at least 10% exploration
                action = tournament_agent.act(state, action_pool)
                tournament_agent.epsilon = old_epsilon  # Restore original epsilon
                
                next_state, reward, done, info = env.step(action)
            
            move_time += time.time() - step_start
            state = next_state
            
            if done:
                steps.append(step + 1)
                move_times.append(move_time)
                train_times.append(train_time)
                break
            
            # More aggressive timeout handling
            if step >= MAX_STEPS_PER_EPISODE - 1:
                done = True
                # reward = DRAW_PENALTY # This is now handled above if winner is DRAW
                info['winner'] = 'DRAW'
                info['timeout'] = True
                
                # If agent is about to time out, give it a negative reward
                if current_player == PLAYER_B:
                    # Apply step penalty before final reward adjustment for timeout
                    current_reward = STEP_PENALTY + DRAW_PENALTY 
                    current_reward = _validate_reward(current_reward)
                    agent.remember(state, action, current_reward, next_state, done)
                    episode_reward += current_reward
                break
        
        # Epsilon decay - keep some exploration in Phase 2
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
        
        scores.append(episode_reward)
        if info.get('winner') == PLAYER_B:
            wins.append(1)
            losses.append(0)
            draws.append(0)
        elif info.get('winner') == PLAYER_A:
            wins.append(0)
            losses.append(1)
            draws.append(0)
        else:
            wins.append(0)
            losses.append(0)
            draws.append(1)
        
        # Calculate metrics
        avg_score = np.mean(scores) if scores else 0.0
        win_rate = np.mean(wins) if wins else 0.0
        loss_rate = np.mean(losses) if losses else 0.0
        draw_rate = np.mean(draws) if draws else 0.0
        avg_steps = np.mean(steps) if steps else 0.0
        avg_move_time = np.mean(move_times) if move_times else 0.0
        avg_train_time = np.mean(train_times) if train_times else 0.0

        # Track best win rate model
        if win_rate > best_win_rate and len(wins) >= 50:
            best_win_rate = win_rate
            agent.save(BEST_MODEL_FILE)
            print(f"New best win rate model saved: {win_rate:.3f}")
            win_rate_no_improvement = 0
        else:
            win_rate_no_improvement += 1
        
        # If win rate hasn't improved for a while, reload the best model and continue
        if win_rate_no_improvement >= 2000 and os.path.exists(BEST_MODEL_FILE):
            try:
                print(f"Win rate stagnant for {win_rate_no_improvement} episodes. Reloading best model.")
                agent.load(BEST_MODEL_FILE)
                win_rate_no_improvement = 0
                # Reinitialize tournament agent from best model
                tournament_agent = copy.deepcopy(agent)
            except Exception as e:
                print(f"Error loading best model: {e}")

        if wandb_enabled:
            try:
                wandb.log({
                    "episode": e,
                    "episode_reward": episode_reward,
                    "avg_score": avg_score,
                    "win_rate": win_rate,
                    "loss_rate": loss_rate,
                    "draw_rate": draw_rate,
                    "epsilon": agent.epsilon,
                    "memory_size": len(agent),
                    "steps": agent_steps,
                    "avg_steps": avg_steps,
                    "avg_move_time": avg_move_time,
                    "avg_train_time": avg_train_time,
                    "timeout_rate": 1.0 if info.get('timeout', False) else 0.0,
                    "best_win_rate": best_win_rate
                })
            except Exception as ex:
                print(f"\nWarning: Failed to log to wandb: {ex}")
                wandb_enabled = False

        # Tournament with more focus on winning strategies
        if e % TOURNAMENT_FREQ == 0:
            print(f"\nRunning tournament at episode {e}")
            pre_weights = agent.model.get_weights()
            best_agent, tournament_best_score, tournament_matches = run_tournament(agent, direct_phase2=direct_phase2)
            best_weights = best_agent.model.get_weights()
            
            # Use more aggressive weight blending based on performance
            if tournament_best_score > 0:  # Only blend if tournament showed improvement
                blend_rate = 0.4  # More aggressive adoption of winning strategies
                new_weights = [pw * (1 - blend_rate) + bw * blend_rate 
                             for pw, bw in zip(pre_weights, best_weights)]
                agent.model.set_weights(new_weights)
                tournament_agent = copy.deepcopy(agent)
                tournament_agent.save(TOURNAMENT_MODEL_FILE)
                print(f"Tournament completed - updated agents with blended weights (blend rate: {blend_rate})")
            else:
                print(f"Tournament showed no improvement. Keeping current weights.")
            
            if wandb_enabled:
                try:
                    wandb.log({
                        "tournament_episode": e,
                        "tournament_best_score": tournament_best_score,
                        "tournament_matches": tournament_matches
                    })
                except Exception as err:  # Changed 'e' to 'err' to avoid shadowing
                    print(f"\nWarning: Failed to log tournament metrics to wandb: {err}")
                    wandb_enabled = False
        
        if e % SAVE_FREQ == 0:
            save_checkpoint(agent, e)
        
        if e % 100 == 0:
            print(f"Phase 2 - Episode: {e}/{episodes} | "
                  f"Score: {episode_reward:.2f} | "
                  f"Win Rate: {win_rate:.2f} | "
                  f"Loss Rate: {loss_rate:.2f} | "
                  f"Draw Rate: {draw_rate:.2f} | "
                  f"Epsilon: {agent.epsilon:.4f} | "
                  f"Avg Steps: {avg_steps:.2f} | "
                  f"Best Win Rate: {best_win_rate:.2f}")
    
    # At the end, save the best win rate model as the tournament model
    if os.path.exists(BEST_MODEL_FILE):
        import shutil
        shutil.copy(BEST_MODEL_FILE, TOURNAMENT_MODEL_FILE)
        print(f"Copied best win rate model to tournament model file")
    else:
        agent.save(TOURNAMENT_MODEL_FILE)
    
    agent.save_for_tfjs(TFJS_MODEL_DIR)
    print(f"Phase 2 training completed. Final model saved.")
    
    if wandb_enabled:
        try:
            wandb.finish()
        except:
            pass
    return agent

def direct_phase2_training(model_file, episodes=PHASE2_EPISODES, final_model_file="switcharoo_dqn_direct_phase2_final.weights.h5", enable_wandb=True):
    """Run Phase 2 training directly with an existing model file as input."""
    print(f"\n===== STARTING DIRECT PHASE 2 TRAINING WITH MODEL: {model_file} =====")
    
    agent = DQNAgent(
        learning_rate=0.00025,
        epsilon=0.01,
        epsilon_decay=1.0,
        epsilon_min=0.01,
        replay_buffer_size=500000,
        batch_size=64,
        target_update_freq=100
    )
    
    try:
        agent.load(model_file)
        print(f"Successfully loaded model from {model_file}")
    except Exception as e:
        print(f"Error loading model from {model_file}: {e}")
        return None
    
    start_episode = 1
    agent = phase2_training(agent, start_episode, episodes, direct_phase2=True, enable_wandb=enable_wandb)
    
    final_checkpoint_file = CHECKPOINT_FILE.format(episodes)
    agent.save(final_checkpoint_file)
    agent.save(final_model_file)
    print(f"Direct Phase 2 training completed. Final models saved to {final_checkpoint_file} and {final_model_file}")
    
    return agent