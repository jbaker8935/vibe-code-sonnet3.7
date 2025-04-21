"""Phase 1 training implementation: Training against random opponent."""
import time
import numpy as np
from collections import deque
import wandb
from random import choice

from env_const import (PLAYER_A, PLAYER_B, PLAYER_B_ID)
from env_util import (_evaluate_board_jit, _calculate_progress_reward, _apply_move_jit)

from game_env import (SwitcharooEnv)
from train_dqn import save_checkpoint, safe_replay
from utils import _action_index_to_move, _validate_reward, init_wandb
from config import (PHASE1_EPISODES, MAX_STEPS_PER_EPISODE, REPLAY_FREQUENCY,
                   BASE_MODEL_FILE, SAVE_FREQ, initial_position)

def get_opponent_action_fast(board_state, legal_actions, opponent_epsilon):
    """Ultra-fast opponent action selection."""
    if np.random.random() < opponent_epsilon:
        return legal_actions[np.random.randint(0, len(legal_actions) - 1)]
    
    best_score = float('-inf')
    best_action = legal_actions[0]
    
    for action in legal_actions:
        board_copy = board_state.copy()
        move = _action_index_to_move(action)
        if move is None:
            continue
        
        start_r, start_c, end_r, end_c = move
        if 0 <= start_r < 8 and 0 <= start_c < 8 and 0 <= end_r < 8 and 0 <= end_c < 8:
            _apply_move_jit(board_copy, start_r, start_c, end_r, end_c)
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
    
    eval_actions = legal_actions if len(legal_actions) <= 5 else np.random.choice(legal_actions, 5, replace=False)
    return get_opponent_action_fast(env.board, eval_actions, opponent_epsilon)

def phase1_training(agent, start_episode=1, episodes=PHASE1_EPISODES, enable_wandb=True):
    """Phase 1: Train against a progressively stronger opponent."""
    env = SwitcharooEnv()
    scores = deque(maxlen=100)
    wins = deque(maxlen=100)
    losses = deque(maxlen=100)
    draws = deque(maxlen=100)
    steps = deque(maxlen=100)
    move_times = deque(maxlen=100)
    train_times = deque(maxlen=100)
    best_avg_score = float('-inf')
    
    log_freq = 10
    wandb_enabled = init_wandb(enable_wandb, "phase1_training", agent)
    
    opponent_epsilon_start = 0.3
    opponent_epsilon_end = 0.01
    opponent_epsilon_decay = (opponent_epsilon_end / opponent_epsilon_start) ** (1 / episodes)
    opponent_epsilon = opponent_epsilon_start
    
    agent_epsilon_decay = (agent.epsilon_min / agent.epsilon) ** (1 / (episodes * 0.75))

    for e in range(start_episode, episodes + 1):
        move_time = train_time = 0
        episode_start = time.time()
        state = env.reset(choice(initial_position))
        episode_reward = 0
        agent_steps = 0
        
        for step in range(MAX_STEPS_PER_EPISODE):
            current_player = env.current_player
            step_start = time.time()
            
            if current_player == PLAYER_B:
                legal_actions = env.get_legal_action_indices(player=PLAYER_B)
                if not legal_actions:
                    break
                
                filtered_actions = []
                for action in legal_actions:
                    board_copy = env.board.copy()
                    move = _action_index_to_move(action)
                    if move is None:
                        continue

                    start_r, start_c, end_r, end_c = move
                    if 0 <= start_r < 8 and 0 <= start_c < 8 and 0 <= end_r < 8 and 0 <= end_c < 8:
                        _apply_move_jit(board_copy, start_r, start_c, end_r, end_c)
                        score = _evaluate_board_jit(board_copy, PLAYER_B_ID)

                        if score == -100.0:
                            agent.remember(state, action, -100, state, False)
                        else:
                            filtered_actions.append(action)

                action = agent.act(state, filtered_actions if filtered_actions else legal_actions)
                next_state, reward, done, info = env.step(action)
                
                if done:
                    if info.get('winner') == PLAYER_B:
                        reward += 150  # Increased reward for winning to encourage aggressive play
                        # Bonus for winning in fewer moves
                        steps_bonus = (MAX_STEPS_PER_EPISODE - step) * 0.5
                        reward += steps_bonus
                    elif info.get('winner') == PLAYER_A:
                        reward -= 50
                    else:
                        reward -= 40  # Increased penalty for draws to discourage defensive play
                else:
                    progress_reward = _calculate_progress_reward(env.board, PLAYER_B_ID) * 7.0  # Reduced progress reward
                    reward += progress_reward

                reward = _validate_reward(reward)
                agent.remember(state, action, reward, next_state, done)
                episode_reward += reward
            else:
                action = get_opponent_action(env, opponent_epsilon=opponent_epsilon)
                if action is None:
                    break
                
                move_time += time.time() - step_start
                next_state, _, done, info = env.step(action)
            
            state = next_state
            
            if len(agent.memory) > agent.batch_size and agent_steps % REPLAY_FREQUENCY == 0:
                train_start = time.time()
                safe_replay(agent)
                train_time += time.time() - train_start
            
            if done:
                steps.append(step+1)
                train_times.append(train_time)
                move_times.append(move_time)
                break
        
        if len(agent.memory) > agent.batch_size:
            safe_replay(agent)
        
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent_epsilon_decay
            
        opponent_epsilon *= opponent_epsilon_decay
        
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
        avg_score = np.mean(scores)
        win_rate = np.mean(wins)
        loss_rate = np.mean(losses)
        draw_rate = np.mean(draws)
        avg_steps = np.mean(steps)
        avg_move_time = np.mean(move_times)
        avg_train_time = np.mean(train_times)

        if wandb_enabled and e % log_freq == 0:
            try:
                wandb.log({
                    "episode": e,
                    "episode_reward": episode_reward,
                    "avg_score": avg_score,
                    "win_rate": win_rate,
                    "loss_rate": loss_rate,
                    "draw_rate": draw_rate,
                    "avg_steps": avg_steps,
                    "avg_move_time": avg_move_time,
                    "avg_train_time": avg_train_time,
                    "epsilon": agent.epsilon,
                    "opponent_epsilon": opponent_epsilon,
                    "memory_size": len(agent.memory),
                    "steps": agent_steps,
                    "time_per_episode": time.time() - episode_start
                })
            except Exception as e:
                print(f"\nWarning: Failed to log to wandb: {e}")
                wandb_enabled = False

        if avg_score > best_avg_score and len(scores) >= 50:
            best_avg_score = avg_score
            agent.save(BASE_MODEL_FILE)
            print(f"New best model saved with avg score: {best_avg_score:.2f}")

        if e % SAVE_FREQ == 0:
            save_checkpoint(agent, e)

        if e % 100 == 0:
            print(f"Phase 1 - Episode: {e}/{episodes} | "
                  f"Score: {episode_reward:.2f} | "
                  f"Avg Steps: {avg_steps:.2f} | "
                  f"Avg Move Time: {avg_move_time:.4f} | "
                  f"Avg Train Time: {avg_train_time:.4f} | "
                  f"Win Rate: {win_rate:.2f} | "
                  f"Loss Rate: {loss_rate:.2f} | "
                  f"Draw Rate: {draw_rate:.2f} | "
                  f"Epsilon: {agent.epsilon:.4f} | "
                  f"Opponent Epsilon: {opponent_epsilon:.4f} | "
                  f"Avg Score: {avg_score:.2f}")
    
    agent.save(BASE_MODEL_FILE)
    print(f"Phase 1 training completed. Final model saved to {BASE_MODEL_FILE}")
    
    if wandb_enabled:
        try:
            wandb.finish()
        except:
            pass
    return agent