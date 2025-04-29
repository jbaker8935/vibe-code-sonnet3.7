"""Phase 2 training implementation: Tournament-based self-play training."""
import time
import copy
import numpy as np
from collections import deque
import wandb
from random import choice

from dqn_agent import DQNAgent
from env_const import (PLAYER_A, PLAYER_B, PLAYER_B_ID)
from env_util import (_evaluate_board_jit, _calculate_progress_reward, _apply_move_jit) 
from game_env import (SwitcharooEnv)
from train_dqn import save_checkpoint
from tournament import run_tournament
from utils import _action_index_to_move, _validate_reward, init_wandb
from config import (PHASE2_EPISODES, MAX_STEPS_PER_EPISODE, REPLAY_FREQUENCY,
                   TOURNAMENT_FREQ, TOURNAMENT_MODEL_FILE, SAVE_FREQ,
                   TFJS_MODEL_DIR, CHECKPOINT_FILE, initial_position)

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

    for e in range(start_episode, episodes + 1):
        move_time = train_time = 0
        episode_start = time.time()
        state = env.reset(choice(initial_position))
        episode_reward = 0
        agent_steps = 0
        
        for step in range(MAX_STEPS_PER_EPISODE):
            step_start = time.time()
            current_player = env.current_player
            
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
                reward = _validate_reward(reward)
                agent.remember(state, action, reward, next_state, done)
                episode_reward += reward
                agent_steps += 1

                train_start = time.time()
                train_time += time.time() - train_start
            else:
                legal_actions = env.get_legal_action_indices(player=PLAYER_A)
                if not legal_actions:
                    break
                
                action = tournament_agent.act(state, legal_actions)
                next_state, reward, done, info = env.step(action)
            
            move_time += time.time() - step_start
            state = next_state
            
            if done:
                steps.append(step + 1)
                move_times.append(move_time)
                train_times.append(train_time)
                break
            
            if step >= MAX_STEPS_PER_EPISODE - 1:
                done = True
                reward = -50.0
                info['winner'] = 'DRAW'
                info['timeout'] = True
                break
        
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
                    "memory_size": len(agent),  # Use len(agent)
                    "steps": agent_steps,
                    "avg_steps": avg_steps,
                    "avg_move_time": avg_move_time,
                    "avg_train_time": avg_train_time,
                    "timeout_rate": 1.0 if info.get('timeout', False) else 0.0
                })
            except Exception as ex:
                print(f"\nWarning: Failed to log to wandb: {ex}")
                wandb_enabled = False

        if e % TOURNAMENT_FREQ == 0:
            print(f"\nRunning tournament at episode {e}")
            pre_weights = agent.model.get_weights()
            best_agent, tournament_best_score, tournament_matches = run_tournament(agent, direct_phase2=direct_phase2)
            best_weights = best_agent.model.get_weights()
            blend_rate = 0.2
            new_weights = [pw * (1 - blend_rate) + bw * blend_rate 
                         for pw, bw in zip(pre_weights, best_weights)]
            agent.model.set_weights(new_weights)
            tournament_agent = copy.deepcopy(agent)
            tournament_agent.save(TOURNAMENT_MODEL_FILE)
            print(f"Tournament completed - updated agents with blended weights")
            
            if wandb_enabled:
                try:
                    wandb.log({
                        "tournament_episode": e,
                        "tournament_best_score": tournament_best_score,
                        "tournament_matches": tournament_matches
                    })
                except Exception as e:
                    print(f"\nWarning: Failed to log tournament metrics to wandb: {e}")
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
                  f"Avg Move Time: {avg_move_time:.4f} | "
                  f"Avg Train Time: {avg_train_time:.4f}")
    
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