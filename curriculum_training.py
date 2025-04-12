import numpy as np
import random
import time
import os
import copy
import tensorflow as tf
import argparse
from collections import deque
import wandb
from numba import njit
from random import choice  # Add this import for selecting random initial positions

from game_env import (SwitcharooEnv, PLAYER_A, PLAYER_B, PLAYER_B_ID, PLAYER_A_ID,
                     _evaluate_board_jit, _calculate_progress_reward, _apply_move_jit)
from dqn_agent import DQNAgent
from train_dqn import save_checkpoint, safe_replay

# Add configuration block at the start of the script
def configure_tensorflow():
    """Configure TensorFlow settings for optimal performance."""
    # Set memory growth and optimization flags before any other TF operations
    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
    os.environ['TF_GPU_THREAD_COUNT'] = '1'
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'
    
    # Use float32 for better numerical stability
    tf.keras.backend.set_floatx('float32')
    
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            # Enable memory growth
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
                
            # Set memory limit only, removed preallocate option
            for device in physical_devices:
                tf.config.set_logical_device_configuration(
                    device,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=3 * 1024)]  # 3GB limit
                )
            
            # Enable XLA and other optimizations
            tf.config.optimizer.set_jit(True)
            tf.config.optimizer.set_experimental_options({
                'layout_optimizer': True,
                'constant_folding': True,
                'shape_optimization': True,
                'remapping': True,
                'arithmetic_optimization': True,
                'dependency_optimization': True,
                'loop_optimization': True,
                'function_optimization': True,
                'debug_stripper': True,
            })
            
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
            print("Falling back to CPU...")
            
    # Optimize for CPU if no GPU
    else:
        tf.config.optimizer.set_jit(True)
        tf.config.threading.set_inter_op_parallelism_threads(4)
        tf.config.threading.set_intra_op_parallelism_threads(4)

# Add this call right after the imports
configure_tensorflow()

# Curriculum Training Configuration
PHASE1_EPISODES = 50000      # Episodes for Phase 1 (random opponent) 50000
PHASE2_EPISODES = 100000     # Episodes for Phase 2 (self-play) 100000
MAX_STEPS_PER_EPISODE = 300  # Maximum steps per episode
REPLAY_FREQUENCY = 2          # Frequency of replay buffer sampling

# Tournament Configuration
TOURNAMENT_FREQ = 1000       # How often to run tournaments
NUM_VARIANTS = 4             # Number of agent variants for tournament
NOISE_SCALE = 0.05          # Scale of Gaussian noise to apply to weights
TOURNAMENT_MATCHES = 20      # Matches per pair in tournament
SAVE_FREQ = 5000             # Save model weights every N episodes

# File paths for saving models
BASE_MODEL_FILE = "switcharoo_dqn_curriculum_phase1.weights.h5"
TOURNAMENT_MODEL_FILE = "switcharoo_dqn_tournament_best.weights.h5"
CHECKPOINT_FILE = "switcharoo_dqn_checkpoint_e{}.weights.h5"
TFJS_MODEL_FILE = "./switcharoo_tfjs_model/tf_model.weights.h5"

WANDB_PROJECT = "switcharoo-dqn"
WANDB_ENTITY = "farmerjohn1958-self"  # Replace with your wandb username

initial_position_base = [
    """\
BBBB
BBBB
....
....
....
....
AAAA
AAAA""",
    """\
....
BBBB
BBBB
....
....
AAAA
AAAA
....""",
    """\
....
....
BBBB
BBBB
AAAA
AAAA
....
....""",
    """\
BB..
BB..
BB..
BB..
AA..
AA..
AA..
AA..""",
    """\
..BB
..BB
..BB
..BB
..AA
..AA
..AA
..AA""",
    """\
.BB.
.BB.
.BB.
.BB.
.AA.
.AA.
.AA.
.AA.""",
    """\
BB..
BB..
BB..
BB..
..AA
..AA
..AA
..AA""",
    """\
BBBB
BB..
BB..
....
....
..AA
..AA
AAAA""",
    """\
..BB
..BB
BBBB
....
....
AAAA
AA..
AA..""",
    """\
B...
BB..
BBB.
BB..
..AA
.AAA
..AA
...A""",
    """\
B..B
.BB.
.BB.
B..B
A..A
.AA.
.AA.
A..A"""
]

initial_position = initial_position_base

@njit(cache=True)
def _action_index_to_move(action_index):
    """Convert action index to move coordinates (row1, col1, row2, col2)."""
    if action_index is None:
        return None
    start_r = action_index // 512
    start_c = (action_index % 512) // 64
    end_r = (action_index % 64) // 8
    end_c = action_index % 8
    return start_r, start_c, end_r, end_c

@njit(cache=True)
def get_opponent_action_fast(board_state, legal_actions, opponent_epsilon):
    """Ultra-fast opponent action selection."""
    if random.random() < opponent_epsilon:
        return legal_actions[random.randint(0, len(legal_actions) - 1)]
    
    best_score = float('-inf')
    best_action = legal_actions[0]
    
    # Use our local _action_index_to_move function
    for action in legal_actions:
        board_copy = board_state.copy()
        move = _action_index_to_move(action)
        if move is None:
            continue
        
        start_r, start_c, end_r, end_c = move
        if 0 <= start_r < 8 and 0 <= start_c < 8 and 0 <= end_r < 8 and 0 <= end_c < 8:
            _apply_move_jit(board_copy, start_r, start_c, end_r, end_c)
            score = _evaluate_board_jit(board_copy, PLAYER_A_ID)
            
            if score > best_score:
                best_score = score
                best_action = action
    
    return best_action

def get_opponent_action(env, opponent_epsilon=0.3):
    """Improved policy for the opponent using JIT-compiled evaluation."""
    legal_actions = env.get_legal_action_indices(player=PLAYER_A)
    if not legal_actions:
        return None
    
    # Fast path for random moves
    if random.random() < opponent_epsilon:
        return random.choice(legal_actions)
    
    # Sample subset of moves for evaluation
    eval_actions = legal_actions if len(legal_actions) <= 5 else random.sample(legal_actions, 5)
    return get_opponent_action_fast(env.board, eval_actions, opponent_epsilon)

def create_agent_variant(base_agent, noise_scale=NOISE_SCALE, epsilon=0.05):
    """Create a variant of the base agent by adding Gaussian noise to its weights."""
    variant = DQNAgent(epsilon=epsilon)  # Variable epsilon based on parameter
    
    # Get the base agent's weights and apply noise
    weights = base_agent.model.get_weights()
    noisy_weights = []
    
    for w in weights:
        # Add small Gaussian noise scaled by parameter magnitude
        noise = np.random.normal(0, noise_scale * (np.abs(w).mean() + 1e-8), w.shape)
        noisy_weights.append(w + noise)
    
    # Set the variant's weights
    variant.model.set_weights(noisy_weights)
    variant.update_target_model()
    return variant

def run_match(env, agent_a, agent_b, max_steps=MAX_STEPS_PER_EPISODE):
    """Run a single match between two agents and return the winner."""
    state = env.reset(choice(initial_position))  # Pass random initial position
    
    for step in range(max_steps):
        current_player = env.current_player
        
        if (current_player == PLAYER_A):
            # Agent A's turn
            legal_actions = env.get_legal_action_indices(player=PLAYER_A)
            if not legal_actions:
                return PLAYER_B  # B wins if A has no moves
            
            action = agent_a.act(state, legal_actions)
        else:
            # Agent B's turn
            legal_actions = env.get_legal_action_indices(player=PLAYER_B)
            if not legal_actions:
                return PLAYER_A  # A wins if B has no moves
            
            action = agent_b.act(state, legal_actions)
        
        next_state, _, done, info = env.step(action)
        state = next_state
        
        if done:
            winner = info.get('winner')
            return winner
    
    # If max steps reached, it's a draw
    return 'DRAW'

def run_tournament(base_agent, num_variants=NUM_VARIANTS, matches_per_pair=TOURNAMENT_MATCHES, direct_phase2=False):
    """Run a round-robin tournament between variants of the base agent."""
    print("\n----- STARTING TOURNAMENT -----")
    
    # Create variants - use epsilon=0.01 for direct_phase2_training
    epsilon_value = 0.01 if direct_phase2 else 0.05
    variants = [create_agent_variant(base_agent, epsilon=epsilon_value) for _ in range(num_variants)]
    variants.append(base_agent)  # Include the base agent in the tournament
    
    # Initialize scores dictionary
    scores = {i: 0 for i in range(len(variants))}
    
    env = SwitcharooEnv()
    total_matches = 0
    
    # Run round-robin matches
    for i in range(len(variants)):
        for j in range(len(variants)):
            if i == j:  # Skip playing against self
                continue
                
            agent_a = variants[i]
            agent_b = variants[j]
            
            # Play multiple matches per pair
            for _ in range(matches_per_pair):
                winner = run_match(env, agent_a, agent_b)
                
                if winner == PLAYER_A:
                    scores[i] += 1
                elif winner == PLAYER_B:
                    scores[j] += 1
                # Draw doesn't award points
                
                total_matches += 1
    
    # Find the best variant
    best_idx = max(scores, key=scores.get)
    best_score = scores[best_idx]
    
    print(f"Tournament completed: {total_matches} matches played")
    print(f"Best variant: {best_idx} with score {best_score}")
    
    for i, score in scores.items():
        print(f"Variant {i}: {score} points")
    
    return variants[best_idx], best_score, total_matches

def init_wandb(enable_wandb, run_name, agent, group_name=None): # Add agent parameter
    """Initialize wandb with error handling"""
    if not enable_wandb:
        return False

    try:
        # Ensure agent is not None before accessing attributes
        if agent is None:
             print("\nWarning: Agent object is None, cannot initialize wandb config.")
             return False

        wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name=run_name,
            group=group_name,
            config={
                # Access attributes from the passed agent object
                "learning_rate": agent.learning_rate,
                "gamma": agent.gamma,
                "epsilon": agent.epsilon,
                "epsilon_decay": agent.epsilon_decay,
                "epsilon_min": agent.epsilon_min,
                "batch_size": agent.batch_size,
                "replay_buffer_size": agent.memory.maxlen,
                "target_update_freq": agent.target_update_freq,
                "gradient_clip_norm": agent.optimizer.clipnorm if hasattr(agent.optimizer, 'clipnorm') else None # Check if clipnorm exists
            }
        )
        return True
    except Exception as e:
        print(f"\nWarning: Failed to initialize wandb: {e}")
        print("Continuing training without wandb logging...")
        return False

@njit(cache=True)
def _validate_reward(reward):
    """JIT-compiled reward validation."""
    if np.isnan(reward) or np.isinf(reward):
        return 0.0
    return reward

def phase1_training(agent, start_episode=1, episodes=PHASE1_EPISODES, enable_wandb=True):
    """Phase 1: Train against a progressively stronger opponent."""
    # Preallocate memory for better performance
    env = SwitcharooEnv()
    scores = deque(maxlen=100)
    wins = deque(maxlen=100)
    losses = deque(maxlen=100)
    draws = deque(maxlen=100)
    steps = deque(maxlen=100)
    move_times = deque(maxlen=100)
    train_times = deque(maxlen=100) 
    best_avg_score = float('-inf')
    
    # Reduce wandb logging frequency
    log_freq = 10  # Only log every 10 episodes
    
    # Initialize wandb with fewer metrics, passing the agent object
    wandb_enabled = init_wandb(enable_wandb, "phase1_training", agent) # Pass agent here
    
    # Batch training flag
    should_train = False
    
    # Adjusted opponent epsilon decay for progressive difficulty
    opponent_epsilon_start = 0.9
    opponent_epsilon_end = 0.2  # Opponent becomes less random but not fully deterministic
    opponent_epsilon_decay = (opponent_epsilon_end / opponent_epsilon_start) ** (1 / episodes)
    opponent_epsilon = opponent_epsilon_start
    
    # Adjust agent's epsilon decay for more exploration
    agent_epsilon_decay = (agent.epsilon_min / agent.epsilon) ** (1 / (episodes * 0.75))  # Slower decay

    for e in range(start_episode, episodes + 1):
        move_time = 0
        eval_time = 0
        train_time = 0
        episode_start = time.time()
        
        state = env.reset(choice(initial_position))
        episode_reward = 0
        agent_steps = 0
        
        for step in range(MAX_STEPS_PER_EPISODE):
            current_player = env.current_player
            step_start = time.time()
            
            if current_player == PLAYER_B:  # Agent's turn
                legal_actions = env.get_legal_action_indices(player=PLAYER_B)
                if not legal_actions:
                    break
                
                # Filter legal actions to exclude losing moves
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
                            # Remember this move as a bad move with -100 reward
                            agent.remember(state, action, -100, state, False)
                        else:
                            filtered_actions.append(action)

                # Use filtered actions for the agent's decision
                action = agent.act(state, filtered_actions if filtered_actions else legal_actions)
                next_state, reward, done, info = env.step(action)
                
                # Reward shaping: Penalize draws, incentivize faster wins, and reward progress
                if done:
                    if info.get('winner') == PLAYER_B:
                        reward += 100  # Win bonus
                    elif info.get('winner') == PLAYER_A:
                        reward -= 50  # Loss penalty
                    else:  # Draw
                        reward -= 20  # Draw penalty
                else:
                    # Use the new progress reward function
                    progress_reward = _calculate_progress_reward(env.board, PLAYER_B_ID) * 10.0
                    reward += progress_reward

                reward = _validate_reward(reward)  # Use JIT for reward validation
                agent.remember(state, action, reward, next_state, done)
                episode_reward += reward
            else:
                # Use improved opponent policy
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
        
        # Additional training at episode end
        if len(agent.memory) > agent.batch_size:
            safe_replay(agent)
        
        # Apply epsilon decay after each episode
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent_epsilon_decay
            
        # Decay opponent epsilon
        opponent_epsilon *= opponent_epsilon_decay
        
        # Track performance
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
            
        avg_score = np.mean(scores) if scores else 0.0
        win_rate = np.mean(wins) if wins else 0.0
        loss_rate = np.mean(losses) if losses else 0.0
        draw_rate = np.mean(draws) if draws else 0.0
        avg_steps = np.mean(steps) if steps else 0.0
        avg_move_time = np.mean(move_times) if move_times else 0.0
        avg_train_time = np.mean(train_times) if train_times else 0.0

        # Log metrics to wandb
        if wandb_enabled and e % log_freq == 0:
            try:
                wandb.log({
                    "episode": e,
                    "episode_reward": episode_reward,
                    "avg_score": avg_score, # Use safe average
                    "win_rate": win_rate,   # Use safe average
                    "loss_rate": loss_rate, # Use safe average
                    "draw_rate": draw_rate, # Use safe average
                    "avg_steps": avg_steps, # Use safe average
                    "avg_move_time": avg_move_time, # Use safe average
                    "avg_train_time": avg_train_time, # Use safe average
                    "epsilon": agent.epsilon,
                    "opponent_epsilon": opponent_epsilon,
                    "memory_size": len(agent.memory),
                    "steps": agent_steps, # This should be the steps in the current episode
                    "time_per_episode": time.time() - episode_start
                })
            except Exception as ex: # Renamed variable to avoid conflict
                print(f"\nWarning: Failed to log to wandb: {ex}")
                wandb_enabled = False

        # Save best model
        if avg_score > best_avg_score and len(scores) >= 50: # Check length before comparing
            best_avg_score = avg_score
            agent.save(BASE_MODEL_FILE)
            print(f"New best model saved with avg score: {best_avg_score:.2f}")

        # Periodic checkpoint
        if e % SAVE_FREQ == 0:
            save_checkpoint(agent, e)

        # Display progress
        if e % 100 == 0:
            # Recalculate safe averages for display
            win_rate_disp = np.mean(wins) if wins else 0.0
            loss_rate_disp = np.mean(losses) if losses else 0.0
            draw_rate_disp = np.mean(draws) if draws else 0.0
            avg_score_disp = np.mean(scores) if scores else 0.0
            avg_steps_disp = np.mean(steps) if steps else 0.0
            avg_move_time_disp = np.mean(move_times) if move_times else 0.0
            avg_train_time_disp = np.mean(train_times) if train_times else 0.0

            total_time = time.time() - episode_start
            print(f"Phase 1 - Episode: {e}/{episodes} | "
                  f"Score: {episode_reward:.2f} | "
                  f"Avg Steps: {avg_steps_disp:.2f} | "
                  f"Avg Move Time: {avg_move_time_disp:.4f} | "
                  f"Avg Train Time: {avg_train_time_disp:.4f} | "
                  f"Win Rate: {win_rate_disp:.2f} | "
                  f"Loss Rate: {loss_rate_disp:.2f} | "
                  f"Draw Rate: {draw_rate_disp:.2f} | "
                  f"Epsilon: {agent.epsilon:.4f} | "
                  f"Opponent Epsilon: {opponent_epsilon:.4f} | "
                  f"Avg Score: {avg_score_disp:.2f}")

    
    # Save final Phase 1 model
    agent.save(BASE_MODEL_FILE)
    print(f"Phase 1 training completed. Final model saved to {BASE_MODEL_FILE}")
    
    if wandb_enabled:
        try:
            wandb.finish()
        except:
            pass
    return agent

def phase2_training(agent, start_episode=1, episodes=PHASE2_EPISODES, direct_phase2=False, enable_wandb=True):
    """Phase 2: Tournament self-play training."""
    # Initialize wandb, passing the agent object
    wandb_enabled = init_wandb(enable_wandb, "phase2_training", agent) # Pass agent here

    print("\n===== STARTING PHASE 2: TOURNAMENT SELF-PLAY =====")
    
    env = SwitcharooEnv()
    scores = deque(maxlen=100)
    wins = deque(maxlen=100)
    losses = deque(maxlen=100)  # Track losses
    draws = deque(maxlen=100)   # Track draws
    tournament_agent = copy.deepcopy(agent)
    steps = deque(maxlen=100)
    move_times = deque(maxlen=100)
    train_times = deque(maxlen=100)

    for e in range(start_episode, episodes + 1):
        move_time = 0
        train_time = 0
        episode_start = time.time()
        state = env.reset(choice(initial_position))  # Pass random initial position
        episode_reward = 0
        agent_steps = 0
        
        for step in range(MAX_STEPS_PER_EPISODE):
            step_start = time.time()
            current_player = env.current_player
            
            if current_player == PLAYER_B:  # Learning agent's turn
                legal_actions = env.get_legal_action_indices(player=PLAYER_B)
                if not legal_actions:
                    break
                
                # Filter legal actions to exclude losing moves
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
                            # Remember this move as a bad move with -100 reward
                            agent.remember(state, action, -100, state, False)
                        else:
                            filtered_actions.append(action)

                # Use filtered actions for the agent's decision
                action = agent.act(state, filtered_actions if filtered_actions else legal_actions)
                next_state, reward, done, info = env.step(action)
                reward = _validate_reward(reward)  # Use JIT for reward validation
                agent.remember(state, action, reward, next_state, done)
                episode_reward += reward
                agent_steps += 1

                train_start = time.time()
                if len(agent.memory) > agent.batch_size and agent_steps % REPLAY_FREQUENCY == 0:
                    safe_replay(agent)
                train_time += time.time() - train_start
            else:  # Tournament agent's turn (opponent)
                legal_actions = env.get_legal_action_indices(player=PLAYER_A)
                if not legal_actions:
                    break
                
                # Tournament agent plays as opponent
                action = tournament_agent.act(state, legal_actions)
                next_state, reward, done, info = env.step(action)
            
            move_time += time.time() - step_start
            state = next_state
            
            if done:
                steps.append(step + 1)
                move_times.append(move_time)
                train_times.append(train_time)
                break
            
            # Force end game if taking too long
            if step >= MAX_STEPS_PER_EPISODE - 1:
                done = True
                reward = -50.0  # Heavy penalty for timeout
                info['winner'] = 'DRAW'
                info['timeout'] = True
                break
        
        avg_steps = np.mean(steps) if steps else 0.0
        avg_move_time = np.mean(move_times) if move_times else 0.0
        avg_train_time = np.mean(train_times) if train_times else 0.0

        # Additional training at episode end
        if len(agent.memory) > agent.batch_size:
            safe_replay(agent)
            
        # Apply epsilon decay after each episode
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
        
        # Track performance
        scores.append(episode_reward)
        # Track game outcome
        if info.get('winner') == PLAYER_B:
            wins.append(1)
            losses.append(0)
            draws.append(0)
        elif info.get('winner') == PLAYER_A:
            wins.append(0)
            losses.append(1)
            draws.append(0)
        else:  # Draw
            wins.append(0)
            losses.append(0)
            draws.append(1)
        
        # Calculate averages safely before logging
        win_rate = np.mean(wins) if wins else 0.0
        loss_rate = np.mean(losses) if losses else 0.0
        draw_rate = np.mean(draws) if draws else 0.0
        avg_score = np.mean(scores) if scores else 0.0

        if wandb_enabled:
            try:
                wandb.log({
                    "episode": e,
                    "episode_reward": episode_reward,
                    "avg_score": avg_score, # Use safe average
                    "win_rate": win_rate,   # Use safe average
                    "loss_rate": loss_rate, # Use safe average
                    "draw_rate": draw_rate, # Use safe average
                    "epsilon": agent.epsilon,
                    "memory_size": len(agent.memory),
                    "steps": agent_steps, # This should be the steps in the current episode
                    "avg_steps": avg_steps, # Use safe average
                    "avg_move_time": avg_move_time, # Use safe average
                    "avg_train_time": avg_train_time, # Use safe average
                    "timeout_rate": 1.0 if info.get('timeout', False) else 0.0
                })
            except Exception as ex: # Renamed variable to avoid conflict
                print(f"\nWarning: Failed to log to wandb: {ex}")
                wandb_enabled = False

        # Run tournament and update best agent
        if e % TOURNAMENT_FREQ == 0:
            print(f"\nRunning tournament at episode {e}")
            best_agent, tournament_best_score, tournament_matches = run_tournament(agent, direct_phase2=direct_phase2)
            
            # Update tournament agent with best weights
            tournament_agent = copy.deepcopy(best_agent)
            tournament_agent.save(TOURNAMENT_MODEL_FILE)
            
            # Also update our main agent to the best one
            agent = copy.deepcopy(best_agent)
            
            print(f"Tournament completed - new best agent selected")
            
            # Additional tournament metrics
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
        
        # Periodic checkpoint
        if e % SAVE_FREQ == 0:
            save_checkpoint(agent, e)
        
        # Display progress
        if e % 100 == 0:
            # Recalculate safe averages for display
            win_rate_disp = np.mean(wins) if wins else 0.0
            loss_rate_disp = np.mean(losses) if losses else 0.0
            draw_rate_disp = np.mean(draws) if draws else 0.0
            avg_score_disp = np.mean(scores) if scores else 0.0
            avg_steps_disp = np.mean(steps) if steps else 0.0
            avg_move_time_disp = np.mean(move_times) if move_times else 0.0
            avg_train_time_disp = np.mean(train_times) if train_times else 0.0

            print(f"Phase 2 - Episode: {e}/{episodes} | "
                  f"Score: {episode_reward:.2f} | "
                  f"Win Rate: {win_rate_disp:.2f} | "
                  f"Loss Rate: {loss_rate_disp:.2f} | "
                  f"Draw Rate: {draw_rate_disp:.2f} | "
                  f"Epsilon: {agent.epsilon:.4f} | "
                  f"Avg Steps: {avg_steps_disp:.2f} | "
                  f"Avg Move Time: {avg_move_time_disp:.4f} | "
                  f"Avg Train Time: {avg_train_time_disp:.4f}") # Use safe average
    
    # Save final model
    agent.save(TOURNAMENT_MODEL_FILE)
    agent.save_for_tfjs(TFJS_MODEL_FILE)
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
    
    # Initialize agent *before* calling phase2_training (which calls init_wandb)
    agent = DQNAgent(
        learning_rate=0.00025,
        epsilon=0.01,
        epsilon_decay=1.0,
        epsilon_min=0.01,
        replay_buffer_size=500000,
        batch_size=64,
        target_update_freq=100
        # gradient_clip_norm=1.0 # Make sure this matches agent definition if used
    )
    
    # Load the provided model file
    try:
        agent.load(model_file)
        print(f"Successfully loaded model from {model_file}")
    except Exception as e:
        print(f"Error loading model from {model_file}: {e}")
        return None
    
    # Run Phase 2 training (this will call init_wandb internally)
    start_episode = 1
    agent = phase2_training(agent, start_episode, episodes, direct_phase2=True, enable_wandb=enable_wandb)
    
    # Save the final model
    final_checkpoint_file = CHECKPOINT_FILE.format(episodes)
    agent.save(final_checkpoint_file)
    agent.save(final_model_file)
    print(f"Direct Phase 2 training completed. Final models saved to {final_checkpoint_file} and {final_model_file}")
    
    return agent

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Curriculum Training for Switcharoo DQN Agent")
    parser.add_argument("--phase2-only", action="store_true", help="Run only Phase 2 training")
    parser.add_argument("--model-file", type=str, help="Path to an existing model file to load for direct Phase 2 training")
    parser.add_argument("--episodes", type=int, default=PHASE2_EPISODES, help=f"Number of episodes for Phase 2 training (default: {PHASE2_EPISODES})")
    parser.add_argument("--final-model-file", type=str, default="switcharoo_dqn_direct_phase2_final.weights.h5", 
                        help="Filename for the final model (default: switcharoo_dqn_direct_phase2_final.weights.h5)")
    parser.add_argument("--wandb-project", type=str, default=WANDB_PROJECT,
                       help="Weights & Biases project name")
    parser.add_argument("--wandb-entity", type=str, default=WANDB_ENTITY,
                       help="Weights & Biases entity/username")
    parser.add_argument("--disable-wandb", action="store_true",
                       help="Disable Weights & Biases logging")
    parser.add_argument("--profile", action="store_true",
                       help="Enable profiling using cProfile")
    parser.add_argument("--profile-output", type=str, default="program.prof",
                       help="Output file for profiling data (default: program.prof)")
    parser.add_argument("--resume", action="store_true",
                       help="Resume training from latest checkpoint if available")
    args = parser.parse_args()

    # Update wandb settings from args
    WANDB_PROJECT = args.wandb_project
    WANDB_ENTITY = args.wandb_entity
    enable_wandb = not args.disable_wandb
    
    # Add these environment variable settings before parsing arguments
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging verbosity
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'  # Use async GPU memory allocator

    # If profiling is enabled, import cProfile and wrap the main training process
    if args.profile:
        import cProfile
        import pstats
        profiler = cProfile.Profile()
        profiler.enable()
        
    try:
        # Direct Phase 2 Training Mode
        if args.phase2_only and args.model_file:
            print(f"Starting direct Phase 2 training with model: {args.model_file}")
            # direct_phase2_training now handles agent creation and loading internally
            direct_phase2_training(
                model_file=args.model_file,
                episodes=args.episodes,
                final_model_file=args.final_model_file,
                enable_wandb=enable_wandb
            )
        else:
            # Standard Curriculum Training (Phase 1 + Phase 2)
            # Initialize agent *before* calling phase1_training or phase2_training
            agent = DQNAgent(
                learning_rate=0.001, # Or adjust as needed for Phase 1
                epsilon=1.0,
                epsilon_decay=.9995, # Adjust if needed
                epsilon_min=0.01,
                replay_buffer_size=500000,
                batch_size=64,
                target_update_freq=100
                # gradient_clip_norm=1.0 # Make sure this matches agent definition if used
            )
            
            # Check for existing checkpoints only if resume flag is set
            start_episode = 1
            if args.resume:
                try:
                    # Specify the directory where checkpoints are stored
                    checkpoint_dir = "./"  # Update this to the correct directory if needed
                    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("switcharoo_dqn_checkpoint_e")]

                    if checkpoints:
                        # Sort checkpoints by episode number
                        latest_checkpoint = max(
                            checkpoints,
                            key=lambda x: int(x.split('_e')[-1].split('.weights')[0])
                        )
                        start_episode = int(latest_checkpoint.split('_e')[-1].split('.weights')[0]) + 1
                        agent.load(os.path.join(checkpoint_dir, latest_checkpoint))
                        print(f"Resuming from checkpoint {latest_checkpoint} at episode {start_episode}")
                    else:
                        print("No checkpoints found. Starting from scratch.")
                except Exception as e:
                    print(f"Error loading checkpoint: {e}")
                    start_episode = 1
            else:
                print("Starting new training session from scratch.")
            
            # Check if Phase 1 is completed
            phase1_completed = os.path.exists(BASE_MODEL_FILE)
            
            try:
                # Phase 1: Train against random opponent
                if not phase1_completed:
                    agent = phase1_training(agent, start_episode, enable_wandb=enable_wandb)
                    start_episode = 1  # Reset episode counter for Phase 2
                else:
                    print(f"Phase 1 model found: {BASE_MODEL_FILE}. Loading...")
                    agent.load(BASE_MODEL_FILE)
                
                # Phase 2: Tournament self-play
                agent.epsilon = 0.01  # Set epsilon to a fixed value for Phase 2
                agent.epsilon_decay = 1.0  # No decay in Phase 2
                agent.epsilon_min = 0.01  # Minimum epsilon for Phase 2
                agent = phase2_training(agent, start_episode, enable_wandb=enable_wandb)
                
                print("Curriculum training completed successfully!")
                
            except KeyboardInterrupt:
                print("\nTraining interrupted by user")
                save_checkpoint(agent, start_episode, emergency=True)
            except Exception as e:
                print(f"\nTraining crashed with error: {e}")
                save_checkpoint(agent, start_episode, emergency=True)
                raise
    finally:
        # If profiling was enabled, save and print results
        if args.profile:
            profiler.disable()
            # Save profiling results to file
            profiler.dump_stats(args.profile_output)
            print(f"\nProfiling data saved to {args.profile_output}")
            
            # Print the top 20 time-consuming functions
            stats = pstats.Stats(args.profile_output)
            print("\nTop 20 time-consuming operations:")
            stats.strip_dirs().sort_stats('cumulative').print_stats(20)
            
            print(f"\nTo analyze the profile data in more detail, you can run:")
            print(f"python -m pstats {args.profile_output}")
            print(f"Or visualize it with: snakeviz {args.profile_output}")

