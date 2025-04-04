import numpy as np
import random
import time
import os
import copy
import tensorflow as tf
import argparse
from collections import deque
import wandb

from game_env import SwitcharooEnv, PLAYER_A, PLAYER_B
from dqn_agent import DQNAgent
from train_dqn import save_checkpoint, safe_replay

# Curriculum Training Configuration
PHASE1_EPISODES = 50000      # Episodes for Phase 1 (random opponent) 50000
PHASE2_EPISODES = 100000     # Episodes for Phase 2 (self-play) 100000
MAX_STEPS_PER_EPISODE = 300  # Maximum steps per episode

# Tournament Configuration
TOURNAMENT_FREQ = 1000       # How often to run tournaments
NUM_VARIANTS = 4             # Number of agent variants for tournament
NOISE_SCALE = 0.02          # Scale of Gaussian noise to apply to weights
TOURNAMENT_MATCHES = 20      # Matches per pair in tournament
SAVE_FREQ = 5000             # Save model weights every N episodes

# File paths for saving models
BASE_MODEL_FILE = "switcharoo_dqn_curriculum_phase1.weights.h5"
TOURNAMENT_MODEL_FILE = "switcharoo_dqn_tournament_best.weights.h5"
CHECKPOINT_FILE = "switcharoo_dqn_checkpoint_e{}.weights.h5"
TFJS_MODEL_FILE = "./switcharoo_tfjs_model/tf_model.weights.h5"

WANDB_PROJECT = "switcharoo-dqn"
WANDB_ENTITY = "farmerjohn1958-self"  # Replace with your wandb username

def get_opponent_action(env):
    """Improved policy for the opponent: uses board evaluation instead of random moves."""
    from game_env import _evaluate_board_jit, _apply_move_jit, PLAYER_A_ID
    
    legal_actions = env.get_legal_action_indices(player=PLAYER_A)
    if not legal_actions:
        return None # No legal moves
    
    # Get the current board state
    board_state = env.board.copy()
    
    # If there are many legal actions, sample a subset for faster evaluation
    if len(legal_actions) > 10:
        # Always evaluate some random actions for exploration
        evaluation_actions = random.sample(legal_actions, min(10, len(legal_actions)))
    else:
        evaluation_actions = legal_actions
    
    best_score = float('-inf')
    best_action = None
    
    # Evaluate each action
    for action in evaluation_actions:
        # Convert action index to move
        move = env._action_index_to_move(action)
        if move is None:
            continue
            
        start_r, start_c, end_r, end_c = move
        
        # Create a copy of the board to simulate the move
        board_copy = board_state.copy()
        
        # Apply the move
        _apply_move_jit(board_copy, start_r, start_c, end_r, end_c)
        
        # Evaluate the resulting board state for Player A
        score = _evaluate_board_jit(board_copy, PLAYER_A_ID)
        
        # Track the best action
        if score > best_score:
            best_score = score
            best_action = action
    
    # If evaluation failed or all moves score the same, fall back to random choice
    if best_action is None:
        return random.choice(legal_actions)
        
    return best_action

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
    state = env.reset()
    
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

def init_wandb(enable_wandb, run_name, group_name=None):
    """Initialize wandb with error handling"""
    if not enable_wandb:
        return False
        
    try:
        wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name=run_name,
            group=group_name,
            config={
                "epsilon": agent.epsilon,
                "epsilon_decay": agent.epsilon_decay,
                "epsilon_min": agent.epsilon_min,
                "batch_size": agent.batch_size,
                "replay_buffer_size": agent.memory.maxlen,
                "target_update_freq": agent.target_update_freq
            }
        )
        return True
    except Exception as e:
        print(f"\nWarning: Failed to initialize wandb: {e}")
        print("Continuing training without wandb logging...")
        return False

def phase1_training(agent, start_episode=1, episodes=PHASE1_EPISODES, enable_wandb=True):
    """Phase 1: Train against a random opponent."""
    # Initialize wandb
    wandb_enabled = init_wandb(enable_wandb, "phase1_training")

    print("\n===== STARTING PHASE 1: TRAINING AGAINST RANDOM OPPONENT =====")
    
    env = SwitcharooEnv()
    scores = deque(maxlen=100)
    wins = deque(maxlen=100)
    best_avg_score = float('-inf')
    
    for e in range(start_episode, episodes + 1):
        state = env.reset()
        episode_reward = 0
        agent_steps = 0
        episode_start = time.time()
        
        for step in range(MAX_STEPS_PER_EPISODE):
            current_player = env.current_player
            
            if current_player == PLAYER_B:  # Agent's turn
                legal_actions = env.get_legal_action_indices(player=PLAYER_B)
                if not legal_actions:
                    break
                
                action = agent.act(state, legal_actions)
                next_state, reward, done, info = env.step(action)
                
                if np.isnan(reward) or np.isinf(reward):
                    reward = 0.0
                
                agent.remember(state, action, reward, next_state, done)
                episode_reward += reward
                agent_steps += 1
            else:
                action = get_opponent_action(env)
                if action is None:
                    break
                
                next_state, reward, done, info = env.step(action)
            
            state = next_state
            
            if len(agent.memory) > agent.batch_size and agent_steps % 1 == 0:
                safe_replay(agent)
            
            if done:
                break
        
        # Additional training at episode end
        if len(agent.memory) > agent.batch_size:
            safe_replay(agent)
        
        # Apply epsilon decay after each episode
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
        
        # Track performance
        scores.append(episode_reward)
        wins.append(1 if info.get('winner') == PLAYER_B else 0)
        avg_score = np.mean(scores)
        win_rate = np.mean(wins)
        
        # Log metrics to wandb only if initialization succeeded
        if wandb_enabled:
            try:
                wandb.log({
                    "episode": e,
                    "episode_reward": episode_reward,
                    "avg_score": avg_score,
                    "win_rate": win_rate,
                    "epsilon": agent.epsilon,
                    "memory_size": len(agent.memory),
                    "steps": agent_steps,
                    "time_per_episode": time.time() - episode_start
                })
            except Exception as e:
                print(f"\nWarning: Failed to log to wandb: {e}")
                wandb_enabled = False

        # Save best model
        if avg_score > best_avg_score and len(scores) >= 50:
            best_avg_score = avg_score
            agent.save(BASE_MODEL_FILE)
            print(f"New best model saved with avg score: {best_avg_score:.2f}")
        
        # Periodic checkpoint
        if e % SAVE_FREQ == 0:
            save_checkpoint(agent, e)
        
        # Display progress
        if e % 100 == 0:
            print(f"Phase 1 - Episode: {e}/{episodes} | "
                  f"Score: {episode_reward:.2f} | "
                  f"Win Rate: {win_rate:.2f} | "
                  f"Epsilon: {agent.epsilon:.4f} | "
                  f"Avg Score: {avg_score:.2f}")
    
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
    # Initialize wandb
    wandb_enabled = init_wandb(enable_wandb, "phase2_training")

    print("\n===== STARTING PHASE 2: TOURNAMENT SELF-PLAY =====")
    
    env = SwitcharooEnv()
    scores = deque(maxlen=100)
    wins = deque(maxlen=100)
    losses = deque(maxlen=100)  # Track losses
    draws = deque(maxlen=100)   # Track draws
    tournament_agent = copy.deepcopy(agent)
    
    for e in range(start_episode, episodes + 1):
        state = env.reset()
        episode_reward = 0
        agent_steps = 0
        
        for step in range(MAX_STEPS_PER_EPISODE):
            current_player = env.current_player
            
            if current_player == PLAYER_B:  # Learning agent's turn
                legal_actions = env.get_legal_action_indices(player=PLAYER_B)
                if not legal_actions:
                    break
                
                action = agent.act(state, legal_actions)
                next_state, reward, done, info = env.step(action)
                
                if np.isnan(reward) or np.isinf(reward):
                    reward = 0.0
                
                agent.remember(state, action, reward, next_state, done)
                episode_reward += reward
                agent_steps += 1
            else:  # Tournament agent's turn (opponent)
                legal_actions = env.get_legal_action_indices(player=PLAYER_A)
                if not legal_actions:
                    break
                
                # Tournament agent plays as opponent
                action = tournament_agent.act(state, legal_actions)
                next_state, reward, done, info = env.step(action)
            
            state = next_state
            
            if len(agent.memory) > agent.batch_size and agent_steps % 1 == 0:
                safe_replay(agent)
            
            if done:
                break
            
            # Force end game if taking too long
            if step >= MAX_STEPS_PER_EPISODE - 1:
                done = True
                reward = -50.0  # Heavy penalty for timeout
                info['winner'] = 'DRAW'
                info['timeout'] = True
                break
        
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
        
        # Log metrics to wandb only if initialization succeeded
        win_rate = np.mean(wins)
        loss_rate = np.mean(losses)
        draw_rate = np.mean(draws)
        avg_score = np.mean(scores)

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
                    "memory_size": len(agent.memory),
                    "steps": agent_steps,
                    "timeout_rate": 1.0 if info.get('timeout', False) else 0.0
                })
            except Exception as e:
                print(f"\nWarning: Failed to log to wandb: {e}")
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
            win_rate = np.mean(wins)
            loss_rate = np.mean(losses)
            draw_rate = np.mean(draws)
            avg_score = np.mean(scores)
            print(f"Phase 2 - Episode: {e}/{episodes} | "
                  f"Score: {episode_reward:.2f} | "
                  f"Win Rate: {win_rate:.2f} | "
                  f"Loss Rate: {loss_rate:.2f} | "
                  f"Draw Rate: {draw_rate:.2f} | "
                  f"Epsilon: {agent.epsilon:.4f} | "
                  f"Avg Score: {avg_score:.2f}")
    
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
    
    # Initialize agent with the fixed epsilon value of 0.01
    agent = DQNAgent(
        epsilon=0.01,  # Fixed epsilon value for direct Phase 2 training
        epsilon_decay=1.0,  # No decay since we want to keep epsilon fixed
        epsilon_min=0.01,
        replay_buffer_size=1000000,
        batch_size=64,
        target_update_freq=100
    )
    
    # Load the provided model file
    try:
        agent.load(model_file)
        print(f"Successfully loaded model from {model_file}")
    except Exception as e:
        print(f"Error loading model from {model_file}: {e}")
        return None
    
    # Run Phase 2 training
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
    parser.add_argument("--resume", action="store_true",
                       help="Resume training from latest checkpoint if available")
    args = parser.parse_args()

    # Update wandb settings from args
    WANDB_PROJECT = args.wandb_project
    WANDB_ENTITY = args.wandb_entity
    enable_wandb = not args.disable_wandb
    
    # Direct Phase 2 Training Mode
    if args.phase2_only and args.model_file:
        print(f"Starting direct Phase 2 training with model: {args.model_file}")
        direct_phase2_training(
            model_file=args.model_file,
            episodes=args.episodes,
            final_model_file=args.final_model_file,
            enable_wandb=enable_wandb
        )
    else:
        # Standard Curriculum Training (Phase 1 + Phase 2)
        # Initialize agent for curriculum learning
        agent = DQNAgent(
            epsilon=0.10,
            epsilon_decay=.999,
            epsilon_min=0.01,
            replay_buffer_size=1000000,
            batch_size=64,
            target_update_freq=100
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
            agent = phase2_training(agent, start_episode, enable_wandb=enable_wandb)
            
            print("Curriculum training completed successfully!")
            
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            save_checkpoint(agent, start_episode, emergency=True)
        except Exception as e:
            print(f"\nTraining crashed with error: {e}")
            save_checkpoint(agent, start_episode, emergency=True)
            raise
