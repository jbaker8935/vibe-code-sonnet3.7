import numpy as np
import random
from collections import deque
import time
import os
import tensorflow as tf
from numba import njit

from game_env import SwitcharooEnv, PLAYER_A, PLAYER_B, _scale_reward_jit, _board_in_history_jit
from dqn_agent import DQNAgent

# Training Configuration
EPISODES = 150000
MAX_STEPS_PER_EPISODE = 300 # Prevent infinitely long games
SAVE_FREQ = 10000 # Save model weights every N episodes
TRAIN_FREQ = 1  # Changed: replay every 4 agent steps instead of every step
MODEL_WEIGHTS_FILE = "switcharoo_dqn.weights.h5"
CHECKPOINT_FILE = "switcharoo_dqn_checkpoint_e{}.weights.h5"  # Add episode number to filename
TFJS_MODEL_DIR = "./switcharoo_tfjs_model" # Directory for SavedModel format
TFJS_MODEL_FILE = "./switcharoo_tfjs_model/tf_model.weights.h5" # Directory for SavedModel format

def save_checkpoint(agent, episode, emergency=False):
    """Save a checkpoint with the current episode number."""
    filename = CHECKPOINT_FILE.format(episode)
    agent.save(filename)
    if emergency:
        print(f"\nEmergency checkpoint saved to {filename}")
    return filename

# --- Opponent Policy (Player A) ---
def get_opponent_action(env, opponent_epsilon=0.3):
    """Improved policy for the opponent (Player A) using board evaluation.
    
    Args:
        env: Game environment
        opponent_epsilon: Probability of choosing a random move (0.0 to 1.0)
    """
    from game_env import _evaluate_board_jit, _apply_move_jit, PLAYER_A_ID
    
    legal_actions = env.get_legal_action_indices(player=PLAYER_A)
    if not legal_actions:
        return None # No legal moves
    
    # Random move with probability opponent_epsilon
    if random.random() < opponent_epsilon:
        return random.choice(legal_actions)
    
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

@njit(cache=True)
def _validate_reward(reward):
    """JIT-compiled reward validation."""
    if np.isnan(reward) or np.isinf(reward):
        return 0.0
    return reward

# Safely replay the experience to handle tensor conversion errors
def safe_replay(agent):
    try:
        loss = agent.replay()
        # We can log the loss here if needed, but we need to convert to numpy OUTSIDE
        # the tf.function context
        # if loss is not None:
        #     loss_value = loss.numpy()  # Only convert to numpy if needed for logging
    except ValueError as e:
        if "Expected values" in str(e) and "to be a dense tensor with shape" in str(e):
            # This is the shape mismatch error we're seeing
            print("Warning: Skipped replay batch due to tensor shape mismatch.")
            # Optional: Clear some memory to avoid the same problematic batch
            if len(agent.memory) > agent.batch_size * 2:  # Only clear if we have enough samples
                for _ in range(agent.batch_size):
                    agent.memory.popleft()  # Remove some old experiences
        else:
            # For other value errors, raise them
            raise e
    except AttributeError as e:
        if "'SymbolicTensor' object has no attribute 'numpy'" in str(e):
            # This is the error we just fixed
            print("Warning: SymbolicTensor numpy() error. Update your code.")
        else:
            raise e

# --- Training Loop ---
if __name__ == "__main__":
    env = SwitcharooEnv()
    agent = DQNAgent(epsilon=1.0,epsilon_decay=0.9995, 
                 replay_buffer_size=1000000, batch_size=64) 

    # Try to find the latest checkpoint
    latest_checkpoint = None
    start_episode = 1
    try:
        checkpoints = [f for f in os.listdir() if f.startswith("switcharoo_dqn_checkpoint_e")]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_e')[-1].split('.weights')[0]))
            start_episode = int(latest_checkpoint.split('_e')[-1].split('.weights')[0]) + 1
            agent.load(latest_checkpoint)
            print(f"Resuming from checkpoint {latest_checkpoint} at episode {start_episode}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        start_episode = 1  # Ensure we start from episode 1 if loading fails

    scores = deque(maxlen=100) # Track scores for the last 100 episodes
    episode_lengths = deque(maxlen=100)
    start_time = time.time()

    print(f"Starting training for {EPISODES} episodes...")

    try:
        for e in range(start_episode, EPISODES + 1):
            state = env.reset()
            episode_reward = 0
            agent_steps = 0 # Count steps taken by the agent (Player B)
            episode_start = time.time()  # Timer for episode

            for step in range(MAX_STEPS_PER_EPISODE):
                current_player = env.current_player

                if current_player == PLAYER_B:  # Agent's turn
                    legal_actions = env.get_legal_action_indices(player=PLAYER_B)
                    if not legal_actions:
                        print(f"Episode {e}: Agent (Player B) has no legal moves. Ending episode.")
                        break  # End episode if agent is stuck

                    action = agent.act(state, legal_actions)
                    next_state, reward, done, info = env.step(action)
                    reward = _validate_reward(reward)  # Use JIT for reward validation
                    
                    # Apply reward scaling based on episode progress
                    reward = _scale_reward_jit(reward, step, MAX_STEPS_PER_EPISODE)
                    
                    agent.remember(state, action, reward, next_state, done)
                    episode_reward += reward
                    agent_steps += 1

                else: # Player A's turn (Opponent)
                    action = get_opponent_action(env, opponent_epsilon=0.3)
                    if action is None:
                        # Opponent has no moves, should result in a win/draw handled by env.step
                        break # End episode if opponent is stuck

                    # We don't need the return values here as we don't train Player A
                    next_state, reward, done, info = env.step(action)
                    # Note: We don't add Player A's experience to the agent's memory

                state = next_state

                # Train the agent (replay) after each step (or less frequently if needed)
                if len(agent.memory) > agent.batch_size and agent_steps % TRAIN_FREQ == 0:
                    safe_replay(agent)

                if done:
                    break # Exit step loop if game ended

            # --- End of Episode ---
            # Use safe replay at the end of episode
            if len(agent.memory) > agent.batch_size:
                safe_replay(agent)
                
            scores.append(episode_reward)
            episode_lengths.append(step + 1)
            avg_score = np.mean(scores)
            avg_length = np.mean(episode_lengths)
            episode_time = time.time() - episode_start

            print(f"Episode: {e}/{EPISODES} | Steps: {step+1} (Agent: {agent_steps}) | Score: {episode_reward:.2f} | Time: {episode_time:.2f}s | Winner: {env.winner} | Epsilon: {agent.epsilon:.4f} | Avg Score (100): {avg_score:.2f} | Avg Len (100): {avg_length:.1f}")

            # Save model weights periodically
            if e % SAVE_FREQ == 0:
                save_checkpoint(agent, e)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        save_checkpoint(agent, e, emergency=True)
    except Exception as ex:
        print(f"\nTraining crashed with error: {ex}")
        save_checkpoint(agent, e, emergency=True)
        raise
    finally:
        # Save final weights and TF.js compatible model
        agent.save(MODEL_WEIGHTS_FILE)
        agent.save_for_tfjs(TFJS_MODEL_FILE)
        print("\nFinal model weights saved.")
        print(f"To convert the model for TensorFlow.js, navigate to the directory containing '{TFJS_MODEL_DIR}' and run:")
        print(f"tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_layers_model {TFJS_MODEL_DIR} ./tfjs_final_model")
