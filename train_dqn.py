import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # only warnings & errors
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # disable oneDNN notices


import numpy as np
import random
from collections import deque
import time
import os
import tensorflow as tf
from numba import njit

from game_env import SwitcharooEnv
from env_const import PLAYER_A, PLAYER_B
from env_util import _scale_reward_jit, _board_in_history_jit
from dqn_agent import DQNAgent

# Training Configuration
EPISODES = 150000
MAX_STEPS_PER_EPISODE = 300 # Prevent infinitely long games
SAVE_FREQ = 10000 # Save model weights every N episodes
TRAIN_FREQ = 8  # Changed: replay every 4 agent steps instead of every step
MODEL_WEIGHTS_FILE = "switcharoo_dqn.weights.h5"
CHECKPOINT_FILE = "switcharoo_dqn_checkpoint_e{}.weights.h5"  # Add episode number to filename
TFJS_MODEL_DIR = "./switcharoo_tfjs_model" # Directory for SavedModel format

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

# --- Training Loop ---
if __name__ == "__main__":
    env = SwitcharooEnv()
    agent = DQNAgent(epsilon=1.0,epsilon_decay=0.9995, 
                 replay_buffer_size=1000000, batch_size=64, use_per=True) 

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

    print("Creating training dataset...")
    # Create the dataset and iterator outside the loop
    train_dataset = agent.create_tf_dataset()
    train_iterator = iter(train_dataset)
    print("Dataset created.")

    print(f"Starting training for {EPISODES} episodes...")

    try:
        for e in range(start_episode, EPISODES + 1):
            state = env.reset()
            episode_reward = 0
            agent_steps = 0 # Count steps taken by the agent (Player B)
            episode_start = time.time()  # Timer for episode
            total_loss = 0.0
            train_steps = 0

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

                # Train the agent using the dataset iterator
                # Check if enough memory and time to train based on TRAIN_FREQ
                current_buffer_size = len(agent)
                if current_buffer_size >= agent.batch_size and agent_steps > 0 and agent_steps % TRAIN_FREQ == 0:
                    try:
                        # Get the next batch from the dataset iterator
                        batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones, batch_weights, batch_indices = next(train_iterator)

                        # Perform a training step
                        loss, td_errors = agent._train_step(batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones, batch_weights)
                        total_loss += loss.numpy()
                        train_steps += 1

                        # Update priorities if using PER
                        if agent.use_per:
                            agent.update_priorities(batch_indices.numpy(), td_errors.numpy())

                        # Update target network periodically
                        agent.update_counter += 1
                        if agent.update_counter >= agent.target_update_freq:
                            agent.update_target_model()
                            agent.update_counter = 0

                    except StopIteration:
                        # Should not happen with an infinite generator, but handle defensively
                        print("Warning: Training dataset iterator exhausted unexpectedly. Recreating.")
                        train_iterator = iter(train_dataset)
                    except Exception as train_ex:
                        print(f"Error during training step: {train_ex}")
                        # Optionally add more robust error handling here

                if done:
                    break # Exit step loop if game ended

            # --- End of Episode ---
            # Optional: Perform a few extra training steps at the end of the episode

            # Decay epsilon
            if agent.epsilon > agent.epsilon_min:
                agent.epsilon *= agent.epsilon_decay
                agent.epsilon = max(agent.epsilon_min, agent.epsilon) # Ensure it doesn't go below min

            # Anneal PER beta if using PER
            if agent.use_per:
                agent.per_beta = min(1.0, agent.per_beta + agent.per_beta_increment)

            scores.append(episode_reward)
            episode_lengths.append(step + 1)
            avg_score = np.mean(scores)
            avg_length = np.mean(episode_lengths)
            avg_loss = total_loss / train_steps if train_steps > 0 else 0
            episode_time = time.time() - episode_start

            print(f"Episode: {e}/{EPISODES} | Steps: {step+1} (Agent: {agent_steps}) | Score: {episode_reward:.2f} | Time: {episode_time:.2f}s | Winner: {env.winner} | Epsilon: {agent.epsilon:.4f} | Avg Score (100): {avg_score:.2f} | Avg Len (100): {avg_length:.1f} | Avg Loss: {avg_loss:.4f}")

            # Save model weights periodically
            if e % SAVE_FREQ == 0:
                save_checkpoint(agent, e)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        save_checkpoint(agent, e, emergency=True)
    except Exception as ex:
        print(f"\nTraining crashed with error: {ex}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        save_checkpoint(agent, e, emergency=True)
        raise
    finally:
        # Save final weights and TF.js compatible model
        agent.save(MODEL_WEIGHTS_FILE)
        # Ensure the TFJS directory exists before saving
        if not os.path.exists(TFJS_MODEL_DIR):
             os.makedirs(TFJS_MODEL_DIR)
        agent.save_for_tfjs(TFJS_MODEL_DIR) # Pass directory to save_for_tfjs
        print("\nFinal model weights saved.")
        print(f"To convert the model for TensorFlow.js, navigate to the directory containing '{TFJS_MODEL_DIR}' and run:")
        print(f"tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model {TFJS_MODEL_DIR} ./tfjs_final_model")