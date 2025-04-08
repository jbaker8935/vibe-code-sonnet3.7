import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from numba import njit
import numba

# Import constants and functions from game_env
from game_env import (
    ROWS, COLS, NUM_ACTIONS, PLAYER_B_ID,
    _evaluate_board_batch, _apply_move_jit, _evaluate_board_jit
)

# Configure GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print("GPU is available:")
    for device in physical_devices:
        print(f" - {device}")
    try:
        # Configure GPU to use memory growth (prevents OOM errors)
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("No GPU available. Running on CPU.")

# Import constants from the environment
from game_env import ROWS, COLS, NUM_ACTIONS, HISTORY_LENGTH

# State size calculation for binary representation
BINARY_BOARD_SIZE = 5  # 5 uint32 values per board
STATE_SIZE = BINARY_BOARD_SIZE + (2 * BINARY_BOARD_SIZE) + 1  # Current + 2 previous moves + player

ACTION_SIZE = NUM_ACTIONS

from binary_board import board_to_binary, binary_to_board

class DQNAgent:
    def __init__(self, state_size=STATE_SIZE, action_size=ACTION_SIZE,
                 learning_rate=0.001, gamma=0.99, epsilon=1.0,
                 epsilon_decay=0.999, epsilon_min=0.01,
                 replay_buffer_size=100000, batch_size=64,  # Increased buffer size from 10000 to 100000
                 target_update_freq=100): # Update target network every 100 steps/episodes
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=replay_buffer_size)
        self.gamma = gamma    # discount rate
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.update_counter = 0 # Counter for target network updates
        
        # Store optimizer and loss function as instance variables BEFORE building models
        self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.loss_function = tf.keras.losses.MeanSquaredError()

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model() # Initialize target model weights

    def _build_model(self):
        """Neural network that processes binary board representation."""
        input_layer = layers.Input(shape=(self.state_size,))
        
        # Current board is 5 uint32 values
        current_board = layers.Lambda(lambda x: x[:, :5])(input_layer)
        
        # History is 2 boards × 5 uint32 values each
        history_size = 2 * 5
        history_boards = layers.Lambda(lambda x: x[:, 5:5+history_size])(input_layer)
        
        # Player input is the last value
        player_input = layers.Lambda(lambda x: x[:, -1:])(input_layer)
        
        # Process current binary board
        x1 = layers.Dense(256, activation='relu')(current_board)
        x1 = layers.Dense(128, activation='relu')(x1)
        
        # Process history binary boards - reshape to (batch_size, 2, 5)
        history_reshaped = layers.Reshape((2, 5))(history_boards)
        
        # Add attention mechanism
        attention = layers.Dense(32, activation='tanh')(history_reshaped)
        attention = layers.Dense(1, activation='sigmoid')(attention)
        x2 = layers.Multiply()([history_reshaped, attention])
        
        # Process temporal patterns with smaller LSTM due to reduced history
        x2 = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x2)
        x2 = layers.Bidirectional(layers.LSTM(32))(x2)
        
        # Combine features
        combined = layers.Concatenate()([x1, x2, player_input])
        x = layers.Dense(256, activation='relu')(combined)
        x = layers.Dense(128, activation='relu')(x)
        
        output = layers.Dense(self.action_size, activation='linear')(x)
        
        model = keras.Model(inputs=input_layer, outputs=output)
        model.compile(loss='mse', optimizer=self.optimizer)
        return model

    def update_target_model(self):
        """Copies weights from the main model to the target model."""
        self.target_model.set_weights(self.model.get_weights())
        # print("Target model updated.")

    def remember(self, state, action, reward, next_state, done):
        """Stores experience in the replay buffer."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, legal_actions_indices):
        """
        Chooses an action using epsilon-greedy policy, considering only legal moves.
        Uses board evaluation as a tiebreaker when multiple actions have the same Q-value.
        """
        from binary_board import binary_to_board
        from game_env import _evaluate_board_jit, _apply_move_jit, PLAYER_B_ID
        
        if not legal_actions_indices:
            print("Warning: No legal actions available in act method.")
            return np.random.randint(self.action_size)

        if np.random.rand() <= self.epsilon:
            return random.choice(legal_actions_indices)
        else:
            state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
            state_tensor = tf.expand_dims(state_tensor, 0)
            # Use the optimized prediction function
            act_values = self.predict_batch(state_tensor)

            # Handle NaN values
            q_values = act_values.numpy()[0]
            if np.isnan(q_values).any():
                print("Warning: NaN detected in Q-values. Using fallback values.")
                q_values = np.zeros_like(q_values)

            legal_q_values = np.take(q_values, legal_actions_indices)
            max_q = np.max(legal_q_values)
            
            # Find actions with Q-values close to the maximum
            best_indices = np.where(np.isclose(legal_q_values, max_q, rtol=1e-5))[0]
            
            # If multiple actions have effectively the same Q-value, use board evaluation
            if len(best_indices) > 1:
                # Get the current binary board state and convert to regular board
                current_binary = state[:5]  # First 5 values are current board state
                board_state = binary_to_board(current_binary.astype(np.uint32))
                
                best_score = float('-inf')
                best_action_idx = legal_actions_indices[best_indices[0]]
                
                for idx in best_indices:
                    action_idx = legal_actions_indices[idx]
                    move = self._get_move_from_action(action_idx)
                    if move is None:
                        continue
                        
                    start_r, start_c, end_r, end_c = move
                    board_copy = board_state.copy()
                    
                    _apply_move_jit(board_copy, start_r, start_c, end_r, end_c)
                    score = _evaluate_board_jit(board_copy, PLAYER_B_ID)
                    
                    if score > best_score:
                        best_score = score
                        best_action_idx = action_idx
                        
                return best_action_idx
                
            return legal_actions_indices[best_indices[0]]
    
    def _get_move_from_action(self, action_index):
        """Converts an action index to (start_r, start_c, end_r, end_c) coordinates."""
        from game_env import DIRECTIONS, COLS, ROWS
        
        if not (0 <= action_index < self.action_size):
            return None

        direction_index = action_index % 8
        start_cell_index = action_index // 8

        start_r = start_cell_index // COLS
        start_c = start_cell_index % COLS

        dr, dc = DIRECTIONS[direction_index]
        end_r, end_c = start_r + dr, start_c + dc

        # Validate coordinates
        if not (0 <= start_r < ROWS and 0 <= start_c < COLS and 
                0 <= end_r < ROWS and 0 <= end_c < COLS):
            return None

        return start_r, start_c, end_r, end_c

    @tf.function
    def replay(self):
        """Experience replay for training the model."""
        if len(self.memory) < self.batch_size:
            return None

        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        # Convert to tensors with explicit shapes and dtypes
        states = tf.convert_to_tensor(np.vstack(states), dtype=tf.float32)
        next_states = tf.convert_to_tensor(np.vstack(next_states), dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        
        # Use target network for stable Q-learning
        next_q_values = self.target_model(next_states, training=False)
        max_next_q_values = tf.reduce_max(next_q_values, axis=1)
        target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        # Standard practice: update only the Q-values for actions taken
        with tf.GradientTape() as tape:
            q_values = self.model(states, training=True)
            one_hot_actions = tf.one_hot(actions, self.action_size)
            predicted_values = tf.reduce_sum(q_values * one_hot_actions, axis=1)
            loss = self.loss_function(target_q_values, predicted_values)
            
        # Apply gradients
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        # Update target network periodically 
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.update_target_model()
            
        return loss  # Return loss tensor directly, not numpy()
        
    # Add a new optimized batch prediction method
    @tf.function
    def predict_batch(self, states_tensor):
        """TF function optimized prediction for multiple states at once."""
        return self.model(states_tensor, training=False)

    def act_batch(self, states, legal_actions_list):
        """Batch version of act() for improved performance."""
        if np.random.rand() <= self.epsilon:
            return [random.choice(actions) for actions in legal_actions_list]
        
        # Get Q-values for all states at once
        states_tensor = tf.convert_to_tensor(states)
        q_values_batch = self.model(states_tensor, training=False).numpy()
        
        selected_actions = []
        for i, (q_values, legal_actions) in enumerate(zip(q_values_batch, legal_actions_list)):
            # Filter for legal actions
            legal_q_values = np.take(q_values, legal_actions)
            max_q = np.max(legal_q_values)
            best_indices = np.where(np.isclose(legal_q_values, max_q, rtol=1e-5))[0]
            
            if len(best_indices) > 1:
                # Use vectorized board evaluation for tiebreaking
                board_states = np.tile(states[i][:-1].reshape(8, 4), (len(best_indices), 1, 1))
                board_states = (board_states * 4).astype(np.int8)
                
                # Generate all potential next states in parallel
                move_indices = [legal_actions[idx] for idx in best_indices]
                next_boards = self._get_next_boards_batch(board_states, move_indices)
                
                # Evaluate all boards in parallel using imported function
                scores = _evaluate_board_batch(next_boards, PLAYER_B_ID)
                best_action_idx = move_indices[np.argmax(scores)]
            else:
                best_action_idx = legal_actions[best_indices[0]]
                
            selected_actions.append(best_action_idx)
        
        return selected_actions

    # Note: Remove the njit decorator from this method since it's a class method
    # and can't be easily JIT compiled. The core operations are still JIT compiled
    # through the imported functions.
    def _get_next_boards_batch(self, board_states, move_indices):
        """Generate next board states for multiple moves in parallel."""
        from game_env import _simulate_moves_batch_jit, DIRECTIONS, COLS, ROWS
        
        num_boards = len(move_indices)
        moves = []
        
        # Convert action indices to moves
        for i in range(num_boards):
            action_index = move_indices[i]
            direction_index = action_index % 8
            start_cell_index = action_index // 8
            
            start_r = start_cell_index // COLS
            start_c = start_cell_index % COLS
            
            dr, dc = DIRECTIONS[direction_index]
            end_r, end_c = start_r + dr, start_c + dc
            
            # Skip invalid moves (shouldn't happen with legal actions)
            if not (0 <= start_r < ROWS and 0 <= start_c < COLS and 
                    0 <= end_r < ROWS and 0 <= end_c < COLS):
                continue
                
            moves.append((start_r, start_c, end_r, end_c))
        
        # Use the JIT-optimized function for parallel simulation
        return _simulate_moves_batch_jit(board_states[0], moves, PLAYER_B_ID)

    def load(self, name):
        """Loads model weights from a file."""
        try:
            self.model.load_weights(name)
            self.update_target_model() # Ensure target model is also updated
            print(f"Model weights loaded from {name}")
        except Exception as e:
            print(f"Error loading model weights from {name}: {e}")


    def save(self, name):
        """Saves model weights to a file."""
        try:
            self.model.save_weights(name)
            print(f"Model weights saved to {name}")
        except Exception as e:
            print(f"Error saving model weights to {name}: {e}")

    def save_for_tfjs(self, file_path):
        """Saves the model in TensorFlow.js format."""
        try:
            # TFJS requires saving the model architecture + weights
            self.model.export(file_path) # Save in SavedModel format first
            # Conversion to TFJS format usually happens via a command-line tool
            # But saving as SavedModel is the prerequisite.
            # The user will need to run: tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_layers_model ./saved_model_dir ./tfjs_model_dir
            print(f"Model saved in SavedModel format at {file_path}. ")
            print(f"Convert to TF.js using: tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_layers_model {file_path} ./tfjs_model")

        except Exception as e:
            print(f"Error saving model for TF.js at {file_path}: {e}")


# Example Usage (Optional - typically used in the training script)
if __name__ == "__main__":
    agent = DQNAgent()
    print("DQN Agent Initialized.")
    print("Model Summary:")
    agent.model.summary()

    # Example state (replace with actual state from env)
    dummy_state = np.random.rand(STATE_SIZE)
    dummy_legal_actions = [10, 25, 100, 200] # Example legal action indices

    action = agent.act(dummy_state, dummy_legal_actions)
    print(f"Chosen action for dummy state: {action}")

    # Example remember and replay (usually done in training loop)
    dummy_next_state = np.random.rand(STATE_SIZE)
    agent.remember(dummy_state, action, 1.0, dummy_next_state, False)
    agent.replay()
    print("Example replay step completed.")

    # Example save/load
    agent.save("dqn_test.weights.h5")
    agent.load("dqn_test.weights.h5")
    agent.save_for_tfjs("./dqn_tfjs_test_model")
