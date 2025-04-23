import numpy as np
import random
from collections import deque
import tensorflow as tf
from keras import layers
import keras


# Import constants and functions from game_env
from game_env import (
    ROWS, COLS, NUM_ACTIONS, PLAYER_B_ID
    
)
from env_util import (_evaluate_board_batch,_apply_move_jit, _evaluate_board_jit)

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
STATE_SIZE = 6  # 5 for the binary board + 1 for the player indicator

ACTION_SIZE = NUM_ACTIONS

from binary_board import board_to_binary, binary_to_board

class DQNAgent:
    def __init__(self, state_size=STATE_SIZE, action_size=ACTION_SIZE,
                 learning_rate=0.001, gamma=0.99, epsilon=1.0,
                 epsilon_decay=0.999, epsilon_min=0.01,
                 replay_buffer_size=100000, batch_size=64,  # Increased buffer size from 10000 to 100000
                 target_update_freq=100, # Update target network every 100 steps/episodes
                 gradient_clip_norm=1.0): # Add gradient clipping parameter
        self.state_size = state_size
        self.action_size = action_size
        # Use Prioritized Experience Replay (PER)
        self.memory = []
        self.memory_capacity = replay_buffer_size
        self.priorities = np.zeros(replay_buffer_size, dtype=np.float32)
        self.memory_index = 0
        self.gamma = gamma    # discount rate
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.update_counter = 0 # Counter for target network updates
        
        # Store optimizer and loss function as instance variables BEFORE building models
        # Add clipnorm to the optimizer
        self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate, clipnorm=gradient_clip_norm)
        # Use Huber loss instead of MSE
        self.loss_function = tf.keras.losses.Huber()

        # Debugging: Log the state size during initialization
        print(f"Initializing DQNAgent with state_size: {self.state_size}")

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model() # Initialize target model weights

    def _build_model(self):
        """Neural network that processes binary board representation."""
        input_layer = layers.Input(shape=(self.state_size,))
        
        # Current board is 5 uint32 values
        current_board = layers.Lambda(lambda x: x[:, :5])(input_layer)
        
        # Player input is the last value
        player_input = layers.Lambda(lambda x: x[:, -1:])(input_layer)
        
        # Process current binary board
        x1 = layers.Dense(256, activation='relu')(current_board)
        x1 = layers.Dense(128, activation='relu')(x1)
        
        # Combine features
        combined = layers.Concatenate()([x1, player_input])
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
        """Stores experience in the replay buffer with initial priority."""
        # Calculate initial priority as max priority or a default high value
        if len(self.memory) == 0:
            priority = 1.0
        else:
            priority = np.max(self.priorities[:len(self.memory)])
        
        if len(self.memory) < self.memory_capacity:
            self.memory.append((state, action, reward, next_state, done))
        else:
            self.memory[self.memory_index] = (state, action, reward, next_state, done)
        
        self.priorities[self.memory_index] = priority
        self.memory_index = (self.memory_index + 1) % self.memory_capacity

    def act(self, state, legal_actions_indices, top_k=3, log_q_values=False):
        """
        Chooses an action using epsilon-greedy policy, considering only legal moves.
        Introduces randomness by sampling from top-k moves based on Q-values.
        Logs Q-value distribution for analysis if enabled.
        """
        if not legal_actions_indices:
            print("Warning: No legal actions available in act method.")
            return np.random.randint(self.action_size)

        if np.random.rand() <= self.epsilon:
            return random.choice(legal_actions_indices)
        else:
            state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
            state_tensor = tf.expand_dims(state_tensor, 0)
            act_values = self.predict_batch(state_tensor)

            q_values = act_values.numpy()[0]
            if np.isnan(q_values).any():
                print("Warning: NaN detected in Q-values. Using fallback values.")
                q_values = np.zeros_like(q_values)

            # Example randomness threshold calculation
            randomness_threshold = 0.05 * max(q_values)  # 5% of the highest Q-value

            # Filter Q-values for legal actions
            legal_q_values = np.take(q_values, legal_actions_indices)

            # Log Q-value distribution if enabled
            if log_q_values:
                print(f"Legal Q-values: {legal_q_values}")
                print(f"Legal actions: {legal_actions_indices}")

            # Sort legal actions by Q-value in descending order
            sorted_indices = np.argsort(legal_q_values)[::-1]
            sorted_legal_actions = [legal_actions_indices[i] for i in sorted_indices]
            sorted_q_values = legal_q_values[sorted_indices]

            # Select top-k moves
            top_k = min(top_k, len(sorted_legal_actions))
            top_actions = sorted_legal_actions[:top_k]
            top_q_values = sorted_q_values[:top_k]

            # Log top-k Q-values if enabled
            if log_q_values:
                print(f"Top-{top_k} Q-values: {top_q_values}")
                print(f"Top-{top_k} actions: {top_actions}")

            # If the difference between the best and second-best Q-value is small, introduce randomness
            if len(top_q_values) > 1 and (top_q_values[0] - top_q_values[1]) < randomness_threshold:
                return random.choice(top_actions)

            # Otherwise, choose the best action
            return top_actions[0]
    
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
    def _replay_step(self, states, actions, rewards, next_states, dones):
        """Performs a single gradient update step. Decorated with tf.function."""
        with tf.GradientTape() as tape:
            q_values = self.model(states, training=True)
            one_hot_actions = tf.one_hot(actions, self.action_size, dtype=tf.float32)
            predicted_values = tf.reduce_sum(q_values * one_hot_actions, axis=1)

            next_q_values = self.target_model(next_states, training=False)
            max_next_q_values = tf.reduce_max(next_q_values, axis=1)
            target_q_values = rewards + (1.0 - dones) * self.gamma * max_next_q_values

            loss = self.loss_function(target_q_values, predicted_values)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        td_errors = tf.abs(target_q_values - predicted_values)
        return loss, td_errors

    def replay(self):
        """Experience replay for training the model with PER."""
        if len(self.memory) < self.batch_size:
            return None

        priorities = self.priorities[:len(self.memory)]
        prob = priorities / np.sum(priorities)
        indices = np.random.choice(len(self.memory), size=self.batch_size, p=prob)
        minibatch = [self.memory[idx] for idx in indices]

        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = tf.convert_to_tensor(np.vstack(states), dtype=tf.float32)
        next_states = tf.convert_to_tensor(np.vstack(next_states), dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        loss, td_errors = self._replay_step(states, actions, rewards, next_states, dones)

        td_errors = td_errors.numpy()
        for i, idx in enumerate(indices):
            self.priorities[idx] = td_errors[i] + 1e-6

        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.update_target_model()
            
        return loss.numpy()
        
    @tf.function
    def predict_batch(self, states_tensor):
        """TF function optimized prediction for multiple states at once."""
        return self.model(states_tensor, training=False)

    def act_batch(self, states, legal_actions_list):
        """Batch version of act() for improved performance."""
        if np.random.rand() <= self.epsilon:
            return [random.choice(actions) for actions in legal_actions_list]
        
        states_tensor = tf.convert_to_tensor(states)
        q_values_batch = self.model(states_tensor, training=False).numpy()
        
        selected_actions = []
        for i, (q_values, legal_actions) in enumerate(zip(q_values_batch, legal_actions_list)):
            legal_q_values = np.take(q_values, legal_actions)
            max_q = np.max(legal_q_values)
            best_indices = np.where(np.isclose(legal_q_values, max_q, rtol=1e-5))[0]
            
            if len(best_indices) > 1:
                board_states = np.tile(states[i][:-1].reshape(8, 4), (len(best_indices), 1, 1))
                board_states = (board_states * 4).astype(np.int8)
                
                move_indices = [legal_actions[idx] for idx in best_indices]
                next_boards = self._get_next_boards_batch(board_states, move_indices)
                
                scores = _evaluate_board_batch(next_boards, PLAYER_B_ID)
                best_action_idx = move_indices[np.argmax(scores)]
            else:
                best_action_idx = legal_actions[best_indices[0]]
                
            selected_actions.append(best_action_idx)
        
        return selected_actions

    def _get_next_boards_batch(self, board_states, move_indices):
        """Generate next board states for multiple moves in parallel."""
        from game_env import _simulate_moves_batch_jit, DIRECTIONS, COLS, ROWS
        
        num_boards = len(move_indices)
        moves = []
        
        for i in range(num_boards):
            action_index = move_indices[i]
            direction_index = action_index % 8
            start_cell_index = action_index // 8
            
            start_r = start_cell_index // COLS
            start_c = start_cell_index % COLS
            
            dr, dc = DIRECTIONS[direction_index]
            end_r, end_c = start_r + dr, start_c + dc
            
            if not (0 <= start_r < ROWS and 0 <= start_c < COLS and 
                    0 <= end_r < ROWS and 0 <= end_c < COLS):
                continue
                
            moves.append((start_r, start_c, end_r, end_c))
        
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
            self.model.export(file_path) # Save in SavedModel format first
            print(f"Model saved in SavedModel format at {file_path}. ")
            print(f"Convert to TF.js using: tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model {file_path} ./tfjs_model/web_model")

        except Exception as e:
            print(f"Error saving model for TF.js at {file_path}: {e}")


if __name__ == "__main__":
    agent = DQNAgent()
    print("DQN Agent Initialized.")
    print("Model Summary:")
    agent.model.summary()

    dummy_state = np.random.rand(STATE_SIZE)
    dummy_legal_actions = [10, 25, 100, 200]

    action = agent.act(dummy_state, dummy_legal_actions)
    print(f"Chosen action for dummy state: {action}")

    dummy_next_state = np.random.rand(STATE_SIZE)
    agent.remember(dummy_state, action, 1.0, dummy_next_state, False)
    agent.replay()
    print("Example replay step completed.")

    agent.save("dqn_test.weights.h5")
    agent.load("dqn_test.weights.h5")
    agent.save_for_tfjs("./dqn_tfjs_test_model")
