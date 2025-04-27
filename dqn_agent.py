import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # only warnings & errors
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # disable oneDNN notices

import numpy as np
import random
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
                 learning_rate=1e-6,  # Reduced learning rate further
                 gamma=0.99, epsilon=1.0,
                 epsilon_decay=0.999, epsilon_min=0.01,
                 replay_buffer_size=100000, batch_size=64,
                 target_update_freq=100,
                 gradient_clip_norm=1.0,
                 use_per=True,  # Flag to enable/disable PER
                 per_alpha=0.5,  # Reduced priority exponent
                 per_beta=0.4,   # Importance sampling exponent
                 per_beta_increment=0.001, # Annealing beta
                 per_epsilon=1e-6): # Small constant added to priorities
        self.state_size = state_size
        self.action_size = action_size
        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size

        # Use NumPy arrays for experience replay buffer
        self.states = np.zeros((replay_buffer_size, state_size), dtype=np.float32)
        self.actions = np.zeros(replay_buffer_size, dtype=np.int32)
        self.rewards = np.zeros(replay_buffer_size, dtype=np.float32)
        self.next_states = np.zeros((replay_buffer_size, state_size), dtype=np.float32)
        self.dones = np.zeros(replay_buffer_size, dtype=np.float32)
        self.priorities = np.zeros(replay_buffer_size, dtype=np.float32)

        self.memory_index = 0
        self.memory_full = False

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.target_update_freq = target_update_freq
        self.update_counter = 0

        self.use_per = use_per  # Store the flag

        # PER parameters (only relevant if use_per is True)
        self.per_alpha = per_alpha
        self.per_beta = per_beta
        self.per_beta_increment = per_beta_increment
        self.per_epsilon = per_epsilon

        self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate, clipnorm=gradient_clip_norm)
        self.loss_function = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)

        print(f"Initializing DQNAgent with state_size: {self.state_size}")

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        """Neural network that processes binary board representation."""
        input_layer = layers.Input(shape=(self.state_size,))

        # Separate board and player inputs
        board_input_raw = layers.Lambda(lambda x: x[:, :5])(input_layer)
        player_input = layers.Lambda(lambda x: x[:, -1:])(input_layer)

        # Normalize the board input (scale uint32 values to ~[0, 1])
        # Using 2**32 - 1 which is the max uint32 value
        max_uint32 = tf.constant(4294967295.0, dtype=tf.float32)
        normalized_board_input = layers.Lambda(lambda x: x / max_uint32)(board_input_raw)

        # Process normalized board input
        x1 = layers.Dense(64, activation='relu')(normalized_board_input)
        x1 = layers.Dense(32, activation='relu')(x1)

        # Combine processed board input with player input
        combined = layers.Concatenate()([x1, player_input])
        x = layers.Dense(64, activation='relu')(combined)
        x = layers.Dense(32, activation='relu')(x)

        output = layers.Dense(self.action_size, activation='linear')(x)

        model = keras.Model(inputs=input_layer, outputs=output)
        return model

    def update_target_model(self):
        """Copies weights from the main model to the target model."""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        """Stores experience in the NumPy arrays and updates priority."""
        # Clip reward to prevent instability
        clipped_reward = np.clip(reward, -10.0, 10.0)

        # Set priority to 1.0 if PER is disabled or buffer is empty
        current_max_priority = 1.0
        if self.use_per and (self.memory_index > 0 or self.memory_full):
            current_max_priority = np.max(self.priorities)

        self.states[self.memory_index] = state
        self.actions[self.memory_index] = action
        # Store the clipped reward
        self.rewards[self.memory_index] = clipped_reward
        self.next_states[self.memory_index] = next_state
        self.dones[self.memory_index] = float(done)
        self.priorities[self.memory_index] = current_max_priority

        self.memory_index += 1
        if self.memory_index >= self.replay_buffer_size:
            self.memory_index = 0
            self.memory_full = True

    def act(self, state, legal_actions_indices, top_k=3, log_q_values=False):
        if not legal_actions_indices:
            print("Warning: No legal actions available in act method.")
            return np.random.randint(self.action_size)

        if np.random.rand() <= self.epsilon:
            return random.choice(legal_actions_indices)
        else:
            state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)
            state_tensor = tf.expand_dims(state_tensor, 0)
            act_values_raw = self.predict_batch(state_tensor)
            act_values = tf.cast(act_values_raw, tf.float32)

            q_values = act_values.numpy()[0]
            if np.isnan(q_values).any():
                q_values = np.full(self.action_size, -np.inf, dtype=np.float32)
                q_values[legal_actions_indices] = 0.0

            randomness_threshold = 0.05 * max(q_values)

            legal_q_values = np.take(q_values, legal_actions_indices)

            if log_q_values:
                print(f"Legal Q-values: {legal_q_values}")
                print(f"Legal actions: {legal_actions_indices}")

            sorted_indices = np.argsort(legal_q_values)[::-1]
            sorted_legal_actions = [legal_actions_indices[i] for i in sorted_indices]
            sorted_q_values = legal_q_values[sorted_indices]

            top_k = min(top_k, len(sorted_legal_actions))
            top_actions = sorted_legal_actions[:top_k]
            top_q_values = sorted_q_values[:top_k]

            if log_q_values:
                print(f"Top-{top_k} Q-values: {top_q_values}")
                print(f"Top-{top_k} actions: {top_actions}")

            if len(top_q_values) > 1 and (top_q_values[0] - top_q_values[1]) < randomness_threshold:
                return random.choice(top_actions)

            return top_actions[0]

    def _get_move_from_action(self, action_index):
        from game_env import DIRECTIONS, COLS, ROWS

        if not (0 <= action_index < self.action_size):
            return None

        direction_index = action_index % 8
        start_cell_index = action_index // 8

        start_r = start_cell_index // COLS
        start_c = start_cell_index % COLS

        dr, dc = DIRECTIONS[direction_index]
        end_r, end_c = start_r + dr, start_c + dc

        if not (0 <= start_r < ROWS and 0 <= start_c < COLS and
                0 <= end_r < ROWS and 0 <= end_c < COLS):
            return None

        return start_r, start_c, end_r, end_c

    @tf.function # Re-enabled after debugging
    def _train_step(self, states, actions, rewards, next_states, dones, importance_weights):
        with tf.GradientTape() as tape:
            q_values = self.model(states, training=True)
            next_q_values_target = self.target_model(next_states, training=False)
            next_q_values_main = self.model(next_states, training=False)

            next_actions = tf.argmax(next_q_values_main, axis=1, output_type=tf.int32)
            next_actions_one_hot = tf.one_hot(next_actions, self.action_size, dtype=tf.float32)
            target_next_q = tf.reduce_sum(next_q_values_target * next_actions_one_hot, axis=1)

            target_q_values = rewards + self.gamma * target_next_q * (1.0 - dones)

            actions_one_hot = tf.one_hot(actions, self.action_size, dtype=tf.float32)
            predicted_values = tf.reduce_sum(q_values * actions_one_hot, axis=1)

            td_errors = tf.abs(target_q_values - predicted_values)
            element_wise_loss = self.loss_function(target_q_values, predicted_values)

            # Use importance weights only if PER is enabled
            if self.use_per:
                weighted_loss = element_wise_loss * importance_weights
            else:
                weighted_loss = element_wise_loss # No weighting if PER is off

            loss = tf.reduce_mean(weighted_loss)

        if tf.math.is_nan(loss) or tf.math.is_inf(loss):
            tf.print("Warning: NaN or Inf detected in loss. Skipping gradient update.")
            return loss, tf.zeros_like(td_errors, dtype=tf.float32)

        grads = tape.gradient(loss, self.model.trainable_variables)
        grads_have_nan_inf = tf.constant(False, dtype=tf.bool)
        for grad in grads:
            if grad is not None and grad.dtype.is_floating:
                is_invalid = tf.reduce_any(tf.math.is_nan(grad)) or tf.reduce_any(tf.math.is_inf(grad))
                grads_have_nan_inf = grads_have_nan_inf or is_invalid

        if grads_have_nan_inf:
            tf.print("Warning: NaN or Inf detected in gradients. Skipping gradient update.")
            return loss, tf.zeros_like(td_errors, dtype=tf.float32)
        else:
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return loss, td_errors

    def _experience_generator(self):
        while True:
            current_size = self.replay_buffer_size if self.memory_full else self.memory_index
            if current_size < self.batch_size:
                continue

            if self.use_per:
                # --- PER Sampling Logic ---
                valid_priorities = self.priorities[:current_size].astype(np.float64)
                scaled_priorities = np.power(valid_priorities + self.per_epsilon, self.per_alpha)
                prob_sum = np.sum(scaled_priorities)

                if prob_sum <= 0:
                    probabilities = np.ones(current_size, dtype=np.float64) / current_size
                else:
                    probabilities = scaled_priorities / prob_sum

                if np.isnan(probabilities).any():
                    print("Warning: NaN detected in sampling probabilities. Using uniform distribution.")
                    probabilities = np.ones(current_size, dtype=np.float64) / current_size

                # Ensure probabilities sum to 1
                prob_sum_check = probabilities.sum()
                if not np.isclose(prob_sum_check, 1.0):
                    probabilities /= prob_sum_check # Renormalize if needed

                sampled_indices = np.random.choice(current_size, self.batch_size, p=probabilities, replace=True)

                total_samples = current_size
                weights = np.power(total_samples * probabilities[sampled_indices], -self.per_beta).astype(np.float64)
                max_weight = np.max(weights)
                weights /= max_weight if max_weight > 0 else 1.0
                batch_weights = weights.astype(np.float32)
                # --- End PER Sampling --- 
            else:
                # --- Uniform Sampling Logic ---
                sampled_indices = np.random.choice(current_size, self.batch_size, replace=True)
                batch_weights = tf.ones(self.batch_size, dtype=tf.float32) # Uniform weights are 1
                # --- End Uniform Sampling ---

            batch_states = self.states[sampled_indices].astype(np.float32)
            batch_actions = self.actions[sampled_indices].astype(np.int32)
            batch_rewards = self.rewards[sampled_indices].astype(np.float32)
            batch_next_states = self.next_states[sampled_indices].astype(np.float32)
            batch_dones = self.dones[sampled_indices].astype(np.float32)
            batch_indices = sampled_indices.astype(np.int64)

            yield (batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones, batch_weights, batch_indices)

    def create_tf_dataset(self):
        output_signature = (
            tf.TensorSpec(shape=(None, self.state_size), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(None, self.state_size), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int64)
        )

        dataset = tf.data.Dataset.from_generator(
            self._experience_generator,
            output_signature=output_signature
        )

        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def update_priorities(self, indices, td_errors):
        # Only update priorities if PER is enabled
        if not self.use_per:
            return

        current_size = self.replay_buffer_size if self.memory_full else self.memory_index
        valid_mask = indices < current_size
        indices_in_bounds = indices[valid_mask]
        td_errors_in_bounds = td_errors[valid_mask]

        if len(indices_in_bounds) == 0:
            return

        new_priorities = np.abs(td_errors_in_bounds) + self.per_epsilon
        self.priorities[indices_in_bounds] = new_priorities

    @tf.function
    def predict_batch(self, states_tensor):
        return self.model(states_tensor, training=False)

    def act_batch(self, states, legal_actions_list):
        if np.random.rand() <= self.epsilon:
            return [random.choice(legal_actions) if legal_actions else np.random.randint(self.action_size)
                    for legal_actions in legal_actions_list]

        states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
        q_values_batch_raw = self.predict_batch(states_tensor)
        q_values_batch = tf.cast(q_values_batch_raw, tf.float32).numpy()

        selected_actions = []
        for i, (q_values, legal_actions) in enumerate(zip(q_values_batch, legal_actions_list)):
            if not legal_actions:
                selected_actions.append(np.random.randint(self.action_size) if self.action_size > 0 else 0)
                continue

            if np.isnan(q_values).any():
                selected_actions.append(random.choice(legal_actions))
                continue

            legal_q_values = np.take(q_values, legal_actions)

            if np.isnan(legal_q_values).any() or np.isinf(legal_q_values).any():
                selected_actions.append(random.choice(legal_actions))
                continue

            max_q = np.max(legal_q_values)
            best_indices_in_legal = np.where(np.isclose(legal_q_values, max_q, rtol=1e-5, atol=1e-8))[0]

            best_action_indices = [legal_actions[idx] for idx in best_indices_in_legal]

            if len(best_action_indices) > 1:
                chosen_action = random.choice(best_action_indices)
            else:
                chosen_action = best_action_indices[0]

            selected_actions.append(chosen_action)

        return selected_actions

    def _get_next_boards_batch(self, board_states, move_indices):
        from game_env import _simulate_moves_batch_jit, DIRECTIONS, COLS, ROWS

        num_boards = len(move_indices)
        moves = []

        for i in range(num_boards):
            action_index = move_indices[i]
            move = self._get_move_from_action(action_index)
            if move:
                moves.append(move)

        if not moves:
            return np.array([])

        if len(moves) != len(board_states):
            if board_states.shape[0] == 1:
                board_states_batch = np.repeat(board_states, len(moves), axis=0)
            else:
                print(f"Warning: Mismatch between number of moves ({len(moves)}) and boards ({len(board_states)})")
                return np.array([])
        else:
            board_states_batch = board_states

        board_states_batch = board_states_batch.astype(np.int8)

        return _simulate_moves_batch_jit(board_states_batch, moves, PLAYER_B_ID)

    def load(self, name):
        try:
            self.model.load_weights(name)
            self.update_target_model()
            print(f"Model weights loaded from {name}")
        except Exception as e:
            print(f"Error loading model weights from {name}: {e}")

    def save(self, name):
        try:
            self.model.save_weights(name)
            print(f"Model weights saved to {name}")
        except Exception as e:
            print(f"Error saving model weights to {name}: {e}")

    def save_for_tfjs(self, file_path):
        try:
            tf.saved_model.save(self.model, file_path)
            print(f"Model saved in SavedModel format at {file_path}. ")
            print(f"Convert to TF.js using: tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model {file_path} ./tfjs_model/web_model")

        except Exception as e:
            print(f"Error saving model for TF.js at {file_path}: {e}")

    def __len__(self):
        return self.replay_buffer_size if self.memory_full else self.memory_index


if __name__ == "__main__":
    agent = DQNAgent()
    print("DQN Agent Initialized.")
    print("Model Summary:")
    agent.model.summary()

    dummy_state = np.random.rand(STATE_SIZE).astype(np.float32)
    dummy_legal_actions = [10, 25, 100, 200]

    action = agent.act(dummy_state, dummy_legal_actions)
    print(f"Chosen action for dummy state: {action}")

    dummy_next_state = np.random.rand(STATE_SIZE).astype(np.float32)
    agent.remember(dummy_state, action, 1.0, dummy_next_state, False)

    for i in range(agent.batch_size + 10):
        state = np.random.rand(STATE_SIZE).astype(np.float32)
        action = random.choice(dummy_legal_actions)
        reward = np.random.rand()
        next_state = np.random.rand(STATE_SIZE).astype(np.float32)
        done = random.choice([True, False])
        agent.remember(state, action, reward, next_state, done)
        agent.priorities[i] = np.random.rand() + 0.1

    print(f"Buffer size: {len(agent)}")

    if len(agent) >= agent.batch_size:
        print("Creating dataset...")
        dataset = agent.create_tf_dataset()
        iterator = iter(dataset)
        print("Getting one batch from dataset...")
        try:
            s, a, r, ns, d, w, idx = next(iterator)
            print(f"Sampled batch shapes: s={s.shape}, a={a.shape}, r={r.shape}, ns={ns.shape}, d={d.shape}, w={w.shape}, idx={idx.shape}")

            print("Performing one train step...")
            loss, td_errors = agent._train_step(s, a, r, ns, d, w)
            print(f"Example train step completed. Loss: {loss.numpy()}")

            print("Updating priorities...")
            agent.update_priorities(idx.numpy(), td_errors.numpy())
            print(f"Priorities updated for indices: {idx.numpy()[:5]}...")
            print(f"New priorities sample: {agent.priorities[idx.numpy()[:5]]}")

        except Exception as e:
            print(f"Error during dataset iteration or training step: {e}")
            import traceback
            traceback.print_exc()

    else:
        print("Not enough samples in buffer to create dataset.")

    agent.save("dqn_test.weights.h5")
    agent.load("dqn_test.weights.h5")
    agent.save_for_tfjs("./dqn_tfjs_test_model")
