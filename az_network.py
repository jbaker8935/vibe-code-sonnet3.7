# Neural Network for AlphaZero will be implemented here.
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

from config import (AZ_NN_INPUT_DEPTH, AZ_NN_RESIDUAL_BLOCKS, AZ_NN_FILTERS,
                    AZ_NN_POLICY_HEAD_UNITS, AZ_NN_VALUE_HEAD_UNITS,
                    AZ_LEARNING_RATE, AZ_L2_REGULARIZATION,
                    AZ_VALUE_LOSS_WEIGHT, AZ_POLICY_LOSS_WEIGHT)
from env_const import NUM_ACTIONS

def build_residual_block(input_tensor, filters):
    """Builds a residual block using Dense layers for 1D input."""
    # Path 1
    x = layers.Dense(filters, kernel_regularizer=l2(AZ_L2_REGULARIZATION), activation='relu')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(filters, kernel_regularizer=l2(AZ_L2_REGULARIZATION))(x)
    x = layers.BatchNormalization()(x)

    # Path 2 (skip connection)
    # Ensure the skip connection has the same number of filters if input_tensor filters != filters
    if input_tensor.shape[-1] != filters:
        input_tensor_reshaped = layers.Dense(filters, kernel_regularizer=l2(AZ_L2_REGULARIZATION))(input_tensor)
    else:
        input_tensor_reshaped = input_tensor

    x = layers.Add()([input_tensor_reshaped, x])
    x = layers.ReLU()(x)
    return x

def build_alpha_zero_network(input_shape=(AZ_NN_INPUT_DEPTH,), num_actions=NUM_ACTIONS):
    """Builds the AlphaZero-style neural network."""
    input_tensor = layers.Input(shape=input_shape)

    # Initial Dense layer to process the flat input vector
    x = layers.Dense(AZ_NN_FILTERS, kernel_regularizer=l2(AZ_L2_REGULARIZATION))(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Common Body with Dense Residual Blocks
    common_representation = x
    for _ in range(AZ_NN_RESIDUAL_BLOCKS):
        common_representation = build_residual_block(common_representation, AZ_NN_FILTERS)

    # Policy Head
    policy_head = layers.Dense(AZ_NN_POLICY_HEAD_UNITS, kernel_regularizer=l2(AZ_L2_REGULARIZATION))(common_representation)
    policy_head = layers.BatchNormalization()(policy_head)
    policy_head = layers.ReLU()(policy_head)
    policy_logits = layers.Dense(num_actions, name='policy_logits', kernel_regularizer=l2(AZ_L2_REGULARIZATION))(policy_head)
    policy_output = layers.Activation('softmax', name='policy_output')(policy_logits)

    # Value Head
    value_head = layers.Dense(AZ_NN_VALUE_HEAD_UNITS, kernel_regularizer=l2(AZ_L2_REGULARIZATION))(common_representation)
    value_head = layers.BatchNormalization()(value_head)
    value_head = layers.ReLU()(value_head)
    value_output = layers.Dense(1, activation='tanh', name='value_output', kernel_regularizer=l2(AZ_L2_REGULARIZATION))(value_head)

    model = Model(inputs=input_tensor, outputs=[policy_output, value_output])

    optimizer = Adam(learning_rate=AZ_LEARNING_RATE)

    model.compile(
        optimizer=optimizer,
        loss={
            'policy_output': tf.keras.losses.CategoricalCrossentropy(), # from_logits=False as softmax is applied
            'value_output': tf.keras.losses.MeanSquaredError()
        },
        loss_weights={
            'policy_output': AZ_POLICY_LOSS_WEIGHT,
            'value_output': AZ_VALUE_LOSS_WEIGHT
        },
        metrics={
            'policy_output': 'accuracy',
            'value_output': 'mae'
        }
    )
    return model

class AlphaZeroNetwork:
    def __init__(self, model_path=None):
        self.model = build_alpha_zero_network()
        if model_path:
            try:
                self.load_model(model_path)
                print(f"Loaded model from {model_path}")
            except Exception as e:
                print(f"Could not load model from {model_path}: {e}. Initializing a new model.")

    def predict(self, state):
        """Takes a game state, returns policy probabilities and value."""
        if not isinstance(state, tf.Tensor):
            state = tf.convert_to_tensor(state, dtype=tf.float32)
        if state.ndim == 1:
            state = tf.expand_dims(state, axis=0)
        
        policy_probs, value = self.model.predict_on_batch(state)
        return policy_probs[0], value[0][0]

    def save_model(self, filepath):
        """Saves the model weights."""
        self.model.save_weights(filepath)
        print(f"Model weights saved to {filepath}")

    def load_model(self, filepath):
        """Loads the model weights."""
        self.model.load_weights(filepath)
        print(f"Model weights loaded from {filepath}")

if __name__ == '__main__':
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Input depth: {AZ_NN_INPUT_DEPTH}, Num Actions: {NUM_ACTIONS}")
    az_nn = AlphaZeroNetwork()
    az_nn.model.summary()

    dummy_state_np = tf.random.uniform((AZ_NN_INPUT_DEPTH,)).numpy() # Use numpy array as often states come like this
    print(f"\nDummy state (numpy) shape: {dummy_state_np.shape}")

    policy_probs, value = az_nn.predict(dummy_state_np)
    print(f"Policy probabilities shape: {policy_probs.shape}")
    print(f"Policy probabilities (first 10): {policy_probs[:10]}")
    print(f"Value: {value}")

    TEST_MODEL_PATH = "az_test_model.weights.h5"
    az_nn.save_model(TEST_MODEL_PATH)
    az_nn_loaded = AlphaZeroNetwork() # Create a new instance before loading
    az_nn_loaded.load_model(TEST_MODEL_PATH)
    policy_probs_loaded, value_loaded = az_nn_loaded.predict(dummy_state_np)

    tf.debugging.assert_near(policy_probs, policy_probs_loaded, rtol=1e-5, atol=1e-5)
    tf.debugging.assert_near(value, value_loaded, rtol=1e-5, atol=1e-5)
    print("\nModel saving and loading test passed.")

    dummy_target_policy = tf.random.uniform((1, NUM_ACTIONS))
    dummy_target_policy = dummy_target_policy / tf.reduce_sum(dummy_target_policy, axis=1, keepdims=True)
    dummy_target_value = tf.random.uniform((1, 1), minval=-1.0, maxval=1.0)
    dummy_state_batch = tf.expand_dims(dummy_state_np, axis=0)

    print(f"\nDummy state batch shape: {dummy_state_batch.shape}")
    print(f"Dummy target policy shape: {dummy_target_policy.shape}")
    print(f"Dummy target value shape: {dummy_target_value.shape}")

    history = az_nn.model.fit(
        dummy_state_batch,
        {'policy_output': dummy_target_policy, 'value_output': dummy_target_value},
        verbose=0
    )
    print(f"Dummy training step loss: {history.history['loss'][0]}")
    print("Neural network structure and basic functions implemented and tested.")
