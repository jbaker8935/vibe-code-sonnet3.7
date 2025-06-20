# Neural Network for AlphaZero will be implemented here.
import tensorflow as tf
import config # Add this import

# # Try to force CPU execution to rule out GPU-specific issues
# # try:
# #     tf.config.set_visible_devices([], 'GPU')
# #     print("INFO: TensorFlow GPU explicitly disabled. Using CPU.")
# # except (RuntimeError, ValueError) as e: # ValueError can occur if GPUs are already initialized
# #     print(f"INFO: Could not disable GPU, possibly already initialized or no GPU found: {e}")

from tensorflow.keras import layers, Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

from config import (AZ_NN_INPUT_DEPTH, AZ_NN_RESIDUAL_BLOCKS, AZ_NN_FILTERS,
                    AZ_NN_POLICY_HEAD_UNITS, AZ_NN_VALUE_HEAD_UNITS,
                    AZ_LEARNING_RATE, AZ_L2_REGULARIZATION,
                    AZ_VALUE_LOSS_WEIGHT, AZ_POLICY_LOSS_WEIGHT,
                    AZ_DISABLE_ATTENTION, AZ_RECOVERY_MODE, AZ_GRADIENT_CLIP_NORM)
from env_const import NUM_ACTIONS

def build_alpha_zero_network(input_shape=(8, 4, 18), num_actions=NUM_ACTIONS, learning_rate_schedule=None):
    """Builds an AlphaZero network that leverages board history and spatial structure."""
    input_tensor = layers.Input(shape=input_shape)

    # Initial feature extraction with Conv2D
    x = layers.Conv2D(
        AZ_NN_FILTERS,
        kernel_size=(3, 3),
        padding='same',
        activation='relu',
        kernel_initializer='he_normal',
        kernel_regularizer=l2(AZ_L2_REGULARIZATION) if AZ_L2_REGULARIZATION > 0 else None,
        name='initial_conv2d'
    )(input_tensor)

    # Optionally add more conv layers or residual blocks
    for i in range(AZ_NN_RESIDUAL_BLOCKS):
        res_input = x
        residual_tensor = layers.Conv2D(
            AZ_NN_FILTERS,
            kernel_size=(3, 3),
            padding='same',
            activation='relu',
            kernel_initializer='he_normal',
            kernel_regularizer=l2(AZ_L2_REGULARIZATION) if AZ_L2_REGULARIZATION > 0 else None,
            name=f'res_block_{i}_conv1'
        )(x)
        residual_tensor = layers.Conv2D(
            AZ_NN_FILTERS,
            kernel_size=(3, 3),
            padding='same',
            activation='relu',
            kernel_initializer='he_normal',
            kernel_regularizer=l2(AZ_L2_REGULARIZATION) if AZ_L2_REGULARIZATION > 0 else None,
            name=f'res_block_{i}_conv2'
        )(residual_tensor)
        x = layers.Add(name=f'res_block_{i}_add')([res_input, residual_tensor])
        # Optionally add layer norm
        # x = layers.LayerNormalization(name=f'res_block_{i}_layernorm')(x)

    x = layers.Flatten(name='flatten')(x)
    shared_trunk_output = layers.Dense(
        AZ_NN_FILTERS,
        activation='relu',
        kernel_initializer='he_normal',
        kernel_regularizer=l2(AZ_L2_REGULARIZATION) if AZ_L2_REGULARIZATION > 0 else None,
        name='shared_dense'
    )(x)
    shared_trunk_output = layers.Activation('linear', name='final_shared_output')(shared_trunk_output)

    # Policy head
    policy_head = layers.Dense(
        AZ_NN_POLICY_HEAD_UNITS,
        activation='relu',
        kernel_initializer='he_normal',
        kernel_regularizer=l2(AZ_L2_REGULARIZATION) if AZ_L2_REGULARIZATION > 0 else None,
        name='policy_head_dense'
    )(shared_trunk_output)
    policy_logits = layers.Dense(
        num_actions,
        name='policy_logits',
        kernel_initializer='he_normal',
        kernel_regularizer=l2(AZ_L2_REGULARIZATION) if AZ_L2_REGULARIZATION > 0 else None
    )(policy_head)
    policy_output = layers.Activation('softmax', name='policy_output')(policy_logits)

    # Value head
    value_head = layers.Dense(
        AZ_NN_VALUE_HEAD_UNITS,
        activation='relu',
        kernel_initializer='he_normal',
        kernel_regularizer=l2(AZ_L2_REGULARIZATION) if AZ_L2_REGULARIZATION > 0 else None,
        name='value_head_dense'
    )(shared_trunk_output)
    value_output = layers.Dense(
        1,
        activation='tanh',
        name='value_output',
        kernel_initializer='he_normal',
        kernel_regularizer=l2(AZ_L2_REGULARIZATION) if AZ_L2_REGULARIZATION > 0 else None
    )(value_head)

    model = Model(inputs=input_tensor, outputs=[policy_output, value_output])

    current_lr = learning_rate_schedule if learning_rate_schedule is not None else AZ_LEARNING_RATE
    optimizer = Adam(
        learning_rate=current_lr,
        clipnorm=AZ_GRADIENT_CLIP_NORM,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        amsgrad=False
    )

    model.compile(
        optimizer=optimizer,
        loss={
            'policy_output': 'categorical_crossentropy',
            'value_output': 'mse'
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
    def __init__(self, model_path=None, learning_rate_schedule=None):
        self.model = build_alpha_zero_network(learning_rate_schedule=learning_rate_schedule)
        if model_path:
            try:
                self.load_model(model_path)
                print(f"Loaded model from {model_path}")
            except Exception as e:
                print(f"Could not load model from {model_path}: {e}. Initializing a new model.")
        
        # For activation logging
        self._activation_model = None
        self._activation_layer_names = [] # Initialize as empty list

    def _get_activation_model(self):
        if self._activation_model is None:
            try:
                layer_outputs = []
                # Use the centrally defined layer names for activations
                requested_layer_names = self.get_activation_layer_names()
                found_layer_names = [] # Store names of layers actually found
                
                if not requested_layer_names:
                    print("Warning: No activation layer names provided by get_activation_layer_names(). Activation logging will be disabled.")
                    return None

                print(f"[Activation Logging] Attempting to find layers: {requested_layer_names}")

                for layer_name_to_find in requested_layer_names:
                    try:
                        layer = self.model.get_layer(name=layer_name_to_find)
                        layer_outputs.append(layer.output)
                        found_layer_names.append(layer_name_to_find) # Add the name of the found layer
                    except ValueError:
                        print(f"Warning: Layer \'{layer_name_to_find}\' not found in model. Skipping for activation logging.")
                
                if not layer_outputs:
                    print("Warning: No specified activation layers were found in the model. Activation model not created.")
                    # Print all available layer names for debugging
                    all_layer_names = [l.name for l in self.model.layers]
                    print(f"Available layer names in the model: {all_layer_names}")
                    return None

                # Update self._activation_layer_names to only include layers that were actually found
                self._activation_layer_names = found_layer_names # Use the list of names that were successfully found

                self._activation_model = Model(inputs=self.model.input, outputs=layer_outputs)
                print(f"Activation model created with output layers: {self._activation_layer_names}")
            except Exception as e:
                print(f"Error creating activation model: {e}")
                self._activation_model = None # Ensure it's None if creation fails
        return self._activation_model

    def get_activation_layer_names(self):
        """Returns a list of layer names whose activations might be interesting to log."""
        # Check if a specific list is provided in config
        if hasattr(config, 'AZ_ACTIVATION_LAYER_NAMES_TO_LOG') and config.AZ_ACTIVATION_LAYER_NAMES_TO_LOG:
            names_to_log = config.AZ_ACTIVATION_LAYER_NAMES_TO_LOG
        else:
            # Default list of layers to log activations from
            # These names should correspond to the 'name' argument in layer constructors
            names_to_log = [
                "initial_dense", # The Dense layer itself, its output is input to next or activation
                # To get ReLU output, the ReLU layer must be named, or this must be post-activation
                # Assuming initial_dense has activation='relu', its output *is* the ReLU output.
            ]

            # Add ReLU layers from residual blocks if they exist and are named
            # The current build_alpha_zero_network names Dense layers, and Add layers.
            # ReLU activations are part of Dense layers or separate Activation layers.
            # Let's refine this to target actual named layers or common patterns.

            # Example: If Dense layers have relu activation, their name is enough.
            # If there are separate ReLU layers, they need names like 'initial_relu'.
            # For now, let's assume Dense layers with 'relu' activation are targeted by their Dense name.

            for i in range(config.AZ_NN_RESIDUAL_BLOCKS):
                names_to_log.append(f'res_block_{i}_dense1') # Output of first Dense in res block (is ReLU)
                names_to_log.append(f'res_block_{i}_dense2') # Output of second Dense in res block (is ReLU)
                names_to_log.append(f'res_block_{i}_add')    # Output of the Add layer

            names_to_log.append("final_shared_output") # This is an Activation('linear') layer

            if config.AZ_DISABLE_ATTENTION:
                if hasattr(config, 'AZ_POLICY_HEAD_TYPE') and config.AZ_POLICY_HEAD_TYPE == "default":
                    names_to_log.append("policy_head_dense") # Output of Dense in policy head (is ReLU)
                if hasattr(config, 'AZ_VALUE_HEAD_TYPE') and config.AZ_VALUE_HEAD_TYPE == "default":
                    names_to_log.append("value_head_dense")  # Output of Dense in value head (is ReLU)
            else:
                # Add attention-related layers if needed, assuming they are named
                names_to_log.append('policy_head_dense_pre_attention')
                names_to_log.append('policy_attention_weights')
                names_to_log.append('policy_head_dense_after_attention')
                names_to_log.append('value_head_dense_pre_attention')
                names_to_log.append('value_attention_weights')
                names_to_log.append('value_head_dense_after_attention')

            names_to_log.append("policy_logits")
            names_to_log.append("value_output")

        # Ensure no duplicates and maintain order
        return list(dict.fromkeys(names_to_log))

    def get_activations(self, state_batch):
        activation_model = self._get_activation_model()
        if activation_model is None:
            return {} # Return empty dict if no model

        if not isinstance(state_batch, tf.Tensor):
            state_batch = tf.convert_to_tensor(state_batch, dtype=tf.float32)
        if state_batch.ndim == 1: # Assuming input_shape is (AZ_NN_INPUT_DEPTH,)
             state_batch = tf.expand_dims(state_batch, axis=0)
        
        activations_list = activation_model.predict_on_batch(state_batch)
        
        # If only one layer is watched, activations_list might not be a list
        if len(self._activation_layer_names) == 1 and not isinstance(activations_list, list):
            activations_list = [activations_list]
            
        return dict(zip(self._activation_layer_names, activations_list))

    def predict(self, state):
        """Takes a game state, returns policy probabilities and value."""
        import numpy as np
        # Acceptable shapes: (8, 4, 18) or (1, 8, 4, 18)
        expected_shape = (8, 4, 18)
        batch_shape = (1, 8, 4, 18)
        if isinstance(state, np.ndarray):
            if state.shape == expected_shape:
                state = np.expand_dims(state, axis=0)  # Add batch dimension
            elif state.shape == batch_shape:
                pass  # Already batched
            else:
                print(f"[DIAGNOSTIC] State type: {type(state)}; shape: {state.shape}; first 10: {state.flatten()[:10]}")
                import traceback; traceback.print_stack()
                raise ValueError(f"AlphaZeroNetwork.predict: State shape {state.shape} does not match expected {expected_shape} or {batch_shape}. State (first 10): {state.flatten()[:10]}")
        elif hasattr(state, 'shape'):
            if tuple(state.shape) == expected_shape:
                state = tf.expand_dims(state, axis=0)
            elif tuple(state.shape) == batch_shape:
                pass
            else:
                raise ValueError(f"AlphaZeroNetwork.predict: State shape {state.shape} does not match expected {expected_shape} or {batch_shape}.")

        if not isinstance(state, tf.Tensor):
            state = tf.convert_to_tensor(state, dtype=tf.float32)

        try:
            # First try predict_on_batch which is more efficient
            policy_probs, value = self.model.predict_on_batch(state)
        except Exception as e:
            # Fall back to standard predict method if predict_on_batch fails
            print(f"Warning: predict_on_batch failed, falling back to predict: {e}")
            outputs = self.model.predict(state, verbose=0)
            if isinstance(outputs, list) and len(outputs) == 2:
                policy_probs, value = outputs
            else:
                # Alternative output format
                policy_probs = outputs["policy_output"]
                value = outputs["value_output"]
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

    # Dummy state now must match new input depth (current + 2 prior boards)
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
