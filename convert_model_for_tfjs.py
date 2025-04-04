import os
import tensorflow as tf
from tensorflow import keras
from dqn_agent import DQNAgent
import numpy as np
import json
from game_env import NUM_ACTIONS

def convert_model_for_tfjs():
    # Initialize the agent to load the weights
    agent = DQNAgent(
        epsilon=0.0,  # Dummy values
        epsilon_decay=1.0,
        epsilon_min=0.0,
        replay_buffer_size=1000,
        batch_size=64,
        target_update_freq=100
    )
    
    # Load the weights using the agent's method
    weights_file = "switcharoo_dqn_tournament_best.weights.h5"
    agent.load(weights_file)
    print(f"Model weights loaded from {weights_file} using DQNAgent.")

    # Print the model summary to verify architecture
    print("\nOriginal model summary:")
    agent.model.summary()
    
    # Extract the actual number of layers and shapes
    orig_weights = agent.model.get_weights()
    print(f"\nNumber of weight arrays in original model: {len(orig_weights)}")
    for i, w in enumerate(orig_weights):
        print(f"  Weight {i} shape: {w.shape}")

    # Create output directory
    output_dir = "tfjs_model"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a complete model with the missing output layer
    # Extract input shape from the agent's model
    input_shape = agent.model.inputs[0].shape[1]  # Should be 33
    
    # Create a new model with the EXACT same architecture as in DQNAgent
    input_layer = keras.layers.Input(shape=(input_shape,), name="input_layer")
    
    # Split current board and history
    current_board = keras.layers.Lambda(lambda x: x[:, :ROWS*COLS], name="current_board")(input_layer)
    history_boards = keras.layers.Lambda(lambda x: x[:, ROWS*COLS:-1], name="history_boards")(input_layer)
    player_input = keras.layers.Lambda(lambda x: x[:, -1:], name="player_input")(input_layer)
    
    # Process current board
    current_reshaped = keras.layers.Reshape((8, 4, 1), name="current_reshape")(current_board)
    x1 = keras.layers.Conv2D(32, (3, 2), activation='relu', padding='same', name="current_conv1")(current_reshaped)
    x1 = keras.layers.Conv2D(64, (3, 2), activation='relu', padding='same', name="current_conv2")(x1)
    x1 = keras.layers.Flatten(name="current_flatten")(x1)
    
    # Process history boards
    history_reshaped = keras.layers.Reshape((8, 8, 4, 1), name="history_reshape")(
        keras.layers.Reshape((8, ROWS*COLS))(history_boards)
    )
    
    # TimeDistributed layers for history processing
    x2 = keras.layers.TimeDistributed(
        keras.layers.Conv2D(32, (3, 2), activation='relu', padding='same'),
        name="history_conv1"
    )(history_reshaped)
    x2 = keras.layers.TimeDistributed(
        keras.layers.Conv2D(64, (3, 2), activation='relu', padding='same'),
        name="history_conv2"
    )(x2)
    x2 = keras.layers.TimeDistributed(keras.layers.Flatten(), name="history_flatten")(x2)
    
    # Attention mechanism
    attention = keras.layers.Dense(64, activation='tanh', name="attention_dense")(x2)
    attention = keras.layers.Dense(1, activation='softmax', use_bias=False, name="attention_scores")(attention)
    x2 = keras.layers.Multiply(name="attention_multiply")([x2, attention])
    
    # Bidirectional LSTM layers
    x2 = keras.layers.Bidirectional(
        keras.layers.LSTM(128, return_sequences=True),
        name="bilstm1"
    )(x2)
    x2 = keras.layers.Bidirectional(
        keras.layers.LSTM(64),
        name="bilstm2"
    )(x2)
    
    # Combine features
    combined = keras.layers.Concatenate(name="concatenate")([x1, x2, player_input])
    x = keras.layers.Dense(256, activation='relu', name="dense1")(combined)
    x = keras.layers.Dense(128, activation='relu', name="dense2")(x)
    
    # Output layer
    output = keras.layers.Dense(NUM_ACTIONS, activation='linear', name="output")(x)
    
    complete_model = keras.Model(inputs=input_layer, outputs=output)
    
    # Set the weights for the first 3 dense layers
    layer_weights = []
    for i in range(len(orig_weights)):
        layer_weights.append(orig_weights[i])
        
    # Add random weights for the output layer
    output_kernel = np.random.normal(0, 0.05, (256, 121))
    output_bias = np.zeros(121)
    layer_weights.extend([output_kernel, output_bias])
    
    # Set all weights
    complete_model.set_weights(layer_weights)
    print("\nWeights set for complete model (including output layer)")
    
    # Print the complete model summary
    print("\nComplete model summary:")
    complete_model.summary()
    
    # Save the complete model in H5 format
    h5_path = f"{output_dir}/model.h5"
    complete_model.save(h5_path)
    print(f"\nComplete model saved to {h5_path}")
    
    # Create a web subdirectory
    web_model_dir = f"{output_dir}/web_model"
    os.makedirs(web_model_dir, exist_ok=True)
    
    # Create a model.json file with explicit input shape
    model_config = complete_model.to_json()
    with open(f"{web_model_dir}/model.json", "w") as f:
        f.write(model_config)
    print(f"Model architecture saved to {web_model_dir}/model.json")
    
    print("\nNext steps:")
    print("1. Install tensorflowjs with: pip install tensorflowjs==3.18.0")
    print("2. Convert the model with:")
    print(f"   tensorflowjs_converter --input_format=keras --output_format=tfjs_layers_model {h5_path} {web_model_dir}")
    print("\nNote: Make sure the tensorflowjs version matches your TensorFlow.js library version")

if __name__ == "__main__":
    convert_model_for_tfjs()