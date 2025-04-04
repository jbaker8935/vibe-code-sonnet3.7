import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
from game_env import NUM_ACTIONS, HISTORY_LENGTH

def convert_model_for_tfjs():
    """Convert the DQN model to TensorFlow.js format with correct binary board shapes."""
    
    # Calculate input shape based on binary representation
    BINARY_BOARD_SIZE = 5  # 5 uint32 values per board state
    STATE_SIZE = BINARY_BOARD_SIZE + (HISTORY_LENGTH * BINARY_BOARD_SIZE) + 1  # Current + history + player
    
    # Create a new model with the binary board architecture
    input_layer = keras.layers.Input(shape=(STATE_SIZE,))
    
    # Split inputs into current board, history, and player indicator
    current_board = keras.layers.Lambda(lambda x: x[:, :BINARY_BOARD_SIZE])(input_layer)
    history_boards = keras.layers.Lambda(lambda x: x[:, BINARY_BOARD_SIZE:-1])(input_layer)
    player_input = keras.layers.Lambda(lambda x: x[:, -1:])(input_layer)
    
    # Process current binary board
    x1 = keras.layers.Dense(256, activation='relu')(current_board)
    x1 = keras.layers.Dense(128, activation='relu')(x1)
    
    # Process history - reshape to (batch_size, HISTORY_LENGTH, BINARY_BOARD_SIZE)
    history_reshaped = keras.layers.Reshape((HISTORY_LENGTH, BINARY_BOARD_SIZE))(history_boards)
    
    # Add attention mechanism
    attention = keras.layers.Dense(64, activation='tanh')(history_reshaped)
    attention = keras.layers.Dense(1, activation='sigmoid')(attention)
    x2 = keras.layers.Multiply()([history_reshaped, attention])
    
    # Process temporal patterns
    x2 = keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True))(x2)
    x2 = keras.layers.Bidirectional(keras.layers.LSTM(64))(x2)
    
    # Combine features
    combined = keras.layers.Concatenate()([x1, x2, player_input])
    x = keras.layers.Dense(256, activation='relu')(combined)
    x = keras.layers.Dense(128, activation='relu')(x)
    
    # Output layer - one value per possible action
    output = keras.layers.Dense(NUM_ACTIONS, activation='linear')(x)
    
    # Create and compile model
    model = keras.Model(inputs=input_layer, outputs=output)
    model.compile(optimizer='adam', loss='mse')
    
    # Print model summary for verification
    print("\nModel Architecture:")
    model.summary()
    
    # Create output directories
    output_dir = "tfjs_model"
    os.makedirs(output_dir, exist_ok=True)
    web_model_dir = f"{output_dir}/web_model"
    os.makedirs(web_model_dir, exist_ok=True)
    
    # Save the model in H5 format
    h5_path = f"{output_dir}/model.h5"
    model.save(h5_path)
    print(f"\nModel saved to {h5_path}")
    
    # Save model architecture to JSON
    model_config = model.to_json()
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