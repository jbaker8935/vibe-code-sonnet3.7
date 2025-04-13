import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from dqn_agent import DQNAgent
import numpy as np
import json
from game_env import NUM_ACTIONS, ROWS, COLS, HISTORY_LENGTH

# State size calculation
BINARY_BOARD_SIZE = 5
STATE_SIZE = BINARY_BOARD_SIZE + (HISTORY_LENGTH * BINARY_BOARD_SIZE) + 1
ACTION_SIZE = NUM_ACTIONS

def create_model():
    """Creates a model matching the DQNAgent architecture exactly by copying its _build_model method."""
    input_layer = layers.Input(shape=(STATE_SIZE,))

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

    output = layers.Dense(ACTION_SIZE, activation='linear')(x)

    model = keras.Model(inputs=input_layer, outputs=output)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse'
    )
    return model

def convert_model_for_tfjs():
    """Converts the DQNAgent model for use with TensorFlow.js."""
    # Initialize the agent to get its model
    agent = DQNAgent(
        epsilon=0.0,
        epsilon_decay=1.0,
        epsilon_min=0.0,
        replay_buffer_size=1000,
        batch_size=64,
        target_update_freq=100
    )
    
    # Use the agent's model directly instead of creating a new one
    weights_file = "switcharoo_dqn_resumed_phase2.weights.h5"
    agent.load(weights_file)
    model = agent.model  # Use the agent's model directly
    
    # Create output directory
    output_dir = "tfjs_model"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the model in H5 format
    h5_path = f"{output_dir}/model.h5"
    model.save(h5_path, save_format='h5')
    print(f"\nModel saved to {h5_path}")
    
    print("\nNext steps:")
    print("1. Install tensorflowjs with: pip install tensorflowjs==3.18.0")
    print("2. Convert the model with:")
    print(f"   tensorflowjs_converter --input_format=keras --output_format=tfjs_layers_model {h5_path} {output_dir}/web_model")

if __name__ == "__main__":
    convert_model_for_tfjs()