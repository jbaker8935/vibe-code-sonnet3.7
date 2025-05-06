"""Utility functions for Switcharoo DQN training."""

import numpy as np
import wandb
from game_env import DIRECTIONS, COLS, ROWS

def _action_index_to_move(action_index):
    """Converts an action index to (start_r, start_c, end_r, end_c)."""
    if not (0 <= action_index < 8 * ROWS * COLS):
        return None  # Invalid index

    direction_index = action_index % 8
    start_cell_index = action_index // 8

    start_r = start_cell_index // COLS
    start_c = start_cell_index % COLS

    dr, dc = DIRECTIONS[direction_index]
    end_r, end_c = start_r + dr, start_c + dc

    # Basic validation (coordinate check)
    if not (0 <= start_r < ROWS and 0 <= start_c < COLS and
            0 <= end_r < ROWS and 0 <= end_c < COLS):
         return None  # Invalid move

    return start_r, start_c, end_r, end_c

def _validate_reward(reward):
    """Validate and clip reward values."""
    if np.isnan(reward) or np.isinf(reward):
        return 0.0
    return np.clip(reward, -250.0, 250.0)  # Wider clipping range

def init_wandb(enable_wandb, project_name, agent):
    """Initialize Weights & Biases logging if enabled."""
    if not enable_wandb:
        return False
        
    try:
        import wandb
        
        # Initialize wandb
        config = {
            "learning_rate": agent.learning_rate,
            "epsilon": agent.epsilon,
            "epsilon_decay": agent.epsilon_decay,
            "epsilon_min": agent.epsilon_min,
            "gamma": agent.gamma,
            "batch_size": agent.batch_size,
            "replay_buffer_size": agent.replay_buffer_size,
            "per_enabled": agent.use_per,
            "per_alpha": agent.per_alpha if agent.use_per else None,
            "per_beta": agent.per_beta if agent.use_per else None
        }
        
        wandb.init(project=project_name, config=config)
        print(f"Wandb initialized for project: {project_name}")
        return True
        
    except Exception as e:
        print(f"Failed to initialize wandb: {e}")
        return False