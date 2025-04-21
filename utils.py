"""Utility functions for the Switcharoo game training."""
import numpy as np
from numba import njit
import wandb
from config import WANDB_PROJECT, WANDB_ENTITY

@njit(cache=True)
def _action_index_to_move(action_index):
    """Convert action index to move coordinates (row1, col1, row2, col2)."""
    if action_index is None:
        return None
    start_r = action_index // 512
    start_c = (action_index % 512) // 64
    end_r = (action_index % 64) // 8
    end_c = action_index % 8
    return start_r, start_c, end_r, end_c

@njit(cache=True)
def _validate_reward(reward):
    """JIT-compiled reward validation."""
    if np.isnan(reward) or np.isinf(reward):
        return 0.0
    return reward

def init_wandb(enable_wandb, run_name, agent, group_name=None):
    """Initialize wandb with error handling"""
    if not enable_wandb:
        return False

    try:
        if agent is None:
            print("\nWarning: Agent object is None, cannot initialize wandb config.")
            return False

        wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name=run_name,
            group=group_name,
            config={
                "learning_rate": agent.learning_rate,
                "gamma": agent.gamma,
                "epsilon": agent.epsilon,
                "epsilon_decay": agent.epsilon_decay,
                "epsilon_min": agent.epsilon_min,
                "batch_size": agent.batch_size,
                "replay_buffer_size": agent.memory.maxlen,
                "target_update_freq": agent.target_update_freq,
                "gradient_clip_norm": agent.optimizer.clipnorm if hasattr(agent.optimizer, 'clipnorm') else None
            }
        )
        return True
    except Exception as e:
        print(f"\nWarning: Failed to initialize wandb: {e}")
        print("Continuing training without wandb logging...")
        return False