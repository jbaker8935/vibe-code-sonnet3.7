"""Tournament functionality for the Switcharoo game."""
import numpy as np
import random
from random import choice
from game_env import SwitcharooEnv, PLAYER_A, PLAYER_B
from config import (initial_position, NUM_VARIANTS, TOURNAMENT_MATCHES, 
                   MAX_STEPS_PER_EPISODE, NOISE_SCALE, TOURNAMENT_MODEL_FILE)
from dqn_agent import DQNAgent # Import DQNAgent to use its class definition

def create_agent_variant(base_agent, noise_scale=NOISE_SCALE, epsilon=0.0):
    """Create a variant of the base agent by adding Gaussian noise to its weights."""
    variant = DQNAgent( # Use DQNAgent class directly
        state_size=base_agent.state_size,
        action_size=base_agent.action_size,
        learning_rate=base_agent.learning_rate,
        gamma=base_agent.gamma,
        epsilon=epsilon,
        epsilon_decay=base_agent.epsilon_decay,
        epsilon_min=base_agent.epsilon_min,
        replay_buffer_size=base_agent.replay_buffer_size, # Use replay_buffer_size
        batch_size=base_agent.batch_size,
        target_update_freq=base_agent.target_update_freq,
        gradient_clip_norm=base_agent.optimizer.clipnorm if hasattr(base_agent.optimizer, 'clipnorm') else None,
        use_per=base_agent.use_per, # Pass PER settings
        per_alpha=base_agent.per_alpha,
        per_beta=base_agent.per_beta,
        per_beta_increment=base_agent.per_beta_increment,
        per_epsilon=base_agent.per_epsilon
    )

    # Get the base agent's weights and apply noise
    weights = base_agent.model.get_weights()
    noisy_weights = []

    for w in weights:
        noise = np.random.normal(0, noise_scale * (np.abs(w).mean() + 1e-8), w.shape)
        noisy_weights.append(w + noise)

    variant.model.set_weights(noisy_weights)
    variant.update_target_model()

    # Prefill replay buffer with 25% of base agent memory
    base_memory_size = len(base_agent) # Use __len__ method
    sample_size = int(base_memory_size * 0.25)
    if sample_size > 0:
        # Sample indices from the base agent's buffer
        sampled_indices = random.sample(range(base_memory_size), sample_size)

        # Copy experiences from base agent's arrays to variant's arrays
        variant.states[:sample_size] = base_agent.states[sampled_indices]
        variant.actions[:sample_size] = base_agent.actions[sampled_indices]
        variant.rewards[:sample_size] = base_agent.rewards[sampled_indices]
        variant.next_states[:sample_size] = base_agent.next_states[sampled_indices]
        variant.dones[:sample_size] = base_agent.dones[sampled_indices]
        variant.priorities[:sample_size] = base_agent.priorities[sampled_indices] # Copy priorities too

        variant.memory_index = sample_size
        if sample_size >= variant.replay_buffer_size:
             variant.memory_full = True # Mark as full if sample filled it

    return variant

def run_match(env, agent_a, agent_b, max_steps=MAX_STEPS_PER_EPISODE):
    """Run a single match between two agents and return the winner."""
    state = env.reset(choice(initial_position))
    
    for step in range(max_steps):
        current_player = env.current_player
        
        if current_player == PLAYER_A:
            legal_actions = env.get_legal_action_indices(player=PLAYER_A)
            if not legal_actions:
                return PLAYER_B
            action = agent_a.act(state, legal_actions)
        else:
            legal_actions = env.get_legal_action_indices(player=PLAYER_B)
            if not legal_actions:
                return PLAYER_A
            action = agent_b.act(state, legal_actions)
        
        next_state, _, done, info = env.step(action)
        state = next_state
        
        if done:
            return info.get('winner')
    
    return 'DRAW'

def run_tournament(base_agent, num_variants=NUM_VARIANTS, matches_per_pair=TOURNAMENT_MATCHES, direct_phase2=False):
    """Run a round-robin tournament between variants of the base agent."""
    print("\n----- STARTING TOURNAMENT -----")
    
    epsilon_value = 0.01 if direct_phase2 else 0.05
    variants = [create_agent_variant(base_agent, epsilon=epsilon_value) for _ in range(num_variants)]
    variants.append(base_agent)
    
    scores = {i: 0 for i in range(len(variants))}
    env = SwitcharooEnv()
    total_matches = 0
    
    for i in range(len(variants)):
        for j in range(len(variants)):
            if i == j:
                continue
                
            agent_a = variants[i]
            agent_b = variants[j]
            
            for _ in range(matches_per_pair):
                winner = run_match(env, agent_a, agent_b)
                
                if winner == PLAYER_A:
                    scores[i] += 1
                elif winner == PLAYER_B:
                    scores[j] += 1
                
                total_matches += 1
    
    best_idx = max(scores, key=scores.get)
    best_score = scores[best_idx]
    
    print(f"Tournament completed: {total_matches} matches played")
    print(f"Best variant: {best_idx} with score {best_score}")
    
    for i, score in scores.items():
        print(f"Variant {i}: {score} points")
    
    return variants[best_idx], best_score, total_matches