"""Tournament system for Switcharoo agents."""
import copy
import random
import numpy as np
import time
import os

from game_env import SwitcharooEnv, DIRECTIONS, COLS, ROWS
from env_const import PLAYER_A, PLAYER_B, PLAYER_A_ID, PLAYER_B_ID
from dqn_agent import DQNAgent
from config import (NUM_VARIANTS, NOISE_SCALE, TOURNAMENT_MATCHES, initial_position)
from binary_board import board_to_binary

def create_variant_agents(base_agent, num_variants=NUM_VARIANTS, noise_scale=NOISE_SCALE):
    """Create variant agents by adding Gaussian noise to the base agent's weights."""
    variants = [copy.deepcopy(base_agent)]  # Include the original agent
    
    # Get the base weights
    base_weights = base_agent.model.get_weights()
    
    # Create variants with different noise patterns
    for i in range(1, num_variants):
        variant = copy.deepcopy(base_agent)
        
        # Apply adaptive noise based on weight magnitudes
        noisy_weights = []
        for w in base_weights:
            # Scale noise by the magnitude of weights
            weight_scale = np.mean(np.abs(w)) + 1e-8
            noise = np.random.normal(0, noise_scale * weight_scale, w.shape)
            noisy_weights.append(w + noise)
        
        variant.model.set_weights(noisy_weights)
        variant.epsilon = max(0.05, base_agent.epsilon)  # Ensure some exploration
        variants.append(variant)
    
    # Create one variant with higher epsilon for exploration
    if len(variants) < num_variants:
        explorer = copy.deepcopy(base_agent)
        explorer.epsilon = 0.2  # Higher exploration rate
        variants.append(explorer)
    
    # Create one variant with focused aggression (bias toward forward moves)
    if len(variants) < num_variants:
        aggressor = copy.deepcopy(base_agent)
        # We'll implement the aggressive behavior in the play_match function
        aggressor.is_aggressive = True  # Just a marker attribute
        variants.append(aggressor)
    
    return variants

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

def play_match(env, agent1, agent2, max_steps=150, board_position=None):
    """Play a single match between two agents."""
    state = env.reset(board_position)
    done = False
    step_count = 0
    
    # Track board positions to detect repetitive play
    position_history = []
    
    while not done and step_count < max_steps:
        current_player = env.current_player
        
        # Get the appropriate agent for the current player
        agent = agent1 if current_player == PLAYER_A else agent2
        legal_actions = env.get_legal_action_indices()
        
        if not legal_actions:
            break  # No legal moves, end game
        
        # Check for repetitive positions (to avoid infinite loops)
        current_binary = board_to_binary(env.board)
        binary_tuple = tuple(map(int, current_binary))  # Convert to tuple for hashability
        
        # If board position repeats too many times, randomly choose a move
        if position_history.count(binary_tuple) >= 2:
            action = random.choice(legal_actions)
        else:
            # For aggressive variant, modify action selection
            if hasattr(agent, 'is_aggressive') and agent.is_aggressive:
                # Score actions based on forward progress
                scored_actions = []
                for action in legal_actions:
                    move = _action_index_to_move(action)
                    if not move:
                        continue
                    
                    start_r, start_c, end_r, end_c = move
                    # Determine forward direction based on player
                    if current_player == PLAYER_A:
                        progress = end_r - start_r  # Down is forward for A
                    else:
                        progress = start_r - end_r  # Up is forward for B
                    
                    # Higher score for forward moves
                    score = progress * 0.5
                    scored_actions.append((action, score))
                
                # Sort by score (higher first)
                scored_actions.sort(key=lambda x: x[1], reverse=True)
                
                # 70% chance to pick from top half of moves
                if scored_actions and random.random() < 0.7:
                    top_half = scored_actions[:max(1, len(scored_actions)//2)]
                    action = random.choice([a for a, _ in top_half])
                else:
                    action = agent.act(state, legal_actions)
            else:
                # Normal action selection
                action = agent.act(state, legal_actions)
        
        # Record this position
        position_history.append(binary_tuple)
        if len(position_history) > 20:  # Limit history length
            position_history.pop(0)
        
        # Take the action
        state, _, done, info = env.step(action)
        step_count += 1
    
    # Determine the winner
    winner = info.get('winner', None)
    if winner == PLAYER_A:
        return 1  # Agent 1 wins
    elif winner == PLAYER_B:
        return -1  # Agent 2 wins
    else:
        return 0  # Draw

def run_tournament(agent, direct_phase2=False, num_variants=NUM_VARIANTS, 
                  noise_scale=NOISE_SCALE, matches_per_pair=TOURNAMENT_MATCHES):
    """Run a tournament between variant agents."""
    variants = create_variant_agents(agent, num_variants, noise_scale)
    env = SwitcharooEnv()
    
    # Track scores for each variant
    scores = [0] * len(variants)
    win_counts = [0] * len(variants)
    draw_counts = [0] * len(variants)
    loss_counts = [0] * len(variants)
    total_matches = 0
    
    # Sample some board positions for diversity
    board_positions = random.sample(initial_position, min(5, len(initial_position)))
    
    # Play matches between all pairs of variants
    print(f"Running tournament with {len(variants)} variants, {matches_per_pair} matches per pair")
    tournament_start = time.time()
    
    match_count = 0
    
    for i in range(len(variants)):
        for j in range(i+1, len(variants)):  # Don't play against self
            for _ in range(matches_per_pair):
                # Randomly select a board position
                board_position = random.choice(board_positions)
                
                # Play match with agents in both positions
                # First with i as Player A, j as Player B
                result1 = play_match(env, variants[i], variants[j], 
                                   board_position=board_position)
                
                # Then with j as Player A, i as Player B
                result2 = play_match(env, variants[j], variants[i],
                                   board_position=board_position)
                
                # Update scores based on results
                if result1 == 1:  # i wins as Player A
                    scores[i] += 1
                    win_counts[i] += 1
                    loss_counts[j] += 1
                elif result1 == -1:  # j wins as Player B
                    scores[j] += 1
                    win_counts[j] += 1
                    loss_counts[i] += 1
                else:  # Draw
                    draw_counts[i] += 1
                    draw_counts[j] += 1
                
                if result2 == 1:  # j wins as Player A
                    scores[j] += 1
                    win_counts[j] += 1
                    loss_counts[i] += 1
                elif result2 == -1:  # i wins as Player B
                    scores[i] += 1
                    win_counts[i] += 1
                    loss_counts[j] += 1
                else:  # Draw
                    draw_counts[i] += 1
                    draw_counts[j] += 1
                
                total_matches += 2  # Counted both matches
                match_count += 2
    
    tournament_time = time.time() - tournament_start
    
    # Find the best variant
    best_idx = np.argmax(scores)
    best_agent = variants[best_idx]
    
    # Calculate win rates
    win_rates = [wins / (total_matches * len(variants) / 2) for wins in win_counts]
    draw_rates = [draws / (total_matches * len(variants) / 2) for draws in draw_counts]
    
    # Print tournament results
    print("\nTournament Results:")
    print(f"Total matches: {total_matches} in {tournament_time:.2f} seconds")
    print(f"{'Variant':<10} {'Score':<10} {'Win Rate':<10} {'Draw Rate':<10}")
    for i in range(len(variants)):
        marker = "* " if i == best_idx else "  "
        print(f"{marker}{i:<8} {scores[i]:<10} {win_rates[i]:.3f}      {draw_rates[i]:.3f}")
    
    print(f"\nBest variant: {best_idx} with score {scores[best_idx]}")
    
    return best_agent, scores[best_idx], total_matches

if __name__ == "__main__":
    # For testing the tournament system
    base_agent = DQNAgent(epsilon=0.1)
    best_agent, score, matches = run_tournament(base_agent)
    print(f"Tournament testing complete. Best agent score: {score}, Total matches: {matches}")