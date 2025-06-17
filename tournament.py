"""Tournament functionality for the Switcharoo game."""
import numpy as np
import random
from random import choice
from game_env import SwitcharooEnv, PLAYER_A, PLAYER_B
from config import (initial_position, NUM_VARIANTS, TOURNAMENT_MATCHES, 
                   MAX_STEPS_PER_EPISODE, NOISE_SCALE, TOURNAMENT_MODEL_FILE)
from dqn_agent import DQNAgent # Import DQNAgent to use its class definition
import os
from tensorflow import keras
import argparse
from time import sleep
import glob
from az_network import AlphaZeroNetwork
from game_env_jit import SwitcharooEnvJitWrapper
from env_const import NUM_ACTIONS

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

def run_pretrained_tournament(models_dir, matches_per_pair=10, max_steps=MAX_STEPS_PER_EPISODE):
    """Run a round-robin tournament between agents loaded from .h5 files in a directory, with progress bar."""
    # Find all .h5 files in the directory
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.h5')]
    if not model_files:
        print(f"No .h5 model files found in {models_dir}")
        return
    print(f"Found {len(model_files)} models: {model_files}")

    # Load all agents
    agents = []
    for fname in model_files:
        agent = DQNAgent()
        agent.load(models_dir + '/' + fname)
        agent.epsilon = 0.01  # Ensure evaluation mode
        agents.append((fname, agent))

    n = len(agents)
    scores = {fname: 0 for fname, _ in agents}
    env = SwitcharooEnv()
    total_matches = 0
    total_pairings = n * (n - 1)
    total_pair_matches = matches_per_pair * 2
    total_expected_matches = (n * (n - 1) // 2) * total_pair_matches
    match_counter = 0

    def print_progress(agent1, agent2, match_num, total, reverse=False):
        bar_len = 30
        filled_len = int(round(bar_len * match_num / float(total)))
        bar = '=' * filled_len + '-' * (bar_len - filled_len)
        pos = "(reversed)" if reverse else ""
        print(f"\r[{bar}] {match_num}/{total}  {agent1} vs {agent2} {pos} ", end='', flush=True)

    for i in range(n):
        for j in range(i+1, n):
            agent_a_name, agent_a = agents[i]
            agent_b_name, agent_b = agents[j]
            # Normal position
            for m in range(1, matches_per_pair + 1):
                print_progress(agent_a_name, agent_b_name, m, matches_per_pair, reverse=False)
                winner = run_match(env, agent_a, agent_b, max_steps)
                if winner == PLAYER_A:
                    scores[agent_a_name] += 1
                elif winner == PLAYER_B:
                    scores[agent_b_name] += 1
                total_matches += 1
            print()  # Newline after progress bar
            # Reversed position
            for m in range(1, matches_per_pair + 1):
                print_progress(agent_b_name, agent_a_name, m, matches_per_pair, reverse=True)
                winner = run_match(env, agent_b, agent_a, max_steps)
                if winner == PLAYER_A:
                    scores[agent_b_name] += 1
                elif winner == PLAYER_B:
                    scores[agent_a_name] += 1
                total_matches += 1
            print()  # Newline after progress bar

    print(f"Tournament completed: {total_matches} matches played")
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    print("Results:")
    for fname, score in sorted_scores:
        print(f"{fname}: {score} points")
    print(f"Winner: {sorted_scores[0][0]} with {sorted_scores[0][1]} points")
    return sorted_scores

def get_az_model_paths(models_dir=".", pattern="switcharoo_az_checkpoint_iter*.weights.h5"):
    search_path = os.path.join(models_dir, pattern)
    return sorted(glob.glob(search_path))

def play_az_game(model_a, model_b, verbose=False):
    env = SwitcharooEnvJitWrapper()
    state = env.reset()
    done = False
    current_player = 0  # 0: model_a, 1: model_b
    while not done:
        if current_player == 0:
            policy, _ = model_a.predict(state)
        else:
            policy, _ = model_b.predict(state)
        legal_moves = env.get_legal_action_indices()
        policy_masked = policy.copy()
        mask = np.zeros_like(policy_masked)
        mask[legal_moves] = 1
        policy_masked *= mask
        if policy_masked.sum() == 0:
            action = np.random.choice(legal_moves)
        else:
            policy_masked /= policy_masked.sum()
            action = np.random.choice(len(policy_masked), p=policy_masked)
        state, reward, done, info = env.step(action)
        current_player = 1 - current_player
        if verbose:
            env.render()
    return info['winner']

def run_az_tournament(model_paths, games_per_pair=2):
    n = len(model_paths)
    results = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            print(f"\nMatch: {model_paths[i]} vs {model_paths[j]}")
            model_a = AlphaZeroNetwork(model_path=model_paths[i])
            model_b = AlphaZeroNetwork(model_path=model_paths[j])
            wins_a = 0
            wins_b = 0
            draws = 0
            for g in range(games_per_pair):
                if g % 2 == 0:
                    winner = play_az_game(model_a, model_b)
                else:
                    winner = play_az_game(model_b, model_a)
                    if winner == 'A':
                        winner = 'B'
                    elif winner == 'B':
                        winner = 'A'
                if winner == 'A':
                    wins_a += 1
                elif winner == 'B':
                    wins_b += 1
                else:
                    draws += 1
            print(f"  {model_paths[i]} wins: {wins_a}, {model_paths[j]} wins: {wins_b}, Draws: {draws}")
            results[i, j] = wins_a
    return results, model_paths

def calculate_elo(results, games_per_pair=2, initial_elo=1500, k=32):
    n = results.shape[0]
    elos = np.full(n, initial_elo, dtype=float)
    total_games = games_per_pair * 2
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            wins_i = results[i, j]
            wins_j = results[j, i]
            draws = total_games - wins_i - wins_j
            for _ in range(wins_i):
                expected_i = 1 / (1 + 10 ** ((elos[j] - elos[i]) / 400))
                elos[i] += k * (1 - expected_i)
                elos[j] += k * (0 - (1 - expected_i))
            for _ in range(wins_j):
                expected_j = 1 / (1 + 10 ** ((elos[i] - elos[j]) / 400))
                elos[j] += k * (1 - expected_j)
                elos[i] += k * (0 - (1 - expected_j))
            for _ in range(draws):
                expected_i = 1 / (1 + 10 ** ((elos[j] - elos[i]) / 400))
                elos[i] += k * (0.5 - expected_i)
                elos[j] += k * (0.5 - (1 - expected_i))
    return elos

def print_az_results(results, model_paths, elos=None):
    n = len(model_paths)
    print("\nAlphaZero Round Robin Results (entries = wins as Player A):")
    header = [p for p in model_paths]
    print("\t" + "\t".join(header))
    for i in range(n):
        row = [model_paths[i]] + [str(results[i, j]) for j in range(n)]
        print("\t".join(row))
    if elos is not None:
        print("\nELO Ratings:")
        sorted_indices = np.argsort(-elos)
        for rank, idx in enumerate(sorted_indices, 1):
            print(f"{rank}. {model_paths[idx]}: {elos[idx]:.1f}")
        print(f"\nTop model: {model_paths[sorted_indices[0]]} (ELO {elos[sorted_indices[0]]:.1f})")

def run_alphazero_tournament(models_dir=".", games_per_pair=2):
    model_paths = get_az_model_paths(models_dir)
    if len(model_paths) < 2:
        print(f"Not enough AlphaZero models found for tournament in {models_dir}.")
        return
    results, model_paths = run_az_tournament(model_paths, games_per_pair=games_per_pair)
    elos = calculate_elo(results, games_per_pair=games_per_pair)
    print_az_results(results, model_paths, elos=elos)

# Add CLI option for AlphaZero tournament
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Switcharoo Tournament Runner")
    parser.add_argument('--pretrained_tournament', action='store_true', help='Run round-robin tournament between all .h5 agents in a directory')
    parser.add_argument('--models_dir', type=str, default='.', help='Directory containing .h5 agent files')
    parser.add_argument('--matches_per_pair', type=int, default=10, help='Matches per agent pair (per side)')
    parser.add_argument('--alphazero_tournament', action='store_true', help='Run round-robin tournament between all AlphaZeroNetwork models in the workspace')
    parser.add_argument('--az_games_per_pair', type=int, default=2, help='Games per AlphaZero model pair (per side)')
    args = parser.parse_args()

    if args.pretrained_tournament:
        run_pretrained_tournament(args.models_dir, args.matches_per_pair)
    elif args.alphazero_tournament:
        run_alphazero_tournament(models_dir=args.models_dir, games_per_pair=args.az_games_per_pair)
    else:
        print("No mode selected. Use --pretrained_tournament or --alphazero_tournament to run a round-robin tournament.")