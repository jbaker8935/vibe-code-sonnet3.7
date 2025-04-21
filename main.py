"""Main entry point for the Switcharoo game training."""
import os
import argparse
import cProfile
import pstats

from dqn_agent import DQNAgent
from phase1_training import phase1_training
from phase2_training import phase2_training, direct_phase2_training
from train_dqn import save_checkpoint
from tensorflow_config import configure_tensorflow
from config import (PHASE2_EPISODES, WANDB_PROJECT, WANDB_ENTITY,
                   BASE_MODEL_FILE, CHECKPOINT_FILE)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Curriculum Training for Switcharoo DQN Agent")
    parser.add_argument("--phase2-only", action="store_true",
                       help="Run only Phase 2 training")
    parser.add_argument("--model-file", type=str,
                       help="Path to an existing model file to load for direct Phase 2 training")
    parser.add_argument("--episodes", type=int, default=PHASE2_EPISODES,
                       help=f"Number of episodes for Phase 2 training (default: {PHASE2_EPISODES})")
    parser.add_argument("--final-model-file", type=str,
                       default="switcharoo_dqn_direct_phase2_final.weights.h5",
                       help="Filename for the final model")
    parser.add_argument("--wandb-project", type=str, default=WANDB_PROJECT,
                       help="Weights & Biases project name")
    parser.add_argument("--wandb-entity", type=str, default=WANDB_ENTITY,
                       help="Weights & Biases entity/username")
    parser.add_argument("--disable-wandb", action="store_true",
                       help="Disable Weights & Biases logging")
    parser.add_argument("--profile", action="store_true",
                       help="Enable profiling using cProfile")
    parser.add_argument("--profile-output", type=str, default="program.prof",
                       help="Output file for profiling data (default: program.prof)")
    parser.add_argument("--resume", action="store_true",
                       help="Resume training from latest checkpoint if available")
    args = parser.parse_args()

    # Configure TensorFlow
    configure_tensorflow()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging verbosity
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

    # Set up profiler if enabled
    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()

    try:
        # Direct Phase 2 Training Mode
        if args.phase2_only and args.model_file:
            print(f"Starting direct Phase 2 training with model: {args.model_file}")
            direct_phase2_training(
                model_file=args.model_file,
                episodes=args.episodes,
                final_model_file=args.final_model_file,
                enable_wandb=not args.disable_wandb
            )
        else:
            # Standard Curriculum Training (Phase 1 + Phase 2)
            agent = DQNAgent(
                learning_rate=0.00025,
                epsilon=0.01,
                epsilon_decay=.9995,
                epsilon_min=0.01,
                replay_buffer_size=500000,
                batch_size=64,
                target_update_freq=100
            )

            # Check for existing checkpoints
            start_episode = 1
            if args.resume:
                try:
                    checkpoint_dir = "./"
                    checkpoints = [f for f in os.listdir(checkpoint_dir)
                                 if f.startswith("switcharoo_dqn_checkpoint_e")]

                    if checkpoints:
                        latest_checkpoint = max(
                            checkpoints,
                            key=lambda x: int(x.split('_e')[-1].split('.weights')[0])
                        )
                        start_episode = int(latest_checkpoint.split('_e')[-1].split('.weights')[0]) + 1
                        agent.load(os.path.join(checkpoint_dir, latest_checkpoint))
                        print(f"Resuming from checkpoint {latest_checkpoint} at episode {start_episode}")
                    else:
                        print("No checkpoints found. Starting from scratch.")
                except Exception as e:
                    print(f"Error loading checkpoint: {e}")
                    start_episode = 1
            else:
                print("Starting new training session from scratch.")

            # Check if Phase 1 is completed
            phase1_completed = os.path.exists(BASE_MODEL_FILE)

            try:
                # Phase 1: Train against random opponent
                if not phase1_completed:
                    agent = phase1_training(agent, start_episode,
                                         enable_wandb=not args.disable_wandb)
                    start_episode = 1  # Reset episode counter for Phase 2
                else:
                    print(f"Phase 1 model found: {BASE_MODEL_FILE}. Loading...")
                    agent.load(BASE_MODEL_FILE)

                # Phase 2: Tournament self-play
                agent.epsilon = 0.01
                agent.epsilon_decay = 1.0
                agent.epsilon_min = 0.01
                agent = phase2_training(agent, start_episode,
                                     enable_wandb=not args.disable_wandb)

                print("Curriculum training completed successfully!")

            except KeyboardInterrupt:
                print("\nTraining interrupted by user")
                save_checkpoint(agent, start_episode, emergency=True)
            except Exception as e:
                print(f"\nTraining crashed with error: {e}")
                save_checkpoint(agent, start_episode, emergency=True)
                raise

    finally:
        # Save and print profiling results if enabled
        if args.profile:
            profiler.disable()
            profiler.dump_stats(args.profile_output)
            print(f"\nProfiling data saved to {args.profile_output}")

            stats = pstats.Stats(args.profile_output)
            print("\nTop 20 time-consuming operations:")
            stats.strip_dirs().sort_stats('cumulative').print_stats(20)

            print(f"\nTo analyze the profile data in more detail, you can run:")
            print(f"python -m pstats {args.profile_output}")
            print(f"Or visualize it with: snakeviz {args.profile_output}")

if __name__ == "__main__":
    main()