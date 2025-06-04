import re
import os
import sys
import json
try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    print("Some analysis libraries not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "matplotlib", "seaborn"])
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

# Regex patterns to capture key training metrics
iteration_start_re = re.compile(r"===== Iteration (\d+)/(\d+) =====")
curriculum_phase_re = re.compile(r"🎯 Curriculum Phase: (\w+)")
overfit_test_header_re = re.compile(r"\[Overfit Test\] Training on a single minibatch...")
overfit_step_re = re.compile(r"Step (\d+): loss=([\d.]+), policy_loss=([\d.]+), value_loss=([\d.]+)")
training_finished_re = re.compile(r"Network training finished. Avg Total Loss: ([\d.]+) \(P: ([\d.]+), V: ([\d.]+), P_Acc: ([\d.]+), V_MAE: ([\d.]+)\), LR: ([\d.e+-]+)")
evaluation_results_re = re.compile(r"Evaluation Results: Candidate Wins: (\d+), Best Wins: (\d+), Draws: (\d+)")
evaluation_win_rate_re = re.compile(r"Candidate Win Rate \(vs Best, excluding draws\): ([\d.]+)%")
model_updated_re = re.compile(r"Candidate model is better.*Updating best model")
model_not_updated_re = re.compile(r"Candidate model is not significantly better|keeping previous best model", re.IGNORECASE)
self_play_finished_re = re.compile(r"Self-play finished. Added (\d+) experiences to replay buffer. Buffer size: (\d+)")
evaluation_skipped_re = re.compile(r"--- Skipping Model Evaluation \(First Iteration\) ---")
iteration_completed_re = re.compile(r"Iteration (\d+) completed in ([\d.]+) seconds")

def initialize_iteration_data(iteration_num):
    """Initializes a dictionary to store data for an iteration."""
    return {
        "iteration": iteration_num,
        "curriculum_phase": None,
        'overfit_initial_loss': None, 'overfit_initial_policy_loss': None, 'overfit_initial_value_loss': None,
        'overfit_final_loss': None, 'overfit_final_policy_loss': None, 'overfit_final_value_loss': None,
        'train_avg_total_loss': None, 'train_avg_policy_loss': None, 'train_avg_value_loss': None,
        'train_avg_policy_acc': None, 'train_avg_value_mae': None, 'train_lr': None,
        'eval_candidate_wins': None, 'eval_best_wins': None, 'eval_draws': None,
        'eval_win_rate': None, 'eval_model_updated': None, 'eval_skipped': False,
        'self_play_new_experiences': None, 'self_play_buffer_size': None,
        'iteration_duration': None
    }

def analyze_log(log_file_path):
    """Parses the log file and extracts training metrics for each iteration."""
    iterations_data = []
    current_iteration_data = None
    capturing_overfit_steps = False
    overfit_step_0_captured = False

    with open(log_file_path, 'r') as f:
        for line in f:
            line = line.strip()

            # Start of a new iteration
            match = iteration_start_re.search(line)
            if match:
                if current_iteration_data:
                    iterations_data.append(current_iteration_data)
                iter_num = int(match.group(1))
                current_iteration_data = initialize_iteration_data(iter_num)
                capturing_overfit_steps = False 
                overfit_step_0_captured = False
                continue

            if not current_iteration_data:
                continue

            # Curriculum phase
            match = curriculum_phase_re.search(line)
            if match:
                current_iteration_data['curriculum_phase'] = match.group(1)
                continue

            # Start of Overfit Test section
            if overfit_test_header_re.search(line):
                capturing_overfit_steps = True
                overfit_step_0_captured = False 
                continue

            # Overfit Test Steps
            if capturing_overfit_steps:
                match = overfit_step_re.search(line)
                if match:
                    step = int(match.group(1))
                    loss = float(match.group(2))
                    p_loss = float(match.group(3))
                    v_loss = float(match.group(4))
                    
                    if not overfit_step_0_captured and step == 0:
                        current_iteration_data['overfit_initial_loss'] = loss
                        current_iteration_data['overfit_initial_policy_loss'] = p_loss
                        current_iteration_data['overfit_initial_value_loss'] = v_loss
                        overfit_step_0_captured = True
                    
                    current_iteration_data['overfit_final_loss'] = loss
                    current_iteration_data['overfit_final_policy_loss'] = p_loss
                    current_iteration_data['overfit_final_value_loss'] = v_loss
                elif "[Overfit Test] After training:" in line:
                    capturing_overfit_steps = False

            # Training Finished
            match = training_finished_re.search(line)
            if match:
                current_iteration_data['train_avg_total_loss'] = float(match.group(1))
                current_iteration_data['train_avg_policy_loss'] = float(match.group(2))
                current_iteration_data['train_avg_value_loss'] = float(match.group(3))
                current_iteration_data['train_avg_policy_acc'] = float(match.group(4))
                current_iteration_data['train_avg_value_mae'] = float(match.group(5))
                current_iteration_data['train_lr'] = match.group(6)
                capturing_overfit_steps = False 
                continue

            # Evaluation Skipped
            match = evaluation_skipped_re.search(line)
            if match:
                current_iteration_data['eval_skipped'] = True
                continue

            # Evaluation Results
            match = evaluation_results_re.search(line)
            if match:
                current_iteration_data['eval_candidate_wins'] = int(match.group(1))
                current_iteration_data['eval_best_wins'] = int(match.group(2))
                current_iteration_data['eval_draws'] = int(match.group(3))
                continue

            # Evaluation Win Rate
            match = evaluation_win_rate_re.search(line)
            if match:
                current_iteration_data['eval_win_rate'] = float(match.group(1))
                continue

            # Model Updated
            if model_updated_re.search(line):
                current_iteration_data['eval_model_updated'] = True
                continue
            
            if model_not_updated_re.search(line):
                current_iteration_data['eval_model_updated'] = False
                continue
            
            # Self-play stats
            match = self_play_finished_re.search(line)
            if match:
                current_iteration_data['self_play_new_experiences'] = int(match.group(1))
                current_iteration_data['self_play_buffer_size'] = int(match.group(2))
                continue

            # Iteration duration
            match = iteration_completed_re.search(line)
            if match:
                current_iteration_data['iteration_duration'] = float(match.group(2))
                continue

        # Add the final iteration
        if current_iteration_data:
            iterations_data.append(current_iteration_data)

    return iterations_data

def load_wandb_summary(wandb_summary_path):
    """Load final metrics from wandb summary."""
    try:
        with open(wandb_summary_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading wandb summary: {e}")
        return {}

def analyze_curriculum_phases(df):
    """Analyze performance across curriculum phases."""
    print("\n" + "="*80)
    print("📚 CURRICULUM PHASE ANALYSIS")
    print("="*80)
    
    if 'curriculum_phase' in df.columns:
        phase_groups = df.groupby('curriculum_phase')
        for phase, group in phase_groups:
            print(f"\n🎯 {phase.upper()}:")
            print(f"   Iterations: {group['iteration'].min()} - {group['iteration'].max()}")
            print(f"   Average Policy Accuracy: {group['train_avg_policy_acc'].mean():.3f}")
            print(f"   Average Value MAE: {group['train_avg_value_mae'].mean():.3f}")
            print(f"   Model Updates: {group['eval_model_updated'].sum()}/{len(group)}")
            if not group['eval_win_rate'].isna().all():
                print(f"   Average Win Rate: {group['eval_win_rate'].mean():.1f}%")

def analyze_training_convergence(df):
    """Analyze training convergence patterns."""
    print("\n" + "="*80)
    print("📈 TRAINING CONVERGENCE ANALYSIS")
    print("="*80)
    
    # Loss trends
    print(f"\n📉 Loss Trends:")
    print(f"   Initial Total Loss: {df['train_avg_total_loss'].iloc[0]:.3f}")
    print(f"   Final Total Loss: {df['train_avg_total_loss'].iloc[-1]:.3f}")
    print(f"   Loss Reduction: {((df['train_avg_total_loss'].iloc[0] - df['train_avg_total_loss'].iloc[-1]) / df['train_avg_total_loss'].iloc[0] * 100):.1f}%")
    
    # Policy accuracy trends
    print(f"\n🎯 Policy Learning:")
    print(f"   Initial Policy Accuracy: {df['train_avg_policy_acc'].iloc[0]:.3f}")
    print(f"   Final Policy Accuracy: {df['train_avg_policy_acc'].iloc[-1]:.3f}")
    print(f"   Improvement: {((df['train_avg_policy_acc'].iloc[-1] - df['train_avg_policy_acc'].iloc[0]) / df['train_avg_policy_acc'].iloc[0] * 100):.1f}%")
    
    # Value prediction accuracy
    print(f"\n📊 Value Prediction:")
    print(f"   Initial Value MAE: {df['train_avg_value_mae'].iloc[0]:.3f}")
    print(f"   Final Value MAE: {df['train_avg_value_mae'].iloc[-1]:.3f}")
    print(f"   Improvement: {((df['train_avg_value_mae'].iloc[0] - df['train_avg_value_mae'].iloc[-1]) / df['train_avg_value_mae'].iloc[0] * 100):.1f}%")

def analyze_model_evolution(df):
    """Analyze how the model evolved through iterations."""
    print("\n" + "="*80)
    print("🔄 MODEL EVOLUTION ANALYSIS")
    print("="*80)
    
    updates = df[df['eval_model_updated'] == True]
    print(f"\n📈 Model Updates: {len(updates)}/{len(df)} iterations ({len(updates)/len(df)*100:.1f}%)")
    
    if len(updates) > 0:
        print(f"   Update Iterations: {list(updates['iteration'])}")
        print(f"   Average Win Rate for Updates: {updates['eval_win_rate'].mean():.1f}%")
    
    # Recent performance
    recent_iterations = df.tail(20)
    recent_win_rate = recent_iterations['eval_win_rate'].mean()
    print(f"\n🏆 Recent Performance (last 20 iterations):")
    print(f"   Average Win Rate: {recent_win_rate:.1f}%")
    print(f"   Model Updates: {recent_iterations['eval_model_updated'].sum()}")

def generate_training_plots(df):
    """Generate visualization plots for training analysis."""
    try:
        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'AlphaZero Training Analysis -  {len(df)} Iterations', fontsize=16, fontweight='bold')
        
        # Loss curves
        axes[0,0].plot(df['iteration'], df['train_avg_total_loss'], 'b-', alpha=0.7, label='Total Loss')
        axes[0,0].plot(df['iteration'], df['train_avg_policy_loss'], 'r-', alpha=0.7, label='Policy Loss')
        axes[0,0].plot(df['iteration'], df['train_avg_value_loss'], 'g-', alpha=0.7, label='Value Loss')
        axes[0,0].set_title('Training Loss Curves')
        axes[0,0].set_xlabel('Iteration')
        axes[0,0].set_ylabel('Loss')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Policy accuracy
        axes[0,1].plot(df['iteration'], df['train_avg_policy_acc'], 'purple', linewidth=2)
        axes[0,1].set_title('Policy Accuracy Over Time')
        axes[0,1].set_xlabel('Iteration')
        axes[0,1].set_ylabel('Policy Accuracy')
        axes[0,1].grid(True, alpha=0.3)
        
        # Value MAE
        axes[0,2].plot(df['iteration'], df['train_avg_value_mae'], 'orange', linewidth=2)
        axes[0,2].set_title('Value Prediction MAE')
        axes[0,2].set_xlabel('Iteration')
        axes[0,2].set_ylabel('Mean Absolute Error')
        axes[0,2].grid(True, alpha=0.3)
        
        # Win rates
        win_rate_data = df[df['eval_win_rate'].notna()]
        axes[1,0].plot(win_rate_data['iteration'], win_rate_data['eval_win_rate'], 'red', linewidth=2, marker='o', markersize=3)
        axes[1,0].axhline(y=60, color='gray', linestyle='--', alpha=0.7, label='60% Threshold')
        axes[1,0].set_title('Evaluation Win Rate vs Best Model')
        axes[1,0].set_xlabel('Iteration')
        axes[1,0].set_ylabel('Win Rate (%)')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Model updates
        updates = df[df['eval_model_updated'] == True]
        axes[1,1].scatter(updates['iteration'], [1]*len(updates), color='green', s=50, alpha=0.8)
        axes[1,1].set_title('Model Update Events')
        axes[1,1].set_xlabel('Iteration')
        axes[1,1].set_ylabel('Model Updated')
        axes[1,1].set_ylim(0.5, 1.5)
        axes[1,1].grid(True, alpha=0.3)
        
        # Buffer growth
        axes[1,2].plot(df['iteration'], df['self_play_buffer_size'], 'brown', linewidth=2)
        axes[1,2].set_title('Replay Buffer Size Growth')
        axes[1,2].set_xlabel('Iteration')
        axes[1,2].set_ylabel('Buffer Size')
        axes[1,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('alphazero_training_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\n📊 Training plots saved as 'alphazero_training_analysis.png'")
        
    except Exception as e:
        print(f"Error generating plots: {e}")

def display_results(data, wandb_summary=None):
    """Display comprehensive analysis results."""
    if not data:
        print("No iteration data parsed from the log.")
        return

    df = pd.DataFrame(data)
    
    print("\n" + "="*80)
    print("🚀 ALPHAZERO TRAINING RESULTS ANALYSIS")
    print("="*80)
    print(f"📊 Training Summary: {len(df)} iterations completed")
    print(f"🎯 Final Iteration: {df['iteration'].max()}")
    
    if wandb_summary:
        print(f"\n📈 Final Metrics from W&B:")
        policy_acc = wandb_summary.get('train/policy_accuracy', 'N/A')
        value_mae = wandb_summary.get('train/value_mae', 'N/A')
        total_loss = wandb_summary.get('train/total_loss', 'N/A')
        current_phase = wandb_summary.get('curriculum/current_phase', 'N/A')
        runtime_hours = wandb_summary.get('_runtime', 0) / 3600
        
        if isinstance(policy_acc, (int, float)):
            print(f"   - Policy Accuracy: {policy_acc:.4f}")
        else:
            print(f"   - Policy Accuracy: {policy_acc}")
            
        if isinstance(value_mae, (int, float)):
            print(f"   - Value MAE: {value_mae:.4f}")
        else:
            print(f"   - Value MAE: {value_mae}")
            
        if isinstance(total_loss, (int, float)):
            print(f"   - Total Loss: {total_loss:.4f}")
        else:
            print(f"   - Total Loss: {total_loss}")
            
        print(f"   - Current Phase: {current_phase}")
        print(f"   - Total Runtime: {runtime_hours:.2f} hours")
    
    # Detailed analysis
    analyze_curriculum_phases(df)
    analyze_training_convergence(df)
    analyze_model_evolution(df)
    
    # Generate plots
    generate_training_plots(df)
    
    print(f"\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)

def main():
    """Main analysis function."""
    log_path = "/home/john/code/vibe-code-sonnet3.7/wandb/latest-run/files/output.log"
    wandb_path = "/home/john/code/vibe-code-sonnet3.7/wandb/latest-run/files/wandb-summary.json"
    
    if not os.path.exists(log_path):
        print(f"Log file not found: {log_path}")
        return
    
    print("🔍 Analyzing AlphaZero training logs...")
    data = analyze_log(log_path)
    
    wandb_summary = None
    if os.path.exists(wandb_path):
        wandb_summary = load_wandb_summary(wandb_path)
    
    display_results(data, wandb_summary)

if __name__ == "__main__":
    main()
