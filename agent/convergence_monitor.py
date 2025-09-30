import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logging
from typing import List, Dict, Tuple, Optional
import wandb
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class ConvergenceMonitor:
    """Monitor and analyze convergence of RL training and hyperparameter optimization."""

    def __init__(self,
                 patience: int = 10,
                 min_improvement: float = 0.01,
                 smoothing_window: int = 5):
        """
        Initialize convergence monitor.

        Args:
            patience: Number of evaluations without improvement before declaring convergence
            min_improvement: Minimum relative improvement required to reset patience counter
            smoothing_window: Window size for smoothing reward curves
        """
        self.patience = patience
        self.min_improvement = min_improvement
        self.smoothing_window = smoothing_window

        # Tracking variables
        self.rewards_history = []
        self.best_reward = float('-inf')
        self.no_improvement_count = 0
        self.converged = False
        self.convergence_step = None

        # Statistics tracking
        self.stats = {
            'mean_rewards': [],
            'std_rewards': [],
            'max_rewards': [],
            'smoothed_rewards': []
        }

    def update(self, reward: float, step: int) -> Dict[str, any]:
        """
        Update convergence monitoring with new reward.

        Args:
            reward: Latest reward value
            step: Current training step

        Returns:
            Dictionary with convergence statistics
        """
        self.rewards_history.append((step, reward))

        # Check for improvement
        if reward > self.best_reward * (1 + self.min_improvement):
            self.best_reward = reward
            self.no_improvement_count = 0
            logger.info(f"New best reward: {reward:.4f} at step {step}")
        else:
            self.no_improvement_count += 1

        # Check for convergence
        if self.no_improvement_count >= self.patience and not self.converged:
            self.converged = True
            self.convergence_step = step
            logger.info(f"Convergence detected at step {step}")

        # Update statistics
        recent_rewards = [r for _, r in self.rewards_history[-self.smoothing_window:]]
        self.stats['mean_rewards'].append(np.mean(recent_rewards))
        self.stats['std_rewards'].append(np.std(recent_rewards))
        self.stats['max_rewards'].append(max(recent_rewards))

        # Smoothed rewards using moving average
        if len(self.rewards_history) >= self.smoothing_window:
            smoothed = np.mean([r for _, r in self.rewards_history[-self.smoothing_window:]])
            self.stats['smoothed_rewards'].append(smoothed)
        else:
            self.stats['smoothed_rewards'].append(reward)

        return {
            'converged': self.converged,
            'best_reward': self.best_reward,
            'no_improvement_count': self.no_improvement_count,
            'convergence_step': self.convergence_step,
            'current_reward': reward,
            'smoothed_reward': self.stats['smoothed_rewards'][-1]
        }

    def get_convergence_metrics(self) -> Dict[str, any]:
        """Get comprehensive convergence metrics."""
        if not self.rewards_history:
            return {}

        rewards = [r for _, r in self.rewards_history]
        steps = [s for s, _ in self.rewards_history]

        # Calculate convergence rate
        if len(rewards) > 1:
            total_improvement = self.best_reward - rewards[0]
            convergence_rate = total_improvement / len(rewards) if total_improvement > 0 else 0
        else:
            convergence_rate = 0

        # Calculate stability metrics
        recent_rewards = rewards[-min(20, len(rewards)):]  # Last 20 evaluations
        stability = 1 / (1 + np.std(recent_rewards)) if len(recent_rewards) > 1 else 0

        return {
            'converged': self.converged,
            'convergence_step': self.convergence_step,
            'best_reward': self.best_reward,
            'final_reward': rewards[-1],
            'total_evaluations': len(rewards),
            'convergence_rate': convergence_rate,
            'stability_score': stability,
            'improvement_ratio': (self.best_reward - rewards[0]) / abs(rewards[0]) if rewards[0] != 0 else 0,
            'reward_variance': np.var(rewards),
            'reward_trend': np.polyfit(range(len(rewards)), rewards, 1)[0] if len(rewards) > 1 else 0
        }

    def plot_convergence(self, save_path: Optional[str] = None) -> str:
        """
        Plot convergence analysis.

        Args:
            save_path: Path to save the plot

        Returns:
            Path to saved plot
        """
        if not self.rewards_history:
            logger.warning("No data to plot")
            return None

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        steps = [s for s, _ in self.rewards_history]
        rewards = [r for _, r in self.rewards_history]

        # 1. Raw rewards over time
        ax1.plot(steps, rewards, 'b-', alpha=0.6, label='Raw Rewards')
        ax1.plot(steps, self.stats['smoothed_rewards'], 'r-', linewidth=2, label='Smoothed Rewards')
        ax1.axhline(y=self.best_reward, color='g', linestyle='--', label=f'Best: {self.best_reward:.3f}')
        if self.convergence_step:
            ax1.axvline(x=self.convergence_step, color='orange', linestyle='--', label=f'Convergence: {self.convergence_step}')
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Reward')
        ax1.set_title('Reward Convergence')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Improvement tracking
        improvements = [(rewards[i] - rewards[0]) / abs(rewards[0]) * 100 if rewards[0] != 0 else 0
                       for i in range(len(rewards))]
        ax2.plot(steps, improvements, 'g-', linewidth=2)
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Improvement (%)')
        ax2.set_title('Cumulative Improvement')
        ax2.grid(True, alpha=0.3)

        # 3. Reward distribution
        ax3.hist(rewards, bins=20, alpha=0.7, edgecolor='black')
        ax3.axvline(x=np.mean(rewards), color='r', linestyle='--', label=f'Mean: {np.mean(rewards):.3f}')
        ax3.axvline(x=self.best_reward, color='g', linestyle='--', label=f'Best: {self.best_reward:.3f}')
        ax3.set_xlabel('Reward')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Reward Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Convergence indicators
        no_improvement_counts = []
        current_count = 0
        best_so_far = float('-inf')
        for reward in rewards:
            if reward > best_so_far * (1 + self.min_improvement):
                best_so_far = reward
                current_count = 0
            else:
                current_count += 1
            no_improvement_counts.append(current_count)

        ax4.plot(steps, no_improvement_counts, 'purple', linewidth=2)
        ax4.axhline(y=self.patience, color='r', linestyle='--', label=f'Patience: {self.patience}')
        ax4.set_xlabel('Training Step')
        ax4.set_ylabel('Steps Without Improvement')
        ax4.set_title('Convergence Indicator')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path is None:
            save_path = f"convergence_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Convergence plot saved to {save_path}")
        return save_path

    def save_metrics(self, filepath: str):
        """Save convergence metrics to file."""
        metrics = self.get_convergence_metrics()
        metrics['rewards_history'] = self.rewards_history
        metrics['stats'] = self.stats

        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Convergence metrics saved to {filepath}")

    @classmethod
    def load_metrics(cls, filepath: str) -> 'ConvergenceMonitor':
        """Load convergence metrics from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        monitor = cls()
        monitor.rewards_history = data['rewards_history']
        monitor.stats = data['stats']
        monitor.converged = data['converged']
        monitor.convergence_step = data['convergence_step']
        monitor.best_reward = data['best_reward']

        return monitor

def analyze_optimization_convergence(study_results: Dict, save_plots: bool = True) -> Dict:
    """
    Analyze convergence of hyperparameter optimization.

    Args:
        study_results: Results from Optuna study
        save_plots: Whether to save convergence plots

    Returns:
        Convergence analysis results
    """
    if 'study' not in study_results:
        logger.error("No study found in results")
        return {}

    study = study_results['study']
    trials = study.trials

    # Extract trial data
    trial_numbers = [t.number for t in trials if t.state.name == 'COMPLETE']
    trial_values = [t.value for t in trials if t.state.name == 'COMPLETE' and t.value is not None]

    if not trial_values:
        logger.warning("No completed trials with values found")
        return {}

    # Create convergence monitor for optimization
    monitor = ConvergenceMonitor(patience=5, min_improvement=0.005)

    # Track optimization convergence
    for i, (trial_num, value) in enumerate(zip(trial_numbers, trial_values)):
        monitor.update(value, trial_num)

    # Generate plots
    if save_plots:
        plot_path = monitor.plot_convergence()

        # Additional optimization-specific plot
        plt.figure(figsize=(12, 8))

        # Best value over trials
        best_values = []
        current_best = float('-inf')
        for value in trial_values:
            if value > current_best:
                current_best = value
            best_values.append(current_best)

        plt.subplot(2, 2, 1)
        plt.plot(trial_numbers, trial_values, 'bo-', alpha=0.6, label='Trial Values')
        plt.plot(trial_numbers, best_values, 'r-', linewidth=2, label='Best So Far')
        plt.xlabel('Trial Number')
        plt.ylabel('Objective Value')
        plt.title('Optimization Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Parameter evolution (if available)
        if hasattr(study, 'trials') and len(study.trials) > 0:
            # Get parameter names
            param_names = list(study.trials[0].params.keys())[:4]  # Show first 4 params

            for i, param_name in enumerate(param_names):
                plt.subplot(2, 2, i + 2)
                param_values = [t.params.get(param_name, 0) for t in trials if t.state.name == 'COMPLETE']
                plt.plot(trial_numbers, param_values, 'go-', alpha=0.6)
                plt.xlabel('Trial Number')
                plt.ylabel(param_name)
                plt.title(f'{param_name} Evolution')
                plt.grid(True, alpha=0.3)

        plt.tight_layout()
        opt_plot_path = f"optimization_convergence_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(opt_plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Optimization plot saved to {opt_plot_path}")

    # Get convergence metrics
    convergence_metrics = monitor.get_convergence_metrics()

    # Add optimization-specific metrics
    convergence_metrics.update({
        'optimization_efficiency': len([v for v in trial_values if v > np.mean(trial_values)]) / len(trial_values),
        'parameter_stability': study_results.get('convergence_info', {}).get('converged', False),
        'trials_completed': len(trial_values),
        'best_trial_number': trial_numbers[np.argmax(trial_values)]
    })

    return convergence_metrics

if __name__ == "__main__":
    # Example usage
    monitor = ConvergenceMonitor(patience=10, min_improvement=0.01)

    # Simulate training data
    np.random.seed(42)
    rewards = []
    base_reward = 100

    for step in range(100):
        # Simulate improving rewards with noise
        improvement = step * 0.5 + np.random.normal(0, 5)
        reward = base_reward + improvement
        rewards.append(reward)

        stats = monitor.update(reward, step)

        if step % 10 == 0:
            print(f"Step {step}: Reward {reward:.2f}, Converged: {stats['converged']}")

    # Generate analysis
    plot_path = monitor.plot_convergence()
    metrics = monitor.get_convergence_metrics()

    print("\nFinal Convergence Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")