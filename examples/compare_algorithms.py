"""Compare DQN and PPO performance for Kubernetes autoscaling."""

import os
import sys
import wandb
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import logging
import argparse
from datetime import datetime
import json
import time
import pandas as pd
from agent.metrics_callback import AutoscalingMetricsCallback
from agent.ppo import K8sSimulationEnv

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_and_evaluate(algorithm, env, timesteps, learning_rate, batch_size, is_simulated=False):
    """Train and evaluate an algorithm."""
    # Initialize WandB
    wandb.init(
        project="k8s-autoscaling-comparison",
        config={
            "algorithm": algorithm,
            "timesteps": timesteps,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "simulation": is_simulated
        }
    )
    
    # Initialize model
    if algorithm == "dqn":
        model = DQN(
            policy="MlpPolicy",
            env=env,
            learning_rate=learning_rate,
            buffer_size=100000,
            learning_starts=5000,
            batch_size=batch_size,
            tau=0.01,
            gamma=0.95,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=5000,
            exploration_fraction=0.3,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.1,
            max_grad_norm=10,
            verbose=1,
            tensorboard_log=f"./{algorithm}_tensorboard/"
        )
    else:  # ppo
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=learning_rate,
            n_steps=2048,
            batch_size=batch_size,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            clip_range_vf=None,
            normalize_advantage=True,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            use_sde=False,
            sde_sample_freq=-1,
            target_kl=None,
            tensorboard_log=f"./{algorithm}_tensorboard/",
            verbose=1
        )
    
    # Initialize metrics callback
    metrics_callback = AutoscalingMetricsCallback(
        prometheus_url="http://localhost:9090",
        algorithm=algorithm,
        is_simulated=is_simulated,
        verbose=1
    )
    
    # Train the model
    model.learn(
        total_timesteps=timesteps,
        callback=metrics_callback,
        progress_bar=True
    )
    
    # Evaluate the model
    mean_reward, std_reward = evaluate_policy(
        model, 
        env, 
        n_eval_episodes=10,
        deterministic=True
    )
    
    # Save the model
    model.save(f"{algorithm}_k8s_autoscaler")
    wandb.save(f"{algorithm}_k8s_autoscaler.zip")
    
    # Close WandB
    wandb.finish()
    
    return mean_reward, std_reward

def plot_comparison(dqn_rewards, ppo_rewards, save_path="algorithm_comparison.png"):
    """Plot comparison of algorithm performance."""
    plt.figure(figsize=(12, 6))
    
    # Plot rewards
    plt.subplot(1, 2, 1)
    plt.plot(dqn_rewards, label="DQN", color="blue")
    plt.plot(ppo_rewards, label="PPO", color="red")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Rewards")
    plt.legend()
    
    # Plot moving averages
    plt.subplot(1, 2, 2)
    window = 10
    dqn_ma = pd.Series(dqn_rewards).rolling(window=window).mean()
    ppo_ma = pd.Series(ppo_rewards).rolling(window=window).mean()
    plt.plot(dqn_ma, label="DQN (MA)", color="blue", linestyle="--")
    plt.plot(ppo_ma, label="PPO (MA)", color="red", linestyle="--")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"Moving Average Rewards (Window={window})")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    """Main comparison function."""
    parser = argparse.ArgumentParser(description="Compare DQN and PPO for Kubernetes autoscaling")
    parser.add_argument("--timesteps", type=int, default=100000, help="Total timesteps for training")
    parser.add_argument("--learning-rate", type=float, default=0.0003, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--simulate", action="store_true", help="Use simulated environment")
    args = parser.parse_args()
    
    # Create environment
    env = K8sSimulationEnv()
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    
    # Train and evaluate DQN
    logger.info("Training DQN...")
    dqn_mean, dqn_std = train_and_evaluate(
        "dqn",
        env,
        args.timesteps,
        args.learning_rate,
        args.batch_size,
        is_simulated=args.simulate
    )
    logger.info(f"DQN - Mean reward: {dqn_mean:.2f} +/- {dqn_std:.2f}")
    
    # Train and evaluate PPO
    logger.info("Training PPO...")
    ppo_mean, ppo_std = train_and_evaluate(
        "ppo",
        env,
        args.timesteps,
        args.learning_rate,
        args.batch_size,
        is_simulated=args.simulate
    )
    logger.info(f"PPO - Mean reward: {ppo_mean:.2f} +/- {ppo_std:.2f}")
    
    # Load training rewards from WandB
    dqn_rewards = wandb.run.history()["train/episode_reward"].tolist()
    ppo_rewards = wandb.run.history()["train/episode_reward"].tolist()
    
    # Plot comparison
    plot_comparison(dqn_rewards, ppo_rewards)
    
    # Print summary
    logger.info("\nAlgorithm Comparison Summary:")
    logger.info(f"DQN - Mean reward: {dqn_mean:.2f} +/- {dqn_std:.2f}")
    logger.info(f"PPO - Mean reward: {ppo_mean:.2f} +/- {ppo_std:.2f}")
    logger.info(f"Best algorithm: {'DQN' if dqn_mean > ppo_mean else 'PPO'}")
    logger.info(f"Comparison plot saved to: algorithm_comparison.png")

if __name__ == "__main__":
    main() 