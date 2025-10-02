import argparse
import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList, BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import get_linear_fn
from wandb.integration.sb3 import WandbCallback
import torch
from stable_baselines3.dqn import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from kubernetes import client, config
from kubernetes.client.rest import ApiException
import subprocess
import signal
import atexit
import threading
from queue import Queue
import requests
from prometheus_api_client import PrometheusConnect
import pandas as pd
from agent.metrics_callback import AutoscalingMetricsCallback

from agent.environment_simulated import MicroK8sEnvSimulated
from agent.environment import MicroK8sEnv
from agent.traffic_simulation import TrafficSimulator

import wandb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("dqn.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

ENV_NAME = "microk8s_env"
EXP_NAME = "dqn_autoscaling_rewardshape"
LR = 3e-4
RUNID = f"dqn_{ENV_NAME[:5]}_lr{LR:.0e}_{datetime.now().strftime('%m%d%H%M')}"

class VisualizationCallback(BaseCallback):
    """Custom callback for plotting metrics."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.resource_history = []
        self.action_history = []
        
    def _on_step(self) -> bool:
        info = self.locals['infos'][0]
        custom_metrics = info.get('custom_metrics', {})
        self.resource_history.append(custom_metrics.get('cpu_utilization', 0))
        self.action_history.append(self.locals['actions'][0])
        
        if len(self.resource_history) % 100 == 0:
            try:
                wandb.log({
                    "resource_trends": wandb.plot.line_series(
                        xs=range(len(self.resource_history)),
                        ys=[self.resource_history],
                        keys=["CPU Utilization"],
                        title="Resource Trends"
                    ),
                    "train/global_step": self.model.num_timesteps
                })
            except wandb.Error as e:
                logger.warning("Failed to log resource trends: %s", e)
                
        return True

class DQNAgent:
    """DQN agent with comprehensive autoscaling metrics tracking."""

    def __init__(self, env, environment: MicroK8sEnv, is_simulated: bool = False, **kwargs):
        """
        Initialize DQN agent with metrics tracking.
        
        Args:
            env: RL environment
            is_simulated: Whether using simulated environment
            kwargs: Additional configs
                - model_dir: Directory to save models
                - learning_rate: DQN learning rate
                - buffer_size: Replay buffer size
                - batch_size: Training batch size
                - gamma: Discount factor
        """
        try:

            self.env = make_vec_env(lambda:environment, n_envs=1)
            self.is_simulated = is_simulated
            self.eval_env = make_vec_env(lambda: environment, n_envs=1)
            self.model = None  # Initialize model attribute

            self.model_dir = kwargs.get('model_dir', './models/dqn')
            os.makedirs(self.model_dir, exist_ok=True)

            # Generate unique run ID
            self.run_id = self._generate_run_id(kwargs)
            
            # Initialize WandB with autoscaling-specific config
            self._init_wandb(kwargs)
            
            # Initialize DQN with tuned parameters
            self._init_model(kwargs)
            
            # Traffic Pattern Visualization
            steps = 50
            traffic_simulator = TrafficSimulator()
            traffic_data = [[step, traffic_simulator.get_load(step)] for step in range(steps)]
            traffic_table = wandb.Table(data=traffic_data, columns=["step", "load"])
            
            wandb.log({
                "traffic_pattern": wandb.plot.line(
                    traffic_table, "step", "load",
                    title="Simulated Traffic Pattern"
                )
            })

            logger.info("DQN agent initialized (ID: %s)", self.run_id)
            
        except Exception as e:
            logger.error("Initialization failed: %s", str(e))
            raise

    def _generate_run_id(self, config: Dict[str, Any]) -> str:
        """Generate unique run ID with config parameters."""
        timestamp = datetime.now().strftime("%m%d_%H%M")  # Shorter timestamp
        return (
            f"dqn_{'sim' if self.is_simulated else 'real'}_"
            f"lr{config.get('learning_rate', 0.0005):.0e}_"
            f"t{timestamp}"
        )

    def _init_wandb(self, config: Dict[str, Any]):
        """Initialize WandB with autoscaling-specific settings."""
        try:
            wandb.init(
                project="microk8s_rl_autoscaling",
                # name=RUNID,
                # id=RUNID,
                config={
                    "algorithm": "DQN",
                    "environment": "simulated" if self.is_simulated else "real",
                    "learning_rate": config.get('learning_rate', 0.0005),
                    "buffer_size": config.get('buffer_size', 100000),
                    "batch_size": config.get('batch_size', 64),
                    "gamma": config.get('gamma', 0.99),
                    "exploration_fraction": config.get('exploration_fraction', 0.5),
                    "exploration_final_eps": config.get('exploration_final_eps', 0.1),
                    "exploration_rate": config.get('exploration_rate', 0.2),
                    "seed": config.get('seed', 42)
                },
                tags=["autoscaling", "DQN", "simulated" if self.is_simulated else "real"],
                sync_tensorboard=False,
                tensorboard=False,
                mode="offline"
            )
            
            # Define custom metric grouping
            wandb.define_metric("dqn/*", step_metric="train/global_step")
            wandb.define_metric("cluster/*", step_metric="train/global_step")
            wandb.define_metric("actions/*", step_metric="train/global_step")
            
        except Exception as e:
            logger.error("WandB initialization failed: %s", str(e))
            raise

    def _init_model(self, config: Dict[str, Any]):
        """Initialize DQN model with exploration tracking."""
        try:
            # Extract network architecture parameters
            net_arch_size = config.get('net_arch_size', 64)
            net_arch_layers = config.get('net_arch_layers', 2)
            net_arch = [net_arch_size] * net_arch_layers

            # Handle learning rate - can be float or function
            learning_rate = config.get('learning_rate', 0.0005)
            if isinstance(learning_rate, float):
                # Use linear schedule if single learning rate provided
                learning_rate = get_linear_fn(learning_rate, learning_rate * 0.2, 1.0)

            self.model = DQN(
                policy=MlpPolicy,
                env=self.env,
                learning_rate=learning_rate,
                buffer_size=config.get('buffer_size', 100000),
                batch_size=config.get('batch_size', 64),
                gamma=config.get('gamma', 0.99),
                tau=config.get('tau', 0.1),
                train_freq=config.get('train_freq', 4),
                gradient_steps=config.get('gradient_steps', 2),
                learning_starts=config.get('learning_starts', 5000),
                target_update_interval=config.get('target_update_interval', 2000),
                exploration_fraction=config.get('exploration_fraction', 0.2),
                exploration_initial_eps=1.0,
                exploration_final_eps=config.get('exploration_final_eps', 0.07),
                max_grad_norm=config.get('max_grad_norm', 10),
                verbose=1,
                seed=config.get('seed', 42),
                tensorboard_log=f"{self.model_dir}/tensorboard",
                policy_kwargs=dict(
                    net_arch=net_arch,
                    activation_fn=torch.nn.ReLU
                ),
                device="auto",
                optimize_memory_usage=False,
            )
             
        except Exception as e:
            logger.error("Model initialization failed: %s", str(e))
            raise

    def train(self, total_timesteps: int = 50000, eval_episodes: int = 100):
        """Train with comprehensive metrics tracking."""
        try:
            logger.info("Starting training for %d timesteps", total_timesteps)
            
            callbacks = self._setup_callbacks(eval_episodes)
            tb_log_name = f"dqn_{datetime.now().strftime('%m%d_%H%M')}"

            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks,
                log_interval=10,
                tb_log_name=tb_log_name
            )
            
            self.save()
            logger.info("Training completed")
            
        except KeyboardInterrupt:
            logger.warning("Training interrupted by user")
            self.save()
            raise
        except Exception as e:
            logger.error("Training failed: %s", str(e))
            raise

    def _setup_callbacks(self, eval_episodes: int) -> CallbackList:
        """Configure callbacks with metrics tracking."""
        metrics_callback = AutoscalingMetricsCallback(
            prometheus_url="http://localhost:9090",
            algorithm="dqn",
            verbose=1,
            is_simulated=self.is_simulated
        )

        return CallbackList([
            CheckpointCallback(
                save_freq=10000,
                save_path=self.model_dir,
                name_prefix="dqn_model"
            ),
            EvalCallback(
                eval_env=self.eval_env,
                best_model_save_path=f"{self.model_dir}/best_model",
                log_path=f"{self.model_dir}/eval_logs",
                eval_freq=10000,
                n_eval_episodes=eval_episodes,
                deterministic=True
            ),
            WandbCallback(
                model_save_path=f"{self.model_dir}/wandb",
                verbose=2,
                log="all"
            ),
            metrics_callback,
            VisualizationCallback()
        ])

    def evaluate(self, episodes: int = 100) -> float:
        """Evaluate with detailed metrics tracking."""
        try:
            logger.info("Starting evaluation with %d episodes", episodes)
            rewards = []
            action_history = []
            cluster_metrics = {
                'cpu': [], 'memory': [], 'pods': [], 'latency': [],'swap':[],'optimal_pods':[]
            }

            for episode in range(episodes):
                # Handle environment reset
                reset_result = self.eval_env.reset()
                if isinstance(reset_result, tuple):
                    obs = reset_result[0]  # Gym API returns (obs, info)
                else:
                    obs = reset_result     # VecEnv API returns obs only
                done = False
                episode_reward = 0
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    step_result = self.eval_env.step(action)
                    
                    # Handle both Gym API and VecEnv API step returns
                    if isinstance(step_result, tuple) and len(step_result) == 1:
                        # VecEnv API: ((obs, reward, terminated, truncated, info」の), ...)
                        step_result = step_result[0]
                    
                    # Handle Gymnasium (5 values) or Gym (4 values) return
                    if len(step_result) == 5:
                        # Gymnasium API: obs, reward, terminated, truncated, info
                        obs, reward, terminated, truncated, info = step_result
                        done = terminated or truncated
                    elif len(step_result) == 4:
                        # Gym API: obs, reward, done, info
                        obs, reward, done, info = step_result
                    else:
                        raise ValueError(f"Unexpected step result format: {step_result}")

                    episode_reward += reward
                    action_history.append(action)  # Ensure action is appended

                    # Extract info and log for debugging
                    current_info = info[0] if isinstance(info, list) and len(info) > 0 else info
                    if not isinstance(current_info, dict):
                        logger.warning("Invalid info format: %s ",current_info)
                        current_info = {}
                    logger.debug("Current info: %s ",current_info)

                    # Map custom_metrics to cluster_metrics
                    custom_metrics = current_info.get('custom_metrics', {})
                    cluster_metrics['cpu'].append(custom_metrics.get('cpu_utilization', 0))
                    cluster_metrics['memory'].append(custom_metrics.get('memory_utilization', 0))
                    cluster_metrics['pods'].append(custom_metrics.get('pod_count', 0))
                    cluster_metrics['latency'].append(custom_metrics.get('latency', 0))
                    cluster_metrics['swap'].append(custom_metrics.get('swap_usage', 0))

                # Log episode reward after episode completes
                rewards.append(episode_reward)
                wandb.log({"eval/episode_reward": episode_reward})

            # Log aggregated metrics with safeguards for empty lists
            avg_reward = np.mean(rewards)
            action_counts = {
                "scale_down": np.sum(np.array(action_history) == 0),
                "no_change": np.sum(np.array(action_history) == 1),
                "scale_up": np.sum(np.array(action_history) == 2)
            }
            
            # Create action distribution table
            action_table = wandb.Table(
                columns=["Action", "Count"],
                data=[
                    ["Scale Down", action_counts["scale_down"]],
                    ["No Change", action_counts["no_change"]],
                    ["Scale Up", action_counts["scale_up"]]
                ]
            )
            
            wandb.log({
                "eval/mean_reward": avg_reward,
                "eval/std_reward": np.std(rewards),
                "eval/cpu_mean": np.mean(cluster_metrics['cpu']) if cluster_metrics['cpu'] else 0,
                "eval/memory_mean": np.mean(cluster_metrics['memory']) if cluster_metrics['memory'] else 0,
                "eval/swap_mean": np.mean(cluster_metrics['swap']) if cluster_metrics['swap'] else 0,
                "eval/pod_mean": np.mean(cluster_metrics['pods']) if cluster_metrics['pods'] else 0,
                "eval/scale_ups": action_counts["scale_up"],
                "eval/scale_downs": action_counts["scale_down"],
                "eval/action_distribution": action_counts,
                "eval_summary/action_distribution": wandb.plot.bar(
                    action_table, "Action", "Count",
                    title="Action Distribution"
                ),
                "train/global_step": self.model.num_timesteps,
                "eval_summary/resource_trend": wandb.plot.line_series(
                    xs=range(len(cluster_metrics['cpu'])),
                    ys=[cluster_metrics['cpu']],
                    keys=["CPU Utilization"],
                    title="Resource Trends"
                ),
                 "eval_summary/pod_comparison": wandb.plot.line_series(
                    xs=range(len(cluster_metrics['pods'])),
                    ys=[cluster_metrics['pods'], cluster_metrics['optimal_pods']],
                    keys=["Actual Pods", "Optimal Pods"],
                    title="Pod Count Comparison"
                )
            })

            # Debug action history
            logger.info("Scale ups: %d, Scale downs: %d", action_counts["scale_up"], action_counts["scale_down"])

            logger.info("Evaluation completed. Avg reward: %.2f", avg_reward)
            return avg_reward

        except Exception as e:
            logger.error("Evaluation failed: %s", str(e))
            raise
             
    def save(self, path: Optional[str] = None) -> None:
        """Save model with error handling."""
        try:
            path = path or f"{self.model_dir}/dqn_final"
            self.model.save(path)
            logger.info("Model saved to %s", path)
        except Exception as e:
            logger.error("Model save failed: %s", str(e))
            raise

    def load(self, path: Optional[str] = None) -> None:
        """Load model with error handling."""
        try:
            path = path or f"{self.model_dir}/dqn_final"
            self.model = DQN.load(path, env=self.env)
            logger.info("Model loaded from %s", path)
        except Exception as e:
            logger.error("Model load failed: %s", str(e))
            raise

def main():
    """Main execution with argument parsing."""
    try:
        parser = argparse.ArgumentParser(description='MicroK8s DQN Autoscaler')
        parser.add_argument('--simulate', action='store_true', help='Use simulated environment')
        parser.add_argument('--timesteps', type=int, default=200000, help='Training timesteps')
        parser.add_argument('--eval-episodes', type=int, default=20, help='Evaluation episodes')
        parser.add_argument('--traffic-seed', type=int, default=42, help='Random seed for traffic simulator (default: 42)')

        # Optimizable hyperparameters - Using best parameters from optimization
        # Best parameters found via Optuna optimization
        # Source: /best_dqn_params_latest.json (project root)
        # Optimization result: Best value = -4.0025 (converged)
        # To re-optimize: python -m agent.dqn_optimization --simulate --trials 20
        parser.add_argument('--learning-rate', type=float, default=0.0003320947521799842, help='Learning rate')
        parser.add_argument('--buffer-size', type=int, default=140978, help='Replay buffer size')
        parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
        parser.add_argument('--gamma', type=float, default=0.9732108019691487, help='Discount factor')
        parser.add_argument('--tau', type=float, default=0.0380978024125337, help='Soft update coefficient')
        parser.add_argument('--train-freq', type=int, default=4, help='Training frequency')
        parser.add_argument('--gradient-steps', type=int, default=1, help='Gradient steps per update')
        parser.add_argument('--learning-starts', type=int, default=5125, help='Steps before learning starts')
        parser.add_argument('--target-update-interval', type=int, default=1049, help='Target network update interval')
        parser.add_argument('--exploration-fraction', type=float, default=0.3025188836389893, help='Exploration fraction')
        parser.add_argument('--exploration-final-eps', type=float, default=0.1448672884499737, help='Final exploration epsilon')
        parser.add_argument('--max-grad-norm', type=int, default=16, help='Maximum gradient norm')
        parser.add_argument('--net-arch-size', type=int, default=128, help='Network layer size')
        parser.add_argument('--net-arch-layers', type=int, default=2, help='Number of network layers')

        # Load optimized parameters from file
        parser.add_argument('--load-optimized', type=str, help='Load optimized parameters from JSON file')

        args = parser.parse_args()

        # Load optimized parameters if provided
        if args.load_optimized:
            import json
            try:
                with open(args.load_optimized, 'r') as f:
                    optimized_params = json.load(f)
                logger.info(f"Loading optimized parameters from {args.load_optimized}")
                logger.info(f"Optimized parameters: {optimized_params}")

                # Override command line args with optimized parameters
                for key, value in optimized_params.items():
                    # Convert hyphenated keys to underscore format
                    arg_key = key.replace('_', '-')
                    if hasattr(args, key):
                        setattr(args, key, value)
                        logger.info(f"Set {key}: {value}")
            except Exception as e:
                logger.error(f"Failed to load optimized parameters: {e}")

        # Instantiate environments separately for training and evaluation with consistent seed
        if args.simulate:
            env = MicroK8sEnvSimulated(seed=args.traffic_seed)
        else:
            env = MicroK8sEnv()

        # Prepare configuration dictionary
        config = {
            'learning_rate': args.learning_rate,
            'buffer_size': args.buffer_size,
            'batch_size': args.batch_size,
            'gamma': args.gamma,
            'tau': args.tau,
            'train_freq': args.train_freq,
            'gradient_steps': args.gradient_steps,
            'learning_starts': args.learning_starts,
            'target_update_interval': args.target_update_interval,
            'exploration_fraction': args.exploration_fraction,
            'exploration_final_eps': args.exploration_final_eps,
            'max_grad_norm': args.max_grad_norm,
            'net_arch_size': args.net_arch_size,
            'net_arch_layers': args.net_arch_layers
        }

        logger.info("Training with parameters:")
        for key, value in config.items():
            logger.info(f"  {key}: {value}")

        # Initialize agent with both environments and custom parameters
        agent = DQNAgent(
            env=env,
            environment=env,
            is_simulated=args.simulate,
            **config
        )
    
        agent.load()

        # Train with more episodes for better scaling policy learning
        agent.train(
            total_timesteps=args.timesteps,
            eval_episodes=args.eval_episodes
        )

        # Evaluate with more episodes to ensure stable scaling
        agent.evaluate(episodes=args.eval_episodes)

    except Exception as e:
        logger.critical("Fatal error: %s", str(e))
        raise


if __name__ == "__main__":
    main()