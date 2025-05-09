import argparse
import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList, BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from wandb.integration.sb3 import WandbCallback

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
                name=RUNID,
                id=RUNID,
                config={
                    "algorithm": "DQN",
                    "environment": "simulated" if self.is_simulated else "real",
                    "learning_rate": config.get('learning_rate', 0.0005),
                    "buffer_size": config.get('buffer_size', 100000),
                    "batch_size": config.get('batch_size', 64),
                    "gamma": config.get('gamma', 0.99),
                    "exploration_fraction": config.get('exploration_fraction', 0.2),
                    "exploration_final_eps": config.get('exploration_final_eps', 0.005),
                },
                tags=["autoscaling", "DQN", "simulated" if self.is_simulated else "real"],
                sync_tensorboard=False,
                tensorboard=False
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
            self.model = DQN(
                policy="MlpPolicy",
                env=self.env,
                learning_rate=config.get('learning_rate', 0.0005),
                buffer_size=config.get('buffer_size', 100000),
                batch_size=config.get('batch_size', 100),
                gamma=config.get('gamma', 0.99),
                tau=0.01,
                train_freq=4,
                learning_starts=config.get('learning_starts', 10000),
                target_update_interval=config.get('target_update_interval', 10000),
                exploration_fraction=config.get('exploration_fraction', 0.2),
                exploration_final_eps=config.get('exploration_final_eps', 0.005),
                tensorboard_log=f"{self.model_dir}/tensorboard",
                verbose=1,
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
        class AutoscalingMetricsCallback(BaseCallback):
            def __init__(self, verbose=0):
                super().__init__(verbose)
                self.action_history = []
                self.episode_count = 0

            def _on_step(self) -> bool:
                # Log DQN internal metrics every 10 steps
                if self.n_calls % 10 == 0:
                    self._log_dqn_metrics()
                
                # Track actions for scaling analysis
                if len(self.locals['actions']) > 0:
                    self.action_history.extend(self.locals['actions'])
                
                return True

            def _on_rollout_end(self) -> None:
                # Log cluster metrics at the end of each rollout
                if len(self.locals['infos']) > 0:
                    self._log_cluster_metrics(self.locals['infos'][0])
                
                # Log scaling actions statistics
                if len(self.action_history) > 0:
                    self._log_scaling_actions()
                    self.action_history = []
                
                self.episode_count += 1

            def _log_dqn_metrics(self):
                """Log DQN-specific exploration and Q-value metrics."""
                wandb.log({
                    "dqn/epsilon": self.model.exploration_rate,
                    "dqn/exploration_steps": self.model.num_timesteps,
                    "dqn/q_value_mean": np.mean(self.model.replay_buffer.rewards[-1000:]) if hasattr(self.model, 'replay_buffer') else 0,
                }, commit=False)

            def _log_cluster_metrics(self, info: Dict[str, Any]):
                """Log cluster resource utilization metrics."""
                if 'cluster_metrics' in info:
                    metrics = info['cluster_metrics']
                    wandb.log({
                        "cluster/cpu_util": metrics.get('actual_cpu', 0),
                        "cluster/memory_util": metrics.get('actual_memory', 0), 
                        "cluster/pod_count": metrics.get('actual_pods', 0),
                        "cluster/latency": metrics.get('latency', 0)
                    }, commit=False)

            def _log_scaling_actions(self):
                """Analyze and log scaling action patterns."""
                actions = np.array(self.action_history)
                action_counts = {
                    "scale_down": np.sum(actions == 0),
                    "no_change": np.sum(actions == 1),
                    "scale_up": np.sum(actions == 2)
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
                    "actions/scale_up": action_counts["scale_up"],
                    "actions/no_change": action_counts["no_change"],
                    "actions/scale_down": action_counts["scale_down"],
                    "actions/mean_action": np.mean(actions),
                    "eval_summary/action_distribution": wandb.plot.bar(
                        action_table, "Action", "Count",
                        title="Action Distribution"
                    )
                }, commit=False)

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
            AutoscalingMetricsCallback(),
            VisualizationCallback()
        ])

    def evaluate(self, episodes: int = 100) -> float:
        """Evaluate with detailed metrics tracking."""
        try:
            logger.info("Starting evaluation with %d episodes", episodes)
            rewards = []
            action_history = []
            cluster_metrics = {
                'cpu': [], 'memory': [], 'pods': [], 'latency': [],'swap':[]
            }

            for episode in range(episodes):
                # Handle environment reset
                reset_result = self.env.reset()
                if isinstance(reset_result, tuple):
                    obs = reset_result[0]  # Gym API returns (obs, info)
                else:
                    obs = reset_result     # VecEnv API returns obs only
                done = False
                episode_reward = 0
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    step_result = self.env.step(action)
                    
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

                    # cluster_metrics = current_info.get('cluster_metrics', {})
                    # cluster_metrics['cpu'].append(cluster_metrics.get('cpu_utilization', 0))
                    # cluster_metrics['memory'].append(cluster_metrics.get('memory_utilization', 0))
                    # cluster_metrics['pods'].append(cluster_metrics.get('pod_count', 0))
                    # cluster_metrics['latency'].append(cluster_metrics.get('latency', 0))
                    # cluster_metrics['swap'].append(cluster_metrics.get('swap_usage', 0))

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
        parser.add_argument('--timesteps', type=int, default=50000, help='Training timesteps')
        parser.add_argument('--eval-episodes', type=int, default=100, help='Evaluation episodes')
        args = parser.parse_args()

        # Instantiate environments separately for training and evaluation
        if args.simulate:
            env = MicroK8sEnvSimulated()
        else:
            env = MicroK8sEnv()

        # Initialize agent with both environments
        agent = DQNAgent(env=env, environment=env, is_simulated=args.simulate)

        # Train and evaluate
        agent.train(
            total_timesteps=args.timesteps,
            eval_episodes=args.eval_episodes
        )

        agent.evaluate(episodes=args.eval_episodes)

    except Exception as e:
        logger.critical("Fatal error: %s", str(e))
        raise


if __name__ == "__main__":
    main()