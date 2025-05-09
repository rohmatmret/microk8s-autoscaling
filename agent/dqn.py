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
from agent.hybird_simulation import HybridTrafficSimulator

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

    def __init__(self, verbose=0, log_freq=1000):
        super().__init__(verbose)
        self.resource_history = []
        self.action_history = []
        self.pod_history = []
        self.optimal_pod_history = []
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        # Defensive: check for 'infos' and 'actions' in locals
        infos = self.locals.get('infos', [])
        actions = self.locals.get('actions', [])
        if not infos or not actions:
            return True
        info = infos[0]
        custom_metrics = info.get('custom_metrics', {})
        self.resource_history.append(custom_metrics.get('cpu_utilization', 0))
        self.action_history.append(actions[0])
        self.pod_history.append(custom_metrics.get('pod_count', 0))
        self.optimal_pod_history.append(custom_metrics.get('optimal_pods', 0))

        if len(self.resource_history) % self.log_freq == 0:
            try:
                wandb.log({
                    "resource_trends": wandb.plot.line_series(
                        xs=range(len(self.resource_history)),
                        ys=[self.resource_history],
                        keys=["CPU Utilization"],
                        title="Resource Trends"
                    ),
                    "pod_comparison": wandb.plot.line_series(
                        xs=range(len(self.pod_history)),
                        ys=[self.pod_history, self.optimal_pod_history],
                        keys=["Actual Pods", "Optimal Pods"],
                        title="Pod Count Comparison"
                    ),
                    "train/global_step": getattr(self.model, "num_timesteps", 0)
                })
            except Exception as e:
                logger.warning("Failed to log trends: %s", e)

        return True

class DQNAgent:
    """DQN agent with comprehensive autoscaling metrics tracking."""

    def __init__(self, env, environment: MicroK8sEnv, is_simulated: bool = False, seed: int = None, enable_visualization: bool = True, **kwargs):
        """
        Initialize DQN agent with metrics tracking.

        Args:
            env: RL environment (not used, kept for compatibility)
            environment: MicroK8sEnv instance (used for vectorized env)
            is_simulated: Whether using simulated environment
            seed: Random seed for reproducibility
            enable_visualization: Whether to enable VisualizationCallback
            kwargs: Additional configs
        """
        self.is_simulated = is_simulated
        self.model_dir = kwargs.get('model_dir', './models/dqn')
        os.makedirs(self.model_dir, exist_ok=True)
        self.model = None
        self.enable_visualization = enable_visualization
        self.seed = seed

        # Set up environments
        self.env = make_vec_env(lambda: environment, n_envs=1, seed=seed)
        self.eval_env = make_vec_env(lambda: environment, n_envs=1, seed=seed)

        # Generate unique run ID
        self.run_id = self._generate_run_id(kwargs)

        # Initialize WandB
        try:
            self._init_wandb(kwargs)
        except Exception as e:
            logger.error("WandB initialization failed: %s", str(e))
            raise

        # Initialize DQN
        try:
            self._init_model(kwargs)
        except Exception as e:
            logger.error("DQN model initialization failed: %s", str(e))
            raise

        # Traffic Pattern Visualization
        if self.is_simulated and self.enable_visualization:
            try:
                steps_count = 100
                traffic_simulator = HybridTrafficSimulator(seed=seed)
                steps, loads, events = traffic_simulator.get_visualization_data(max_steps=steps_count)
                traffic_table = wandb.Table(
                    data=[[s, l, e] for s, l, e in zip(steps, loads, events)],
                    columns=["step", "load", "event_type"]
                )
                wandb.log({
                    "traffic_pattern": wandb.plot.line(
                        traffic_table, "step", "load",
                        title="Simulated Traffic Pattern"
                    ),
                    "traffic_events": traffic_table
                })
            except Exception as e:
                logger.warning("Traffic pattern visualization failed: %s", str(e))

        logger.info("DQN agent initialized (ID: %s)", self.run_id)

    def _generate_run_id(self, config: Dict[str, Any]) -> str:
        timestamp = datetime.now().strftime("%m%d_%H%M")
        return (
            f"dqn_{'sim' if self.is_simulated else 'real'}_"
            f"lr{config.get('learning_rate', 0.0005):.0e}_"
            f"t{timestamp}"
        )

    def _init_wandb(self, config: Dict[str, Any]):
        try:
            wandb.init(
                project="microk8s_rl_autoscaling",
                name=self.run_id,
                id=self.run_id,
                config={
                    "algorithm": "DQN",
                    "environment": "simulated" if self.is_simulated else "real",
                    "learning_rate": config.get('learning_rate', 0.0005),
                    "buffer_size": config.get('buffer_size', 100000),
                    "batch_size": config.get('batch_size', 64),
                    "gamma": config.get('gamma', 0.99),
                    "exploration_fraction": config.get('exploration_fraction', 0.2),
                    "exploration_final_eps": config.get('exploration_final_eps', 0.005),
                    "seed": self.seed
                },
                tags=["autoscaling", "DQN", "simulated" if self.is_simulated else "real"],
                sync_tensorboard=False,
                tensorboard=False,
            )

            wandb.define_metric("dqn/*", step_metric="train/global_step")
            wandb.define_metric("cluster/*", step_metric="train/global_step")
            wandb.define_metric("actions/*", step_metric="train/global_step")

        except Exception as e:
            logger.error("WandB initialization failed: %s", str(e))
            raise

    def _init_model(self, config: Dict[str, Any]):
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
                policy_kwargs=dict(net_arch=[256, 256, 256]),
                verbose=1,
                seed=self.seed
            )

        except Exception as e:
            logger.error("Model initialization failed: %s", str(e))
            raise

    def train(self, total_timesteps: int = 50000, eval_episodes: int = 100):
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
        class AutoscalingMetricsCallback(BaseCallback):
            def __init__(self, verbose=0, log_freq=100):
                super().__init__(verbose)
                self.action_history = []
                self.episode_count = 0
                self.log_freq = log_freq

            def _on_step(self) -> bool:
                if self.n_calls % self.log_freq == 0:
                    self._log_dqn_metrics()

                actions = self.locals.get('actions', [])
                if len(actions) > 0:
                    self.action_history.extend(actions)

                return True

            def _on_rollout_end(self) -> None:
                infos = self.locals.get('infos', [])
                if len(infos) > 0:
                    self._log_cluster_metrics(infos[0])

                if len(self.action_history) > 0:
                    self._log_scaling_actions()
                    self.action_history = []

                self.episode_count += 1

            def _log_dqn_metrics(self):
                rewards = getattr(self.model, "replay_buffer", None)
                q_value_mean = 0
                if rewards is not None and hasattr(rewards, "rewards"):
                    q_value_mean = np.mean(rewards.rewards[-1000:]) if len(rewards.rewards) > 0 else 0
                wandb.log({
                    "dqn/epsilon": getattr(self.model, "exploration_rate", 0),
                    "dqn/exploration_steps": getattr(self.model, "num_timesteps", 0),
                    "dqn/q_value_mean": q_value_mean,
                }, commit=False)

            def _log_cluster_metrics(self, info: Dict[str, Any]):
                if 'cluster_metrics' in info:
                    metrics = info['cluster_metrics']
                    wandb.log({
                        "cluster/cpu_util": metrics.get('actual_cpu', 0),
                        "cluster/memory_util": metrics.get('actual_memory', 0),
                        "cluster/pod_count": metrics.get('actual_pods', 0),
                        "cluster/latency": metrics.get('latency', 0)
                    }, commit=False)

            def _log_scaling_actions(self):
                actions = np.array(self.action_history)
                action_counts = {
                    "scale_down": int(np.sum(actions == 0)),
                    "no_change": int(np.sum(actions == 1)),
                    "scale_up": int(np.sum(actions == 2))
                }

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
                    "actions/mean_action": float(np.mean(actions)) if len(actions) > 0 else 0,
                    "eval_summary/action_distribution": wandb.plot.bar(
                        action_table, "Action", "Count",
                        title="Action Distribution"
                    )
                }, commit=False)

        callbacks = [
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
            AutoscalingMetricsCallback(log_freq=100)
        ]

        if self.enable_visualization:
            callbacks.append(VisualizationCallback(log_freq=1000))

        return CallbackList(callbacks)

    def evaluate(self, episodes: int = 100) -> float:
        try:
            logger.info("Starting evaluation with %d episodes", episodes)
            rewards = []
            action_history = []
            cluster_metrics = {
                'cpu': [], 'memory': [], 'pods': [], 'latency': [], 'swap': [], 'optimal_pods': []
            }

            for episode in range(episodes):
                reset_result = self.env.reset()
                if isinstance(reset_result, tuple):
                    obs = reset_result[0]
                else:
                    obs = reset_result
                done = False
                episode_reward = 0
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    step_result = self.env.step(action)

                    # Unwrap step_result if needed
                    if isinstance(step_result, tuple) and len(step_result) == 1:
                        step_result = step_result[0]

                    # Handle both 5-tuple and 4-tuple step results
                    if isinstance(step_result, (list, tuple)) and len(step_result) == 5:
                        obs, reward, terminated, truncated, info = step_result
                        done = terminated or truncated
                    elif isinstance(step_result, (list, tuple)) and len(step_result) == 4:
                        obs, reward, done, info = step_result
                    else:
                        raise ValueError(f"Unexpected step result format: {step_result}")

                    episode_reward += reward
                    action_history.append(action)

                    # Defensive: info may be a list or dict
                    current_info = info[0] if isinstance(info, list) and len(info) > 0 else info
                    if not isinstance(current_info, dict):
                        logger.warning("Invalid info format: %s ", current_info)
                        current_info = {}
                    logger.debug("Current info: %s ", current_info)

                    custom_metrics = current_info.get('custom_metrics', {})
                    cluster_metrics['cpu'].append(custom_metrics.get('cpu_utilization', 0))
                    cluster_metrics['memory'].append(custom_metrics.get('memory_utilization', 0))
                    cluster_metrics['pods'].append(custom_metrics.get('pod_count', 0))
                    cluster_metrics['latency'].append(custom_metrics.get('latency', 0))
                    cluster_metrics['swap'].append(custom_metrics.get('swap_usage', 0))
                    cluster_metrics['optimal_pods'].append(custom_metrics.get('optimal_pods', 0))

                rewards.append(episode_reward)
                wandb.log({"eval/episode_reward": episode_reward})

            avg_reward = float(np.mean(rewards)) if rewards else 0
            std_reward = float(np.std(rewards)) if rewards else 0
            action_arr = np.array(action_history)
            action_counts = {
                "scale_down": int(np.sum(action_arr == 0)),
                "no_change": int(np.sum(action_arr == 1)),
                "scale_up": int(np.sum(action_arr == 2))
            }

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
                "eval/std_reward": std_reward,
                "eval/cpu_mean": float(np.mean(cluster_metrics['cpu'])) if cluster_metrics['cpu'] else 0,
                "eval/memory_mean": float(np.mean(cluster_metrics['memory'])) if cluster_metrics['memory'] else 0,
                "eval/swap_mean": float(np.mean(cluster_metrics['swap'])) if cluster_metrics['swap'] else 0,
                "eval/pod_mean": float(np.mean(cluster_metrics['pods'])) if cluster_metrics['pods'] else 0,
                "eval/optimal_pod_mean": float(np.mean(cluster_metrics['optimal_pods'])) if cluster_metrics['optimal_pods'] else 0,
                "eval/scale_ups": action_counts["scale_up"],
                "eval/scale_downs": action_counts["scale_down"],
                "eval/action_distribution": action_counts,
                "eval_summary/action_distribution": wandb.plot.bar(
                    action_table, "Action", "Count",
                    title="Action Distribution"
                ),
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

            logger.info("Scale ups: %d, Scale downs: %d", action_counts["scale_up"], action_counts["scale_down"])
            logger.info("Evaluation completed. Avg reward: %.2f", avg_reward)
            return avg_reward

        except Exception as e:
            logger.error("Evaluation failed: %s", str(e))
            raise

    def save(self, path: Optional[str] = None) -> None:
        try:
            path = path or f"{self.model_dir}/dqn_final"
            self.model.save(path)
            logger.info("Model saved to %s", path)
        except Exception as e:
            logger.error("Model save failed: %s", str(e))
            raise

    def load(self, path: Optional[str] = None) -> None:
        try:
            path = path or f"{self.model_dir}/dqn_final"
            self.model = DQN.load(path, env=self.env)
            logger.info("Model loaded from %s", path)
        except Exception as e:
            logger.error("Model load failed: %s", str(e))
            raise

def main():
    try:
        parser = argparse.ArgumentParser(description='MicroK8s DQN Autoscaler')
        parser.add_argument('--simulate', action='store_true', help='Use simulated environment')
        parser.add_argument('--timesteps', type=int, default=100000, help='Training timesteps')
        parser.add_argument('--eval-episodes', type=int, default=100, help='Evaluation episodes')
        parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
        parser.add_argument('--no-visualization', action='store_true', help='Disable visualization callback')
        args = parser.parse_args()

        if args.simulate:
            env = MicroK8sEnvSimulated(seed=args.seed, enable_visualization=not args.no_visualization)
        else:
            env = MicroK8sEnv()

        agent = DQNAgent(
            env=env,
            environment=env,
            is_simulated=args.simulate,
            seed=args.seed,
            # enable_visualization=not args.no_visualization
            enable_visualization=False
        )

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