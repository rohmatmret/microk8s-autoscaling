# agent/ppo.py
import datetime
import argparse
import os
import logging
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList,BaseCallback
from stable_baselines3.common.env_util import make_vec_env
import wandb
from wandb.integration.sb3 import WandbCallback
from agent.environment import MicroK8sEnv
from agent.environment_simulated import MicroK8sEnvSimulated
from agent.system_callback_metrics import SystemMetricsCallback
from agent.traffic_simulation import TrafficSimulator
import torch as th
import sys
import gym
from gym import spaces
import time
from kubernetes import client, config
from prometheus_api_client import PrometheusConnect
from agent.metrics_callback import AutoscalingMetricsCallback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("ppo.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

EXP_NAME = "ppo_ac_rewardshape"
LR = 3e-4
ENV_NAME = "microk8s_env"
RUNID = f"{EXP_NAME}_lr{LR:.0e}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

class VisualizationCallback(BaseCallback):
    """Callback to log resource utilization and scaling metrics to wandb."""
    def __init__(self, env, log_freq=1000):
        super().__init__()
        self.env = env
        self.log_freq = log_freq
        self.scaling_actions = []
        
    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            try:
                # Resource Utilization Dashboard
                state = self.env.api.get_cluster_state()
                wandb.log({
                    "resources/cpu": state["cpu"],
                    "resources/memory": state["memory"]/500e6,
                    "resources/swap": state["swap"]/200e6,
                    "resources/latency": state["latency"],
                    "scaling/pod_count": state["pods"],
                    "scaling/desired_pods": state["desired_replicas"],
                    "scaling/lag": state["desired_replicas"] - state["pods"],
                    "train/global_step": self.num_timesteps
                })
            except Exception as e:
                logger.warning("Failed to log visualization metrics: %s ",str(e))
        return True
    
class CustomEvalCallback(EvalCallback):
    def _on_step(self) -> bool:
        result = super()._on_step()
        if result and self.eval_env is not None:
            # Log additional metrics after each evaluation
            if hasattr(self, "last_eval_metrics"):
                wandb.log({
                    "eval/mean_reward": self.last_eval_metrics["mean_reward"],
                    "eval/mean_ep_length": self.last_eval_metrics["mean_ep_length"],
                    "train/global_step": self.num_timesteps
                })
                
            if hasattr(self.eval_env, "get_cluster_metrics"):  # Add this method to your env
                metrics = self.eval_env.get_cluster_metrics()
                wandb.log({
                    "cluster/cpu_util": metrics["cpu"],
                    "cluster/memory_util": metrics["memory"],
                    "cluster/pod_count": metrics["pods"],
                })
                
        return result
    
class ScalingMetricsCallback(BaseCallback):
    """Callback to track scaling actions and their distribution."""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.scale_ups = 0
        self.scale_downs = 0
        self.no_changes = 0
        
    def _on_step(self) -> bool:
        if len(self.locals['actions']) > 0:
            action = self.locals['actions'][0]
            if action == 1:  # Scale up
                self.scale_ups += 1
            elif action == 2:  # Scale down
                self.scale_downs += 1
            else:  # No change
                self.no_changes += 1
                
            # Log to wandb every 1000 steps
            if self.n_calls % 1000 == 0:
                wandb.log({
                    "scaling/scale_ups": self.scale_ups,
                    "scaling/scale_downs": self.scale_downs,
                    "scaling/no_changes": self.no_changes,
                    "train/global_step": self.num_timesteps
                })
        return True

    def _on_rollout_end(self) -> None:
        # Log final scaling metrics at the end of each rollout
        logger.info("[Training] Scale ups: %d, Scale downs: %d, No changes: %d", 
                   self.scale_ups, self.scale_downs, self.no_changes)

class PPOAgent:
    """PPO agent for MicroK8s autoscaling."""

    def __init__(
        self,
        environment: MicroK8sEnv,
        learning_rate: float = 0.0003,
        n_steps: int = 2048,
        batch_size: int = 64,
        gamma: float = 0.99,
        model_dir: str = "./models/ppo"
    ):
        self.env = environment
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

        # Initialize wandb
        wandb.init(
            project="microk8s_rl_autoscaling",
            resume="allow",
            config={
                "algorithm": "PPO",
                "environment": ENV_NAME,
                "reward_shaping": True,
                "learning_rate": learning_rate,
                "gamma": gamma,
                "gae_lambda": 0.95,
                "n_steps": n_steps,
                "batch_size": batch_size,
                "n_epochs": 10,
                "normalize_advantage": True,
                "clip_range": 0.2,
                "vf_coef": 0.5,
                "policy": "MlpPolicy",
            },
            tensorboard=True,
            tags=["PPO", "autoscaling", "reward shaping", ENV_NAME],
            notes="Eksperimen dengan reward shaping dan PPO baseline."
        )
        wandb.define_metric("custom/*", step_metric="train/global_step")
        wandb.define_metric("train/global_step")
        wandb.define_metric("custom/cpu_utilization", step_metric="train/global_step")
        wandb.define_metric("custom/memory_utilization", step_metric="train/global_step")
        wandb.define_metric("custom/pod_count", step_metric="train/global_step")

        wandb.define_metric("scaling/*", step_metric="train/global_step")
        wandb.define_metric("resources/*", step_metric="train/global_step")

        wandb.config.update({
            "custom_metrics": {
                "target_cpu_range": [0.4, 0.8],
                "target_memory_range": [0.3, 0.7],
                "ideal_pod_count": 5,
                "swap":0
            }
        })
        wandb.Settings(init_timeout=180)

        # Create evaluation environment with higher base load
        self.eval_env = make_vec_env(
            lambda: MicroK8sEnvSimulated(),
            n_envs=1
        )
        
        # Update traffic simulator parameters for evaluation
        for env in self.eval_env.envs:
            env.env.traffic_simulator.base_load = 150
            env.env.traffic_simulator.max_spike = 50
            env.env.traffic_simulator.daily_amplitude = 0.5
            env.env.traffic_simulator.spike_probability = 0.01

        # Initialize PPO model
        self.model = PPO(
            policy="MlpPolicy",
            env=self.env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            gamma=gamma,
            gae_lambda=0.95,
            clip_range=0.2,
            n_epochs=10,
            verbose=1,
            ent_coef=0.05,  # Added entropy coefficient
            max_grad_norm=0.5,  # Add gradient clipping
            policy_kwargs=dict(net_arch=[64,64]),
            seed=42,
            device="cuda"  # ✅ Gunakan GPU

        )
        logger.info("Initialized PPO with learning_rate=%.4f, gamma=%.2f", learning_rate, gamma)

    def train(self, total_timesteps: int = 50000, checkpoint_freq: int = 10000, 
        eval_episodes=10) -> None:
        
        """Train the PPO model."""

        try:
            # --- Callback Setup ---
            callbacks = []
            
            vis_callback = VisualizationCallback(self.env)
            callbacks.append(vis_callback)
            
            # Add scaling metrics callback
            scaling_callback = ScalingMetricsCallback()
            callbacks.append(scaling_callback)
            
            checkpoint_callback = CheckpointCallback(
                save_freq=checkpoint_freq,
                save_path=self.model_dir,
                name_prefix="ppo_model"
            )
            callbacks.append(checkpoint_callback)

            eval_callback = CustomEvalCallback(
                eval_env=self.eval_env,
                best_model_save_path=f"{self.model_dir}/best_model",
                log_path=f"{self.model_dir}/eval_logs",
                eval_freq=5000,
                deterministic=True,
                render=False,
                n_eval_episodes=eval_episodes
            )
            callbacks.append(eval_callback)

            wandb_callback = WandbCallback(
                model_save_path=f"{self.model_dir}/wandb",
                verbose=2,
                log="all",
                gradient_save_freq=100,
                model_save_freq=checkpoint_freq,
            )
            
            callbacks.append(wandb_callback)
            
            system_callback = SystemMetricsCallback(self.env,
                                log_freq=1000,
                                metrics_to_track=[
                                    'cpu_utilization',
                                    'memory_utilization',
                                    'pod_count',
                                    'scaling_lag'
                                ]
                            )
            
            callbacks.append(system_callback)

            # callback = CallbackList([checkpoint_callback, eval_callback, wandb_callback,system_callback])
            class ProgressLogger(BaseCallback):
                "progress"
                def __init__(self, check_interval=1000):
                    super().__init__()
                    self.check_interval = check_interval
                
                def _on_step(self) -> bool:
                    if self.n_calls % self.check_interval == 0:
                        logger.info("Progress: %.1f%%", 100 * self.num_timesteps / total_timesteps)
                        if hasattr(self.model, 'env'):
                            try:
                                wandb.log({
                                    "progress": self.num_timesteps/total_timesteps,
                                    "timesteps": self.num_timesteps
                                })
                            except:
                                logger.warning("Failed to log progress to W&B")
                    return True
                    
            callbacks.append(ProgressLogger())

            self.model.learn(
                total_timesteps=total_timesteps,
                callback=CallbackList(callbacks),
                log_interval=10
            )
            self.save()
            logger.info("Training completed for %d timesteps", total_timesteps)
        except Exception as e:
            logger.error("Training failed: %s", e)
            raise

    def evaluate(self, episodes: int = 10) -> float:
        """Enhanced evaluation with detailed metrics"""
        try:
            logger.info("Starting evaluation with %d episodes", episodes)
            rewards = []
            action_history = []
            cluster_metrics = {
                'cpu': [], 'memory': [], 'pods': [], 'latency': [], 'swap': []
            }
            episode_metrics = []  # Track metrics per episode

            for episode in range(episodes):
                obs = self.eval_env.reset()
                done = False
                episode_reward = 0
                episode_actions = []
                episode_cpu = []
                
                while not done:
                    # Get action probabilities for analysis
                    action, state = self.model.predict(obs, deterministic=True)
                    # Convert observation to tensor for action probabilities
                    obs_tensor = th.from_numpy(obs).float()
                    action_probs = self.model.policy.get_distribution(obs_tensor).distribution.probs.detach().cpu().numpy()
                    
                    action_history.append(action[0])
                    episode_actions.append(action[0])
                    
                    obs, reward, done, info = self.eval_env.step(action)
                    episode_reward += reward
                    
                    # Track cluster metrics
                    if isinstance(info, list) and len(info) > 0:
                        info = info[0]
                    if 'custom_metrics' in info:
                        metrics = info['custom_metrics']
                        cpu_util = metrics.get('cpu_utilization', 0)
                        episode_cpu.append(cpu_util)
                        cluster_metrics['cpu'].append(cpu_util)
                        cluster_metrics['memory'].append(metrics.get('memory_utilization', 0))
                        cluster_metrics['pods'].append(metrics.get('pod_count', 0))
                        cluster_metrics['latency'].append(metrics.get('latency', 0))
                        cluster_metrics['swap'].append(metrics.get('swap_usage', 0))
                
                # Log episode-specific metrics
                episode_metrics.append({
                    'reward': episode_reward,
                    'actions': episode_actions,
                    'cpu_mean': np.mean(episode_cpu) if episode_cpu else 0,
                    'action_probs': action_probs
                })
                
                rewards.append(episode_reward)
                wandb.log({"eval/episode_reward": episode_reward})

            # Calculate scaling statistics
            action_counts = {
                "scale_down": np.sum(np.array(action_history) == 2),
                "no_change": np.sum(np.array(action_history) == 0),
                "scale_up": np.sum(np.array(action_history) == 1)
            }
            
            # Log detailed scaling statistics
            logger.info("[Evaluation] Scale ups: %d, Scale downs: %d, No changes: %d", 
                       action_counts["scale_up"], 
                       action_counts["scale_down"],
                       action_counts["no_change"])
            
            # Log CPU utilization statistics
            cpu_mean = np.mean(cluster_metrics['cpu']) if cluster_metrics['cpu'] else 0
            logger.info("[Evaluation] Average CPU utilization: %.2f", cpu_mean)
            
            # Calculate and log average reward
            avg_reward = np.mean(rewards)
            logger.info("Evaluation completed. Avg reward: %.2f", avg_reward)
            
            # Log detailed metrics to wandb
            wandb.log({
                "eval/mean_reward": avg_reward,
                "eval/std_reward": np.std(rewards),
                "eval/cpu_mean": cpu_mean,
                "eval/memory_mean": np.mean(cluster_metrics['memory']) if cluster_metrics['memory'] else 0,
                "eval/swap_mean": np.mean(cluster_metrics['swap']) if cluster_metrics['swap'] else 0,
                "eval/pod_mean": np.mean(cluster_metrics['pods']) if cluster_metrics['pods'] else 0,
                "eval/scale_ups": action_counts["scale_up"],
                "eval/scale_downs": action_counts["scale_down"],
                "eval/no_changes": action_counts["no_change"],
                "train/global_step": self.model.num_timesteps
            })
            
            return avg_reward
            
        except Exception as e:
            logger.error("Evaluation failed: %s", str(e))
            raise

    def save(self) -> None:
        """Save the model."""
        self.model.save(f"{self.model_dir}/ppo_final")
        logger.info("Model saved to %s", self.model_dir)

    def load(self, path: str = None) -> None:
        """Load a trained model."""
        path = path or f"{self.model_dir}/ppo_final"
        self.model = PPO.load(path, env=self.env)
        logger.info("Model loaded from %s", path)

class K8sEnv(gym.Env):
    """Kubernetes environment for PPO training."""
    
    def __init__(self):
        super().__init__()
        
        # Define action space (scale up, no change, scale down)
        self.action_space = spaces.Discrete(3)
        
        # Define observation space
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0]),  # [cpu_util, memory_util, pod_count, latency]
            high=np.array([100, 100, 10, 1000]),
            dtype=np.float32
        )
        
        # Initialize Kubernetes client
        try:
            config.load_kube_config()
            self.k8s_api = client.AppsV1Api()
            self.core_api = client.CoreV1Api()
        except Exception as e:
            logger.error(f"Failed to initialize Kubernetes client: {e}")
            raise
        
        # Initialize Prometheus client
        self.prom = PrometheusConnect(url="http://localhost:9090")
        
        # Set deployment name
        self.deployment_name = "nginx-deployment"
        self.namespace = "default"
        
        # Initialize state
        self.current_pods = 1
        self.max_pods = 10
        self.min_pods = 1
        
    def reset(self):
        """Reset environment to initial state."""
        # Reset deployment to initial state
        try:
            self.k8s_api.patch_namespaced_deployment(
                name=self.deployment_name,
                namespace=self.namespace,
                body={"spec": {"replicas": self.min_pods}}
            )
            self.current_pods = self.min_pods
        except Exception as e:
            logger.error(f"Failed to reset deployment: {e}")
        
        # Wait for pods to be ready
        time.sleep(5)
        
        # Get initial state
        return self._get_state()
    
    def step(self, action):
        """Execute one step in the environment."""
        # Map action to pod count change
        if action == 0:  # Scale down
            new_pods = max(self.current_pods - 1, self.min_pods)
        elif action == 1:  # No change
            new_pods = self.current_pods
        else:  # Scale up
            new_pods = min(self.current_pods + 1, self.max_pods)
        
        # Update deployment
        try:
            self.k8s_api.patch_namespaced_deployment(
                name=self.deployment_name,
                namespace=self.namespace,
                body={"spec": {"replicas": new_pods}}
            )
            self.current_pods = new_pods
        except Exception as e:
            logger.error(f"Failed to update deployment: {e}")
        
        # Wait for scaling to take effect
        time.sleep(5)
        
        # Get new state
        state = self._get_state()
        
        # Calculate reward
        reward = self._calculate_reward(state)
        
        # Check if episode is done
        done = False
        
        # Additional info
        info = {
            'cluster_metrics': {
                'cpu_utilization': state[0],
                'memory_utilization': state[1],
                'pod_count': state[2],
                'latency': state[3]
            }
        }
        
        return state, reward, done, info
    
    def _get_state(self):
        """Get current state from Prometheus metrics."""
        try:
            # Get CPU utilization
            cpu_query = 'rate(container_cpu_usage_seconds_total[1m])'
            cpu_response = self.prom.custom_query(cpu_query)
            cpu_util = float(cpu_response[0]['value'][1]) * 100
            
            # Get memory utilization
            memory_query = 'container_memory_usage_bytes'
            memory_response = self.prom.custom_query(memory_query)
            memory_util = float(memory_response[0]['value'][1]) / 1e9 * 100  # Convert to GB and percentage
            
            # Get latency
            latency_query = 'rate(http_request_duration_seconds_sum[1m])/rate(http_request_duration_seconds_count[1m])'
            latency_response = self.prom.custom_query(latency_query)
            latency = float(latency_response[0]['value'][1]) * 1000  # Convert to ms
            
            return np.array([cpu_util, memory_util, self.current_pods, latency], dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Failed to get state from Prometheus: {e}")
            return np.zeros(4, dtype=np.float32)
    
    def _calculate_reward(self, state):
        """Calculate reward based on current state."""
        cpu_util, memory_util, pod_count, latency = state
        
        # Target metrics
        target_cpu = 70
        target_memory = 70
        target_latency = 100  # ms
        
        # Calculate individual rewards
        cpu_reward = -abs(cpu_util - target_cpu) / 100
        memory_reward = -abs(memory_util - target_memory) / 100
        latency_reward = -abs(latency - target_latency) / 1000
        
        # Penalize too many pods
        pod_penalty = -0.1 * (pod_count - 1)
        
        # Combine rewards
        total_reward = cpu_reward + memory_reward + latency_reward + pod_penalty
        
        return total_reward

class K8sSimulationEnv(K8sEnv):
    """Simulated Kubernetes environment for faster training."""
    
    def __init__(self):
        super().__init__()
        
        # Simulated metrics
        self.cpu_util = 50
        self.memory_util = 50
        self.latency = 100
        
        # Traffic pattern
        self.time_step = 0
        self.base_load = 100
        self.max_spike = 1000
        self.spike_probability = 0.1
    
    def _get_state(self):
        """Get simulated state."""
        # Simulate traffic pattern
        if np.random.random() < self.spike_probability:
            self.cpu_util = min(100, self.cpu_util + np.random.randint(10, 30))
            self.memory_util = min(100, self.memory_util + np.random.randint(5, 15))
            self.latency = min(1000, self.latency + np.random.randint(50, 200))
        else:
            self.cpu_util = max(0, self.cpu_util - np.random.randint(5, 15))
            self.memory_util = max(0, self.memory_util - np.random.randint(2, 8))
            self.latency = max(0, self.latency - np.random.randint(20, 100))
        
        self.time_step += 1
        return np.array([self.cpu_util, self.memory_util, self.current_pods, self.latency], dtype=np.float32)

def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description='MicroK8s Autoscaler PPO Agent')
    parser.add_argument('--simulate', action='store_true', help='Use simulated environment')
    parser.add_argument('--eval-episodes', type=int, default=10, help='Number of evaluation episodes')
    parser.add_argument('--timesteps', type=int, default=100000, help='Total training timesteps')
    parsed_args = parser.parse_args()
    
    # Set default eval_episodes based on simulation mode
    if parsed_args.simulate:
        parsed_args.eval_episodes = 100
    
    return parsed_args

if __name__ == "__main__":
    args = parse_args()
    DEFAULT_EPISODE = 100 if args.simulate else 10
    args.eval_episodes = DEFAULT_EPISODE
    
    # Create the appropriate environment
    if args.simulate:
        traffic_simulator = TrafficSimulator(
            base_load=100,
            max_spike=30,
        )
        env = MicroK8sEnvSimulated()
        print("Using SIMULATED environment")
    else:
        env = MicroK8sEnv()  # Your real environment
        print("Using REAL environment")
    # Initialize and train agent
    agent = PPOAgent(env)
    agent.train(
        total_timesteps=args.timesteps,
        eval_episodes=args.eval_episodes
    )
    # Final evaluation
    agent.evaluate(episodes=args.eval_episodes)