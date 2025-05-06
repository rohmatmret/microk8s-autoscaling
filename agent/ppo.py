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
            # entity="rohmatmret-institute-teknologi-sepuluh-nopember",
            name=RUNID,
            id=RUNID,
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
                "ent_coef": 0.05,
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

        # Create evaluation environment
        self.eval_env = make_vec_env(lambda:MicroK8sEnvSimulated(), n_envs=1)

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
            ent_coef=0.01,  # Added entropy coefficient
            max_grad_norm=0.5,  # Add gradient clipping
        )
        logger.info("Initialized PPO with learning_rate=%.4f, gamma=%.2f", learning_rate, gamma)

    def train(self, total_timesteps: int = 50000, checkpoint_freq: int = 10000, 
        eval_episodes=10) -> None:
        
        """Train the PPO model."""
        try:
            # --- Callback Setup ---
            callbacks = []
            
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
        metrics = {
            'rewards': [],
            'cpu_util': [],
            'memory_util': [],
            'pod_counts': [],
            'swap_usage': [],
            'scaling_actions': []
        }
        for episode in range(episodes):
            obs, _ = self.env.reset()
            done = False
            episode_reward =0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
                
                # Collect metrics
                if 'custom_metrics' in info:
                    custom = info['custom_metrics']
                    metrics['cpu_util'].append(custom.get('cpu_utilization', 0.0))
                    metrics['memory_util'].append(custom.get('memory_utilization', 0.0))
                    metrics['pod_counts'].append(custom.get('pod_count', 0))
                    metrics['swap_usage'].append(custom.get('swap_usage', 0.0))
                    metrics['scaling_actions'].append(custom.get('scaling_action', 0))
                    metrics['rewards'].append(episode_reward)
                elif 'cluster_metrics' in info:
                    custom = info['cluster_metrics']
                    metrics['cpu_util'].append(custom.get('cpu_utilization', 0.0))
                    metrics['memory_util'].append(custom.get('memory_utilization', 0.0))
                    metrics['pod_counts'].append(custom.get('pod_count', 0))
                    metrics['swap_usage'].append(custom.get('swap_usage', 0.0))
                    metrics['scaling_actions'].append(custom.get('scaling_action', 0))
                    metrics['rewards'].append(episode_reward)
                
                logger.info("Episode %d reward: %.2f", episode + 1, episode_reward)
            
        # Log aggregated metrics
        wandb.log({
            "eval/mean_reward": np.mean(metrics['rewards']),
            "eval/std_reward": np.std(metrics['rewards']),
            "eval/mean_cpu": np.mean(metrics['cpu_util']),
            "eval/mean_memory": np.mean(metrics['memory_util']),
            "eval/mean_swap": np.mean(metrics['swap_usage']),
            "eval/mean_pods": np.mean(metrics['pod_counts']),
            "eval/action_distribution": {
                "scale_down": metrics['scaling_actions'].count(-1),
                "no_change": metrics['scaling_actions'].count(0),
                "scale_up": metrics['scaling_actions'].count(1)
            }
        })
        
        return np.mean(metrics['rewards'])

    def save(self) -> None:
        """Save the model."""
        self.model.save(f"{self.model_dir}/ppo_final")
        logger.info("Model saved to %s", self.model_dir)

    def load(self, path: str = None) -> None:
        """Load a trained model."""
        path = path or f"{self.model_dir}/ppo_final"
        self.model = PPO.load(path, env=self.env)
        logger.info("Model loaded from %s", path)

      
      
        
def parse_args():
   
    parser = argparse.ArgumentParser(description='MicroK8s Autoscaler PPO Agent')
    parser.add_argument('--simulate', action='store_true',
        help='Use simulated environment')
    parser.add_argument('--eval-episodes', type=int, default=100,
        help='Number of evaluation episodes')
    parser.add_argument('--timesteps', type=int, default=100000,
        help='Total training timesteps') 

    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()
    DEFAULT_EPISODE = 100 if args.simulate else 10
    args.eval_episodes = DEFAULT_EPISODE
    
    # Create the appropriate environment
    if args.simulate:
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