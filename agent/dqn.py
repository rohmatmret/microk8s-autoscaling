import os
import logging
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
import wandb
from wandb.integration.sb3 import WandbCallback
from agent.environment import MicroK8sEnv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("dqn.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class DQNAgent:
    """DQN agent for MicroK8s autoscaling."""

    def __init__(
        self,
        env: MicroK8sEnv,
        learning_rate: float = 0.001,
        buffer_size: int = 10000,
        batch_size: int = 32,
        gamma: float = 0.95,
        model_dir: str = "./models/dqn"
    ):
        self.env = env
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

        # Initialize wandb
        wandb.init(
            project="microk8s_rl_autoscaling",
            config={
                "learning_rate": learning_rate,
                "buffer_size": buffer_size,
                "batch_size": batch_size,
                "gamma": gamma
            }
        )

        # Initialize DQN model
        self.model = DQN(
            policy="MlpPolicy",
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            batch_size=batch_size,
            gamma=gamma,
            learning_starts=1000,
            train_freq=4,
            target_update_interval=1000,
            exploration_fraction=0.1,
            exploration_final_eps=0.01,
            verbose=1
        )
        logger.info("Initialized DQN with learning_rate=%.4f, gamma=%.2f", learning_rate, gamma)

    def train(self, total_timesteps: int = 50000, checkpoint_freq: int = 10000) -> None:
        """Train the DQN model."""
        try:
            checkpoint_callback = CheckpointCallback(
                save_freq=checkpoint_freq,
                save_path=self.model_dir,
                name_prefix="dqn_model"
            )
            wandb_callback = WandbCallback(
                model_save_path=f"{self.model_dir}/wandb",
                verbose=2
            )
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=[checkpoint_callback, wandb_callback],
                log_interval=10
            )
            self.save()
            logger.info("Training completed for %d timesteps", total_timesteps)
        except Exception as e:
            logger.error("Training failed: %s", e)
            raise

    def evaluate(self, episodes: int = 10) -> float:
        """Evaluate the model and return average reward."""
        total_rewards = []
        for episode in range(episodes):
            obs = self.env.reset()
            done = False
            episode_reward = 0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.env.step(action)
                episode_reward += reward
            total_rewards.append(episode_reward)
            logger.info("Episode %d reward: %.2f", episode + 1, episode_reward)
        avg_reward = np.mean(total_rewards)
        wandb.log({"eval_avg_reward": avg_reward})
        logger.info("Average reward over %d episodes: %.2f", episodes, avg_reward)
        return avg_reward

    def save(self) -> None:
        """Save the model."""
        self.model.save(f"{self.model_dir}/dqn_final")
        logger.info("Model saved to %s", self.model_dir)

    def load(self, path: str = None) -> None:
        """Load a trained model."""
        path = path or f"{self.model_dir}/dqn_final"
        self.model = DQN.load(path, env=self.env)
        logger.info("Model loaded from %s", path)

if __name__ == "__main__":
    env = MicroK8sEnv()
    agent = DQNAgent(env)
    agent.train(total_timesteps=50000)
    agent.evaluate(episodes=10)