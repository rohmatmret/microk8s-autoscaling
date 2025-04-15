# agent/environment.py
import os
import time
import logging
import numpy as np
from typing import Tuple, Dict
import gym
from gym import spaces
from kubernetes.client.rest import ApiException
from agent.kubernetes_api import KubernetesAPI, KubernetesAPIError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("environment.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class MicroK8sEnv(gym.Env):
    """Gym environment for MicroK8s autoscaling with DRL."""

    def __init__(
        self,
        deployment_name: str = os.getenv("DEPLOYMENT_NAME", "nginx-deployment"),
        namespace: str = os.getenv("K8S_NAMESPACE", "default"),
        max_pods: int = 10,
        scaling_delay: int = 15
    ):
        super(MicroK8sEnv, self).__init__()
        self.k8s = KubernetesAPI(max_pods=max_pods, namespace=namespace)
        self.deployment_name = deployment_name
        self.scaling_delay = scaling_delay

        # State: [cpu, memory, latency, pods]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )

        # Action: [scale up, scale down, no-op]
        self.action_space = spaces.Discrete(3)

        self.state = None
        logger.info("Initialized MicroK8sEnv for %s in namespace %s", deployment_name, namespace)

    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        try:
            # Set initial pod count to 2
            self.k8s.safe_scale(self.deployment_name, 2)
            time.sleep(self.scaling_delay)
            self.state = self._get_normalized_state()
            logger.info("Environment reset: state=%s", self.state)
            return self.state
        except KubernetesAPIError as e:
            logger.error("Reset failed: %s", e)
            raise

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute action and return new state, reward, done, and info."""
        try:
            prev_state = self.state
            current_replicas = self.k8s._get_current_replicas(self.deployment_name)

            # Execute action
            if action == 0:  # Scale up
                self.k8s.safe_scale(self.deployment_name, current_replicas + 1)
            elif action == 1:  # Scale down
                self.k8s.safe_scale(self.deployment_name, current_replicas - 1)
            # Action 2: No-op (do nothing)

            time.sleep(self.scaling_delay)  # Wait for scaling effect
            self.state = self._get_normalized_state()

            reward = self._calculate_reward(prev_state, self.state)
            done = self._is_done()
            info = {"action": action, "replicas": current_replicas}

            logger.debug("Step: action=%d, reward=%.2f, state=%s", action, reward, self.state)
            return self.state, reward, done, info

        except KubernetesAPIError as e:
            logger.error("Step failed: %s", e)
            reward = -10.0  # Penalty for failure
            return self.state, reward, False, {"error": str(e)}

    def _get_normalized_state(self) -> np.ndarray:
        """Get and normalize cluster state."""
        raw_state = self.k8s.get_cluster_state()
        normalized = np.array([
            min(raw_state["cpu"] / 100.0, 1.0),  # CPU usage (%)
            min(raw_state["memory"] / 1e9, 1.0),  # Memory (GB, capped)
            min(raw_state["latency"] / 0.2, 1.0),  # Latency (ms, capped at 200ms)
            min(raw_state["pods"] / self.k8s.max_pods, 1.0)  # Pods
        ], dtype=np.float32)
        return normalized

    def _calculate_reward(self, prev_state: np.ndarray, new_state: np.ndarray) -> float:
        """Calculate reward based on state transition."""
        cpu, memory, latency, pods = new_state
        reward = -latency * 0.6 - cpu * 0.3 - memory * 0.1  # Weighted metrics

        # Penalties
        if latency > 0.8:  # High latency (>160ms normalized)
            reward -= 5.0
        if cpu > 0.85:  # High CPU usage
            reward -= 2.0
        if pods > 0.8:  # Too many pods
            reward -= 1.0

        logger.debug("Reward calculated: %.2f (latency=%.2f, cpu=%.2f, pods=%.2f)", 
                     reward, latency, cpu, pods)
        return reward

    def _is_done(self) -> bool:
        """Determine if episode is complete."""
        # Example: End if latency too high or max pods reached
        done = self.state[2] > 1.0 or self.state[3] >= 1.0
        if done:
            logger.info("Episode done: state=%s", self.state)
        return done

    def render(self, mode: str = "human") -> None:
        """Render environment state (for debugging)."""
        logger.info("Current state: cpu=%.2f, memory=%.2f, latency=%.2f, pods=%.2f",
                    self.state[0], self.state[1], self.state[2], self.state[3])