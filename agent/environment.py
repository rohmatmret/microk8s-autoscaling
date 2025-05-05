# agent/environment.py
import os
import time
import logging
import numpy as np
from typing import Tuple, Dict
import gymnasium as gym
from gymnasium import spaces
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

    # def reset(self) -> np.ndarray:
    #     """Reset environment to initial state."""
    #     try:
    #         # Set initial pod count to 2
    #         self.k8s.safe_scale(self.deployment_name, 2)
    #         time.sleep(self.scaling_delay)
    #         self.state = self._get_normalized_state()
    #         logger.info("Environment reset: state=%s", self.state)
    #         return self.state
    #     except KubernetesAPIError as e:
    #         logger.error("Reset failed: %s", e)
    #         raise
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        try:
            # Set initial pod count to 2
            self.k8s.safe_scale(self.deployment_name, 2)
            time.sleep(self.scaling_delay)
            self.state = self._get_normalized_state()
            logger.info("Environment reset: state=%s", self.state)
            return self.state, {}  # Gymnasium expects (obs, info)
        except KubernetesAPIError as e:
            logger.error("Reset failed: %s", e)
            raise

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute action and return (obs, reward, terminated, truncated, info)."""
        try:
            prev_state = self.state
            current_replicas = self.k8s._get_current_replicas(self.deployment_name)

            # Execute action
            if action == 0:  # Scale up
                self.k8s.safe_scale(self.deployment_name, current_replicas + 1)
            elif action == 1:  # Scale down
                self.k8s.safe_scale(self.deployment_name, current_replicas - 1)

            time.sleep(self.scaling_delay)
            self.state = self._get_normalized_state()

            reward = self._calculate_reward(prev_state, self.state)
            terminated = self._is_done()  # True if episode ends
            truncated = False  # SB3 rarely uses this
            info = {"action": action, "replicas": current_replicas}

            logger.debug("Step: action=%d, reward=%.2f, state=%s", action, reward, self.state)
            return self.state, reward, terminated, truncated, info
        except KubernetesAPIError as e:
            logger.error("Step failed: %s", e)
            return self.state, -10.0, False, False, {"error": str(e)}

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
        cpu, memory, latency, swap, pods = new_state
        reward = (
            -latency * 0.5     # User experience utama
            -cpu * 0.2         # Beban sistem
            -memory * 0.15
            -swap * 0.1        # Swap = tekanan RAM
            -pods * 0.05       # Biaya infrastruktur
        )

        # --- Penalti tambahan untuk outlier atau kondisi tidak optimal ---
        if latency > 0.8:
            reward -= 2.0
        if cpu > 0.9:
            reward -= 1.5
        if swap > 0.5:
            reward -= 1.0
        if pods > 0.85:
            reward -= 1.0


        # --- Penalti untuk scaling yang tidak efektif ---
        if prev_state[4] < new_state[4]:  # jumlah pods naik
            if abs(prev_state[2] - new_state[2]) < 0.05:  # latency tidak turun signifikan
                reward -= 1.0
                
        if latency < 0.3 and cpu < 0.5:
            reward += 1.0
                
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