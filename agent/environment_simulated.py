import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any
from .mock_kubernetes_api import MockKubernetesAPI

class MicroK8sEnvSimulated(gym.Env):
    """Simulated MicroK8s environment for autoscaling with Gymnasium compatibility."""

    def __init__(self):
        super().__init__()
        self.api = MockKubernetesAPI()
        
        self.action_space = spaces.Discrete(3)  # (-1, 0, +1)
        
        # self.observation_space = spaces.Box(
        #     low=np.array([1, 1, 0.0, 0.0, 0.0]),
        #     high=np.array([10, 5, 1.0, 500e6, 1.0]),
        #     dtype=np.float32
        # )
        self.observation_space = spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)


        self.state = None
        self.current_step = 0
        self.max_steps = 100

    def reset(self, seed: int = None, options: Dict = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self.api = MockKubernetesAPI()
        if seed is not None:
            self.api.seed(seed)
        self.current_step = 0
        self.state = self._observe()
        return self.state, {}

    def step(self, action):
        self.current_step += 1

        current_state = self.api.get_cluster_state()
        current_pods = current_state["pods"]

        if action == 0:
            target_pods = current_pods
        elif action == 1:
            target_pods = current_pods + 1
        else:
            target_pods = current_pods - 1

        self.api.safe_scale("autoscaler", target_pods)
        next_state = self.api.get_cluster_state()
        obs = self._get_obs(next_state)

        # Reward shaping (CPU & memory in target range, penalize high swap)
        cpu = next_state["cpu"]
        memory = next_state["memory"]
        swap = next_state["swap"]  # in bytes

        reward = 0

        # CPU reward
        if 0.4 <= cpu <= 0.8:
            reward += 1
        else:
            reward -= abs(cpu - 0.6)

        # Memory reward
        normalized_memory = memory / 500e6  # max memory
        if 0.3 <= normalized_memory <= 0.7:
            reward += 1
        else:
            reward -= abs(normalized_memory - 0.5)

        # Swap penalty
        swap_penalty = swap / 200e6  # normalized to [0, 1]
        reward -= swap_penalty * 2  # stronger penalty if swap > 100MB

        terminated = False
        truncated = self.current_step >= self.max_steps
        info = {
            "custom_metrics": {
                "cpu_utilization": cpu,
                "memory_utilization": normalized_memory,
                "pod_count": next_state["pods"],
                "swap_usage": swap,
                "scaling_action": action - 1  # -1, 0, 1
            }
        }

        return obs, reward, terminated, truncated, info

    def _get_obs(self, state):
        return np.array([
            state["cpu"],
            state["memory"] / 500e6,
            state["latency"],
            state["swap"] / 200e6,
            state["nodes"]
            
        ], dtype=np.float32)
        
    # def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
    #     scaling = action - 1  # Map to (-1, 0, +1)
    #     current_pods = int(self.state[0])
    #     desired_replicas = current_pods + scaling

    #     # Attempt to scale
    #     success = self.api.safe_scale("deployment", desired_replicas)

    #     # Observe new state (after internal update with buffer logic)
    #     self.state = self._observe()

    #     reward = self._calculate_reward(self.state, scaling, success)

    #     self.current_step += 1
    #     terminated = False
    #     truncated = self.current_step >= self.max_steps

    #     info = {
    #         "pods": self.state[0],
    #         "nodes": self.state[1],
    #         "scaling_success": success,
    #         "custom_metrics": {
    #             "cpu_utilization": self.state[2],
    #             "memory_utilization": self.state[3] / 500e6,
    #             "latency": self.state[4],
    #             "scaling_action": scaling
    #         }
    #     }

    #     return self.state, reward, terminated, truncated, info

    def _observe(self) -> np.ndarray:
        state = self.api.get_cluster_state()
        return np.array([
            state["pods"],
            state["nodes"],
            state["cpu"],
            state["memory"],
            state["latency"]
        ], dtype=np.float32)

    def _calculate_reward(self, state: np.ndarray, scaling: int, success: bool) -> float:
        cpu_util = state[2]
        memory_util = state[3] / 500e6
        pod_count = state[0]

        cpu_reward = 1.0 if 0.4 <= cpu_util <= 0.8 else -1.0
        memory_reward = 1.0 if 0.3 <= memory_util <= 0.7 else -0.5
        pod_reward = -0.2 * abs(pod_count - 5)

        # Penalize failed scaling actions
        scaling_penalty = -0.5 if scaling != 0 and not success else 0.0

        total_reward = (
            0.4 * cpu_reward +
            0.3 * memory_reward +
            0.3 * pod_reward +
            scaling_penalty
        )
        return float(total_reward)

    def render(self, mode: str = 'human'):
        if mode == 'human':
            print(f"Step {self.current_step} | State: {self.state}")

    def close(self):
        pass
