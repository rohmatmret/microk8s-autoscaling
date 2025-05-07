import random
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any

import psutil
from .mock_kubernetes_api import MockKubernetesAPI

def event_traffic(step):
    base = 100 * (1 + 0.3 * np.sin(2 * np.pi * step / 1440))
    # Special events at steps 1000 and 2500
    if 1000 <= step < 1300:  # 5-minute flash sale
        return base * 25 + random.uniform(-50, 50)
    elif 2500 <= step < 3100:  # 10-minute holiday sale
        return base * 30 + random.uniform(-100, 100)
    return base + random.uniform(-20, 20)
class MicroK8sEnvSimulated(gym.Env):
    """Simulated MicroK8s environment for autoscaling with Gymnasium compatibility."""

    def __init__(self):
        super().__init__()
        self.api = MockKubernetesAPI(traffic_simulator=event_traffic)
        
        self.action_space = spaces.Discrete(3)  # (-1, 0, +1)
        self.pods = 5  # Initial pod count
        self.cpu_util = 0.4
        self.memory_util = 0.3
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

        # Get current state
        current_state = self.api.get_cluster_state()
        current_pods = current_state["pods"]
        desired_pods = current_state.get("desired_replicas", current_pods)  # From MockAPI

        # Apply action (0=no-op, 1=scale up, 2=scale down)
        if action == 0:
            target_pods = current_pods
        elif action == 1:
            target_pods = min(current_pods + 1, self.api.max_pods)
        else:
            target_pods = max(current_pods - 1, 1)  # Never scale below 1 pod

        # Simulate scaling operation and track success
        scaling_success = self.api.safe_scale("autoscaler", target_pods)
        next_state = self.api.get_cluster_state()
        obs = self._get_obs(next_state)

        # --- Reward Engineering ---
        reward = 0
        cpu = next_state["cpu"]
        memory = next_state["memory"] / 500e6  # Normalized to [0,1]
        swap = next_state["swap"] / 200e6     # Normalized to [0,1]
        latency = next_state.get("latency", 0.0)
        actual_pods = next_state["pods"]

        # 1. Target range bonuses (CPU/Memory)
        reward += 1.0 if (0.4 <= cpu <= 0.8) else -abs(cpu - 0.6)
        reward += 1.0 if (0.3 <= memory <= 0.7) else -abs(memory - 0.5)

        # 2. Penalties (Swap, Scaling Issues)
        reward -= swap * 2  # Heavy penalty for swap usage
        
        # Penalize scaling failures
        if not scaling_success:
            reward -= 1.0
            
        # Small penalty for scaling actions to encourage stability
        if action != 0:
            reward -= 0.1

        # 3. Dynamic pod count optimization (replaces ideal_pod_count)
        optimal_pods = max(1, min(
            int(cpu / 0.6),  # Estimate based on CPU target
            int(memory / 0.5)  # Estimate based on memory target
        ))
        reward -= 0.05 * abs(actual_pods - optimal_pods)

        # --- Metrics Tracking ---
        scaling_lag = desired_pods - actual_pods
        scaling_efficiency = 1 - (abs(scaling_lag) / self.api.max_pods)

        # Simulate resource utilization
        self.cpu_util = 0.4 + (self.api.traffic_simulator(self.current_step) / 1000)
        self.memory_util = 0.3 + (self.cpu_util - 0.4)

        # Reward function
        cpu_target = 0.6  # Midpoint of 0.4–0.8
        memory_target = 0.5  # Midpoint of 0.3–0.7
        pod_target = 5
        reward = (
            -abs(self.cpu_util - cpu_target) * 10  # Scale to emphasize CPU
            - abs(self.memory_util - memory_target) * 5
            - abs(self.pods - pod_target) * 2  # Encourage ideal pod count
        )

        done = False
        truncated = False

        info = {
            "custom_metrics": {
                # Resource metrics
                "cpu_utilization": cpu,
                "memory_utilization": memory,
                "swap_usage": swap,
                "latency": latency,
                
                # Scaling metrics
                "pod_count": actual_pods,
                "desired_pods": desired_pods,
                "optimal_pods": optimal_pods,
                "scaling_lag": scaling_lag,
                "scaling_efficiency": scaling_efficiency,
                "scaling_success": float(scaling_success),
                "scaling_action": action - 1,  # -1, 0, 1
                
                # Reward components
                "reward_components": {
                    "cpu_reward": 1.0 if (0.4 <= cpu <= 0.8) else -abs(cpu - 0.6),
                    "memory_reward": 1.0 if (0.3 <= memory <= 0.7) else -abs(memory - 0.5),
                    "swap_penalty": -swap * 2,
                    "scaling_penalty": -0.1 if action != 0 else 0,
                    "optimal_pods_penalty": -0.05 * abs(actual_pods - optimal_pods)
                }
            },
            "cluster_metrics": {
                "actual_cpu": psutil.cpu_percent() / 100,  # Normalized
                "actual_memory": psutil.virtual_memory().percent / 100,
                "actual_pods": self.api.get_current_pod_count(),
            }
        }

        # --- Termination Conditions ---
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        # Early termination if system is unstable
        if swap > 0.8 or cpu > 0.9 or memory > 0.9:  # Dangerous thresholds
            terminated = True
            reward -= 5  # Large penalty for instability

        return obs, reward, terminated, truncated, info
    
    def _get_obs(self, state):
        return np.array([
            state["cpu"],
            state["memory"] / 500e6,
            state["latency"],
            state["swap"] / 200e6,
            state["nodes"],
            # state["pods"]
        ], dtype=np.float32)
        

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

    def get_cluster_state(self):
        return self.api.get_cluster_state()
