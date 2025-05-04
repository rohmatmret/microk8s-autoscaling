# agent/environment_simulated.py
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
        
        # Define action and observation spaces
        # Action space: scaling decision (-1: scale down, 0: no change, 1: scale up)
        self.action_space = spaces.Discrete(3)  # 3 possible actions
        
        # Observation space: [pods, nodes, cpu, memory, latency]
        self.observation_space = spaces.Box(
            low=np.array([1, 1, 0.0, 0.0, 0.0]),
            high=np.array([10, 5, 1.0, 500e6, 1.0]),
            dtype=np.float32
        )
        
        # Initialize state
        self.state = None
        self.current_step = 0
        self.max_steps = 100  # Episode length
        
    def reset(self, seed: int = None, options: Dict = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Reset API and step counter
        self.api = MockKubernetesAPI()
        self.current_step = 0
        
        # Get initial state
        cluster_state = self.api.get_cluster_state()
        self.state = np.array([
            cluster_state["pods"],
            cluster_state["nodes"],
            cluster_state["cpu"],
            cluster_state["memory"],
            cluster_state["latency"]
        ], dtype=np.float32)
        
        return self.state, {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment.
        
        Args:
            action: The action to take (0: -1, 1: 0, 2: +1)
            
        Returns:
            observation: The new state
            reward: The reward for this step
            terminated: Whether the episode is done
            truncated: Whether the episode was truncated
            info: Additional information
        """
        # Convert action to scaling decision (-1, 0, +1)
        scaling = action - 1
        
        # Get current state
        current_pods = self.state[0]
        
        # Calculate new desired replicas
        desired_replicas = current_pods + scaling
        success = self.api.safe_scale("deployment", desired_replicas)
        
        # Get new state
        cluster_state = self.api.get_cluster_state()
        self.state = np.array([
            cluster_state["pods"],
            cluster_state["nodes"],
            cluster_state["cpu"],
            cluster_state["memory"],
            cluster_state["latency"]
        ], dtype=np.float32)
        
        # Calculate reward (customize this based on your needs)
        reward = self._calculate_reward(cluster_state)
        
        # Update step counter
        self.current_step += 1
        
        # Check termination conditions
        terminated = False  # Add your termination conditions
        truncated = self.current_step >= self.max_steps
        
        # Additional info
        info = {
            "pods": cluster_state["pods"],
            "nodes": cluster_state["nodes"],
            "scaling_success": success
        }
        
        info.update({
            "custom_metrics": {
                "cpu_utilization": cluster_state["cpu"],
                "memory_utilization": cluster_state["memory"] / 500e6,
                "pod_count": cluster_state["pods"],
                "scaling_action": action - 1,  # -1, 0, or +1
                "node_count": cluster_state["nodes"],
                "latency": cluster_state["latency"]
            }
        })
        
        return self.state, reward, terminated, truncated, info
    
    def _calculate_reward(self, state: Dict) -> float:
        """Improved reward function"""
        cpu_util = state["cpu"]
        memory_util = state["memory"] / 500e6
        pod_count = state["pods"]
        
        # Reward for being in ideal CPU range (0.4-0.8)
        cpu_reward = 1.0 if 0.4 <= cpu_util <= 0.8 else -1.0
        
        # Reward for being in ideal memory range (0.3-0.7)
        memory_reward = 1.0 if 0.3 <= memory_util <= 0.7 else -0.5
        
        # Scaling reward (encourage moderate pod counts)
        pod_reward = -0.2 * abs(pod_count - 5)
        
        # Combine rewards with weights
        total_reward = (
            0.4 * cpu_reward +
            0.3 * memory_reward +
            0.3 * pod_reward
        )
        return float(total_reward)
    
    def render(self, mode: str = 'human'):
        """Render the environment (optional)."""
        if mode == 'human':
            print(f"Current state: {self.state}")
    
    def close(self):
        """Clean up resources (optional)."""
        pass