import random
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any
import logging

import psutil
from .mock_kubernetes_api import MockKubernetesAPI
from .hybird_simulation import HybridTrafficSimulator

# Configure logging
logger = logging.getLogger(__name__)

class MicroK8sEnvSimulated(gym.Env):
    """Simulated MicroK8s environment for autoscaling with Gymnasium compatibility."""
    MIN_HISTORY_LENGTH = 5  # Minimum number of load history entries required
    DEFAULT_CPU = 0.4
    DEFAULT_MEMORY = 0.3
    DEFAULT_MEMORY_BYTES = DEFAULT_MEMORY * 500e6
    DEFAULT_NODES = 1

    def __init__(self, seed=None, enable_visualization=True):
        super().__init__()
        self.traffic_simulator = HybridTrafficSimulator(
            base_load=100,
            seed=seed,
            event_frequency=0.005,
            min_intensity=5,
            max_intensity=50,
            min_duration=10,
            max_duration=200
        )
        self.api = MockKubernetesAPI(traffic_simulator=self.traffic_simulator.get_load)
        self.enable_visualization = enable_visualization
        
        self.action_space = spaces.Discrete(3)  # (0: no-op, 1: scale-up, 2: scale-down)
        self.pods = 5  # Initial pod count
        self.cpu_util = self.DEFAULT_CPU
        self.memory_util = self.DEFAULT_MEMORY
        self.load_history = []  # Track load for trend calculations
        
        # Observation space: [cpu, memory, latency, swap, nodes, load_mean, load_gradient]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )

        self.state = None
        self.current_step = 0
        self.max_steps = 100

    def reset(self, seed: int = None, options: Dict = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self.api = MockKubernetesAPI(traffic_simulator=self.traffic_simulator.get_load)
        if seed is not None:
            self.api.seed(seed)
            self.traffic_simulator.rng = np.random.default_rng(seed)
        self.current_step = 0
        self.pods = 5
        self.cpu_util = self.DEFAULT_CPU
        self.memory_util = self.DEFAULT_MEMORY
        self.load_history = []
        self.load_history = [self.traffic_simulator.base_load] * self.MIN_HISTORY_LENGTH

        self.state = self._observe()
        logger.debug("Environment reset: step=%d, pods=%d", self.current_step, self.pods)
        return self.state, {}

    def step(self, action):
        self.current_step += 1

        # Get current load and update history
        try:
            # Warm-up period - collect data but don't take actions
            # Warm-up period - collect data but don't take actions
            if self.current_step < self.MIN_HISTORY_LENGTH:
                action = 0  # No-op during warm-up
                logger.debug("Warm-up period - forcing no-op action")

            current_load = self.traffic_simulator.get_load(self.current_step)
            self.load_history.append(current_load)
            logger.debug("Step %d: load=%f, history_length=%d", self.current_step, current_load, len(self.load_history))
            if len(self.load_history) > 50:  # Keep last 50 steps for trend
                self.load_history.pop(0)
        except Exception as e:
            logger.error("Error updating load history: %s", str(e))
            current_load = self.traffic_simulator.base_load

        # Get current state
        try:
            current_state = self.api.get_cluster_state()
            current_pods = current_state["pods"]
            desired_pods = current_state.get("desired_replicas", current_pods)
        except Exception as e:
            logger.error("Error getting cluster state: %s", str(e))
            current_state = {"pods": self.pods, "cpu": self.cpu_util, "memory": self.memory_util * 500e6, "latency": 0, "swap": 0, "nodes": 1}
            current_pods = self.pods
            desired_pods = current_pods

        # Apply action
        try:
            if action == 0:
                target_pods = current_pods
            elif action == 1:
                target_pods = min(current_pods + 1, self.api.max_pods)
            else:
                target_pods = max(current_pods - 1, 1)

            # Simulate scaling operation
            scaling_success = self.api.safe_scale("autoscaler", target_pods)
            next_state = self.api.get_cluster_state()
            obs = self._get_obs(next_state)
        except Exception as e:
            logger.error("Error applying action: %s", str(e))
            next_state = current_state
            obs = self._get_obs(next_state)
            scaling_success = False

        # Reward calculation
        try:
            reward = self._calculate_reward(next_state, action, scaling_success)
        except Exception as e:
            logger.error("Error calculating reward: %s", str(e))
            reward = -1.0

        # Metrics tracking
        scaling_lag = desired_pods - next_state["pods"]
        scaling_efficiency = 1 - (abs(scaling_lag) / self.api.max_pods)
        optimal_pods = max(1, min(
            int(next_state["cpu"] / 0.6),
            int((next_state["memory"] / 500e6) / 0.5)
        ))

        info = {
            "custom_metrics": {
                "cpu_utilization": next_state["cpu"],
                "memory_utilization": next_state["memory"] / 500e6,
                "swap_usage": next_state["swap"] / 200e6,
                "latency": next_state["latency"],
                "pod_count": next_state["pods"],
                "desired_pods": desired_pods,
                "optimal_pods": optimal_pods,
                "scaling_lag": scaling_lag,
                "scaling_efficiency": scaling_efficiency,
                "scaling_success": float(scaling_success),
                "scaling_action": action - 1,
                "load_mean": np.mean(self.load_history) / 5000 if self.load_history else 0,
                "load_gradient": self._calculate_load_gradient(),
                "reward_components": {
                    "cpu_reward": 1.0 if (0.4 <= next_state["cpu"] <= 0.8) else -abs(next_state["cpu"] - 0.6),
                    "memory_reward": 1.0 if (0.3 <= next_state["memory"] / 500e6 <= 0.7) else -abs(next_state["memory"] / 500e6 - 0.5),
                    "swap_penalty": -(next_state["swap"] / 200e6),
                    "scaling_penalty": -0.1 if action != 0 else 0,
                    "traffic_response": self._traffic_response_reward(action, next_state)
                }
            },
            "cluster_metrics": {
                "actual_cpu": psutil.cpu_percent() / 100,
                "actual_memory": psutil.virtual_memory().percent / 100,
                "actual_pods": self.api.get_current_pod_count(),
            }
        }

        # Termination conditions
        terminated = next_state["swap"] / 200e6 > 0.8 or next_state["cpu"] > 0.9 or next_state["memory"] / 500e6 > 0.9
        truncated = self.current_step >= self.max_steps
        if terminated:
            reward -= 5

        logger.debug("Step %d: reward=%f, terminated=%s, truncated=%s", self.current_step, reward, terminated, truncated)
        return obs, reward, terminated, truncated, info

    def _get_obs(self, state: Dict) -> np.ndarray:
        """Return observation with load trend and gradient using validated state."""
        state = self._validate_state(state)
        
        try:
            # Ensure we have enough history
            if len(self.load_history) < self.MIN_HISTORY_LENGTH:
                self.load_history = [self.traffic_simulator.base_load] * self.MIN_HISTORY_LENGTH
                
            load_mean = np.mean(self.load_history) / 5000
            load_gradient = self._calculate_load_gradient()
            
            obs = np.array([
                state["cpu"],
                state["memory"] / 500e6,
                state["latency"],
                state["swap"] / 200e6,
                state["nodes"] / self.api.max_nodes,
                load_mean,
                load_gradient
            ], dtype=np.float32)
            
            logger.debug("Observation: %s", obs)
            return obs
            
        except Exception as e:
            logger.error("Error getting observation: %s", str(e))
            return np.array([
                self.DEFAULT_CPU,
                self.DEFAULT_MEMORY,
                0.0,  # latency
                0.0,  # swap
                1.0,  # nodes
                0.0,  # load_mean
                0.0   # load_gradient
            ], dtype=np.float32)

    def _observe(self) -> np.ndarray:
        """Get current observation with proper error handling."""
        try:
            state = self.api.get_cluster_state()
            return self._get_obs(state)
        except Exception as e:
            logger.error("Error observing state: %s", str(e))
            return self._get_obs({})  # Will use default values

    def _calculate_load_gradient(self):
        """Calculate gradient of load history."""
        try:
            if len(self.load_history) < 2:
                return 0.0
            recent_loads = np.array(self.load_history[-2:])
            gradient = (recent_loads[-1] - recent_loads[-2]) / 5000
            logger.debug("Load gradient: %f", gradient)
            return gradient
        except Exception as e:
            logger.error("Error calculating load gradient: %s", str(e))
            return 0.0

    def _observe(self) -> np.ndarray:
        try:
            state = self.api.get_cluster_state()
            load_mean = np.mean(self.load_history) / 5000 if self.load_history else 0
            load_gradient = self._calculate_load_gradient()
            obs = np.array([
                state["cpu"],
                state["memory"] / 500e6,
                state["latency"],
                state["swap"] / 200e6,
                state["nodes"] / self.api.max_nodes,
                load_mean,
                load_gradient
            ], dtype=np.float32)
            logger.debug("Observation (observe): %s", obs)
            return obs
        except Exception as e:
            logger.error("Error observing state: %s", str(e))
            return np.array([0.4, 0.3, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)

    def _calculate_reward(self, state: Dict, action: int, success: bool) -> float:
        """Complete reward calculation incorporating pod count and utilization."""
        state = self._validate_state(state)
        cpu = state["cpu"]
        memory = state["memory"] / 500e6
        swap = state["swap"] / 200e6
        pods = state["pods"]
        max_pods = self.api.max_pods  # Assuming this is available
        
        # Ensure sufficient load history
        if len(self.load_history) < self.MIN_HISTORY_LENGTH:
            self.load_history = [self.traffic_simulator.base_load] * self.MIN_HISTORY_LENGTH
            
        # Calculate load characteristics
        load_mean = np.mean(self.load_history) / 5000
        load_gradient = self._calculate_load_gradient()
        
        # 1. Core Resource Efficiency (40%)
        cpu_reward = (1.0 if 0.4 <= cpu <= 0.8 else -abs(cpu - 0.6)) * 0.2
        mem_reward = (1.0 if 0.3 <= memory <= 0.7 else -abs(memory - 0.5)) * 0.2
        
        # 2. Pod Utilization Efficiency (30%)
        # Calculate ideal pod count based on current load
        load_based_pods = min(max_pods, max(1, round(load_mean * max_pods * 1.2)))  # 20% headroom
        
        # Current pod utilization efficiency
        pod_utilization = (0.5 - abs(pods - load_based_pods)/max_pods) * 0.3
        
        # 3. Scaling Effectiveness (20%)
        scaling_efficiency = 0.0
        if action != 0:  # Only evaluate if scaling action was taken
            # Reward moving toward ideal pod count
            direction = 1 if (load_based_pods > pods) else -1
            scaling_efficiency = (0.2 if (action == 1 and direction > 0) or 
                                (action == 2 and direction < 0) else -0.1)
        
        # 4. Load Pattern Response (10%)
        load_response = 0.0
        if abs(load_gradient) > 0.01:  # Significant trend
            correct_direction = (load_gradient > 0 and action == 1) or (load_gradient < 0 and action == 2)
            load_response = (0.1 if correct_direction else -0.05)
        
        # 5. Stability Penalties (10%)
        swap_penalty = -swap * 0.1
        scaling_cost = -0.02 if action != 0 else 0  # Small cost for any scaling action
        
        # Combine all components
        total_reward = (
            cpu_reward +
            mem_reward +
            pod_utilization +
            scaling_efficiency +
            load_response +
            swap_penalty +
            scaling_cost
        )
        
        # Clip final reward and ensure float type
        total_reward = float(np.clip(total_reward, -1.0, 1.0))
        
        logger.debug(
            "Pod metrics - Current: %d, Ideal: %d => Utilization: %.2f, ScalingEff: %.2f",
            pods, load_based_pods, pod_utilization, scaling_efficiency
        )
        
        return total_reward
    def _traffic_response_reward(self, action: int, state: Dict) -> float:
        """Reward for responding to traffic patterns."""
        try:
            load_gradient = self._calculate_load_gradient()
            cpu = state["cpu"]
            pods = state["pods"]
            reward = 0

            if load_gradient > 0.01 and action == 1:
                reward += 1.0 if cpu < 0.8 else -1.0
            elif load_gradient < -0.01 and action == 2:
                reward += 0.5 if pods > 1 else -0.5
            elif abs(load_gradient) < 0.005 and action == 0:
                reward += 0.2

            logger.debug("Traffic response reward: %f", reward)
            return reward
        except Exception as e:
            logger.error("Error calculating traffic response reward: %s", str(e))
            return 0.0

    def render(self, mode: str = 'human'):
        if mode == 'human':
            print(f"Step {self.current_step} | State: {self.state}")

    def close(self):
        pass

    def get_cluster_state(self):
        try:
            state = self.api.get_cluster_state()
            logger.debug("Cluster state: %s", state)
            return state
        except Exception as e:
            logger.error("Error getting cluster state: %s", str(e))
            return {"pods": self.pods, "cpu": self.cpu_util, "memory": self.memory_util * 500e6, "latency": 0, "swap": 0, "nodes": 1}
        
    def _validate_state(self, state: Dict) -> Dict:
        """Ensure the state dictionary contains all required keys with valid values."""
        if not isinstance(state, dict):
            state = {}
            
        return {
            "cpu": float(state.get("cpu", self.DEFAULT_CPU)),
            "memory": float(state.get("memory", self.DEFAULT_MEMORY_BYTES)),
            "latency": float(state.get("latency", 0.0)),
            "swap": float(state.get("swap", 0.0)),
            "nodes": int(state.get("nodes", self.DEFAULT_NODES)),
            "pods": int(state.get("pods", self.pods))
        }