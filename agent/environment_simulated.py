"""Simulated MicroK8s environment for reinforcement learning-based autoscaling.

This module implements a Gymnasium-compatible environment that simulates a MicroK8s cluster
for training and evaluating autoscaling policies using reinforcement learning.
"""

import logging
from typing import Tuple, Dict, List
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque
import psutil
from .mock_kubernetes_api import MockKubernetesAPI
from .traffic_simulation import TrafficSimulator

# Configure logging
logger = logging.getLogger(__name__)

class MicroK8sEnvSimulated(gym.Env):
    """Simulated MicroK8s environment for autoscaling with Gymnasium compatibility."""
    MIN_HISTORY_LENGTH = 5  # Minimum number of load history entries required
    MAX_HISTORY_LENGTH = 1000  # Maximum history size
    DEFAULT_CPU = 0.3
    DEFAULT_MEMORY = 0.3
    DEFAULT_MEMORY_BYTES = DEFAULT_MEMORY * 500e6
    DEFAULT_NODES = 1
    LOAD_NORMALIZATION = 5000  # Normalization factor for load values

    def __init__(self, seed=None, enable_visualization=True):
        super().__init__()

        # Store seed for reproducibility
        self.seed = seed if seed is not None else 42

        # Initialize traffic simulator with seed for consistent patterns
        self.traffic_simulator = TrafficSimulator(
            base_load=500,  # 5x increase: 100 -> 500 RPS for production-level load
            max_spike=150,  # 5x increase: 30 -> 150 RPS for realistic spikes
            daily_amplitude=0.3,
            spike_probability=0.005,
            min_spike_duration=10,
            max_spike_duration=30,
            min_load=50,    # 5x increase: 10 -> 50 RPS for realistic minimum
            history_size=self.MAX_HISTORY_LENGTH,
            seed=self.seed  # Use consistent seed for reproducible traffic
        )
        self.api = MockKubernetesAPI(traffic_simulator=self.traffic_simulator.get_load)
        self.enable_visualization = enable_visualization
        
        self.action_space = spaces.Discrete(3)  # (0: no-op, 1: scale-up, 2: scale-down)
        self.pods = 1  # Initial pod count
        self.cpu_util = self.DEFAULT_CPU
        self.memory_util = self.DEFAULT_MEMORY
        
        # Use deque for efficient history management
        self.load_history = deque(maxlen=self.MAX_HISTORY_LENGTH)
        self.load_history.extend([self.traffic_simulator.base_load] * self.MIN_HISTORY_LENGTH)
        
        # Observation space: [cpu, memory, latency, swap, nodes, load_mean, throughput]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            dtype=np.float64
        )

        self.state = None
        self.current_step = 0
        self.max_steps = 100
        self._setup_traffic_events()

    def _setup_traffic_events(self):
        """Setup realistic traffic events."""
        # Morning peak
        self.traffic_simulator.add_event(
            start=360,  # 6 AM
            duration=120,
            intensity=2.0,
            event_type="scheduled"
        )
        
        # Afternoon peak
        self.traffic_simulator.add_event(
            start=720,  # 12 PM
            duration=180,
            intensity=1.8,
            event_type="scheduled"
        )
        
        # Evening peak
        self.traffic_simulator.add_event(
            start=1080,  # 6 PM
            duration=240,
            intensity=2.2,
            event_type="scheduled"
        )
        
        # Random burst events
        for _ in range(3):
            start = np.random.randint(100, 2000)
            self.traffic_simulator.add_event(
                start=start,
                duration=np.random.randint(30, 60),
                intensity=np.random.uniform(1.5, 2.5),
                event_type="burst"
            )

    def reset(self, seed: int = None, options: Dict = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self.api = MockKubernetesAPI(traffic_simulator=self.traffic_simulator.get_load)
        if seed is not None:
            self.api.seed(seed)
            self.traffic_simulator.rng = np.random.default_rng(seed)
        self.current_step = 0
        self.pods = 1
        self.cpu_util = self.DEFAULT_CPU
        self.memory_util = self.DEFAULT_MEMORY
        
        # Reset load history
        self.load_history.clear()
        self.load_history.extend([self.traffic_simulator.base_load] * self.MIN_HISTORY_LENGTH)
        
        # Reset traffic events
        self._setup_traffic_events()

        self.state = self._observe()
        logger.debug("Environment reset: step=%d, pods=%d", self.current_step, self.pods)
        return self.state, {}

    def _calculate_ideal_pods(self, state: Dict, load_mean: float) -> int:
        """Calculate the ideal number of pods based on current state and load."""
        max_pods = self.api.max_pods
        cpu = state["cpu"]
        memory = state["memory"] / 500e6  # Convert bytes to normalized value
        
        # Load-based calculation with dynamic headroom
        load_headroom = 1.2  # Base 20% headroom
        load_gradient = self._calculate_load_gradient()
        if load_gradient > 0.01:  # If load is increasing
            load_headroom += 0.1  # Additional 10% headroom
        load_based_pods = min(max_pods, max(1, round(load_mean * max_pods * load_headroom)))
        
        # Resource-based calculation with dynamic thresholds
        cpu_threshold = 0.6  # Base threshold
        memory_threshold = 0.6
        if state["latency"] > 0.7:  # If latency is high
            cpu_threshold = 0.5  # Be more conservative
            memory_threshold = 0.5
        
        resource_based_pods = max(1, min(max_pods, 
            round(max(cpu / cpu_threshold, memory / memory_threshold))))
        
        # Combine with consideration for current pod count
        current_pods = state["pods"]
        if abs(load_based_pods - current_pods) < abs(resource_based_pods - current_pods):
            ideal_pods = load_based_pods
        else:
            ideal_pods = resource_based_pods
            
        # Add stability factor to prevent flapping
        stability_window = 5  # Consider last 5 steps
        if len(self.load_history) >= stability_window:
            recent_loads = list(self.load_history)[-stability_window:]
            load_std = np.std(recent_loads)
            if load_std < (0.1 * self.LOAD_NORMALIZATION):  # Stable load
                # Prefer to keep current pods if within reasonable range
                if (0.8 * ideal_pods <= current_pods <= 1.2 * ideal_pods):
                    ideal_pods = current_pods
        
        return min(max_pods, max(1, ideal_pods))

    def step(self, action):
        self.current_step += 1

        # Get current load with error handling
        try:
            current_load = self.traffic_simulator.get_load(self.current_step)
            self._update_load_history(current_load)
        except Exception as e:
            logger.error("Error getting load: %s", str(e))
            current_load = self.traffic_simulator.base_load
            self._update_load_history(current_load)

        # Get current state
        try:
            current_state = self.api.get_cluster_state()
            # Ensure swap is present in current state
            if "swap" not in current_state:
                current_state["swap"] = 0.0
            current_pods = current_state["pods"]
            desired_pods = current_state.get("desired_replicas", current_pods)
        except Exception as e:
            current_state = {
                "pods": self.pods,
                "cpu": self.cpu_util,
                "memory": self.memory_util * 500e6,
                "latency": 0.0,
                "swap": 0.0,
                "nodes": 1
            }
            logger.error("Error getting cluster state in step function: %s", str(e))
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
            # Ensure swap is present in next state
            if "swap" not in next_state:
                next_state["swap"] = 0.0
            obs = self._get_obs(next_state)
        except Exception as e:
            logger.error("Error applying action: %s", str(e))
            next_state = current_state
            obs = self._get_obs(next_state)
            scaling_success = False

        # Calculate ideal pods
        load_mean = np.mean(self.load_history) / 5000 if self.load_history else 0
        ideal_pods = self._calculate_ideal_pods(next_state, load_mean)

        # Reward calculation
        try:
            reward = self._calculate_reward(next_state, action, scaling_success, ideal_pods)
        except Exception as e:
            logger.error("Error calculating reward: %s", str(e))
            reward = -1.0

        # Metrics tracking
        scaling_lag = desired_pods - next_state["pods"]
        scaling_efficiency = 1 - (abs(scaling_lag) / self.api.max_pods)

        info = {
            "custom_metrics": {
                "cpu_utilization": next_state["cpu"],
                "memory_utilization": next_state["memory"] / 500e6,
                "swap_usage": next_state.get("swap", 0.0) / 200e6,  # Use get() with default value
                "latency": next_state["latency"],
                "pod_count": next_state["pods"],
                "desired_pods": desired_pods,
                "optimal_pods": ideal_pods,
                "scaling_lag": scaling_lag,
                "scaling_efficiency": scaling_efficiency,
                "scaling_success": float(scaling_success),
                "scaling_action": action - 1,
                "load_mean": load_mean,
                "load_gradient": self._calculate_load_gradient(),
                "ideal_pods_calculation": {
                    "load_based": min(self.api.max_pods, max(1, round(load_mean * self.api.max_pods * 1.2))),
                    "resource_based": max(1, min(self.api.max_pods, 
                        round(max(next_state["cpu"] / 0.6, (next_state["memory"] / 500e6) / 0.6)))),
                    "final_ideal": ideal_pods
                },
                "reward_components": {
                    "cpu_reward": 1.0 if (0.4 <= next_state["cpu"] <= 0.8) else -abs(next_state["cpu"] - 0.6),
                    "memory_reward": 1.0 if (0.3 <= next_state["memory"] / 500e6 <= 0.7) else -abs(next_state["memory"] / 500e6 - 0.5),
                    "swap_penalty": -(next_state.get("swap", 0.0) / 200e6),  # Use get() with default value
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
        terminated = next_state.get("swap", 0.0) / 200e6 > 0.8 or next_state["cpu"] > 0.9 or next_state["memory"] / 500e6 > 0.9
        truncated = self.current_step >= self.max_steps
        if terminated:
            reward -= 5

        logger.debug("Step %d: reward=%f, terminated=%s, truncated=%s", self.current_step, reward, terminated, truncated)
        return obs, reward, terminated, truncated, info

    def _calculate_reward(self, state: Dict, action: int, success: bool, ideal_pods: int) -> float:
        """Complete reward calculation incorporating pod count and utilization."""
        state = self._validate_state(state)
        cpu = state["cpu"]
        memory = state["memory"] / 500e6
        swap = state["swap"] / 200e6
        pods = state["pods"]
        max_pods = self.api.max_pods
        
        pod_diff = ideal_pods - pods
        
        # Base reward starts negative to encourage action
        total_reward = -0.1
        
        # 1. Resource Utilization (40%)
        # CPU utilization reward
        if cpu < 0.3:  # Low utilization
            if action == 2 and pods > 1:  # Reward scale down
                total_reward += 0.4
            elif action == 0 and pods > 1:  # Penalize no action
                total_reward -= 0.2
        elif 0.3 <= cpu <= 0.7:  # Optimal range
            total_reward += 0.3
        elif 0.7 < cpu <= 0.85:  # Warning zone
            if action == 1:  # Reward scale up
                total_reward += 0.3
            elif action == 0:  # Penalize no action
                total_reward -= 0.2
        else:  # High utilization
            if action == 1:  # Strongly reward scale up
                total_reward += 0.4
            else:  # Strongly penalize not scaling up
                total_reward -= 0.4
                
        # Memory utilization reward
        if memory < 0.3 and pods > 1:
            if action == 2:  # Reward scale down
                total_reward += 0.3
            elif action == 0:  # Small penalty for no action
                total_reward -= 0.1
        elif memory > 0.8:
            if action == 1:  # Reward scale up
                total_reward += 0.3
            elif action == 0:  # Penalize no action
                total_reward -= 0.2
                
        # 2. Pod Alignment Reward (30%)
        if pods == ideal_pods:
            total_reward += 0.3
        elif pod_diff > 0:  # Need more pods
            if action == 1:  # Reward scale up
                total_reward += 0.3
            elif action == 0:  # Penalize no action
                total_reward -= 0.2
        else:  # Too many pods
            if cpu < 0.4 and memory < 0.4:  # Only if resources are low
                if action == 2:  # Reward scale down
                    total_reward += 0.3
                elif action == 0 and pods > 1:  # Penalize no action
                    total_reward -= 0.2
                    
        # 3. Load Response (30%)
        load_gradient = self._calculate_load_gradient()
        if abs(load_gradient) > 0.008:
            if load_gradient > 0.008:  # Load increasing
                if cpu > 0.6 or memory > 0.6:  # Only if resources are getting high
                    if action == 1:
                        total_reward += 0.3
                    elif action == 0:
                        total_reward -= 0.2
            elif load_gradient < -0.008:  # Load decreasing
                if cpu < 0.4 and memory < 0.4 and pods > 1:  # Only if resources are low
                    if action == 2:
                        total_reward += 0.3
                    elif action == 0:
                        total_reward -= 0.1
        
        # Critical condition penalties
        if cpu > 0.9 or memory > 0.9:
            total_reward -= 0.5
            if action != 1:  # Extra penalty for not scaling up
                total_reward -= 0.3
        
        # Swap penalty
        if swap > 0.1:
            total_reward -= swap * 0.4
        
        # Scaling cost - smaller for scale down than up
        if action == 1:  # Scale up
            total_reward -= 0.05
        elif action == 2:  # Scale down
            total_reward -= 0.02  # Lower penalty for scaling down
        
        # Clip final reward
        total_reward = float(np.clip(total_reward, -1.0, 1.0))
        
        logger.debug(
            "Step info - CPU: %.2f, Memory: %.2f, Pods: %d, Ideal: %d, Action: %d, Reward: %.2f",
            cpu, memory, pods, ideal_pods, action, total_reward
        )
        
        return total_reward

    def _get_obs(self, state: Dict) -> np.ndarray:
        """Return observation with load trend and gradient using validated state."""
        state = self._validate_state(state)
        
        try:
            # Ensure we have enough history
            if len(self.load_history) < self.MIN_HISTORY_LENGTH:
                self.load_history = deque(maxlen=self.MAX_HISTORY_LENGTH)
                self.load_history.extend([self.traffic_simulator.base_load] * self.MIN_HISTORY_LENGTH)
                
            load_mean = np.mean(self.load_history) / 5000

            # Calculate throughput (requests per second)
            throughput = state.get("throughput", 0.0)
            normalized_throughput = min(throughput / 1000.0, 1.0)  # Normalize assuming max 1000 RPS

            obs = np.array([
                state["cpu"],
                state["memory"] / 500e6,
                state["latency"],
                state["swap"] / 200e6,
                state["nodes"] / self.api.max_nodes,
                load_mean,
                normalized_throughput
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
                0.0   # throughput
            ], dtype=np.float32)

    def _observe(self) -> np.ndarray:
        """Get current observation with proper error handling."""
        try:
            state = self.api.get_cluster_state()
            return self._get_obs(state)
        except Exception as e:
            logger.error("Error observing state: %s", str(e))
            return self._get_obs({})  # Will use default values

    def _calculate_load_gradient(self) -> float:
        """Calculate smoothed load gradient using multiple points."""
        try:
            if len(self.load_history) < self.MIN_HISTORY_LENGTH:
                return 0.0
                
            # Use last 5 points for gradient calculation
            recent_loads = np.array(list(self.load_history)[-5:])
            x = np.arange(len(recent_loads))
            
            # Calculate linear regression
            slope, _ = np.polyfit(x, recent_loads, 1)
            
            # Normalize gradient
            normalized_gradient = slope / self.LOAD_NORMALIZATION
            
            # Clip to [-1, 1] range
            return np.clip(normalized_gradient, -1.0, 1.0)
            
        except Exception as e:
            logger.error("Error calculating load gradient: %s", str(e))
            return 0.0

    def _update_load_history(self, current_load: float):
        """Update load history with proper error handling."""
        try:
            self.load_history.append(current_load)
        except Exception as e:
            logger.error("Error updating load history: %s", str(e))
            # Ensure minimum history length
            if len(self.load_history) < self.MIN_HISTORY_LENGTH:
                self.load_history.extend([self.traffic_simulator.base_load] * 
                                      (self.MIN_HISTORY_LENGTH - len(self.load_history)))

    def _traffic_response_reward(self, action: int, state: Dict) -> float:
        """Reward for responding to traffic patterns."""
        try:
            load_gradient = self._calculate_load_gradient()
            cpu = state["cpu"]
            pods = state["pods"]
            reward = 0

            # More balanced traffic response rewards
            if load_gradient > 0.01:  # Increasing load
                if action == 1:  # Scale up
                    reward += 0.5 if cpu < 0.8 else -0.3
                elif action == 0 and cpu < 0.4:  # No change when utilization is low
                    reward += 0.2
            elif load_gradient < -0.01:  # Decreasing load
                if action == 2:  # Scale down
                    reward += 0.3 if cpu < 0.3 else -0.3  # Reduced reward for scale-down
                elif action == 0 and cpu < 0.3:  # No change when utilization is low
                    reward += 0.2
            elif abs(load_gradient) < 0.005:  # Stable load
                if action == 0:  # No change
                    reward += 0.3
                elif action == 2 and cpu < 0.3:  # Scale down when utilization is low
                    reward += 0.1  # Reduced reward for scale-down

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
            # Ensure swap is present in the state
            if "swap" not in state:
                state["swap"] = 0.0
            logger.debug("Cluster state: %s", state)
            return state
        except Exception as e:
            logger.error("Error getting cluster state: %s", str(e))
            return {
                "pods": self.pods,
                "cpu": self.cpu_util,
                "memory": self.memory_util * 500e6,
                "latency": 0.0,
                "swap": 0.0,  # Ensure swap is present in error case
                "nodes": 1
            }
        
    def _validate_state(self, state: Dict) -> Dict:
        """Ensure the state dictionary contains all required keys with valid values."""
        if not isinstance(state, dict):
            state = {}
            
        return {
            "cpu": float(state.get("cpu", self.DEFAULT_CPU)),
            "memory": float(state.get("memory", self.DEFAULT_MEMORY_BYTES)),
            "latency": float(state.get("latency", 0.0)),
            "swap": float(state.get("swap", 0.0)),  # Ensure swap is always present
            "nodes": int(state.get("nodes", self.DEFAULT_NODES)),
            "pods": int(state.get("pods", self.pods))
        }