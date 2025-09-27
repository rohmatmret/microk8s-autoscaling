import os
import time
import logging
import numpy as np
from typing import Tuple, Dict, Any, Optional
import gymnasium as gym
from gymnasium import spaces
from kubernetes.client.rest import ApiException
from agent.kubernetes_api import KubernetesAPI, KubernetesAPIError
from agent.metrics_callback import AutoscalingMetricsCallback
from prometheus_api_client import PrometheusConnect
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("hybrid_environment.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class HybridMicroK8sEnv(gym.Env):
    """Hybrid environment for MicroK8s autoscaling with DQN-PPO integration."""

    def __init__(
        self,
        deployment_name: str = os.getenv("DEPLOYMENT_NAME", "nginx-deployment"),
        namespace: str = os.getenv("K8S_NAMESPACE", "default"),
        max_pods: int = 10,
        scaling_delay: int = 10,
        metrics_sync_interval: int = 10,
        prometheus_url: str = "http://localhost:9090"
    ):
        super(HybridMicroK8sEnv, self).__init__()
        
        # Kubernetes API
        self.k8s_api = KubernetesAPI(max_pods=max_pods, namespace=namespace)
        self.deployment_name = deployment_name
        self.namespace = namespace
        self.scaling_delay = scaling_delay
        
        # Metrics integration
        self.metrics_sync_interval = metrics_sync_interval
        self.last_metrics_sync = 0
        self.metrics_callback = AutoscalingMetricsCallback()
        
        # Prometheus integration
        try:
            self.prometheus = PrometheusConnect(url=prometheus_url, disable_ssl=True)
            self.prometheus_available = True
            logger.info("Prometheus integration enabled")
        except Exception as e:
            logger.warning(f"Prometheus not available: {e}")
            self.prometheus_available = False
        
        # State space: [cpu, memory, latency, pods, queue_length, throughput]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            dtype=np.float64
        )

        # Action space: [scale up, scale down, no-op]
        self.action_space = spaces.Discrete(3)

        # Environment state
        self.state = None
        self.episode_step = 0
        self.max_episode_steps = 1000
        
        # Metrics tracking
        self.episode_metrics = {
            'cost': [],
            'response_time': [],
            'queue_length': [],
            'throughput': [],
            'cpu_utilization': [],
            'memory_utilization': []
        }
        
        # Reward function parameters (can be overridden by PPO)
        self.reward_params = {
            'latency_weight': 0.5,
            'cpu_weight': 0.2,
            'memory_weight': 0.15,
            'cost_weight': 0.1,
            'throughput_weight': 0.05,
            'latency_threshold': 0.3,
            'cpu_threshold': 0.7,
            'cost_threshold': 0.8
        }
        
        logger.info(f"HybridMicroK8sEnv initialized for {deployment_name} in namespace {namespace}")

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        try:
            # Set initial pod count to 2
            self.k8s_api.safe_scale(self.deployment_name, 2)
            time.sleep(self.scaling_delay)
            
            # Reset episode tracking
            self.episode_step = 0
            self.episode_metrics = {
                'cost': [],
                'response_time': [],
                'queue_length': [],
                'throughput': [],
                'cpu_utilization': [],
                'memory_utilization': []
            }
            
            # Get initial state
            self.state = self._get_normalized_state()
            self.last_metrics_sync = time.time()
            
            logger.info(f"Environment reset: state={self.state}")
            return self.state, {}
            
        except KubernetesAPIError as e:
            logger.error(f"Reset failed: {e}")
            raise

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute action and return (obs, reward, terminated, truncated, info)."""
        try:
            if self.state is None:
                self.state = self._get_normalized_state()
                
            prev_state = self.state.copy()
            current_replicas = self.k8s_api._get_current_replicas(self.deployment_name)

            # Execute action
            if action == 0:  # Scale up
                new_replicas = min(current_replicas + 1, self.k8s_api.max_pods)
                self.k8s_api.safe_scale(self.deployment_name, new_replicas)
            elif action == 1:  # Scale down
                new_replicas = max(current_replicas - 1, 1)
                self.k8s_api.safe_scale(self.deployment_name, new_replicas)
            else:  # No change
                new_replicas = current_replicas

            # Wait for scaling to take effect
            time.sleep(self.scaling_delay)
            
            # Get new state
            self.state = self._get_normalized_state()
            
            # Calculate reward
            reward = self._calculate_reward(prev_state, self.state, action)
            
            # Check termination conditions
            terminated = self._is_done()
            truncated = self.episode_step >= self.max_episode_steps
            
            # Update episode step
            self.episode_step += 1
            
            # Collect metrics
            metrics = self._collect_metrics()
            self._update_episode_metrics(metrics)
            
            # Prepare info
            info = {
                "action": action,
                "replicas": new_replicas,
                "custom_metrics": metrics,
                "episode_step": self.episode_step,
                "reward_params": self.reward_params.copy()
            }

            logger.debug(f"Step: action={action}, reward={reward:.2f}, state={self.state}")
            return self.state, reward, terminated, truncated, info
            
        except KubernetesAPIError as e:
            logger.error(f"Step failed: {e}")
            # Ensure we have a valid state even in error case
            if self.state is None:
                self.state = self._get_normalized_state()
            return self.state, -10.0, False, False, {"error": str(e)}

    def _get_normalized_state(self) -> np.ndarray:
        """Get and normalize cluster state with enhanced metrics."""
        try:
            # Get basic cluster state
            raw_state = self.k8s_api.get_cluster_state()
            
            # Get additional metrics from Prometheus if available
            additional_metrics = self._get_prometheus_metrics()
            
            # Combine metrics
            state = np.array([
                min(raw_state["cpu"] / 100.0, 1.0),  # CPU usage (%)
                min(raw_state["memory"] / 1e9, 1.0),  # Memory (GB, capped)
                min(raw_state["latency"] / 0.2, 1.0),  # Latency (ms, capped at 200ms)
                min(raw_state["pods"] / self.k8s_api.max_pods, 1.0),  # Pods
                additional_metrics.get('queue_length', 0.0),  # Queue length
                additional_metrics.get('throughput', 0.0)  # Throughput
            ], dtype=np.float32)
            
            return state
            
        except Exception as e:
            logger.error(f"Failed to get normalized state: {e}")
            return np.zeros(6, dtype=np.float32)

    def _get_prometheus_metrics(self) -> Dict[str, float]:
        """Get additional metrics from Prometheus."""
        if not self.prometheus_available:
            return {'queue_length': 0.0, 'throughput': 0.0}
        
        try:
            metrics = {}
            
            # Query queue length (requests waiting)
            queue_query = 'sum(rate(nginx_http_requests_total{status!~"5.."}[1m]))'
            queue_result = self.prometheus.custom_query(queue_query)
            if queue_result:
                metrics['queue_length'] = min(float(queue_result[0]['value'][1]) / 100.0, 1.0)
            else:
                metrics['queue_length'] = 0.0
            
            # Query throughput (requests per second)
            throughput_query = 'sum(rate(nginx_http_requests_total[1m]))'
            throughput_result = self.prometheus.custom_query(throughput_query)
            if throughput_result:
                metrics['throughput'] = min(float(throughput_result[0]['value'][1]) / 1000.0, 1.0)
            else:
                metrics['throughput'] = 0.0
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Failed to get Prometheus metrics: {e}")
            return {'queue_length': 0.0, 'throughput': 0.0}

    def _calculate_reward(self, prev_state: np.ndarray, new_state: np.ndarray, action: int) -> float:
        """Calculate reward using configurable parameters."""
        cpu, memory, latency, pods, queue_length, throughput = new_state
        
        # Base reward calculation
        reward = 0.0
        
        # Latency reward (primary concern)
        if latency < self.reward_params['latency_threshold']:
            reward += self.reward_params['latency_weight'] * 2.0
        elif latency < 0.6:
            reward += self.reward_params['latency_weight'] * 1.0
        else:
            reward -= self.reward_params['latency_weight'] * 2.0
        
        # CPU utilization penalty
        if cpu > self.reward_params['cpu_threshold']:
            reward -= self.reward_params['cpu_weight'] * 1.5
        elif cpu > 0.5:
            reward -= self.reward_params['cpu_weight'] * 0.5
        
        # Memory utilization penalty
        if memory > 0.8:
            reward -= self.reward_params['memory_weight'] * 1.0
        elif memory > 0.6:
            reward -= self.reward_params['memory_weight'] * 0.5
        
        # Cost penalty (based on pod count)
        if pods > self.reward_params['cost_threshold']:
            reward -= self.reward_params['cost_weight'] * 1.0
        elif pods > 0.6:
            reward -= self.reward_params['cost_weight'] * 0.5
        
        # Throughput bonus
        if throughput > 0.7:
            reward += self.reward_params['throughput_weight'] * 1.0
        
        # Queue length penalty
        if queue_length > 0.5:
            reward -= 0.2 * queue_length
        
        # Scaling efficiency bonus
        if len(prev_state) > 3 and prev_state[3] > pods:  # Pods decreased
            if cpu < prev_state[0] and latency < prev_state[2]:
                reward += 0.5  # Good scaling down decision
        
        # Action penalty to encourage stability
        if action != 2:  # If not no-op
            reward -= 0.1  # Small penalty for taking action
        
        return reward

    def _is_done(self) -> bool:
        """Determine if episode is complete."""
        if self.state is None:
            return False
            
        # End if latency too high, max pods reached, or resource exhaustion
        done = (
            self.state[2] > 0.95 or  # High latency
            self.state[3] >= 0.95 or  # Max pods
            self.state[0] > 0.95 or  # High CPU
            self.state[1] > 0.95     # High memory
        )
        
        if done:
            logger.info(f"Episode done: state={self.state}")
        
        return done

    def _collect_metrics(self) -> Dict[str, float]:
        """Collect comprehensive metrics for logging."""
        try:
            cluster_state = self.k8s_api.get_cluster_state()
            prometheus_metrics = self._get_prometheus_metrics()
            
            metrics = {
                'cpu_utilization': cluster_state.get('cpu', 0.0) / 100.0,
                'memory_utilization': cluster_state.get('memory', 0.0) / 1e9,
                'latency': cluster_state.get('latency', 0.0) / 0.2,
                'pod_count': cluster_state.get('pods', 0.0),
                'queue_length': prometheus_metrics.get('queue_length', 0.0),
                'throughput': prometheus_metrics.get('throughput', 0.0),
                'cost': cluster_state.get('pods', 0.0) / self.k8s_api.max_pods,
                'response_time': cluster_state.get('latency', 0.0)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
            return {
                'cpu_utilization': 0.0,
                'memory_utilization': 0.0,
                'latency': 0.0,
                'pod_count': 0.0,
                'queue_length': 0.0,
                'throughput': 0.0,
                'cost': 0.0,
                'response_time': 0.0
            }

    def _update_episode_metrics(self, metrics: Dict[str, float]):
        """Update episode metrics tracking."""
        self.episode_metrics['cost'].append(metrics['cost'])
        self.episode_metrics['response_time'].append(metrics['response_time'])
        self.episode_metrics['queue_length'].append(metrics['queue_length'])
        self.episode_metrics['throughput'].append(metrics['throughput'])
        self.episode_metrics['cpu_utilization'].append(metrics['cpu_utilization'])
        self.episode_metrics['memory_utilization'].append(metrics['memory_utilization'])

    def override_reward_params(self, new_params: Dict[str, float]):
        """Override reward function parameters (called by PPO agent)."""
        self.reward_params.update(new_params)
        logger.info(f"Reward parameters updated: {new_params}")

    def get_cluster_metrics(self) -> Dict[str, float]:
        """Get current cluster metrics for external monitoring."""
        return self._collect_metrics()

    def get_episode_summary(self) -> Dict[str, Any]:
        """Get summary of episode metrics."""
        if not self.episode_metrics['cost']:
            return {}
        
        return {
            'avg_cost': np.mean(self.episode_metrics['cost']),
            'avg_response_time': np.mean(self.episode_metrics['response_time']),
            'avg_queue_length': np.mean(self.episode_metrics['queue_length']),
            'avg_throughput': np.mean(self.episode_metrics['throughput']),
            'avg_cpu_utilization': np.mean(self.episode_metrics['cpu_utilization']),
            'avg_memory_utilization': np.mean(self.episode_metrics['memory_utilization']),
            'max_cost': np.max(self.episode_metrics['cost']),
            'max_response_time': np.max(self.episode_metrics['response_time']),
            'min_throughput': np.min(self.episode_metrics['throughput']),
            'episode_length': len(self.episode_metrics['cost'])
        }

    def render(self, mode: str = "human") -> None:
        """Render environment state (for debugging)."""
        if self.state is not None:
            logger.info(
                f"Current state: cpu={self.state[0]:.2f}, memory={self.state[1]:.2f}, "
                f"latency={self.state[2]:.2f}, pods={self.state[3]:.2f}, "
                f"queue={self.state[4]:.2f}, throughput={self.state[5]:.2f}"
            )

    def close(self):
        """Clean up environment resources."""
        logger.info("HybridMicroK8sEnv closed")

    def seed(self, seed=None):
        """Set random seed for reproducibility."""
        np.random.seed(seed)
        return [seed] 