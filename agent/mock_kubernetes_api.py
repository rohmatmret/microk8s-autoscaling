"""Mock Kubernetes API implementation for simulating cluster behavior and autoscaling operations.

This module provides a simulated Kubernetes API that mimics real cluster behavior,
including pod scaling, resource utilization, and traffic-driven metrics.
"""

import random
import numpy as np
from typing import Dict, Any

class MockKubernetesAPI:
    """Enhanced mock Kubernetes API with realistic traffic simulation and scaling behavior"""
    
    def __init__(self, max_pods=10, max_nodes=5, namespace="default", traffic_simulator=None):
        self.max_pods = max_pods
        self.max_nodes = max_nodes
        self.namespace = namespace
        self.current_replicas = 1
        self.active_pods = 1
        self.random = random.Random()
        self.np_random = np.random.default_rng()
        self.scale_buffer = []
        self.traffic_simulator = traffic_simulator or self._default_traffic_simulator()
        
        # Optimized performance characteristics
        self.pod_capacity = 500  # Requests per pod
        self.base_latency = 0.2  # Seconds
        self.pod_startup_time = 15  # Reduced from 60 to 15 seconds
        self.failure_rate = 0.05
        self.scaling_delay =2  # Reduced from 60 to 15 seconds
        self.current_step = 0
        
        # Optimized scaling thresholds
        self.scale_up_threshold = 0.75
        self.scale_down_threshold = 0.65
        self.scale_up_step = 2
        self.scale_down_step = 1

        # State history for trend analysis
        self.cpu_history = []
        self.memory_history = []
        self.latency_history = []

    def _default_traffic_simulator(self):
        """Fallback traffic simulator if none provided"""
        def simulator(step):
            # Base load with diurnal pattern + random noise
            return 100 * (1 + 0.3 * np.sin(2 * np.pi * step / 1440)) + 20 * random.random()
        return simulator

    def seed(self, seed=None):
        self.random.seed(seed)
        self.np_random = np.random.default_rng(seed)
        return [seed]

    def get_current_pod_count(self) -> int:
        return self.active_pods

    def get_desired_pod_count(self) -> int:
        return self.current_replicas

    def get_cluster_state(self) -> Dict[str, Any]:
        """Generate realistic cluster state with traffic-driven metrics"""
        self.current_step += 1
        
        # Calculate traffic-driven metrics
        current_load = self.traffic_simulator(self.current_step)
        effective_capacity = max(1, self.active_pods * self.pod_capacity)
        load_ratio = current_load / effective_capacity

        # Optimized auto-scale based on load ratio
        if load_ratio > self.scale_up_threshold and self.current_replicas < self.max_pods:
            self.current_replicas = min(self.max_pods, self.current_replicas + self.scale_up_step)
        elif load_ratio < self.scale_down_threshold and self.current_replicas > 1:
            self.current_replicas = max(1, self.current_replicas - self.scale_down_step)

        # Apply scaling after delay
        if len(self.scale_buffer) > 0 and len(self.scale_buffer) >= self.scaling_delay:
            self.current_replicas = self.scale_buffer.pop(0)
        
        # Simulate gradual pod changes with faster scaling
        if self.active_pods < self.current_replicas:
            self.active_pods += min(self.scale_up_step, self.current_replicas - self.active_pods)
        elif self.active_pods > self.current_replicas:
            self.active_pods -= min(self.scale_down_step, self.active_pods - self.current_replicas)
        
        # Dynamic resource metrics
        cpu_util = min(1.0, 0.2 + 0.8 * load_ratio + self.random.uniform(-0.05, 0.05))
        memory_util = min(500e6, 80e6 + (current_load * 0.3) + self.random.uniform(-10e6, 10e6))
        latency = max(0.05, self.base_latency + (load_ratio ** 2))
        
        # Track history for trend analysis
        self.cpu_history.append(cpu_util)
        self.memory_history.append(memory_util)
        self.latency_history.append(latency)
        
        return {
            "cpu": cpu_util,
            "memory": memory_util,
            "latency": latency,
            "pods": self.active_pods / self.max_pods,
            "load_ratio": load_ratio
        }

    def safe_scale(self, deployment_name: str, desired_replicas: int) -> bool:
        """Realistic scaling simulation with failure modes"""
        if self.random.random() < self.failure_rate:
            return False
            
        target = max(1, min(desired_replicas, self.max_pods))
        # Simulate cloud provider rate limiting
        max_scale_step = self.max_pods # 10
        if abs(target - self.current_replicas) > max_scale_step:
            target = self.current_replicas + np.sign(target - self.current_replicas) * max_scale_step
        
        self.scale_buffer.append(target)
        return True

    def get_metrics_history(self, window=10) -> Dict[str, Any]:
        """Get trend metrics for RL state observation"""
        return {
            "cpu_trend": np.mean(self.cpu_history[-window:]),
            "memory_trend": np.mean(self.memory_history[-window:]),
            "latency_trend": np.mean(self.latency_history[-window:]),
            "scaling_frequency": len(self.scale_buffer) / window if window > 0 else 0
        }