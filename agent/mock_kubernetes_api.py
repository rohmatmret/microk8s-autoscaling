# agent/mock_kubernetes_api.py
import random
import numpy as np

class MockKubernetesAPI:
    def __init__(self, max_pods=10, max_nodes=5, namespace="default"):
        self.max_pods = max_pods
        self.max_nodes = max_nodes
        self.namespace = namespace
        self.current_replicas = 1
        self.random = random.Random()
        self.np_random = np.random.default_rng()

    def seed(self, seed=None):
        """Set random seed for reproducibility."""
        self.random.seed(seed)
        self.np_random.seed(seed)
        return [seed]

    def get_cluster_state(self):
        """Return simulated cluster state."""
        return {
            "pods": self.current_replicas,
            "nodes": self.random.randint(1, 3),
            "cpu": self.random.uniform(0.1, 0.8),
            "memory": self.random.uniform(100e6, 500e6),
            "latency": self.random.uniform(0.05, 0.4)
        }

    def safe_scale(self, deployment_name: str, desired_replicas: int) -> bool:
        """Simulate scaling operation."""
        self.current_replicas = max(1, min(self.max_pods, desired_replicas))
        return True