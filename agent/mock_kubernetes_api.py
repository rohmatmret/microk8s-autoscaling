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
        self.scale_buffer = []  # Simulate scaling delay

    def seed(self, seed=None):
        """Set random seed for reproducibility."""
        self.random.seed(seed)
        self.np_random = np.random.default_rng(seed)
        return [seed]

    def get_cluster_state(self):
        """Return simulated cluster state with effect of scaling and load."""
        # Simulate scaling delay (2 steps)
        if len(self.scale_buffer) >= 2:
            self.current_replicas = self.scale_buffer.pop(0)

        pods = self.current_replicas
        nodes = min(1 + pods // 3, self.max_nodes)

        # Simulate CPU and memory usage affected by pod count
        cpu_load = max(0.2, 1.0 - (pods * 0.08) + self.random.uniform(-0.05, 0.05))
        memory_load = 80e6 + pods * 30e6 + self.random.uniform(-10e6, 10e6)
        latency = max(0.05, 0.5 - pods * 0.04 + self.random.uniform(-0.02, 0.02))

        cpu = min(cpu_load, 1.0)
        memory = min(max(memory_load, 0.0), 500e6)
        latency = min(latency, 1.0)

        return {
            "pods": pods,
            "nodes": nodes,
            "cpu": cpu,
            "memory": memory,
            "latency": latency,
            "swap": self.random.uniform(0.0, 200e6),  # Contoh swap usage

        }

    def safe_scale(self, deployment_name: str, desired_replicas: int) -> bool:
        """Simulate scaling operation with bounds and possible delay."""
        bounded_replicas = max(1, min(self.max_pods, desired_replicas))

        # Simulate 5% chance of scaling failure
        if self.random.random() < 0.05:
            # print(f"[MockAPI] Scaling FAILED to {bounded_replicas} replicas.")
            return False

        # Simulate delay in applying scaling
        self.scale_buffer.append(bounded_replicas)

        # print(f"[MockAPI] Scaling scheduled to {bounded_replicas} replicas (buffer size: {len(self.scale_buffer)}).")
        return True