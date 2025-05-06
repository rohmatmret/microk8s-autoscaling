import random
import numpy as np

class MockKubernetesAPI:
    "Mock Kubernetes API"
    def __init__(self, max_pods=10, max_nodes=5, namespace="default"):
        self.max_pods = max_pods
        self.max_nodes = max_nodes
        self.namespace = namespace
        self.current_replicas = 1
        self.random = random.Random()
        self.np_random = np.random.default_rng()
        self.scale_buffer = []  # Simulate scaling delay
        self.failure_rate = 0.05  # 5% chance of scaling failure
        self.scaling_delay = 2     # Steps required for scaling to take effect
        self.active_pods = 1       # Actual running pods (accounts for scaling delay)


    def seed(self, seed=None):
        """Set random seed for reproducibility."""
        self.random.seed(seed)
        self.np_random = np.random.default_rng(seed)
        return [seed]
    
    def get_current_pod_count(self) -> int:
        """Returns the CURRENT number of active pods (accounts for scaling delays)."""
        return self.active_pods

    def get_desired_pod_count(self) -> int:
        """Returns the DESIRED number of pods (may not match active pods due to delays)."""
        return self.current_replicas

    def get_cluster_state(self):
        """Return simulated cluster state with effect of scaling and load."""
        # # Simulate scaling delay (2 steps)
        # if len(self.scale_buffer) >= 2:
        #     self.current_replicas = self.scale_buffer.pop(0)

        # pods = self.current_replicas
        # nodes = min(1 + pods // 3, self.max_nodes)

        # # Simulate CPU and memory usage affected by pod count
        # cpu_load = max(0.2, 1.0 - (pods * 0.08) + self.random.uniform(-0.05, 0.05))
        # memory_load = 80e6 + pods * 30e6 + self.random.uniform(-10e6, 10e6)
        # latency = max(0.05, 0.5 - pods * 0.04 + self.random.uniform(-0.02, 0.02))

        # cpu = min(cpu_load, 1.0)
        # memory = min(max(memory_load, 0.0), 500e6)
        # latency = min(latency, 1.0)
        # Apply scaling after delay
        if len(self.scale_buffer) >= self.scaling_delay:
            self.current_replicas = self.scale_buffer.pop(0)
            self.active_pods = self.current_replicas

        # Simulate pod startup time (not all pods become active immediately)
        if self.active_pods < self.current_replicas:
            self.active_pods += 1
        elif self.active_pods > self.current_replicas:
            self.active_pods -= 1

        nodes = min(1 + self.active_pods // 3, self.max_nodes)

        # Simulate dynamic resource usage
        base_cpu = 0.2
        cpu_per_pod = 0.08
        noise = self.random.uniform(-0.05, 0.05)
        cpu_util = min(1.0, max(base_cpu, base_cpu + (self.active_pods * cpu_per_pod) + noise))

        memory_per_pod = 30e6  # 30MB per pod
        memory_util = 80e6 + (self.active_pods * memory_per_pod) + self.random.uniform(-10e6, 10e6)
        memory_util = min(max(memory_util, 0), 500e6)

        # Latency improves with more pods (up to a point)
        latency = max(0.05, 0.5 - (self.active_pods * 0.04) + self.random.uniform(-0.02, 0.02))


        return {
            "pods": self.active_pods,
            "nodes": nodes,
            "cpu": cpu_util,
            "memory": memory_util,
            "latency": latency,
            "swap": self.random.uniform(0.0, 200e6),
            "desired_replicas": self.current_replicas,  # Useful for debugging

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