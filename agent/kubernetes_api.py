# agent/kubernetes_api.py
import os
import time
import logging
from typing import Dict, Optional
from functools import lru_cache
from kubernetes import client, config
from kubernetes.client.rest import ApiException
from prometheus_api_client import PrometheusConnect, PrometheusApiClientException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("kubernetes_api.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class KubernetesAPIError(Exception):
    """Custom exception for Kubernetes API operations."""
    pass

class KubernetesAPI:
    """Manages interactions with Kubernetes and Prometheus for cluster state and scaling."""

    def __init__(
        self,
        max_pods: int = 10,
        max_nodes: int = 5,
        namespace: str = os.getenv("K8S_NAMESPACE", "default"),
        prometheus_url: str = os.getenv("PROMETHEUS_URL", "http://prometheus:9090")
    ):
        """Initialize Kubernetes and Prometheus clients."""
        try:
            config.load_incluster_config()  # Try in-cluster config first
        except config.ConfigException:
            config.load_kube_config()  # Fallback to local kubeconfig
            logger.warning("Using local kubeconfig")

        configuration = client.Configuration()
        configuration.retries = 3
        self.apps_api = client.AppsV1Api(client.ApiClient(configuration))
        self.core_api = client.CoreV1Api(client.ApiClient(configuration))
        self.prometheus = PrometheusConnect(url=prometheus_url, disable_ssl=True)
        self.max_pods = max_pods
        self.max_nodes = max_nodes
        self.namespace = namespace
        logger.info("Initialized KubernetesAPI with namespace=%s", namespace)

    def get_cluster_state(self) -> Dict[str, float]:
        """Retrieve normalized cluster state metrics."""
        try:
            state = {
                "pods": self._get_pod_count(),
                "nodes": self._get_node_count(),
                "cpu": self._get_cpu_usage(),
                "memory": self._get_memory_usage(),
                "latency": self._get_latency(),
            }
            logger.debug("Cluster state: %s", state)
            return state
        except Exception as e:
            logger.error("Failed to get cluster state: %s", e)
            raise KubernetesAPIError(f"Failed to get cluster state: {str(e)}") from e

    @lru_cache(maxsize=32)
    def _get_cpu_usage(self) -> float:
        """Get average CPU usage across pods in namespace (cached for 30s)."""
        query = f'sum(rate(container_cpu_usage_seconds_total{{namespace="{self.namespace}"}}[1m]))'
        return self._query_prometheus(query, "CPU usage")

    @lru_cache(maxsize=32)
    def _get_memory_usage(self) -> float:
        """Get total memory usage in bytes (cached for 30s)."""
        query = f'sum(container_memory_usage_bytes{{namespace="{self.namespace}"}})'
        return self._query_prometheus(query, "Memory usage")

    @lru_cache(maxsize=32)
    def _get_latency(self) -> float:
        """Get 95th percentile latency in seconds (cached for 30s)."""
        query = 'histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[1m]))'
        return self._query_prometheus(query, "Latency")

    def _query_prometheus(self, query: str, metric_name: str) -> float:
        """Helper to query Prometheus with error handling."""
        try:
            result = self.prometheus.custom_query(query)
            value = float(result[0]["value"][1]) if result else 0.0
            return value
        except PrometheusApiClientException as e:
            logger.error("%s query failed: %s", metric_name, e)
            raise KubernetesAPIError(f"{metric_name} query failed: {str(e)}") from e

    def _get_node_count(self) -> int:
        """Get number of worker nodes (excluding control plane)."""
        try:
            nodes = self.core_api.list_node().items
            count = len([
                n for n in nodes
                if "node-role.kubernetes.io/control-plane" not in (n.metadata.labels or {})
            ])
            logger.debug("Node count: %d", count)
            return count
        except ApiException as e:
            logger.error("Failed to get node count: %s", e)
            raise KubernetesAPIError(f"Node count failed: {str(e)}") from e

    def _get_pod_count(self) -> int:
        """Get current pod count in namespace."""
        try:
            pods = self.core_api.list_namespaced_pod(self.namespace)
            count = len(pods.items)
            logger.debug("Pod count: %d", count)
            return count
        except ApiException as e:
            logger.error("Failed to get pod count: %s", e)
            raise KubernetesAPIError(f"Pod count failed: {str(e)}") from e

    def safe_scale(self, deployment_name: str, desired_replicas: int) -> bool:
        """Scale deployment with validation and verification."""
        try:
            current = self._get_current_replicas(deployment_name)
            desired = max(1, min(self.max_pods, desired_replicas))
            logger.info("Scaling %s from %d to %d replicas", deployment_name, current, desired)

            if desired == current:
                logger.debug("No scaling needed")
                return True

            self._scale_deployment(deployment_name, desired)
            return self._verify_pod_scale(deployment_name, desired)
        except ApiException as e:
            logger.error("Scaling failed for %s: %s", deployment_name, e)
            raise KubernetesAPIError(f"Scaling failed: {str(e)}") from e

    def _get_current_replicas(self, deployment_name: str) -> int:
        """Get current replica count for deployment."""
        try:
            deployment = self.apps_api.read_namespaced_deployment(deployment_name, self.namespace)
            replicas = deployment.spec.replicas or 1
            return replicas
        except ApiException as e:
            logger.error("Failed to get replicas for %s: %s", deployment_name, e)
            raise KubernetesAPIError(f"Replica check failed: {str(e)}") from e

    def _scale_deployment(self, deployment_name: str, replicas: int) -> None:
        """Scale deployment to specified replicas."""
        if not 1 <= replicas <= self.max_pods:
            logger.error("Invalid replicas: %d (must be 1-%d)", replicas, self.max_pods)
            raise ValueError(f"Replicas must be between 1 and {self.max_pods}")

        patch = {"spec": {"replicas": replicas}}
        self.apps_api.patch_namespaced_deployment(
            name=deployment_name,
            namespace=self.namespace,
            body=patch
        )
        logger.info("Scaled %s to %d replicas", deployment_name, replicas)

    def _verify_pod_scale(self, deployment_name: str, desired: int) -> bool:
        """Verify pod scaling completion within timeout."""
        timeout = 300  # 5 minutes
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                deployment = self.apps_api.read_namespaced_deployment(deployment_name, self.namespace)
                ready = deployment.status.ready_replicas or 0
                updated = deployment.status.updated_replicas or 0

                if ready == desired and updated == desired:
                    logger.info("Scaling verified: %s has %d ready replicas", deployment_name, desired)
                    return True

                time.sleep(5)
            except ApiException as e:
                logger.error("Verification failed for %s: %s", deployment_name, e)
                raise KubernetesAPIError(f"Verification failed: {str(e)}") from e

        logger.error("Scaling timeout for %s: desired %d replicas", deployment_name, desired)
        raise KubernetesAPIError("Pod scaling timeout")