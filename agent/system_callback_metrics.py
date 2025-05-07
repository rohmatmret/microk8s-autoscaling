import logging
from typing import List, Dict, Optional
import time
import psutil
from stable_baselines3.common.callbacks import BaseCallback
from kubernetes import client, config
import wandb

class SystemMetricsCallback(BaseCallback):
    """
    Enhanced system metrics tracking for MicroK8s RL experiments.
    Tracks both host and Kubernetes cluster metrics with adaptive logging.
    """
    def __init__(
        self,
        env,  #  MicroK8s environment
        log_freq: int = 1000,
        metrics_to_track: List[str] = None,
        k8s_config_path: Optional[str] = None,
        max_retries: int = 3
    ):
        super().__init__()
        self.env = env
        self.log_freq = log_freq
        self.metrics_to_track = metrics_to_track or [
            'cpu', 'memory', 'pods', 'swap', 'scaling_lag'
            'network', 'disk', 'scaling_lag'
        ]
        self.k8s_api = None
        self.last_log_time = time.time()
        self.retry_count = 0
        self.max_retries = max_retries
        self.setup_k8s_client(k8s_config_path)
        self._last_net_metrics = {'bytes_sent': 0, 'bytes_recv': 0}  # Initialize network metrics

        
        # Metrics configuration
        self.metric_handlers = {
            'cpu': self._get_cpu_metrics,
            'memory': self._get_memory_metrics,
            'pods': self._get_pod_metrics,
            'network': self._get_network_metrics,
            'disk': self._get_disk_metrics,
            'scaling_lag': self._get_scaling_lag,
            'k8s_nodes': self._get_k8s_node_metrics
        }

    def setup_k8s_client(self, config_path: Optional[str] = None):
        """Initialize Kubernetes client with retry logic"""
        try:
            if config_path:
                config.load_kube_config(config_file=config_path)
            else:
                config.load_kube_config()  # Default location
            self.k8s_api = client.CoreV1Api()
            self.retry_count = 0
        except (config.ConfigException, client.ApiException) as e:
            if self.retry_count < self.max_retries:
                self.retry_count += 1
                time.sleep(2 ** self.retry_count)  # Exponential backoff
                self.setup_k8s_client(config_path)
            else:
                logging.warning("Failed to initialize K8s client after %d attempts: %s", self.max_retries, e)
                self.k8s_api = None

    def _on_step(self) -> bool:
        """Log system metrics at specified frequency"""
        if self.n_calls % self.log_freq == 0:
            try:
                metrics = {}
                current_time = time.time()
                time_elapsed = current_time - self.last_log_time
                
                # Collect all requested metrics
                for metric in self.metrics_to_track:
                    if handler := self.metric_handlers.get(metric):
                        metrics.update(handler(time_elapsed))
                
                # Add timestamp and step information
                metrics.update({
                    'timesteps': self.num_timesteps,
                    'wall_time': current_time,
                    'fps': self.log_freq / time_elapsed if time_elapsed > 0 else 0
                })
                
                wandb.log(metrics, commit=False)
                self.last_log_time = current_time
                
            except (wandb.Error, psutil.Error) as e:
                logging.error("Error logging system metrics %s:", e)
                
        return True

    # --- Metric Collection Methods ---
    def _get_cpu_metrics(self, _) -> Dict[str, float]:
        return {
            'system/cpu_util': psutil.cpu_percent() / 100,
            'system/cpu_load': psutil.getloadavg()[0] / psutil.cpu_count(),
            'system/cpu_temp': self._get_cpu_temp() if hasattr(psutil, "sensors_temperatures") else 0
        }

    def _get_memory_metrics(self, _) -> Dict[str, float]:
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        return {
            'system/memory_util': mem.percent / 100,
            'system/memory_used_gb': mem.used / (1024**3),
            'system/swap_util': swap.percent / 100,
            'system/swap_used_gb': swap.used / (1024**3)
        }

    def _get_pod_metrics(self, _) -> Dict[str, float]:
        if not hasattr(self.env, 'api'):
            return {}
            
        state = self.env.api.get_cluster_state()
        return {
            'k8s/pod_count': state.get('pods', 0),
            'k8s/desired_pods': state.get('desired_replicas', 0),
            'k8s/nodes': state.get('nodes', 1)
        }

    def _get_scaling_lag(self, _) -> Dict[str, float]:
        if not hasattr(self.env, 'api'):
            return {}
            
        state = self.env.api.get_cluster_state()
        desired = state.get('desired_replicas', state.get('pods', 0))
        actual = state.get('pods', 0)
        return {
            'k8s/scaling_lag': desired - actual,
            'k8s/scaling_efficiency': 1 - (abs(desired - actual) / self.env.api.max_pods)
        }

    def _get_k8s_node_metrics(self, _) -> Dict[str, float]:
        if not self.k8s_api:
            return {}
            
        try:
            nodes = self.k8s_api.list_node()
            if nodes.items:
                node = nodes.items[0]  # Assuming single-node MicroK8s
                return {
                    'k8s/node_cpu_alloc': float(node.status.allocatable['cpu']),
                    'k8s/node_mem_alloc_gb': float(node.status.allocatable['memory'].rstrip('Ki')) / (1024**2)
                }
        except (client.ApiException, ValueError) as e:
            logging.debug("Couldn't fetch K8s node metrics: %s", e)
        return {}

    def _get_network_metrics(self, time_elapsed: float) -> Dict[str, float]:
        net = psutil.net_io_counters()
        metrics = {
            'network/bytes_sent': net.bytes_sent,
            'network/bytes_recv': net.bytes_recv
        }
        if time_elapsed > 0 and hasattr(self, '_last_net_metrics'):
            last = self._last_net_metrics
            metrics.update({
                'network/bandwidth_tx_mbps': (net.bytes_sent - last['bytes_sent']) * 8 / (time_elapsed * 1e6),
                'network/bandwidth_rx_mbps': (net.bytes_recv - last['bytes_recv']) * 8 / (time_elapsed * 1e6)
            })
        self._last_net_metrics = {'bytes_sent': net.bytes_sent, 'bytes_recv': net.bytes_recv}
        return metrics

    def _get_disk_metrics(self, _) -> Dict[str, float]:
        disk = psutil.disk_io_counters()
        usage = psutil.disk_usage('/')
        return {
            'disk/usage_percent': usage.percent / 100,
            'disk/read_mb': disk.read_bytes / (1024**2),
            'disk/write_mb': disk.write_bytes / (1024**2),
            'disk/available_gb': usage.free / (1024**3)
        }

    def _get_cpu_temp(self) -> float:
        """Get CPU temperature in Celsius (if available)"""
        try:
            temps = psutil.sensors_temperatures()
            if 'coretemp' in temps:
                return temps['coretemp'][0].current
        except (psutil.Error, AttributeError):
            pass
        return 0