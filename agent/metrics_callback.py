"""Shared metrics callback for DQN and PPO autoscaling algorithms."""

import wandb
import requests
import time
import logging
from typing import Dict, Any, Optional
from stable_baselines3.common.callbacks import BaseCallback
import torch

logger = logging.getLogger(__name__)

class AutoscalingMetricsCallback(BaseCallback):
    """Callback for logging autoscaling metrics to WandB and Prometheus."""
    
    def __init__(self, 
                 prometheus_url: str = "http://localhost:9090",
                 algorithm: str = "dqn",
                 is_simulated: bool = False,
                 verbose: int = 0):
        """
        Initialize metrics callback.
        
        Args:
            prometheus_url: URL of Prometheus server
            algorithm: Either 'dqn' or 'ppo'
            is_simulated: Whether to use simulated metrics
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.prometheus_url = prometheus_url
        self.algorithm = algorithm
        self.is_simulated = is_simulated
        self.metrics_history = {
            'latency': [],
            'cpu': [],
            'throughput': [],
            'pods': [],
            'rewards': [],
            'q_values': [] if algorithm == 'dqn' else None,
            'value_loss': [] if algorithm == 'ppo' else None,
            'policy_loss': [] if algorithm == 'ppo' else None
        }
    
    def _on_step(self) -> bool:
        """Collect and log metrics every step."""
        try:
            # Collect metrics based on mode
            if self.is_simulated:
                metrics = self._collect_simulated_metrics()
            else:
                metrics = self._collect_prometheus_metrics()
            
            # Collect algorithm-specific metrics
            algo_metrics = self._collect_algorithm_metrics()
            
            # Combine all metrics
            all_metrics = {**metrics, **algo_metrics}
            
            # Log to WandB
            wandb.log(all_metrics)
            
            # Update history
            self._update_metrics_history(all_metrics)
            
            # Create and log plots every 100 steps
            if self.n_calls % 100 == 0:
                self._log_metric_plots()
                
        except Exception as e:
            logger.error(f"Error in metrics callback: {e}")
            
        return True
    
    def _collect_simulated_metrics(self) -> Dict[str, float]:
        """Collect simulated metrics from environment info."""
        try:
            # Get metrics from environment info
            info = self.locals.get('infos', [{}])[0]
            custom_metrics = info.get('custom_metrics', {})
            
            return {
                "metrics/latency_ms": custom_metrics.get('latency', 0),
                "metrics/cpu_usage_percent": custom_metrics.get('cpu_utilization', 0),
                "metrics/throughput_req_per_sec": custom_metrics.get('throughput', 0),
                "metrics/pod_count": custom_metrics.get('pod_count', 0)
            }
            
        except Exception as e:
            logger.error(f"Error collecting simulated metrics: {e}")
            return {}
    
    def _collect_prometheus_metrics(self) -> Dict[str, float]:
        """Collect metrics from Prometheus."""
        try:
            # Latency
            latency_query = 'rate(http_request_duration_seconds_sum[1m])/rate(http_request_duration_seconds_count[1m])'
            latency_response = requests.get(f"{self.prometheus_url}/api/v1/query", 
                                         params={"query": latency_query})
            latency_ms = float(latency_response.json()["data"]["result"][0]["value"][1]) * 1000

            # CPU usage
            cpu_query = 'rate(container_cpu_usage_seconds_total[1m])'
            cpu_response = requests.get(f"{self.prometheus_url}/api/v1/query", 
                                     params={"query": cpu_query})
            cpu_percent = float(cpu_response.json()["data"]["result"][0]["value"][1]) * 100

            # Throughput
            throughput_query = 'rate(http_requests_total[1m])'
            throughput_response = requests.get(f"{self.prometheus_url}/api/v1/query", 
                                           params={"query": throughput_query})
            throughput = float(throughput_response.json()["data"]["result"][0]["value"][1])
            
            # Pod count
            pod_query = 'kube_deployment_status_replicas'
            pod_response = requests.get(f"{self.prometheus_url}/api/v1/query", 
                                     params={"query": pod_query})
            pod_count = float(pod_response.json()["data"]["result"][0]["value"][1])

            return {
                "metrics/latency_ms": latency_ms,
                "metrics/cpu_usage_percent": cpu_percent,
                "metrics/throughput_req_per_sec": throughput,
                "metrics/pod_count": pod_count
            }
            
        except Exception as e:
            logger.error(f"Error collecting Prometheus metrics: {e}")
            return {}
    
    def _collect_algorithm_metrics(self) -> Dict[str, float]:
        """Collect algorithm-specific metrics."""
        metrics = {}
        
        # Common metrics
        if self.locals.get("rewards"):
            metrics["train/episode_reward"] = self.locals["rewards"][-1]
        
        # DQN-specific metrics
        if self.algorithm == "dqn":
            # Get Q-values from current observations
            obs = self.locals.get("obs", None)
            if obs is not None:
                try:
                    with torch.no_grad():
                        q_values = self.model.q_net(obs)
                        metrics["train/q_value_mean"] = q_values.mean().item()
                except Exception as e:
                    logger.debug(f"Could not compute Q-values: {e}")
        
        # PPO-specific metrics
        elif self.algorithm == "ppo":
            if hasattr(self.model, "logger") and hasattr(self.model.logger, "name_to_value"):
                metrics["train/value_loss"] = self.model.logger.name_to_value.get("train/value_loss", 0)
                metrics["train/policy_loss"] = self.model.logger.name_to_value.get("train/policy_loss", 0)
        
        return metrics
    
    def _update_metrics_history(self, metrics: Dict[str, float]):
        """Update metrics history for plotting."""
        for key, value in metrics.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)
    
    def _log_metric_plots(self):
        """Create and log metric plots to WandB."""
        try:
            # Resource utilization plot
            if len(self.metrics_history['cpu']) > 0:
                resource_data = [[i, cpu, latency, throughput] 
                               for i, (cpu, latency, throughput) in enumerate(zip(
                                   self.metrics_history['cpu'],
                                   self.metrics_history['latency'],
                                   self.metrics_history['throughput']
                               ))]
                
                resource_table = wandb.Table(
                    data=resource_data,
                    columns=["step", "cpu", "latency", "throughput"]
                )
                
                wandb.log({
                    "plots/resource_utilization": wandb.plot.line_series(
                        resource_table,
                        "step",
                        ["cpu", "latency", "throughput"],
                        title="Resource Utilization"
                    )
                })
            
            # Pod scaling plot
            if len(self.metrics_history['pods']) > 0:
                pod_data = [[i, pods] for i, pods in enumerate(self.metrics_history['pods'])]
                pod_table = wandb.Table(
                    data=pod_data,
                    columns=["step", "pods"]
                )
                
                wandb.log({
                    "plots/pod_scaling": wandb.plot.line(
                        pod_table,
                        "step",
                        "pods",
                        title="Pod Scaling"
                    )
                })
            
            # Algorithm-specific plots
            if self.algorithm == "dqn" and len(self.metrics_history['q_values']) > 0:
                q_value_data = [[i, q] for i, q in enumerate(self.metrics_history['q_values'])]
                q_value_table = wandb.Table(
                    data=q_value_data,
                    columns=["step", "q_value"]
                )
                wandb.log({
                    "plots/q_values": wandb.plot.line(
                        q_value_table,
                        "step",
                        "q_value",
                        title="Q-Value Mean"
                    )
                })
            
            elif self.algorithm == "ppo" and len(self.metrics_history['value_loss']) > 0:
                loss_data = [[i, v_loss, p_loss] 
                           for i, (v_loss, p_loss) in enumerate(zip(
                               self.metrics_history['value_loss'],
                               self.metrics_history['policy_loss']
                           ))]
                loss_table = wandb.Table(
                    data=loss_data,
                    columns=["step", "value_loss", "policy_loss"]
                )
                wandb.log({
                    "plots/ppo_losses": wandb.plot.line_series(
                        loss_table,
                        "step",
                        ["value_loss", "policy_loss"],
                        title="PPO Losses"
                    )
                })
                
        except Exception as e:
            logger.error(f"Error creating metric plots: {e}") 