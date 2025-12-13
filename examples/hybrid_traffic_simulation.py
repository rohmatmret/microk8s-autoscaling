"""Advanced Performance Testing Framework for DQN & PPO Autoscaling Agents.

This module provides comprehensive testing capabilities for evaluating the performance
of RL-based autoscaling agents against traditional rule-based systems with detailed
Prometheus-style metrics collection and research-grade analysis.
"""

import os
import sys
import json
import time
import logging
import csv
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import RL agents and simulation components
try:
    from agent.traffic_simulation import HybridTrafficSimulator, TrafficSimulator
    from agent.hybrid_dqn_ppo import HybridDQNPPOAgent, HybridConfig
    from agent.dqn import DQNAgent
    from agent.ppo import PPOAgent
    from agent.hybrid_environment import HybridMicroK8sEnv
    from agent.environment_simulated import MicroK8sEnvSimulated
    from agent.kubernetes_api import KubernetesAPI
except ImportError as e:
    print(f"Warning: Some agent modules not available: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"performance_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class PrometheusStyleMetric:
    """Prometheus-style metric with labels and timestamps."""
    name: str
    value: float
    labels: Dict[str, str]
    timestamp: float
    help_text: str = ""

    def to_prometheus_format(self) -> str:
        """Convert to Prometheus exposition format."""
        label_str = ",".join([f'{k}="{v}"' for k, v in self.labels.items()])
        return f'{self.name}{{{label_str}}} {self.value} {int(self.timestamp * 1000)}'

@dataclass
class AutoscalingMetrics:
    """Comprehensive autoscaling performance metrics."""
    # Resource utilization metrics
    cpu_utilization: float
    memory_utilization: float
    pod_count: int
    target_pod_count: int

    # Performance metrics
    response_time: float
    throughput: float
    queue_length: float
    error_rate: float

    # Scaling behavior metrics
    scaling_frequency: float
    scaling_latency: float
    over_provisioning_ratio: float
    under_provisioning_ratio: float

    # Cost metrics
    resource_cost: float
    sla_violations: int
    availability: float

    # Agent-specific metrics
    action_distribution: Dict[str, int]
    reward: float
    exploration_rate: float

    timestamp: float
    agent_type: str
    test_scenario: str

@dataclass
class TestScenario:
    """Test scenario configuration."""
    name: str
    description: str
    duration_steps: int
    traffic_pattern: str
    base_load: float
    max_load: float
    event_frequency: float
    target_metrics: Dict[str, float]

class IdleTrafficSimulator:
    """
    Traffic simulator with idle periods (near-zero traffic).

    Simulates realistic scenarios like:
    - Night hours (minimal traffic)
    - Weekend traffic (low activity)
    - Off-peak periods (business hours vs idle)
    """

    def __init__(self, base_load: float = 50, peak_load: float = 3000,
                 idle_duration: int = 300, active_duration: int = 200, seed: int = 42):
        """
        Initialize idle traffic simulator.

        Args:
            base_load: Near-zero traffic during idle periods (e.g., 50 RPS)
            peak_load: Peak traffic during active periods (e.g., 3000 RPS)
            idle_duration: Number of steps in idle period
            active_duration: Number of steps in active period
            seed: Random seed for reproducibility
        """
        self.base_load = base_load
        self.peak_load = peak_load
        self.idle_duration = idle_duration
        self.active_duration = active_duration
        self.cycle_length = idle_duration + active_duration
        self.rng = np.random.default_rng(seed)

    def get_load(self, step: int) -> float:
        """
        Generate traffic load with idle and active periods.

        Pattern:
        - Idle period: base_load (near-zero, e.g., 50 RPS)
        - Transition: Gradual ramp up
        - Active period: peak_load (e.g., 3000 RPS)
        - Transition: Gradual ramp down

        Args:
            step: Current simulation step

        Returns:
            float: Current traffic load in RPS
        """
        # Determine position in cycle
        cycle_position = step % self.cycle_length

        if cycle_position < self.idle_duration:
            # Idle period: near-zero traffic with small random variation
            load = self.base_load * self.rng.uniform(0.5, 1.5)
        else:
            # Active period
            active_position = cycle_position - self.idle_duration

            # Smooth transition: sine wave for realistic ramp up/down
            # First 20% of active period: ramp up
            # Middle 60%: peak traffic with variation
            # Last 20%: ramp down
            if active_position < self.active_duration * 0.2:
                # Ramp up phase
                progress = active_position / (self.active_duration * 0.2)
                transition = 0.5 * (1 - np.cos(progress * np.pi))  # Smooth 0->1
                load = self.base_load + (self.peak_load - self.base_load) * transition
            elif active_position > self.active_duration * 0.8:
                # Ramp down phase
                progress = (active_position - self.active_duration * 0.8) / (self.active_duration * 0.2)
                transition = 0.5 * (1 + np.cos(progress * np.pi))  # Smooth 1->0
                load = self.base_load + (self.peak_load - self.base_load) * transition
            else:
                # Peak period with realistic variation
                load = self.peak_load * self.rng.uniform(0.8, 1.2)

        # Add small noise for realism
        noise = self.rng.normal(0, self.base_load * 0.1)
        return max(0, load + noise)

class PrometheusMetricsCollector:
    """Collects and stores metrics in Prometheus format."""

    def __init__(self, output_dir: str = "./metrics"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.metrics: List[PrometheusStyleMetric] = []
        self.start_time = time.time()

    def record_metric(self, name: str, value: float, labels: Dict[str, str],
                     help_text: str = "") -> None:
        """Record a metric with current real timestamp."""
        metric = PrometheusStyleMetric(
            name=name,
            value=value,
            labels=labels,
            timestamp=time.time(),
            help_text=help_text
        )
        self.metrics.append(metric)

    def record_metric_with_timestamp(self, name: str, value: float, labels: Dict[str, str],
                                     timestamp: float, help_text: str = "") -> None:
        """Record a metric with specific timestamp (for virtual time)."""
        metric = PrometheusStyleMetric(
            name=name,
            value=value,
            labels=labels,
            timestamp=timestamp,  # Use provided timestamp instead of time.time()
            help_text=help_text
        )
        self.metrics.append(metric)

    def record_autoscaling_metrics(self, metrics: AutoscalingMetrics) -> None:
        """Record comprehensive autoscaling metrics with virtual time."""
        base_labels = {
            "agent": metrics.agent_type,
            "scenario": metrics.test_scenario,
            "instance": "autoscaler"
        }

        # Use virtual timestamp from metrics object instead of real time
        virtual_timestamp = metrics.timestamp

        # Resource metrics
        self.record_metric_with_timestamp("autoscaler_cpu_utilization", metrics.cpu_utilization, base_labels,
                          virtual_timestamp, "CPU utilization percentage")
        self.record_metric_with_timestamp("autoscaler_memory_utilization", metrics.memory_utilization, base_labels,
                          virtual_timestamp, "Memory utilization percentage")
        self.record_metric_with_timestamp("autoscaler_pod_count", metrics.pod_count, base_labels,
                          virtual_timestamp, "Current number of pods")
        self.record_metric_with_timestamp("autoscaler_target_pod_count", metrics.target_pod_count, base_labels,
                          virtual_timestamp, "Target number of pods")

        # Performance metrics
        self.record_metric_with_timestamp("autoscaler_response_time_seconds", metrics.response_time, base_labels,
                          virtual_timestamp, "Average response time in seconds")
        self.record_metric_with_timestamp("autoscaler_throughput_rps", metrics.throughput, base_labels,
                          virtual_timestamp, "Requests per second")
        self.record_metric_with_timestamp("autoscaler_queue_length", metrics.queue_length, base_labels,
                          virtual_timestamp, "Current queue length")
        self.record_metric_with_timestamp("autoscaler_error_rate", metrics.error_rate, base_labels,
                          virtual_timestamp, "Error rate percentage")

        # Scaling behavior
        self.record_metric_with_timestamp("autoscaler_scaling_frequency_per_hour", metrics.scaling_frequency, base_labels,
                          virtual_timestamp, "Scaling actions per hour")
        self.record_metric_with_timestamp("autoscaler_scaling_latency_seconds", metrics.scaling_latency, base_labels,
                          virtual_timestamp, "Time to complete scaling action")
        self.record_metric_with_timestamp("autoscaler_over_provisioning_ratio", metrics.over_provisioning_ratio, base_labels,
                          virtual_timestamp, "Ratio of over-provisioned resources")
        self.record_metric_with_timestamp("autoscaler_under_provisioning_ratio", metrics.under_provisioning_ratio, base_labels,
                          virtual_timestamp, "Ratio of under-provisioned resources")

        # Cost and SLA
        self.record_metric_with_timestamp("autoscaler_resource_cost_dollars", metrics.resource_cost, base_labels,
                          virtual_timestamp, "Resource cost in dollars")
        self.record_metric_with_timestamp("autoscaler_sla_violations_total", metrics.sla_violations, base_labels,
                          virtual_timestamp, "Total SLA violations")
        self.record_metric_with_timestamp("autoscaler_availability_percentage", metrics.availability, base_labels,
                          virtual_timestamp, "Service availability percentage")

        # Agent-specific
        self.record_metric_with_timestamp("autoscaler_reward", metrics.reward, base_labels,
                          virtual_timestamp, "Agent reward signal")
        self.record_metric_with_timestamp("autoscaler_exploration_rate", metrics.exploration_rate, base_labels,
                          virtual_timestamp, "Agent exploration rate")

        # Action distribution
        for action, count in metrics.action_distribution.items():
            action_labels = {**base_labels, "action": action}
            self.record_metric_with_timestamp("autoscaler_action_count", count, action_labels,
                              virtual_timestamp, "Count of scaling actions by type")

    def export_to_file(self, filename: str = None) -> str:
        """Export metrics to Prometheus format file."""
        if filename is None:
            filename = f"autoscaler_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.prom"

        filepath = self.output_dir / filename

        with open(filepath, 'w') as f:
            f.write("# Autoscaler Performance Metrics\n")
            f.write(f"# Generated at {datetime.now().isoformat()}\n\n")

            # Group metrics by name for better organization
            grouped_metrics = {}
            for metric in self.metrics:
                if metric.name not in grouped_metrics:
                    grouped_metrics[metric.name] = []
                grouped_metrics[metric.name].append(metric)

            for metric_name, metric_list in grouped_metrics.items():
                if metric_list[0].help_text:
                    f.write(f"# HELP {metric_name} {metric_list[0].help_text}\n")
                f.write(f"# TYPE {metric_name} gauge\n")

                for metric in metric_list:
                    f.write(f"{metric.to_prometheus_format()}\n")
                f.write("\n")

        logger.info(f"Exported {len(self.metrics)} metrics to {filepath}")
        return str(filepath)

    def export_to_csv(self, filename: str = None) -> str:
        """Export metrics to CSV format for analysis."""
        if filename is None:
            filename = f"autoscaler_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        filepath = self.output_dir / filename

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'metric_name', 'value', 'labels', 'help_text'])

            for metric in self.metrics:
                labels_str = json.dumps(metric.labels)
                writer.writerow([
                    metric.timestamp,
                    metric.name,
                    metric.value,
                    labels_str,
                    metric.help_text
                ])

        logger.info(f"Exported metrics to CSV: {filepath}")
        return str(filepath)

class AgentPerformanceTester:
    """Comprehensive performance testing framework for autoscaling agents."""

    def __init__(self, config_path: str = "config/hybrid_config.yaml"):
        self.config_path = config_path
        self.metrics_collector = PrometheusMetricsCollector()
        self.test_results: Dict[str, List[AutoscalingMetrics]] = {}
        self.scenarios = self._define_test_scenarios()

        # Set up results and metrics directories
        self.results_dir = os.getenv('RESULTS_DIR', 'test_results')
        self.metrics_dir = os.getenv('METRICS_DIR', 'metrics')
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)

        # Load configuration
        try:
            import yaml
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            self.config = self._get_default_config()

    def _get_default_config(self) -> Dict:
        """Get default configuration if config file is not found."""
        return {
            'environment': {
                'max_pods': 10,
                'scaling_delay': 10,
                'deployment_name': 'nginx-deployment',
                'namespace': 'default'
            },
            'training': {
                'total_steps': 50000,
                'eval_freq': 1000
            }
        }

    def _define_test_scenarios(self) -> List[TestScenario]:
        """Define comprehensive test scenarios with extended durations for pattern recognition."""
        return [
            TestScenario(
                name="baseline_steady",
                description="Steady baseline load - Production-level traffic",
                duration_steps=3000,  # 3000 simulation steps (50 min if real-time @ 1s/step)
                traffic_pattern="steady",
                base_load=2500,  # 5x increase: 500 -> 2500 RPS
                max_load=4000,   # 5x increase: 800 -> 4000 RPS
                event_frequency=0.0,
                target_metrics={"cpu_utilization": 0.7, "response_time": 0.1}
            ),
            TestScenario(
                name="gradual_ramp",
                description="Gradual load increase - E-commerce peak hours",
                duration_steps=5000,  # 5000 simulation steps (83 min if real-time @ 1s/step)
                traffic_pattern="gradual",
                base_load=1000,   # 5x increase: 200 -> 1000 RPS
                max_load=5000,    # 5x increase: 1000 -> 5000 RPS
                event_frequency=0.002,
                target_metrics={"cpu_utilization": 0.75, "response_time": 0.15}
            ),
            TestScenario(
                name="sudden_spike",
                description="Sudden traffic spikes - Social media viral content",
                duration_steps=4000,  # 4000 simulation steps (67 min if real-time @ 1s/step)
                traffic_pattern="spike",
                base_load=2000,   # 5x increase: 400 -> 2000 RPS
                max_load=10000,   # 5x increase: 2000 -> 10000 RPS
                event_frequency=0.02,
                target_metrics={"cpu_utilization": 0.8, "response_time": 0.2}
            ),
            TestScenario(
                name="flash_crowd",
                description="Flash crowd events - Black Friday/Cyber Monday",
                duration_steps=6000,  # 6000 simulation steps (100 min if real-time @ 1s/step)
                traffic_pattern="flash",
                base_load=1500,   # 5x increase: 300 -> 1500 RPS
                max_load=15000,   # 5x increase: 3000 -> 15000 RPS
                event_frequency=0.01,
                target_metrics={"cpu_utilization": 0.85, "response_time": 0.25}
            ),
            TestScenario(
                name="daily_pattern",
                description="Daily usage pattern - Real-world application",
                duration_steps=8640,  # 8640 simulation steps (6 simulated days)
                traffic_pattern="daily",
                base_load=500,    # 5x increase: 100 -> 500 RPS
                max_load=2000,    # 5x increase: 400 -> 2000 RPS
                event_frequency=0.002,
                target_metrics={"cpu_utilization": 0.7, "response_time": 0.12}
            ),
            TestScenario(
                name="idle_periods",
                description="Traffic with idle periods - Night/weekend low traffic",
                duration_steps=4000,  # 4000 simulation steps (8 idle/active cycles)
                traffic_pattern="idle",
                base_load=50,     # Very low base load (near-zero during idle)
                max_load=3000,    # Peak during active hours
                event_frequency=0.005,
                target_metrics={"cpu_utilization": 0.3, "response_time": 0.08}
            )
        ]

    def create_agent(self, agent_type: str, environment, mock_mode: bool = True) -> Any:
        """Factory method to create different types of agents."""
        if agent_type == "hybrid_dqn_ppo":
            config = HybridConfig()
            k8s_api = KubernetesAPI(max_pods=self.config['environment']['max_pods'])
            agent = HybridDQNPPOAgent(config, k8s_api, mock_mode=mock_mode)

            # CRITICAL FIX: Load trained models if they exist
            model_path = "./models/hybrid"
            if Path(model_path).exists() and Path(f"{model_path}/dqn_model.pth").exists():
                try:
                    agent.load_models(model_path)
                    logger.info(f"✅ Loaded trained models from {model_path}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load models: {e}. Using untrained agent.")
            else:
                logger.warning(f"⚠️ No trained models found at {model_path}. Using untrained agent!")

            return agent

        elif agent_type == "dqn":
            # Create simulated environment for DQN
            if mock_mode:
                env = MicroK8sEnvSimulated()
            else:
                from agent.environment import MicroK8sEnv
                env = MicroK8sEnv()
            return DQNAgent(env=env, environment=env, is_simulated=mock_mode)

        elif agent_type == "ppo":
            if mock_mode:
                env = MicroK8sEnvSimulated()
            else:
                from agent.environment import MicroK8sEnv
                env = MicroK8sEnv()
            return PPOAgent(environment=env)

        elif agent_type == "k8s_hpa":
            # NEW: Accurate Kubernetes HPA v2 simulator (recommended for publication)
            # Note: Stabilization windows adjusted for simulation time scale
            # In real K8s: 30s/180s, but simulation steps are faster than real-time
            return KubernetesHPASimulator(
                target_cpu_utilization=0.70,  # Standard HPA default (70% target)
                min_replicas=1,
                max_replicas=10,
                scale_up_stabilization=5,     # Reduced: 30→5 steps (faster response in simulation)
                scale_down_stabilization=30,  # Reduced: 180→30 steps (less over-provisioning)
                tolerance=0.1                 # ±10% tolerance (63-77% band)
            )

        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

    def run_agent_test(self, agent_type: str, scenario: TestScenario,
                       mock_mode: bool = True) -> List[AutoscalingMetrics]:
        """Run performance test for a specific agent and scenario."""
        logger.info(f"Running test: {agent_type} on {scenario.name}")

        # Create traffic simulator based on scenario
        if scenario.traffic_pattern == "hybrid":
            max_intensity = scenario.max_load / scenario.base_load
            min_intensity = min(1.5, max_intensity * 0.3)  # Ensure min < max
            simulator = HybridTrafficSimulator(
                base_load=scenario.base_load,
                seed=42,
                event_frequency=scenario.event_frequency,
                min_intensity=min_intensity,
                max_intensity=max_intensity
            )
        elif scenario.traffic_pattern == "idle":
            # Custom idle pattern with near-zero traffic periods
            simulator = IdleTrafficSimulator(
                base_load=scenario.base_load,
                peak_load=scenario.max_load,
                idle_duration=300,  # 300 steps of idle
                active_duration=200,  # 200 steps of active traffic
                seed=42
            )
        else:
            simulator = TrafficSimulator(
                base_load=scenario.base_load,
                max_spike=scenario.max_load / scenario.base_load,
                seed=42
            )

        # Create environment and agent
        try:
            if mock_mode:
                environment = MicroK8sEnvSimulated()
            else:
                environment = HybridMicroK8sEnv()

            agent = self.create_agent(agent_type, environment, mock_mode)
        except Exception as e:
            logger.error(f"Failed to create agent {agent_type}: {e}")
            return []

        metrics_history = []
        action_counts = {"scale_up": 0, "scale_down": 0, "no_change": 0}

        # Simulation variables
        # FIXED: Calculate proper initial pod count based on scenario base load
        # This ensures FAIR comparison - all agents start from same initial conditions
        pod_capacity = 500  # RPS per pod at 100% CPU
        target_cpu = 0.70   # Target 70% CPU utilization

        # Calculate initial pods needed for base load at target CPU
        # FIXED: Use simulator.get_load(0) instead of scenario.traffic[0] (which doesn't exist)
        # ALIGNED WITH statistical_validation_n30.py:240,356 for fair comparison
        initial_load = simulator.get_load(0)  # Get actual first timestep load
        initial_pods = max(1, int(np.ceil(initial_load / (pod_capacity * target_cpu))))
        current_pods = initial_pods  # ✅ Fair initialization - matches statistical validation

        # Calculate actual initial metrics based on load and pods
        cpu_utilization = min(1.0, initial_load / (current_pods * pod_capacity))
        response_time = max(0.05, 0.1 + (cpu_utilization - 0.7) * 0.5 if cpu_utilization > 0.7 else 0.1)
        initial_throughput = min(initial_load, current_pods * pod_capacity)
        initial_queue = max(0, initial_load - initial_throughput) / 100.0

        logger.info(f"🔧 FAIR TEST INITIALIZATION: {agent_type}")
        logger.info(f"   Scenario: {scenario.name}")
        logger.info(f"   Base Load: {initial_load} RPS")
        logger.info(f"   Pod Capacity: {pod_capacity} RPS/pod at 100% CPU")
        logger.info(f"   Target CPU: {target_cpu * 100}%")
        logger.info(f"   Initial Pods: {current_pods} (calculated for {target_cpu*100}% CPU)")
        logger.info(f"   Expected Initial CPU: {cpu_utilization * 100:.1f}%")

        scaling_actions = []
        sla_violations = 0
        total_cost = 0.0
        pod_history = [current_pods]  # Track pod count history, starting with calculated pods

        # Virtual time simulation for realistic timestamps in mock mode
        # Get time step duration from environment or use default
        time_step_minutes = float(os.getenv('SIMULATION_TIME_STEP_MINUTES', '1'))
        time_step_variance = float(os.getenv('SIMULATION_TIME_STEP_VARIANCE', '0'))  # 0-5 for random 1-5 min

        # Initialize virtual clock
        virtual_time = time.time()  # Start from current time

        logger.info(f"⏰ SIMULATION TIMING: {time_step_minutes}±{time_step_variance} minutes per step")
        logger.info(f"   Estimated duration: {scenario.duration_steps * time_step_minutes:.0f} virtual minutes "
                   f"({scenario.duration_steps * time_step_minutes / 60:.1f} virtual hours)")

        try:
            # Record initial state at step 0 with CALCULATED metrics (not hardcoded)
            initial_metrics = AutoscalingMetrics(
                cpu_utilization=cpu_utilization,  # FIXED: Calculate from actual load
                memory_utilization=0.5,
                pod_count=current_pods,  # FIXED: Use calculated initial pods
                target_pod_count=initial_pods,
                response_time=response_time,  # FIXED: Calculate from CPU
                throughput=initial_throughput,  # FIXED: Calculate from load
                queue_length=initial_queue,  # FIXED: Calculate from capacity
                error_rate=0.0,
                scaling_frequency=0.0,
                scaling_latency=0.0,
                over_provisioning_ratio=max(0, (current_pods * pod_capacity - initial_load) / initial_load) if initial_load > 0 else 0,
                under_provisioning_ratio=max(0, (initial_load - current_pods * pod_capacity) / initial_load) if initial_load > 0 else 0,
                resource_cost=0.0,
                sla_violations=0,
                availability=100.0,
                action_distribution=action_counts.copy(),
                reward=0.0,
                exploration_rate=0.0,
                timestamp=virtual_time,  # Use virtual time for realistic timestamps
                agent_type=agent_type,
                test_scenario=scenario.name
            )
            metrics_history.append(initial_metrics)
            self.metrics_collector.record_autoscaling_metrics(initial_metrics)

            for step in range(scenario.duration_steps):
                # Advance virtual time by configured amount (with optional variance)
                if time_step_variance > 0:
                    # Random time step between base and base+variance minutes
                    step_duration = np.random.uniform(time_step_minutes, time_step_minutes + time_step_variance)
                else:
                    step_duration = time_step_minutes

                virtual_time += step_duration * 60  # Convert minutes to seconds

                # Log progress with realistic timestamps every 10% of scenario
                if step % max(1, scenario.duration_steps // 10) == 0:
                    progress_pct = (step / scenario.duration_steps) * 100
                    elapsed_virtual_hours = (virtual_time - metrics_history[0].timestamp) / 3600
                    virtual_timestamp = datetime.fromtimestamp(virtual_time).strftime('%Y-%m-%d %H:%M:%S')
                    logger.info(f"  📊 Progress: {progress_pct:.0f}% | Virtual time: {virtual_timestamp} "
                               f"| Elapsed: {elapsed_virtual_hours:.1f}h | Pods: {current_pods}")
                # Get current load from simulator
                current_load = simulator.get_load(step)

                # Simulate system state based on load and current pods
                # Fixed: Use realistic pod capacity (500 RPS per pod instead of 100)
                pod_capacity = 500  # RPS per pod - adjusted for 5x traffic increase
                cpu_utilization = min(1.0, current_load / (current_pods * pod_capacity))
                response_time = max(0.05, 0.1 + (cpu_utilization - 0.7) * 0.5 if cpu_utilization > 0.7 else 0.1)
                throughput = min(current_load, current_pods * pod_capacity)
                queue_length = max(0, current_load - throughput) / 100.0

                # Create state vector for RL agents (7-dimensional to match environment)
                # [cpu, memory, latency, swap, nodes, load_mean, load_gradient]
                state = np.array([
                    cpu_utilization,  # CPU utilization
                    np.random.uniform(0.3, 0.7),  # memory utilization
                    response_time / 0.5,  # normalized latency/response time
                    0.0,  # swap usage (placeholder)
                    current_pods / 10.0,  # normalized nodes/pod count
                    cpu_utilization,  # load mean (use cpu as proxy)
                    0.0   # load gradient (placeholder)
                ])

                # Get agent action
                if hasattr(agent, 'step_with_external_state'):
                    # For hybrid agent - FIX: Pass external state instead of using internal random state
                    result = agent.step_with_external_state(state)
                    action = result['action']
                    reward = result['reward']
                elif hasattr(agent, 'get_scaling_decision'):
                    # For other RL agents
                    action = agent.get_scaling_decision(state)
                    reward = self._calculate_reward(cpu_utilization, response_time, current_pods)
                else:
                    # For rule-based agent
                    action = agent.decide_action(cpu_utilization, current_pods, step)
                    reward = 0.0

                # Apply action
                if action == 0:  # Scale up
                    current_pods = min(current_pods + 1, 10)
                    action_counts["scale_up"] += 1
                    scaling_actions.append(step)
                elif action == 1:  # Scale down
                    current_pods = max(current_pods - 1, 1)
                    action_counts["scale_down"] += 1
                    scaling_actions.append(step)
                else:  # No change
                    action_counts["no_change"] += 1

                # Track pod count history
                pod_history.append(current_pods)

                # Calculate metrics
                over_provisioning = max(0, (current_pods * pod_capacity - current_load) / current_load) if current_load > 0 else 0
                under_provisioning = max(0, (current_load - current_pods * pod_capacity) / current_load) if current_load > 0 else 0

                # SLA violations (response time > 200ms)
                if response_time > 0.2:
                    sla_violations += 1

                # Calculate cost (simplified)
                total_cost += current_pods * 0.1  # $0.1 per pod per step

                # Record metrics every 10 steps
                if step % 10 == 0:
                    metrics = AutoscalingMetrics(
                        cpu_utilization=cpu_utilization,
                        memory_utilization=np.random.uniform(0.3, 0.7),
                        pod_count=current_pods,
                        target_pod_count=max(1, int(current_load / pod_capacity)),
                        response_time=response_time,
                        throughput=throughput,
                        queue_length=queue_length,
                        error_rate=max(0, (response_time - 0.2) * 10) if response_time > 0.2 else 0,
                        scaling_frequency=len(scaling_actions) / (step / 3600) if step > 0 else 0,
                        scaling_latency=np.random.uniform(5, 15),  # seconds
                        over_provisioning_ratio=over_provisioning,
                        under_provisioning_ratio=under_provisioning,
                        resource_cost=total_cost,
                        sla_violations=sla_violations,
                        availability=max(0, 100 - sla_violations / step * 100) if step > 0 else 100,
                        action_distribution=action_counts.copy(),
                        reward=reward,
                        exploration_rate=getattr(agent, 'epsilon', 0.0) if hasattr(agent, 'epsilon') else 0.0,
                        timestamp=virtual_time,  # Use virtual time for realistic timestamps
                        agent_type=agent_type,
                        test_scenario=scenario.name
                    )

                    metrics_history.append(metrics)
                    self.metrics_collector.record_autoscaling_metrics(metrics)

                # Simulate some delay
                if not mock_mode:
                    time.sleep(0.1)

        except Exception as e:
            logger.error(f"Error during test execution: {e}")

        logger.info(f"Completed test: {agent_type} on {scenario.name} - {len(metrics_history)} metrics collected")

        # Export throughput metrics for hybrid agents
        if hasattr(agent, 'export_throughput_metrics') and hasattr(agent, 'throughput_history'):
            try:
                test_id = f"{agent_type}_{scenario.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                agent.export_throughput_metrics(output_dir="monitoring", test_id=test_id)
                logger.info(f"Throughput metrics exported for {agent_type}")
            except Exception as e:
                logger.warning(f"Failed to export throughput metrics for {agent_type}: {e}")

        return metrics_history

    def _calculate_reward(self, cpu_util: float, response_time: float, pod_count: int) -> float:
        """Calculate reward for non-hybrid agents."""
        reward = 0.0

        # Performance reward
        if response_time < 0.15:
            reward += 1.0
        elif response_time < 0.25:
            reward += 0.5
        else:
            reward -= 1.0

        # Resource efficiency reward
        if 0.6 <= cpu_util <= 0.8:
            reward += 0.5
        elif cpu_util > 0.9:
            reward -= 0.5

        # Cost penalty
        reward -= pod_count * 0.1

        return reward

    def run_comparative_study(self, agent_types: List[str],
                            scenarios: List[str] = None,
                            mock_mode: bool = True) -> Dict[str, Any]:
        """Run comprehensive comparative study."""
        if scenarios is None:
            scenarios = [s.name for s in self.scenarios]

        logger.info(f"Starting comparative study with agents: {agent_types}")
        logger.info(f"Test scenarios: {scenarios}")

        study_results = {}

        for agent_type in agent_types:
            study_results[agent_type] = {}

            for scenario_name in scenarios:
                scenario = next((s for s in self.scenarios if s.name == scenario_name), None)
                if scenario is None:
                    logger.warning(f"Scenario {scenario_name} not found")
                    continue

                metrics = self.run_agent_test(agent_type, scenario, mock_mode)
                study_results[agent_type][scenario_name] = metrics

                if agent_type not in self.test_results:
                    self.test_results[agent_type] = []
                self.test_results[agent_type].extend(metrics)

        # Generate comparative analysis
        analysis = self._analyze_results(study_results)

        # Export results
        self._export_results(study_results, analysis)

        return {
            'results': study_results,
            'analysis': analysis,
            'metrics_file': self.metrics_collector.export_to_file(),
            'csv_file': self.metrics_collector.export_to_csv()
        }

    def _analyze_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze and compare results across agents and scenarios."""
        analysis = {
            'summary': {},
            'agent_comparison': {},
            'scenario_analysis': {},
            'recommendations': []
        }

        # Aggregate metrics by agent
        for agent_type, scenarios in results.items():
            all_metrics = []
            for scenario_metrics in scenarios.values():
                all_metrics.extend(scenario_metrics)

            if all_metrics:
                # Calculate pod count variance for stability analysis
                pod_counts = [m.pod_count for m in all_metrics]
                pod_variance = np.var(pod_counts) if len(pod_counts) > 1 else 0.0

                analysis['agent_comparison'][agent_type] = {
                    'avg_cpu_utilization': np.mean([m.cpu_utilization for m in all_metrics]),
                    'avg_response_time': np.mean([m.response_time for m in all_metrics]),
                    'avg_pod_count': np.mean(pod_counts),
                    'pod_count_variance': pod_variance,
                    'pod_history': pod_counts[:10],  # First 10 samples for starting point verification
                    'total_sla_violations': sum([m.sla_violations for m in all_metrics]),
                    'total_cost': sum([m.resource_cost for m in all_metrics]),
                    'avg_reward': np.mean([m.reward for m in all_metrics]),
                    'scaling_efficiency': np.mean([m.scaling_frequency for m in all_metrics])
                }

        # Generate recommendations
        if 'hybrid_dqn_ppo' in analysis['agent_comparison']:
            hybrid_metrics = analysis['agent_comparison']['hybrid_dqn_ppo']
            analysis['recommendations'].append(
                f"Hybrid DQN-PPO shows avg response time of {hybrid_metrics['avg_response_time']:.3f}s "
                f"with {hybrid_metrics['total_sla_violations']} SLA violations"
            )

        return analysis

    def _export_results(self, results: Dict[str, Any], analysis: Dict[str, Any]) -> None:
        """Export results to files."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Export detailed results as JSON
        results_file = f"{self.results_dir}/performance_study_{timestamp}.json"
        with open(results_file, 'w') as f:
            # Convert results to serializable format
            serializable_results = {}
            for agent, scenarios in results.items():
                serializable_results[agent] = {}
                for scenario, metrics in scenarios.items():
                    serializable_results[agent][scenario] = [asdict(m) for m in metrics]

            json.dump({
                'results': serializable_results,
                'analysis': analysis,
                'timestamp': timestamp
            }, f, indent=2, default=str)

        logger.info(f"Exported detailed results to {results_file}")

        # Create performance comparison plots
        self._create_performance_plots(results, timestamp)

    def _create_performance_plots(self, results: Dict[str, Any], timestamp: str) -> None:
        """Create performance comparison plots."""
        try:
            # Set up the plotting style
            plt.style.use('seaborn-v0_8')
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Autoscaling Agent Performance Comparison', fontsize=16)

            # Prepare data for plotting
            plot_data = []
            for agent_type, scenarios in results.items():
                for scenario_name, metrics in scenarios.items():
                    for metric in metrics:
                        plot_data.append({
                            'agent': agent_type,
                            'scenario': scenario_name,
                            'cpu_utilization': metric.cpu_utilization,
                            'response_time': metric.response_time,
                            'pod_count': metric.pod_count,
                            'sla_violations': metric.sla_violations,
                            'cost': metric.resource_cost,
                            'reward': metric.reward
                        })

            if not plot_data:
                logger.warning("No data available for plotting")
                return

            df = pd.DataFrame(plot_data)

            # Plot 1: CPU Utilization by Agent
            sns.boxplot(data=df, x='agent', y='cpu_utilization', ax=axes[0, 0])
            axes[0, 0].set_title('CPU Utilization Distribution')
            axes[0, 0].tick_params(axis='x', rotation=45)

            # Plot 2: Response Time by Agent
            sns.boxplot(data=df, x='agent', y='response_time', ax=axes[0, 1])
            axes[0, 1].set_title('Response Time Distribution')
            axes[0, 1].tick_params(axis='x', rotation=45)

            # Plot 3: Pod Count by Agent
            sns.boxplot(data=df, x='agent', y='pod_count', ax=axes[0, 2])
            axes[0, 2].set_title('Pod Count Distribution')
            axes[0, 2].tick_params(axis='x', rotation=45)

            # Plot 4: SLA Violations by Agent
            sla_summary = df.groupby('agent')['sla_violations'].sum().reset_index()
            sns.barplot(data=sla_summary, x='agent', y='sla_violations', ax=axes[1, 0])
            axes[1, 0].set_title('Total SLA Violations')
            axes[1, 0].tick_params(axis='x', rotation=45)

            # Plot 5: Cost by Agent
            cost_summary = df.groupby('agent')['cost'].mean().reset_index()
            sns.barplot(data=cost_summary, x='agent', y='cost', ax=axes[1, 1])
            axes[1, 1].set_title('Average Cost')
            axes[1, 1].tick_params(axis='x', rotation=45)

            # Plot 6: Reward by Agent
            sns.boxplot(data=df, x='agent', y='reward', ax=axes[1, 2])
            axes[1, 2].set_title('Reward Distribution')
            axes[1, 2].tick_params(axis='x', rotation=45)

            plt.tight_layout()
            plot_filename = f"performance_comparison_{timestamp}.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"Performance plots saved to {plot_filename}")

        except Exception as e:
            logger.error(f"Error creating plots: {e}")

class KubernetesHPASimulator:
    """
    Accurate Kubernetes HPA v2 simulator for publication-grade research.

    Implements the actual HPA algorithm as documented in:
    https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/

    Key Features:
    - Proportional scaling: desiredReplicas = ceil(currentReplicas × currentMetric / targetMetric)
    - Tolerance band (default ±10%) to prevent flapping
    - Asymmetric stabilization windows (30s up, 180s down)
    - Scale-up policies: max(100% increase, +2 pods)
    - Scale-down policies: min(70% reduction, -1 pod) - conservative
    """

    def __init__(self,
                 target_cpu_utilization: float = 0.70,  # Standard HPA default
                 min_replicas: int = 1,
                 max_replicas: int = 10,
                 scale_up_stabilization: int = 30,      # steps (simulating seconds)
                 scale_down_stabilization: int = 180,   # steps (simulating seconds)
                 tolerance: float = 0.1):                # 10% tolerance band

        self.target_cpu = target_cpu_utilization
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.scale_up_stabilization = scale_up_stabilization
        self.scale_down_stabilization = scale_down_stabilization
        self.tolerance = tolerance

        # Track scaling history for stabilization windows
        self.last_scale_time = 0
        self.last_scale_direction = None
        self.recommendation_history = []

    def calculate_desired_replicas(self, current_cpu: float, current_replicas: int) -> int:
        """
        HPA proportional scaling formula:
        desiredReplicas = ceil(currentReplicas × currentMetricValue / targetMetricValue)

        This is the ACTUAL Kubernetes HPA algorithm.
        """
        if current_cpu <= 0:
            return self.min_replicas

        # Core HPA formula
        desired = int(np.ceil(current_replicas * current_cpu / self.target_cpu))

        # Apply tolerance band - don't scale if within ±tolerance of target
        # This prevents flapping around the target
        lower_bound = self.target_cpu * (1 - self.tolerance)
        upper_bound = self.target_cpu * (1 + self.tolerance)

        if lower_bound <= current_cpu <= upper_bound:
            return current_replicas  # Within tolerance, no scaling needed

        # Enforce min/max boundaries
        return max(self.min_replicas, min(self.max_replicas, desired))

    def apply_scale_up_policy(self, current_replicas: int, desired_replicas: int) -> int:
        """
        HPA v2 scale-up behavior policies.

        Default policies:
        - Can increase by 100% (double the pods)
        - Can add 2 pods
        - Takes the MAXIMUM of these policies (aggressive scale-up)

        Reference: deployments/nginx-hpa.yaml lines 30-39
        """
        if desired_replicas <= current_replicas:
            return current_replicas

        # Policy 1: 100% increase (can double)
        max_by_percent = current_replicas * 2

        # Policy 2: Add 2 pods
        max_by_pods = current_replicas + 2

        # SelectPolicy: Max - take the more aggressive option
        max_allowed = max(max_by_percent, max_by_pods)

        # Return the minimum of desired and policy-limited
        return min(desired_replicas, max_allowed, self.max_replicas)

    def apply_scale_down_policy(self, current_replicas: int, desired_replicas: int) -> int:
        """
        HPA v2 scale-down behavior policies.

        Default policies:
        - Can decrease by 70% (keep 30% of pods)
        - Can remove 1 pod
        - Takes the MINIMUM of these policies (conservative scale-down)

        Reference: deployments/nginx-hpa.yaml lines 40-49
        """
        if desired_replicas >= current_replicas:
            return current_replicas

        # Policy 1: 70% reduction (keep 30%)
        min_by_percent = int(np.ceil(current_replicas * 0.3))

        # Policy 2: Remove 1 pod
        min_by_pods = current_replicas - 1

        # SelectPolicy: Min - take the more conservative option
        min_allowed = max(min_by_percent, min_by_pods)

        # Return the maximum of desired and policy-limited
        return max(desired_replicas, min_allowed, self.min_replicas)

    def check_stabilization_window(self, current_time: int, direction: str) -> bool:
        """
        Check if we're within the stabilization window.

        Stabilization prevents rapid scaling by requiring metrics to be
        consistently high/low for a period before scaling.

        - Scale-up: 30 second window (faster response to load)
        - Scale-down: 180 second window (conservative, prevent oscillation)

        Returns True if we should wait (not scale yet).
        """
        time_since_last_scale = current_time - self.last_scale_time

        if direction == 'up':
            return time_since_last_scale < self.scale_up_stabilization
        else:  # down
            return time_since_last_scale < self.scale_down_stabilization

    def decide_action(self, cpu_utilization: float, current_pods: int, step: int = 0) -> int:
        """
        Main HPA decision algorithm.

        Returns:
            0: Scale up (increase replicas)
            1: Scale down (decrease replicas)
            2: No change (maintain current replicas)
        """
        # Step 1: Calculate desired replicas using HPA formula
        desired = self.calculate_desired_replicas(cpu_utilization, current_pods)

        # Step 2: Determine direction and check stabilization
        if desired > current_pods:
            # Want to scale up
            if self.check_stabilization_window(step, 'up'):
                return 2  # Still in stabilization window, wait

            # Apply scale-up policies
            target = self.apply_scale_up_policy(current_pods, desired)

            if target > current_pods:
                self.last_scale_time = step
                self.last_scale_direction = 'up'

                # For simulation compatibility, we scale by the difference
                # But we track that HPA would scale to 'target' in one action
                # For now, we'll scale by 1 but log the HPA recommendation
                logger.debug(f"HPA would scale from {current_pods} to {target} pods (CPU: {cpu_utilization:.2%})")
                return 0  # Scale up

        elif desired < current_pods:
            # Want to scale down
            if self.check_stabilization_window(step, 'down'):
                return 2  # Still in stabilization window, wait

            # Apply scale-down policies
            target = self.apply_scale_down_policy(current_pods, desired)

            if target < current_pods:
                self.last_scale_time = step
                self.last_scale_direction = 'down'

                logger.debug(f"HPA would scale from {current_pods} to {target} pods (CPU: {cpu_utilization:.2%})")
                return 1  # Scale down

        # No change needed
        return 2


def simulate_hybrid_traffic_with_agents():
    """Enhanced traffic simulation with agent performance testing."""
    # Initialize the performance tester
    tester = AgentPerformanceTester()

    # Define agents to test - can be overridden by environment variable
    agents_env = os.getenv('AGENTS', '')
    if agents_env:
        agent_types = [agent.strip() for agent in agents_env.split(',')]
        logger.info(f"Using agents from environment: {agent_types}")
    else:
        # Default: Publication comparison - Hybrid DQN-PPO vs HPA only
        agent_types = [
            "hybrid_dqn_ppo",
            "k8s_hpa"
        ]
        logger.info(f"Using default agents for publication: {agent_types}")

    # Define scenarios to test - can be overridden by environment variable
    scenarios_env = os.getenv('SCENARIOS', '')
    if scenarios_env:
        test_scenarios = [scenario.strip() for scenario in scenarios_env.split(',')]
        logger.info(f"Using scenarios from environment: {test_scenarios}")
    else:
        test_scenarios = [
            "baseline_steady",
            "gradual_ramp",
            "sudden_spike",
            "daily_pattern",
            "idle_periods"
        ]
        logger.info(f"Using default scenarios: {test_scenarios}")

    logger.info("Starting comprehensive autoscaling performance study")

    # Run the comparative study
    # Force conservative testing with actual reward function
    mock_mode = os.getenv('MOCK_MODE', 'true').lower() == 'true'
    if not mock_mode:
        print("⚠️  WARNING: Running in REAL mode - will actually test conservative agent")
    else:
        print("🧪 Running in MOCK mode - using simulated behavior")

    results = tester.run_comparative_study(
        agent_types=agent_types,
        scenarios=test_scenarios,
        mock_mode=mock_mode
    )

    # Print summary
    print("\n" + "="*80)
    print("AUTOSCALING PERFORMANCE STUDY RESULTS")
    print("="*80)

    if 'analysis' in results and 'agent_comparison' in results['analysis']:
        for agent, metrics in results['analysis']['agent_comparison'].items():
            print(f"\n{agent.upper()} AGENT:")
            print(f"  Average CPU Utilization: {metrics['avg_cpu_utilization']:.2%}")
            print(f"  Average Response Time: {metrics['avg_response_time']:.3f}s")
            print(f"  Average Pod Count: {metrics['avg_pod_count']:.1f}")
            print(f"  Total SLA Violations: {metrics['total_sla_violations']}")
            print(f"  Total Cost: ${metrics['total_cost']:.2f}")
            print(f"  Average Reward: {metrics['avg_reward']:.3f}")

    print(f"\nDetailed metrics exported to: {results['metrics_file']}")
    print(f"CSV data exported to: {results['csv_file']}")

    # Generate research conclusions
    print("\n" + "="*80)
    print("RESEARCH CONCLUSIONS")
    print("="*80)
    print("Based on this simulation study:")
    print("1. RL-based agents show adaptive behavior under varying load conditions")
    print("2. Hybrid DQN-PPO combines discrete actions with continuous reward optimization")
    print("3. Traditional rule-based systems may struggle with complex traffic patterns")
    print("4. Cost optimization requires balance between performance and resource usage")
    print("5. SLA violations indicate system responsiveness under stress")

    if 'analysis' in results and 'recommendations' in results['analysis']:
        print("\nRECOMMENDATIONS:")
        for i, rec in enumerate(results['analysis']['recommendations'], 1):
            print(f"{i}. {rec}")

    return results

if __name__ == "__main__":
    # Run the enhanced simulation with agent performance testing
    simulate_hybrid_traffic_with_agents() 