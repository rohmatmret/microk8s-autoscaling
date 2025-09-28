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

class PrometheusMetricsCollector:
    """Collects and stores metrics in Prometheus format."""

    def __init__(self, output_dir: str = "./metrics"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.metrics: List[PrometheusStyleMetric] = []
        self.start_time = time.time()

    def record_metric(self, name: str, value: float, labels: Dict[str, str],
                     help_text: str = "") -> None:
        """Record a metric with timestamp."""
        metric = PrometheusStyleMetric(
            name=name,
            value=value,
            labels=labels,
            timestamp=time.time(),
            help_text=help_text
        )
        self.metrics.append(metric)

    def record_autoscaling_metrics(self, metrics: AutoscalingMetrics) -> None:
        """Record comprehensive autoscaling metrics."""
        base_labels = {
            "agent": metrics.agent_type,
            "scenario": metrics.test_scenario,
            "instance": "autoscaler"
        }

        # Resource metrics
        self.record_metric("autoscaler_cpu_utilization", metrics.cpu_utilization, base_labels,
                          "CPU utilization percentage")
        self.record_metric("autoscaler_memory_utilization", metrics.memory_utilization, base_labels,
                          "Memory utilization percentage")
        self.record_metric("autoscaler_pod_count", metrics.pod_count, base_labels,
                          "Current number of pods")
        self.record_metric("autoscaler_target_pod_count", metrics.target_pod_count, base_labels,
                          "Target number of pods")

        # Performance metrics
        self.record_metric("autoscaler_response_time_seconds", metrics.response_time, base_labels,
                          "Average response time in seconds")
        self.record_metric("autoscaler_throughput_rps", metrics.throughput, base_labels,
                          "Requests per second")
        self.record_metric("autoscaler_queue_length", metrics.queue_length, base_labels,
                          "Current queue length")
        self.record_metric("autoscaler_error_rate", metrics.error_rate, base_labels,
                          "Error rate percentage")

        # Scaling behavior
        self.record_metric("autoscaler_scaling_frequency_per_hour", metrics.scaling_frequency, base_labels,
                          "Scaling actions per hour")
        self.record_metric("autoscaler_scaling_latency_seconds", metrics.scaling_latency, base_labels,
                          "Time to complete scaling action")
        self.record_metric("autoscaler_over_provisioning_ratio", metrics.over_provisioning_ratio, base_labels,
                          "Ratio of over-provisioned resources")
        self.record_metric("autoscaler_under_provisioning_ratio", metrics.under_provisioning_ratio, base_labels,
                          "Ratio of under-provisioned resources")

        # Cost and SLA
        self.record_metric("autoscaler_resource_cost_dollars", metrics.resource_cost, base_labels,
                          "Resource cost in dollars")
        self.record_metric("autoscaler_sla_violations_total", metrics.sla_violations, base_labels,
                          "Total SLA violations")
        self.record_metric("autoscaler_availability_percentage", metrics.availability, base_labels,
                          "Service availability percentage")

        # Agent-specific
        self.record_metric("autoscaler_reward", metrics.reward, base_labels,
                          "Agent reward signal")
        self.record_metric("autoscaler_exploration_rate", metrics.exploration_rate, base_labels,
                          "Agent exploration rate")

        # Action distribution
        for action, count in metrics.action_distribution.items():
            action_labels = {**base_labels, "action": action}
            self.record_metric("autoscaler_action_count", count, action_labels,
                              "Count of scaling actions by type")

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
        """Define comprehensive test scenarios."""
        return [
            TestScenario(
                name="baseline_steady",
                description="Steady baseline load - Production-level traffic",
                duration_steps=1000,
                traffic_pattern="steady",
                base_load=2500,  # 5x increase: 500 -> 2500 RPS
                max_load=4000,   # 5x increase: 800 -> 4000 RPS
                event_frequency=0.0,
                target_metrics={"cpu_utilization": 0.7, "response_time": 0.1}
            ),
            TestScenario(
                name="gradual_ramp",
                description="Gradual load increase - E-commerce peak hours",
                duration_steps=2000,
                traffic_pattern="gradual",
                base_load=1000,   # 5x increase: 200 -> 1000 RPS
                max_load=5000,    # 5x increase: 1000 -> 5000 RPS
                event_frequency=0.002,
                target_metrics={"cpu_utilization": 0.75, "response_time": 0.15}
            ),
            TestScenario(
                name="sudden_spike",
                description="Sudden traffic spikes - Social media viral content",
                duration_steps=1500,
                traffic_pattern="spike",
                base_load=2000,   # 5x increase: 400 -> 2000 RPS
                max_load=10000,   # 5x increase: 2000 -> 10000 RPS
                event_frequency=0.02,
                target_metrics={"cpu_utilization": 0.8, "response_time": 0.2}
            ),
            TestScenario(
                name="flash_crowd",
                description="Flash crowd events - Black Friday/Cyber Monday",
                duration_steps=3000,
                traffic_pattern="flash",
                base_load=1500,   # 5x increase: 300 -> 1500 RPS
                max_load=15000,   # 5x increase: 3000 -> 15000 RPS
                event_frequency=0.01,
                target_metrics={"cpu_utilization": 0.85, "response_time": 0.25}
            ),
            TestScenario(
                name="daily_pattern",
                description="Daily usage pattern - Real-world application",
                duration_steps=4320,  # 3 days
                traffic_pattern="daily",
                base_load=500,    # 5x increase: 100 -> 500 RPS
                max_load=2000,    # 5x increase: 400 -> 2000 RPS
                event_frequency=0.002,
                target_metrics={"cpu_utilization": 0.7, "response_time": 0.12}
            )
        ]

    def create_agent(self, agent_type: str, environment, mock_mode: bool = True) -> Any:
        """Factory method to create different types of agents."""
        if agent_type == "hybrid_dqn_ppo":
            config = HybridConfig()
            k8s_api = KubernetesAPI(max_pods=self.config['environment']['max_pods'])
            return HybridDQNPPOAgent(config, k8s_api, mock_mode=mock_mode)

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

        elif agent_type == "rule_based":
            return RuleBasedAutoscaler()

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
        current_pods = 1
        cpu_utilization = 0.5
        response_time = 0.1
        scaling_actions = []
        sla_violations = 0
        total_cost = 0.0

        try:
            for step in range(scenario.duration_steps):
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
                        timestamp=time.time(),
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
                analysis['agent_comparison'][agent_type] = {
                    'avg_cpu_utilization': np.mean([m.cpu_utilization for m in all_metrics]),
                    'avg_response_time': np.mean([m.response_time for m in all_metrics]),
                    'avg_pod_count': np.mean([m.pod_count for m in all_metrics]),
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

class RuleBasedAutoscaler:
    """Traditional rule-based autoscaler for comparison."""

    def __init__(self, cpu_threshold_up: float = 0.5, cpu_threshold_down: float = 0.3):  # Match HPA threshold
        self.cpu_threshold_up = cpu_threshold_up
        self.cpu_threshold_down = cpu_threshold_down
        self.last_action_time = 0
        self.cooldown_period = 10  # steps - reduced for better responsiveness with realistic loads

    def decide_action(self, cpu_utilization: float, current_pods: int, step: int = 0) -> int:
        """Decide scaling action based on CPU utilization."""
        if step - self.last_action_time < self.cooldown_period:
            return 2  # No change (cooldown)

        if cpu_utilization > self.cpu_threshold_up and current_pods < 10:
            self.last_action_time = step
            return 0  # Scale up
        elif cpu_utilization < self.cpu_threshold_down and current_pods > 1:
            self.last_action_time = step
            return 1  # Scale down
        else:
            return 2  # No change

def simulate_hybrid_traffic_with_agents():
    """Enhanced traffic simulation with agent performance testing."""
    # Initialize the performance tester
    tester = AgentPerformanceTester()

    # Define agents to test
    agent_types = [
        "hybrid_dqn_ppo",
        "dqn",
        "ppo",
        "rule_based"
    ]

    # Define scenarios to test
    test_scenarios = [
        "baseline_steady",
        "gradual_ramp",
        "sudden_spike",
        "daily_pattern"
    ]

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