# Performance Testing Guide for Autoscaling Agents

This guide provides comprehensive instructions for testing and evaluating the performance of PPO, DQN, and Hybrid autoscaling agents against traditional rule-based systems.

## Overview

The performance testing framework provides:

1. **Comprehensive Agent Testing**: Compare Hybrid DQN-PPO, DQN, PPO, and Rule-based agents
2. **Prometheus-style Metrics**: Industry-standard metrics collection and export
3. **Multiple Test Scenarios**: Steady state, gradual ramp, sudden spikes, and daily patterns
4. **Statistical Analysis**: Significance testing and performance comparisons
5. **Research-grade Reports**: Publication-ready analysis and visualizations

## Quick Start

### Step 1: Run a Quick Performance Test

```bash
# Navigate to the project directory
cd /path/to/microk8s-autoscaling

# Run quick test (recommended for first time)
./scripts/run-performance-test.sh quick
```

This will:
- Test Hybrid DQN-PPO vs Rule-based agents
- Run baseline and spike scenarios
- Generate metrics and basic analysis
- Takes approximately 5-10 minutes

### Step 2: View Results

```bash
# Analyze the most recent test results
./scripts/run-performance-test.sh analyze
```

### Step 3: Generate Research Report

```bash
# Generate comprehensive research report
python3 generate_research_report.py
```

## Detailed Usage

### Test Types

#### 1. Quick Test
```bash
./scripts/run-performance-test.sh quick
```
- **Duration**: ~5-10 minutes
- **Agents**: Hybrid DQN-PPO, Rule-based
- **Scenarios**: Baseline steady, Sudden spike
- **Purpose**: Initial evaluation and system validation

#### 2. Comprehensive Test
```bash
./scripts/run-performance-test.sh comprehensive
```
- **Duration**: ~30-45 minutes
- **Agents**: All agents (Hybrid, DQN, PPO, Rule-based)
- **Scenarios**: All scenarios (Steady, Gradual ramp, Sudden spike, Daily pattern)
- **Purpose**: Complete performance evaluation

#### 3. Real Cluster Test
```bash
./scripts/run-performance-test.sh real
```
- **Duration**: Variable
- **Environment**: Actual Kubernetes cluster
- **Caution**: Requires proper cluster setup and monitoring
- **Purpose**: Production validation

### Understanding Test Scenarios

#### Baseline Steady
- **Load Pattern**: Constant traffic around 100 RPS
- **Duration**: 1000 steps (~16 minutes)
- **Purpose**: Baseline performance measurement
- **Key Metrics**: Steady-state efficiency, resource utilization

#### Gradual Ramp
- **Load Pattern**: Progressive increase from 50 to 300 RPS
- **Duration**: 2000 steps (~33 minutes)
- **Purpose**: Test adaptability to changing load
- **Key Metrics**: Scaling responsiveness, prediction accuracy

#### Sudden Spike
- **Load Pattern**: Random traffic spikes up to 500 RPS
- **Duration**: 1500 steps (~25 minutes)
- **Purpose**: Test emergency scaling capabilities
- **Key Metrics**: Response time under stress, over-provisioning

#### Daily Pattern
- **Load Pattern**: Realistic daily usage cycle
- **Duration**: 4320 steps (3 simulated days)
- **Purpose**: Long-term behavior analysis
- **Key Metrics**: Cost optimization, learning convergence

## Metrics Collection

### Prometheus-style Metrics

The framework collects industry-standard metrics:

```
# Resource Utilization
autoscaler_cpu_utilization{agent="hybrid_dqn_ppo",scenario="sudden_spike"} 0.75
autoscaler_memory_utilization{agent="hybrid_dqn_ppo",scenario="sudden_spike"} 0.60
autoscaler_pod_count{agent="hybrid_dqn_ppo",scenario="sudden_spike"} 5

# Performance Metrics
autoscaler_response_time_seconds{agent="hybrid_dqn_ppo",scenario="sudden_spike"} 0.120
autoscaler_throughput_rps{agent="hybrid_dqn_ppo",scenario="sudden_spike"} 450
autoscaler_queue_length{agent="hybrid_dqn_ppo",scenario="sudden_spike"} 0.15

# Scaling Behavior
autoscaler_scaling_frequency_per_hour{agent="hybrid_dqn_ppo",scenario="sudden_spike"} 8.5
autoscaler_over_provisioning_ratio{agent="hybrid_dqn_ppo",scenario="sudden_spike"} 0.10

# Cost and SLA
autoscaler_resource_cost_dollars{agent="hybrid_dqn_ppo",scenario="sudden_spike"} 125.50
autoscaler_sla_violations_total{agent="hybrid_dqn_ppo",scenario="sudden_spike"} 3
autoscaler_availability_percentage{agent="hybrid_dqn_ppo",scenario="sudden_spike"} 99.2
```

### Output Files

After running tests, you'll find:

```
test_results/
├── performance_study_YYYYMMDD_HHMMSS.json    # Detailed results
├── test_log_YYYYMMDD_HHMMSS.log             # Execution logs
└── research_report_YYYYMMDD_HHMMSS.md       # Generated report

metrics/
├── autoscaler_metrics_YYYYMMDD_HHMMSS.prom  # Prometheus format
└── autoscaler_metrics_YYYYMMDD_HHMMSS.csv   # CSV for analysis

# Generated visualizations
performance_comparison_YYYYMMDD_HHMMSS.png
performance_radar_YYYYMMDD_HHMMSS.png
```

## Direct Python Usage

### Running Individual Agent Tests

```python
from examples.hybrid_traffic_simulation import AgentPerformanceTester

# Initialize tester
tester = AgentPerformanceTester()

# Test specific agent and scenario
results = tester.run_agent_test(
    agent_type="hybrid_dqn_ppo",
    scenario=tester.scenarios[0],  # baseline_steady
    mock_mode=True
)

# Access metrics
for metric in results:
    print(f"CPU: {metric.cpu_utilization:.2%}, "
          f"Response: {metric.response_time:.3f}s, "
          f"Cost: ${metric.resource_cost:.2f}")
```

### Custom Scenario Creation

```python
from examples.hybrid_traffic_simulation import TestScenario

# Create custom test scenario
custom_scenario = TestScenario(
    name="peak_hours",
    description="Peak business hours simulation",
    duration_steps=2000,
    traffic_pattern="custom",
    base_load=200,
    max_load=600,
    event_frequency=0.008,
    target_metrics={"cpu_utilization": 0.75, "response_time": 0.18}
)

# Run test with custom scenario
tester.scenarios.append(custom_scenario)
results = tester.run_agent_test("hybrid_dqn_ppo", custom_scenario)
```

## Agent Access and Testing

### Hybrid DQN-PPO Agent

```python
from agent.hybrid_dqn_ppo import HybridDQNPPOAgent, HybridConfig
from agent.kubernetes_api import KubernetesAPI

# Initialize hybrid agent
config = HybridConfig()
k8s_api = KubernetesAPI(max_pods=10)
agent = HybridDQNPPOAgent(config, k8s_api, mock_mode=True)

# Get scaling decision
state = np.array([0.75, 0.60, 0.15, 0.50, 0.10, 0.85])  # [cpu, mem, latency, pods, queue, throughput]
step_result = agent.step()
action = step_result['action']  # 0=scale_up, 1=scale_down, 2=no_change
```

### Individual DQN Agent

```python
from agent.dqn import DQNAgent
from agent.environment_simulated import MicroK8sEnvSimulated

# Initialize DQN agent
env = MicroK8sEnvSimulated()
agent = DQNAgent(env=env, environment=env, is_simulated=True)

# Get scaling decision
state = np.array([0.75, 0.60, 0.15, 0.50, 0.10, 0.85])
action = agent.model.predict(state.reshape(1, -1))[0]
```

### PPO Agent

```python
from agent.ppo import PPOAgent
from agent.environment_simulated import MicroK8sEnvSimulated

# Initialize PPO agent
env = MicroK8sEnvSimulated()
agent = PPOAgent(environment=env)

# Get scaling decision
state = np.array([0.75, 0.60, 0.15, 0.50, 0.10, 0.85])
action, _ = agent.model.predict(state.reshape(1, -1))
```

## Research Analysis

### Key Performance Indicators

1. **Response Time**: Average request response time (lower is better)
2. **Resource Utilization**: CPU and memory efficiency (target: 60-80%)
3. **Cost Efficiency**: Resource cost per unit of throughput
4. **SLA Compliance**: Percentage of time meeting performance targets
5. **Scaling Frequency**: Number of scaling actions per hour
6. **Over/Under-provisioning**: Resource allocation efficiency

### Expected Results

Based on research findings, expect to see:

- **Hybrid DQN-PPO**: Best overall performance, 15-30% improvement over rule-based
- **DQN**: Good discrete action selection, stable performance
- **PPO**: Excellent adaptability, may have higher variance
- **Rule-based**: Baseline performance, may struggle with complex patterns

### Statistical Significance

The framework includes statistical testing:

```python
# Automatic significance testing in reports
from generate_research_report import ResearchReportGenerator

generator = ResearchReportGenerator()
report = generator.generate_report()
# Report includes p-values and confidence intervals
```

## Production Deployment Considerations

### Safety Mechanisms

1. **Bounded Actions**: Agents cannot scale beyond defined limits
2. **Fallback Systems**: Rule-based backup when RL agents fail
3. **Gradual Rollout**: Progressive deployment with monitoring
4. **Human Override**: Manual intervention capabilities

### Monitoring Integration

```python
# Integration with existing monitoring
from examples.hybrid_traffic_simulation import PrometheusMetricsCollector

collector = PrometheusMetricsCollector()
# Export to your monitoring system
metrics_file = collector.export_to_file()
# Import into Prometheus, Grafana, etc.
```

### Configuration Management

```yaml
# config/hybrid_config.yaml
environment:
  max_pods: 10
  scaling_delay: 10
  deployment_name: "your-app"
  namespace: "production"

hybrid:
  reward_optimization_freq: 100
  batch_evaluation_steps: 50

training:
  total_steps: 50000
  eval_freq: 1000
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **Permission Errors**: Make scripts executable
   ```bash
   chmod +x scripts/run-performance-test.sh
   chmod +x generate_research_report.py
   ```

3. **No Results Found**: Check that tests completed successfully
   ```bash
   ls -la test_results/
   ls -la metrics/
   ```

4. **Plotting Errors**: Install visualization dependencies
   ```bash
   pip install matplotlib seaborn
   ```

### Performance Optimization

1. **Reduce Test Duration**: Modify scenario duration_steps
2. **Parallel Execution**: Run multiple tests concurrently
3. **Memory Usage**: Use mock_mode=True for large-scale testing
4. **Disk Space**: Clean old results periodically

## Research Citation

When using this framework for research, please cite:

```bibtex
@misc{microk8s_rl_autoscaling,
  title={Reinforcement Learning for Kubernetes Autoscaling: A Comparative Study},
  author={Your Name},
  year={2024},
  note={Performance testing framework for RL-based autoscaling agents}
}
```

## Contributing

To extend the framework:

1. **Add New Agents**: Implement in `agent/` directory
2. **Create Scenarios**: Add to `_define_test_scenarios()`
3. **Custom Metrics**: Extend `AutoscalingMetrics` class
4. **Visualization**: Add charts to `_create_performance_plots()`

## Support

For issues and questions:
1. Check this documentation
2. Review log files in `test_results/`
3. Examine metric outputs in `metrics/`
4. Verify agent implementations in `agent/` directory

---

This framework provides research-grade evaluation capabilities for autoscaling agents, supporting the hypothesis that RL-based approaches can outperform traditional rule-based systems in dynamic cloud environments.