# Performance Testing Guide for Autoscaling Agents

This guide provides comprehensive instructions for testing and evaluating the performance of PPO, DQN, and Hybrid autoscaling agents against traditional rule-based systems.

## Overview

The enhanced performance testing framework provides:

1. **Comprehensive Agent Testing**: Compare Hybrid DQN-PPO, DQN, PPO, and Rule-based agents
2. **Publication-Quality Data Collection**: Research-grade data with statistical validation
3. **Real-time System Monitoring**: CPU, memory, and performance tracking
4. **Multiple Export Formats**: Prometheus, CSV, JSON, LaTeX, and Grafana dashboards
5. **Statistical Validation**: Effect size analysis, significance testing, and power assessment
6. **Reproducible Research**: Complete environment documentation and metadata
7. **Grafana Integration**: Auto-generated dashboards with import instructions

## 🚀 **Single Command Complete Workflow**

**For production-grade monitoring with Docker (Simplified Setup):**

```bash
# One command runs everything: test + Prometheus + Grafana + monitoring
./scripts/complete-docker-workflow.sh
```

**What this does automatically:**
- ✅ Runs publication-quality performance tests
- ✅ Deploys Prometheus & Grafana using Docker containers
- ✅ Works with any Docker installation (no Kubernetes required)
- ✅ Configures monitoring stack with auto-scraped metrics
- ✅ Imports dashboard and provides access URLs
- ✅ Creates publication-ready data packages

**Prerequisites:**
- ✅ `Docker` installed and running
- ✅ No Kubernetes cluster required
- ✅ Ports 3000, 8080, 9090 available on localhost

**Access after completion:**
- 📊 **Grafana Dashboard**: `http://localhost:3000` (admin/admin)
- 📈 **Prometheus Metrics**: `http://localhost:9090`
- 🔧 **Metrics Service**: `http://localhost:8080/metrics`
- 📚 **Publication Data**: `publication_data/study_*/`

---

## Quick Start

### Step 1: Choose Your Testing Mode

#### For First-Time Users (Recommended)
```bash
# Navigate to the project directory
cd /path/to/microk8s-autoscaling

# Run quick test with basic monitoring
./scripts/run-performance-test.sh quick
```

#### For Publication-Quality Research
```bash
# Run comprehensive test with enhanced monitoring and validation
./scripts/run-performance-test.sh publication
```

#### For Real-Time Monitoring Focus
```bash
# Run with enhanced system monitoring
./scripts/run-performance-test.sh monitor
```

### Step 2: View and Validate Results

```bash
# Analyze the most recent test results with enhanced insights
./scripts/run-performance-test.sh analyze

# Perform statistical validation (automatic in publication mode)
./scripts/run-performance-test.sh validate
```

### Step 3: Export Publication-Ready Data

```bash
# Export structured data package for research/publication
./scripts/run-performance-test.sh export

# Generate comprehensive research report
python3 generate_research_report.py
```

### Step 4: Import to Grafana (Optional)

```bash
# Check the metrics/ directory for auto-generated dashboard
ls metrics/grafana_dashboard_*.json

# Follow the import instructions in the generated _import_instructions.md file
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
- **Features**: Basic metrics, standard analysis
- **Purpose**: Initial evaluation and system validation

#### 2. Comprehensive Test
```bash
./scripts/run-performance-test.sh comprehensive
```
- **Duration**: ~30-45 minutes
- **Agents**: All agents (Hybrid, DQN, PPO, Rule-based)
- **Scenarios**: All scenarios (Steady, Gradual ramp, Sudden spike, Daily pattern)
- **Features**: Enhanced logging, multiple export formats
- **Purpose**: Complete performance evaluation

#### 3. Publication-Quality Test
```bash
./scripts/run-performance-test.sh publication
```
- **Duration**: ~30-45 minutes
- **Agents**: All agents with comprehensive analysis
- **Features**:
  - Statistical validation and effect size analysis
  - Real-time system monitoring
  - Publication-ready data export (CSV, LaTeX, JSON)
  - Auto-generated Grafana dashboards
  - Complete reproducibility documentation
- **Purpose**: Research-grade evaluation for academic/industry publication

#### 4. Enhanced Monitoring Test
```bash
./scripts/run-performance-test.sh monitor
```
- **Duration**: ~30-45 minutes
- **Agents**: All agents
- **Features**:
  - Real-time CPU, memory, and load monitoring
  - Background system metrics collection
  - Statistical validation
  - Enhanced performance insights
- **Purpose**: Detailed system behavior analysis

#### 5. Real Cluster Test
```bash
./scripts/run-performance-test.sh real
```
- **Duration**: Variable
- **Environment**: Actual Kubernetes cluster
- **Caution**: Requires proper cluster setup and monitoring
- **Purpose**: Production validation

#### 6. Standalone Operations

**Statistical Validation Only**
```bash
./scripts/run-performance-test.sh validate
```
- Analyzes most recent results for statistical significance
- Provides effect size analysis and power assessment
- Generates publication-quality statistical reports

**Export Publication Data**
```bash
./scripts/run-performance-test.sh export
```
- Creates structured data package for research
- Generates LaTeX tables and CSV files
- Includes complete metadata and documentation

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

## Enhanced Monitoring and Validation

### Real-Time System Monitoring

The framework now includes comprehensive system monitoring:

```bash
# Monitoring is automatically enabled in publication and monitor modes
./scripts/run-performance-test.sh publication
./scripts/run-performance-test.sh monitor

# Or enable manually with environment variables
REAL_TIME_MONITORING=true ./scripts/run-performance-test.sh comprehensive
```

**System Metrics Collected:**
- **CPU Utilization**: Real-time processor usage
- **Memory Usage**: Physical memory consumption
- **Load Average**: System load (1-minute average)
- **Network I/O**: Incoming/outgoing network traffic (when available)
- **Disk I/O**: Read/write operations (when available)

**Monitoring Configuration:**
```bash
# Default intervals (configurable)
METRICS_COLLECTION_INTERVAL=5     # 5 seconds
SYSTEM_MONITORING_INTERVAL=10     # 10 seconds
PERFORMANCE_SAMPLING_RATE=1       # 1 second
```

### Statistical Validation Framework

Automated statistical analysis ensures publication-quality results:

#### Sample Size Validation
- **Minimum Requirements**: 30+ samples for normal distribution
- **Power Analysis**: Assessment of statistical power
- **Confidence Intervals**: 95% confidence reporting

#### Effect Size Analysis
- **Cohen's d**: Standardized effect size measurement
- **Percentage Improvement**: Practical significance assessment
- **Baseline Comparison**: Rule-based agent as control group

#### Statistical Significance Testing
```bash
# Automatic validation (included in publication mode)
./scripts/run-performance-test.sh publication

# Standalone validation of existing results
./scripts/run-performance-test.sh validate
```

**Validation Report Includes:**
- Sample size adequacy assessment
- Effect size classification (small/medium/large)
- Statistical power evaluation
- Coefficient of variation analysis
- Recommendations for improvement

### Publication-Quality Data Export

Structured data packages for research and publication:

```bash
# Export publication-ready data package
./scripts/run-performance-test.sh export
```

**Generated Structure:**
```
publication_data/study_TESTID/
├── raw_data/
│   ├── performance_study.json      # Complete test results
│   ├── autoscaler_metrics_*.csv    # Time-series data
│   └── system_metrics_*.csv        # System monitoring data
├── processed_data/
│   └── [Cleaned and aggregated datasets]
├── tables/
│   ├── agent_performance_summary.csv    # Performance comparison
│   ├── agent_performance_summary.tex    # LaTeX table format
│   ├── test_scenarios.csv              # Scenario characteristics
│   └── test_scenarios.tex              # LaTeX scenario table
├── figures/
│   └── [Auto-generated visualizations]
├── supplementary/
│   └── [Additional analysis materials]
├── metadata.json                   # Complete experimental metadata
└── README.md                       # Documentation and usage guide
```

### Environment Documentation

Complete reproducibility support:

**Automatically Generated:**
- System information (OS, hardware, software versions)
- Git commit hash and branch information
- Test configuration and parameters
- Directory structure and file locations
- Timestamp and unique test identifiers

**Environment File Example:**
```
=== ENVIRONMENT DETAILS ===
Test ID: publication_20241028_143022
System: Darwin 24.4.0 x86_64
Python Version: Python 3.11.5
Git Commit: a1b2c3d4e5f6
Git Branch: main

=== TEST CONFIGURATION ===
Test Type: publication
Agents: hybrid_dqn_ppo,dqn,ppo,rule_based
Scenarios: baseline_steady,gradual_ramp,sudden_spike,daily_pattern
Publication Mode: true
Statistical Validation: true
Real-time Monitoring: true
```

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

## Publication Workflow Guide

### Complete Research Workflow

For academic papers, industry reports, or technical documentation:

#### Step 1: Data Collection
```bash
# Run publication-quality test with all enhancements
./scripts/run-performance-test.sh publication

# This automatically enables:
# - Statistical validation
# - Real-time monitoring
# - Publication data export
# - Grafana dashboard generation
```

#### Step 2: Data Validation
```bash
# Review statistical validation results (included in publication output)
# Or run standalone validation
./scripts/run-performance-test.sh validate
```

**Validation Checklist:**
- ✅ Sample size ≥30 for robust statistics
- ✅ Effect size analysis (Cohen's d)
- ✅ Statistical power assessment
- ✅ Coefficient of variation analysis

#### Step 3: Export and Format
```bash
# Export structured data package
./scripts/run-performance-test.sh export

# Generated files ready for publication:
# - LaTeX tables for papers
# - CSV files for additional analysis
# - JSON for programmatic access
# - Grafana dashboards for presentations
```

#### Step 4: Visualization and Reporting
```bash
# Generate comprehensive research report
python3 generate_research_report.py

# Import Grafana dashboard for interactive visualization
# File: metrics/grafana_dashboard_TESTID.json
# Instructions: metrics/grafana_dashboard_TESTID_import_instructions.md
```

### Environment Variables for Customization

Control testing behavior with environment variables:

```bash
# Enable publication mode
export PUBLICATION_MODE=true

# Control monitoring features
export REAL_TIME_MONITORING=true
export STATISTICAL_VALIDATION=true

# Customize export formats
export EXPORT_FORMATS="json,csv,prometheus,grafana"

# Adjust monitoring intervals
export METRICS_COLLECTION_INTERVAL=5      # seconds
export SYSTEM_MONITORING_INTERVAL=10      # seconds
export PERFORMANCE_SAMPLING_RATE=1        # seconds

# Then run any test type
./scripts/run-performance-test.sh comprehensive
```

### Publication-Ready Examples

#### Academic Research Example
```bash
# For peer-reviewed publications
PUBLICATION_MODE=true ./scripts/run-performance-test.sh comprehensive
./scripts/run-performance-test.sh validate
./scripts/run-performance-test.sh export
python3 generate_research_report.py

# Use generated LaTeX tables in your paper:
# publication_data/study_TESTID/tables/agent_performance_summary.tex
# publication_data/study_TESTID/tables/test_scenarios.tex
```

#### Industry Benchmark Example
```bash
# For technical reports and presentations
./scripts/run-performance-test.sh monitor
./scripts/run-performance-test.sh export

# Import Grafana dashboard for executive presentations:
# metrics/grafana_dashboard_TESTID.json
```

#### Reproducible Research Example
```bash
# Document complete experimental setup
./scripts/run-performance-test.sh publication

# All environment details saved in:
# test_results/environment_TESTID.txt
# publication_data/study_TESTID/metadata.json
```

### Quality Assurance Checklist

Before publishing results, verify:

- [ ] **Statistical Power**: Test duration ≥5000 steps across ≥3 scenarios
- [ ] **Effect Size**: Meaningful differences (≥10% improvement)
- [ ] **Reproducibility**: Complete environment documentation
- [ ] **Data Integrity**: All metrics properly collected and validated
- [ ] **Multiple Formats**: Data available in required formats (LaTeX, CSV, JSON)
- [ ] **Visualization**: Grafana dashboards for presentation/demonstration

### Common Publication Scenarios

#### Conference Paper
```bash
# Short evaluation with key comparisons
./scripts/run-performance-test.sh publication
# Use: agent_performance_summary.tex, research_report.md
```

#### Journal Article
```bash
# Comprehensive evaluation with statistical rigor
PUBLICATION_MODE=true ./scripts/run-performance-test.sh comprehensive
./scripts/run-performance-test.sh validate
# Use: Full publication_data package, statistical validation report
```

#### Technical Demo
```bash
# Real-time monitoring with live dashboard
./scripts/run-performance-test.sh monitor
# Use: Grafana dashboard, system monitoring data
```

#### Reproducibility Package
```bash
# Complete experimental package
./scripts/run-performance-test.sh publication
./scripts/run-performance-test.sh export
# Use: Complete publication_data directory with metadata
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

The actual configuration used by the system is defined in `config/hybrid_config.yaml`:

```yaml
# config/hybrid_config.yaml - Actual Configuration
dqn:
  batch_size: 64
  buffer_size: 100000
  epsilon_decay: 0.995
  epsilon_end: 0.07
  epsilon_start: 1.0
  gamma: 0.99
  learning_rate: 0.0005
  target_update_freq: 2000
  tau: 0.1

environment:
  deployment_name: nginx-deployment
  max_pods: 10
  metrics_sync_interval: 10
  namespace: default
  prometheus_url: http://localhost:9090
  scaling_delay: 10

hybrid:
  action_dim: 3
  batch_evaluation_steps: 50
  hidden_dim: 64
  reward_optimization_freq: 100
  state_dim: 6

ppo:
  batch_size: 64
  clip_range: 0.2
  ent_coef: 0.01
  gae_lambda: 0.95
  gamma: 0.99
  learning_rate: 0.0003
  n_steps: 2048

training:
  checkpoint_freq: 5000
  eval_freq: 1000
  log_freq: 100
  total_steps: 50000
```

### Production-Scale Test Scenarios

The framework now includes production-scale scenarios with significantly higher traffic loads:

```python
# Actual test scenarios from hybrid_traffic_simulation.py
scenarios = [
    {
        "name": "baseline_steady",
        "description": "Steady baseline load - Production-level traffic",
        "duration_steps": 1000,
        "base_load": 2500,  # RPS (5x increase from original)
        "max_load": 4000,   # RPS
        "target_cpu": 0.7,
        "target_response_time": 0.1
    },
    {
        "name": "gradual_ramp",
        "description": "Gradual load increase - E-commerce peak hours",
        "duration_steps": 2000,
        "base_load": 1000,  # RPS
        "max_load": 6000,   # RPS (5x increase)
        "event_frequency": 0.003
    },
    {
        "name": "sudden_spike",
        "description": "Traffic spikes - Flash sales and viral content",
        "duration_steps": 1500,
        "base_load": 1500,  # RPS
        "max_load": 12500,  # RPS (5x increase)
        "spike_frequency": 0.008
    },
    {
        "name": "daily_pattern",
        "description": "Realistic daily usage - 24h business cycle",
        "duration_steps": 4320,  # 3 simulated days
        "base_load": 1000,   # RPS
        "peak_load": 8000,   # RPS
        "pattern": "sinusoidal_with_noise"
    }
]
```

## Grafana Integration Guide

### Overview

This section provides a comprehensive guide for importing autoscaling metrics into Grafana for advanced visualization and monitoring. The framework exports metrics in both Prometheus and CSV formats for maximum compatibility.

### Prerequisites

1. **Grafana Installation** (version 8.0+)
   ```bash
   # Docker installation
   docker run -d -p 3000:3000 --name grafana grafana/grafana:latest

   # Or using package manager
   sudo apt-get install -y grafana
   sudo systemctl start grafana-server
   sudo systemctl enable grafana-server
   ```

2. **Prometheus Data Source** (optional but recommended)
   ```bash
   # Install Prometheus
   docker run -d -p 9090:9090 --name prometheus prom/prometheus:latest
   ```

### Method 1: Direct Prometheus Import

#### Step 1: Configure Prometheus to Scrape Metrics

Create a `prometheus.yml` configuration:

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'autoscaling-metrics'
    static_configs:
      - targets: ['localhost:8080']  # Your metrics endpoint
    scrape_interval: 10s
    metrics_path: '/metrics'

  - job_name: 'microk8s-autoscaler'
    kubernetes_sd_configs:
      - role: pod
        namespaces:
          names:
            - default
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app]
        action: keep
        regex: autoscaling-agent
```

#### Step 2: Import Metrics Files to Prometheus

```bash
# Copy generated metrics to Prometheus data directory
cp metrics/autoscaler_metrics_*.prom /prometheus/data/

# Or use HTTP API to import
curl -X POST http://localhost:9090/api/v1/admin/tsdb/snapshot \
  -H "Content-Type: application/json"

# Import metrics via remote write
curl -X POST "http://localhost:9090/api/v1/write" \
  --data-binary @metrics/autoscaler_metrics_$(date +%Y%m%d_%H%M%S).prom
```

#### Step 3: Configure Grafana Data Source

1. Open Grafana at `http://localhost:3000` (admin/admin)
2. Navigate to **Configuration > Data Sources**
3. Add Prometheus data source:
   ```
   URL: http://localhost:9090
   Access: Server (default)
   ```

### Method 2: CSV Import with Grafana CSV Plugin

#### Step 1: Install CSV Data Source Plugin

```bash
# Install CSV plugin
grafana-cli plugins install marcusolsson-csv-datasource

# Restart Grafana
sudo systemctl restart grafana-server
```

#### Step 2: Configure CSV Data Source

1. Go to **Configuration > Data Sources**
2. Add **CSV** data source
3. Configure CSV settings:
   ```
   URL: file:///path/to/metrics/autoscaler_metrics_YYYYMMDD_HHMMSS.csv
   Delimiter: comma
   Skip leading rows: 1
   ```

### Method 3: Direct File Import

#### Step 1: Prepare Metrics for Import

```python
# Convert metrics to Grafana-compatible format
import pandas as pd
import json
from datetime import datetime

def convert_metrics_for_grafana(csv_file_path):
    """Convert CSV metrics to Grafana JSON format."""
    df = pd.read_csv(csv_file_path)

    # Parse labels column
    df['labels_parsed'] = df['labels'].apply(json.loads)

    # Create Grafana-compatible structure
    grafana_data = []

    for _, row in df.iterrows():
        labels = row['labels_parsed']
        grafana_data.append({
            "target": f"{row['metric_name']}_{labels.get('agent', 'unknown')}",
            "datapoints": [[row['value'], int(row['timestamp'] * 1000)]],
            "tags": labels
        })

    return grafana_data

# Usage
grafana_metrics = convert_metrics_for_grafana('metrics/autoscaler_metrics_latest.csv')
with open('grafana_import.json', 'w') as f:
    json.dump(grafana_metrics, f, indent=2)
```

#### Step 2: Import via Grafana API

```bash
# Import dashboard with metrics
curl -X POST \
  http://admin:admin@localhost:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @grafana_dashboard_template.json
```

### Pre-built Dashboard Templates

#### Template 1: Autoscaling Performance Overview

```json
{
  "dashboard": {
    "id": null,
    "title": "Autoscaling Agents Performance Comparison",
    "tags": ["autoscaling", "kubernetes", "rl"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "CPU Utilization by Agent",
        "type": "timeseries",
        "targets": [
          {
            "expr": "autoscaler_cpu_utilization",
            "legendFormat": "{{agent}} - {{scenario}}"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "color": {"mode": "palette-classic"},
            "unit": "percent",
            "min": 0,
            "max": 100
          }
        }
      },
      {
        "id": 2,
        "title": "Response Time Comparison",
        "type": "timeseries",
        "targets": [
          {
            "expr": "autoscaler_response_time_seconds",
            "legendFormat": "{{agent}} - {{scenario}}"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "color": {"mode": "palette-classic"},
            "unit": "s"
          }
        }
      },
      {
        "id": 3,
        "title": "Scaling Actions Frequency",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(autoscaler_scaling_frequency_per_hour[5m])",
            "legendFormat": "{{agent}}"
          }
        ]
      },
      {
        "id": 4,
        "title": "Cost Efficiency",
        "type": "bargauge",
        "targets": [
          {
            "expr": "autoscaler_resource_cost_dollars",
            "legendFormat": "{{agent}}"
          }
        ]
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "10s"
  }
}
```

#### Template 2: Agent Comparison Dashboard

```json
{
  "dashboard": {
    "title": "RL vs Rule-based Agent Analysis",
    "panels": [
      {
        "title": "Performance Radar Chart",
        "type": "piechart",
        "targets": [
          {
            "expr": "avg_over_time(autoscaler_cpu_utilization[1h])",
            "legendFormat": "CPU Efficiency"
          },
          {
            "expr": "avg_over_time(autoscaler_response_time_seconds[1h])",
            "legendFormat": "Response Time"
          },
          {
            "expr": "avg_over_time(autoscaler_resource_cost_dollars[1h])",
            "legendFormat": "Cost"
          }
        ]
      },
      {
        "title": "SLA Violations by Agent",
        "type": "table",
        "targets": [
          {
            "expr": "sum(autoscaler_sla_violations_total) by (agent)",
            "format": "table"
          }
        ]
      }
    ]
  }
}
```

### Advanced Grafana Queries

#### Performance Comparison Queries

```promql
# CPU utilization efficiency (target: 70-80%)
abs(autoscaler_cpu_utilization - 0.75) by (agent, scenario)

# Response time improvement over baseline
(
  autoscaler_response_time_seconds{agent="rule_based"} -
  autoscaler_response_time_seconds{agent!="rule_based"}
) / autoscaler_response_time_seconds{agent="rule_based"} * 100

# Cost savings percentage
(
  autoscaler_resource_cost_dollars{agent="rule_based"} -
  autoscaler_resource_cost_dollars{agent!="rule_based"}
) / autoscaler_resource_cost_dollars{agent="rule_based"} * 100

# Scaling stability (lower is better)
rate(autoscaler_scaling_frequency_per_hour[10m]) by (agent)

# SLA compliance rate
(1 - autoscaler_sla_violations_total / autoscaler_total_requests) * 100 by (agent)
```

#### Agent Performance Rankings

```promql
# Overall performance score (weighted average)
(
  (1 - autoscaler_response_time_seconds / max(autoscaler_response_time_seconds)) * 0.3 +
  (autoscaler_cpu_utilization) * 0.2 +
  (1 - autoscaler_resource_cost_dollars / max(autoscaler_resource_cost_dollars)) * 0.3 +
  (autoscaler_availability_percentage / 100) * 0.2
) by (agent)
```

### Alerting Rules

Create alerting rules for monitoring agent performance:

```yaml
# grafana_alerts.yml
groups:
  - name: autoscaling_alerts
    rules:
      - alert: HighResponseTime
        expr: autoscaler_response_time_seconds > 0.5
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High response time detected"
          description: "Response time is {{ $value }}s for agent {{ $labels.agent }}"

      - alert: ResourceCostSpike
        expr: increase(autoscaler_resource_cost_dollars[1h]) > 100
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Resource cost spike detected"

      - alert: FrequentScaling
        expr: rate(autoscaler_scaling_frequency_per_hour[10m]) > 20
        for: 3m
        labels:
          severity: warning
        annotations:
          summary: "Excessive scaling activity"
```

### Data Export and Backup

#### Automated Export Script

```bash
#!/bin/bash
# export_grafana_data.sh

GRAFANA_URL="http://localhost:3000"
GRAFANA_TOKEN="your_api_token"
BACKUP_DIR="grafana_backups/$(date +%Y%m%d_%H%M%S)"

mkdir -p "$BACKUP_DIR"

# Export dashboards
curl -H "Authorization: Bearer $GRAFANA_TOKEN" \
     "$GRAFANA_URL/api/search?type=dash-db" | \
     jq -r '.[] | .uid' | \
while read uid; do
    curl -H "Authorization: Bearer $GRAFANA_TOKEN" \
         "$GRAFANA_URL/api/dashboards/uid/$uid" > \
         "$BACKUP_DIR/dashboard_$uid.json"
done

# Export data sources
curl -H "Authorization: Bearer $GRAFANA_TOKEN" \
     "$GRAFANA_URL/api/datasources" > \
     "$BACKUP_DIR/datasources.json"

echo "Grafana backup completed in $BACKUP_DIR"
```

### Performance Monitoring Setup

#### Real-time Monitoring Dashboard

```json
{
  "dashboard": {
    "title": "Real-time Autoscaling Monitoring",
    "refresh": "5s",
    "panels": [
      {
        "title": "Live Agent Performance",
        "type": "timeseries",
        "targets": [
          {
            "expr": "autoscaler_cpu_utilization",
            "refId": "A"
          }
        ],
        "options": {
          "legend": {"displayMode": "table"},
          "tooltip": {"mode": "multi"}
        }
      }
    ]
  }
}
```

### Troubleshooting Grafana Integration

#### Common Issues

1. **Metrics Not Showing**
   ```bash
   # Check if metrics file exists
   ls -la metrics/autoscaler_metrics_*.prom

   # Verify Prometheus can read metrics
   curl "http://localhost:9090/api/v1/query?query=autoscaler_cpu_utilization"

   # Check Grafana data source connectivity
   curl -H "Authorization: Bearer $TOKEN" \
        "http://localhost:3000/api/datasources/proxy/1/api/v1/query?query=up"
   ```

2. **Performance Issues**
   ```bash
   # Reduce scrape interval in Prometheus
   # Limit time range in Grafana queries
   # Use recording rules for complex queries
   ```

3. **Data Format Issues**
   ```python
   # Validate metrics format
   def validate_prometheus_format(file_path):
       with open(file_path, 'r') as f:
           for line_num, line in enumerate(f, 1):
               if line.strip() and not line.startswith('#'):
                   try:
                       # Parse Prometheus line format
                       parts = line.strip().split(' ')
                       metric_name = parts[0].split('{')[0]
                       value = float(parts[1])
                       timestamp = int(parts[2]) if len(parts) > 2 else None
                   except Exception as e:
                       print(f"Invalid format at line {line_num}: {e}")
   ```

## Complete Example Workflow

### 🚀 **End-to-End Publication-Quality Testing with Grafana Visualization**

This section provides a complete workflow from running tests to visualizing results in Grafana.

#### **Step 1: Run Publication-Quality Performance Test**

```bash
# Navigate to project directory
cd /path/to/microk8s-autoscaling

# Run comprehensive publication-quality test
./scripts/run-performance-test.sh publication
```

**This command automatically:**
- Enables statistical validation and real-time monitoring
- Tests all agents (Hybrid DQN-PPO, DQN, PPO, Rule-based)
- Runs all scenarios (baseline, gradual ramp, sudden spike, daily pattern)
- Generates publication-ready data packages
- Creates auto-generated Grafana dashboard
- Performs statistical significance testing

#### **Step 2: Set Up Grafana Dashboard**

##### **Option A: Docker Setup (Recommended)**

```bash
# Start Grafana container
docker run -d -p 3000:3000 --name grafana \
  -v grafana-storage:/var/lib/grafana \
  grafana/grafana:latest

# Wait for Grafana to start (about 30 seconds)
echo "Waiting for Grafana to start..."
sleep 30

# Verify Grafana is running
curl -s http://localhost:3000/api/health || echo "Grafana not ready yet"
```

##### **Option B: Native Installation (macOS/Linux)**

```bash
# macOS with Homebrew
brew install grafana
brew services start grafana

# Ubuntu/Debian
sudo apt-get install -y grafana
sudo systemctl start grafana-server
sudo systemctl enable grafana-server

# Access at http://localhost:3000
```

#### **Step 3: Access and Configure Grafana**

```bash
# Open Grafana in browser
echo "🌐 Open Grafana at: http://localhost:3000"
echo "📧 Username: admin"
echo "🔑 Password: admin (you'll be prompted to change it)"

# Check generated dashboard files
echo "📊 Generated dashboard:"
ls metrics/grafana_dashboard_*.json

echo "📋 Import instructions:"
ls metrics/grafana_dashboard_*_import_instructions.md
```

**Manual Steps in Grafana:**
1. **Login**: Go to `http://localhost:3000` (admin/admin)
2. **Import Dashboard**:
   - Click **"+" → Import**
   - Upload the JSON file from `metrics/grafana_dashboard_*.json`
   - Click **Import**

#### **Step 4: Set Up Data Source (Optional - For Real-Time Data)**

```bash
# Start Prometheus for real-time metrics (optional)
docker run -d -p 9090:9090 --name prometheus \
  -v $(pwd)/metrics:/prometheus/data \
  prom/prometheus:latest

# Configure Prometheus data source in Grafana:
# 1. Go to Configuration → Data Sources
# 2. Add Prometheus
# 3. URL: http://localhost:9090
# 4. Save & Test
```

#### **Step 5: View Results and Analysis**

```bash
# Check all generated outputs
echo "📊 Performance Results:"
ls test_results/performance_study_*.json

echo "📈 Metrics Files:"
ls metrics/autoscaler_metrics_*.csv
ls metrics/autoscaler_metrics_*.prom

echo "📚 Publication Data:"
ls -la publication_data/study_*/

echo "🖥️ System Monitoring:"
ls monitoring/system_metrics_*.csv

echo "📋 Environment Documentation:"
ls test_results/environment_*.txt
```

#### **Step 6: Statistical Validation and Export**

```bash
# Perform additional statistical validation
./scripts/run-performance-test.sh validate

# Export publication-ready data package
./scripts/run-performance-test.sh export

# Generate comprehensive research report
python3 generate_research_report.py
```

#### **Step 7: Access Your Complete Analysis**

```bash
# View publication-ready tables (LaTeX format for papers)
echo "📄 LaTeX Tables for Publication:"
find publication_data/ -name "*.tex" -exec echo "  {}" \;

# View CSV data for additional analysis
echo "📊 CSV Data for Analysis:"
find publication_data/ -name "*.csv" -exec echo "  {}" \;

# View comprehensive metadata
echo "📋 Experimental Metadata:"
find publication_data/ -name "metadata.json" -exec echo "  {}" \;

# Check import instructions for Grafana
echo "📖 Grafana Import Guide:"
cat metrics/grafana_dashboard_*_import_instructions.md
```

### 🎯 **Complete Docker Workflow Script**

**Single Command for Complete Workflow:**

```bash
# One command to run everything with Docker monitoring
./scripts/complete-docker-workflow.sh
```

**This automated script performs:**
1. ✅ **Docker environment validation** (checks if Docker is running)
2. ✅ **Publication-quality test execution**
3. ✅ **Prometheus deployment with auto-configuration**
4. ✅ **Grafana deployment with data source setup**
5. ✅ **Automatic dashboard import**
6. ✅ **Service exposure and access URLs**

### 📊 **Complete Workflow Script Options**

```bash
# Basic usage - runs everything
./scripts/complete-docker-workflow.sh

# Show help and options
./scripts/complete-docker-workflow.sh --help

# Check current monitoring status
./scripts/complete-docker-workflow.sh --status

# Start monitoring only (skip publication test)
./scripts/complete-docker-workflow.sh --start-only

# Update metrics with latest data
./scripts/complete-docker-workflow.sh --update-metrics

# Cleanup existing monitoring stack
./scripts/complete-docker-workflow.sh --cleanup
```

### 🚀 **What the Complete Workflow Does**

#### **Step 1-2: Environment Setup**
- Validates Docker installation and availability
- Checks for existing containers and cleans them up
- Creates Docker network for container communication

#### **Step 3: Publication Test**
- Runs `./scripts/run-performance-test.sh publication`
- Generates all metrics, dashboards, and analysis
- Creates publication-ready data packages

#### **Step 4-6: Monitoring Stack**
- Deploys metrics service with real-time data serving
- Deploys Prometheus with autoscaling metrics scraping
- Deploys Grafana with auto-configuration

#### **Step 7-9: Configuration & Access**
- Auto-configures Prometheus data source in Grafana
- Imports generated dashboard automatically
- Provides access URLs and credentials

### 📱 **Access Your Monitoring Stack**

After running the complete workflow:

```bash
# Get access information
./scripts/complete-docker-workflow.sh --status

# Direct access URLs:
# 📊 Grafana: http://localhost:3000 (admin/admin)
# 📈 Prometheus: http://localhost:9090
# 🔧 Metrics Service: http://localhost:8080/metrics
```

### 🔧 **Management Commands**

```bash
# View monitoring stack logs
docker logs autoscaling-grafana
docker logs autoscaling-prometheus
docker logs autoscaling-metrics

# Check container status
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'

# Update metrics with latest test data
./scripts/complete-docker-workflow.sh --update-metrics

# Restart specific services
docker restart autoscaling-grafana
docker restart autoscaling-prometheus
docker restart autoscaling-metrics

# View container resource usage
docker stats autoscaling-grafana autoscaling-prometheus autoscaling-metrics

# Access container shells for debugging
docker exec -it autoscaling-grafana /bin/bash
docker exec -it autoscaling-prometheus /bin/sh

# Delete entire monitoring stack
./scripts/complete-docker-workflow.sh --cleanup
```

### 🎯 **Docker Workflow Features**

The Docker workflow provides several advantages over Kubernetes:

**✅ Simplified Setup:**
- No Kubernetes cluster required
- No kubectl configuration needed
- Works on any machine with Docker

**✅ Faster Deployment:**
- Containers start in seconds
- No complex networking setup
- Direct localhost access

**✅ Easy Management:**
- Simple Docker commands for troubleshooting
- Individual container control
- Quick cleanup and restart

**✅ Development Friendly:**
- Perfect for local testing
- Easy to modify and experiment
- No cluster resource constraints

### 📊 **Alternative Quick Visualization**

If you prefer not to set up Grafana immediately:

```bash
# View generated performance plots
echo "📈 Generated Visualizations:"
ls performance_comparison_*.png
ls performance_radar_*.png

# Open plots (macOS)
open performance_comparison_*.png

# Open plots (Linux)
xdg-open performance_comparison_*.png

# View data in spreadsheet applications
echo "📊 Open CSV data:"
echo "  - Excel/Numbers: $(ls metrics/autoscaler_metrics_*.csv)"
echo "  - Publication tables: $(ls publication_data/study_*/tables/*.csv)"
```

### 🔍 **Docker Workflow Verification Checklist**

Before proceeding with publication or presentation:

```bash
# Verify all components are working
echo "🔍 Docker Workflow Verification Checklist:"
echo "========================================="

# Check Docker is running
if docker info >/dev/null 2>&1; then
    echo "✅ Docker is running"
else
    echo "❌ Docker is not running - start Docker first"
fi

# Check test results
if [ -f "$(ls test_results/performance_study_*.json 2>/dev/null | head -1)" ]; then
    echo "✅ Test results generated"
else
    echo "❌ Test results missing"
fi

# Check Grafana dashboard
if [ -f "$(ls metrics/grafana_dashboard_*.json 2>/dev/null | head -1)" ]; then
    echo "✅ Grafana dashboard generated"
else
    echo "❌ Grafana dashboard missing"
fi

# Check publication data
if [ -d "$(ls -d publication_data/study_* 2>/dev/null | head -1)" ]; then
    echo "✅ Publication data package created"
else
    echo "❌ Publication data package missing"
fi

# Check Docker containers
if docker ps --format '{{.Names}}' | grep -q "autoscaling-grafana"; then
    echo "✅ Grafana container running"
else
    echo "⚠️ Grafana container not running"
fi

if docker ps --format '{{.Names}}' | grep -q "autoscaling-prometheus"; then
    echo "✅ Prometheus container running"
else
    echo "⚠️ Prometheus container not running"
fi

if docker ps --format '{{.Names}}' | grep -q "autoscaling-metrics"; then
    echo "✅ Metrics service container running"
else
    echo "⚠️ Metrics service container not running"
fi

# Check service accessibility
if curl -s http://localhost:3000/api/health >/dev/null 2>&1; then
    echo "✅ Grafana accessible at http://localhost:3000"
else
    echo "⚠️ Grafana not accessible"
fi

if curl -s http://localhost:9090/-/ready >/dev/null 2>&1; then
    echo "✅ Prometheus accessible at http://localhost:9090"
else
    echo "⚠️ Prometheus not accessible"
fi

if curl -s http://localhost:8080/health >/dev/null 2>&1; then
    echo "✅ Metrics service accessible at http://localhost:8080"
else
    echo "⚠️ Metrics service not accessible"
fi

echo ""
echo "🎯 Next Steps:"
echo "1. Open http://localhost:3000 (admin/admin)"
echo "2. Dashboard should be auto-imported, or import from metrics/grafana_dashboard_*.json"
echo "3. Review publication data in publication_data/study_*/"
echo "4. Use LaTeX tables for academic papers"
echo "5. Use CSV data for additional analysis"
echo "6. Use ./scripts/complete-docker-workflow.sh --status for container status"
echo "7. Use ./scripts/complete-docker-workflow.sh --cleanup to stop all containers"
```

This complete workflow provides everything needed to run publication-quality autoscaling performance tests and visualize the results professionally in Grafana.

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   # For enhanced features, also install:
   pip install scipy pandas matplotlib seaborn
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
   ls -la monitoring/          # New monitoring directory
   ls -la publication_data/    # New publication directory
   ```

4. **Plotting Errors**: Install visualization dependencies
   ```bash
   pip install matplotlib seaborn
   ```

5. **Monitoring Issues**: System monitoring troubleshooting
   ```bash
   # Check if monitoring process is running
   ps aux | grep run-performance-test

   # Check monitoring log files
   ls -la monitoring/system_metrics_*.csv

   # Verify monitoring PID file
   cat monitoring/monitoring.pid

   # Manually stop monitoring if stuck
   pkill -f "run-performance-test.sh"
   rm -f monitoring/monitoring.pid
   ```

6. **Statistical Validation Errors**: Requires scipy
   ```bash
   pip install scipy numpy
   # If still failing, check Python version (requires 3.7+)
   python3 --version
   ```

7. **Publication Export Issues**: Check directory permissions
   ```bash
   # Ensure directories are writable
   mkdir -p publication_data monitoring
   chmod 755 publication_data monitoring

   # Check available disk space
   df -h .
   ```

8. **Grafana Dashboard Issues**: Verify JSON format
   ```bash
   # Validate dashboard JSON
   python3 -m json.tool metrics/grafana_dashboard_*.json

   # Check import instructions
   cat metrics/grafana_dashboard_*_import_instructions.md
   ```

9. **Environment Variable Issues**: Check configuration
   ```bash
   # Verify environment variables are set
   echo "Publication Mode: $PUBLICATION_MODE"
   echo "Monitoring: $REAL_TIME_MONITORING"
   echo "Validation: $STATISTICAL_VALIDATION"

   # Reset to defaults if needed
   unset PUBLICATION_MODE REAL_TIME_MONITORING STATISTICAL_VALIDATION
   ```

10. **Large File Issues**: Manage disk space
    ```bash
    # Check file sizes
    du -sh test_results/ metrics/ monitoring/ publication_data/

    # Clean old test results (keep recent ones)
    find test_results/ -name "*.log" -mtime +7 -delete
    find metrics/ -name "*.csv" -mtime +7 -delete
    find monitoring/ -name "*.csv" -mtime +7 -delete
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

## Summary of Enhanced Features

This framework now provides **publication-grade evaluation capabilities** for autoscaling agents with the following enhancements:

### 🚀 **New Command Options**
```bash
./scripts/run-performance-test.sh publication    # Publication-quality testing
./scripts/run-performance-test.sh monitor       # Enhanced monitoring
./scripts/run-performance-test.sh validate      # Statistical validation
./scripts/run-performance-test.sh export        # Publication data export
```

### 📊 **Enhanced Data Collection**
- **Real-time System Monitoring**: CPU, memory, load tracking
- **Statistical Validation**: Effect size analysis, power assessment
- **Multiple Export Formats**: CSV, JSON, LaTeX, Prometheus, Grafana
- **Publication-Ready Tables**: Formatted for academic papers
- **Complete Reproducibility**: Environment documentation and metadata

### 🔬 **Research-Grade Analysis**
- **Statistical Significance Testing**: Automated validation
- **Effect Size Calculation**: Cohen's d and percentage improvements
- **Sample Size Validation**: Ensures robust statistical conclusions
- **Coefficient of Variation**: Measures meaningful differences
- **Power Analysis**: Assesses reliability for publication

### 📚 **Publication Support**
- **Structured Data Packages**: Organized directories for research
- **LaTeX Table Generation**: Ready for academic papers
- **Metadata Documentation**: Complete experimental details
- **Grafana Dashboards**: Auto-generated with import instructions
- **Quality Assurance Checklist**: Publication readiness verification

### 🖥️ **System Integration**
- **Background Monitoring**: Non-intrusive system metrics collection
- **Cross-platform Support**: Works on macOS and Linux
- **Configurable Intervals**: Customizable monitoring frequencies
- **Automatic Cleanup**: Proper process management
- **Environment Variables**: Fine-grained control over features

This enhanced framework supports the hypothesis that RL-based approaches can outperform traditional rule-based systems in dynamic cloud environments, providing the rigor and documentation needed for academic publication and industry benchmarking.