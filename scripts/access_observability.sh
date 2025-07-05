#!/bin/bash

# MicroK8s Observability Access Script
# This script provides easy access to Grafana, Prometheus, and Alertmanager

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if observability is enabled
check_observability() {
    if ! kubectl get namespace observability >/dev/null 2>&1; then
        print_error "Observability namespace not found. Please enable the observability addon:"
        echo "  microk8s enable observability"
        exit 1
    fi
    
    if ! kubectl get pods -n observability | grep -q "grafana.*Running"; then
        print_error "Grafana is not running. Please check the observability stack:"
        echo "  kubectl get pods -n observability"
        exit 1
    fi
}

# Function to get Grafana credentials
get_grafana_credentials() {
    print_status "Getting Grafana credentials..."
    
    # Get admin password (using reset password)
ADMIN_PASSWORD="admin123"
    
    echo "Grafana Login Credentials:"
    echo "  Username: admin"
    echo "  Password: $ADMIN_PASSWORD"
    echo ""
}

# Function to start port forwarding
start_port_forward() {
    local service=$1
    local local_port=$2
    local remote_port=${3:-80}
    
    print_status "Starting port forward for $service on localhost:$local_port..."
    
    # Kill any existing port forward
    pkill -f "kubectl port-forward.*$service" 2>/dev/null || true
    
    # Start new port forward
    kubectl port-forward -n observability svc/$service $local_port:$remote_port &
    local pid=$!
    
    # Wait a moment for port forward to establish
    sleep 2
    
    # Check if port forward is working
    if kill -0 $pid 2>/dev/null; then
        print_success "Port forward started successfully!"
        return 0
    else
        print_error "Failed to start port forward for $service"
        return 1
    fi
}

# Function to import dashboard via Grafana API
import_dashboard_via_api() {
    local dashboard_file=$1
    local grafana_url="http://admin:$ADMIN_PASSWORD@localhost:3000"
    
    print_status "Importing RL Autoscaling Dashboard via API..."
    
    # Wait for Grafana to be ready
    sleep 5
    
    # Import dashboard
    if curl -s -X POST \
        -H "Content-Type: application/json" \
        -d @$dashboard_file \
        "$grafana_url/api/dashboards/db" > /dev/null; then
        print_success "Dashboard imported successfully!"
        echo "Dashboard available at: http://localhost:3000/dashboards"
    else
        print_warning "Failed to import dashboard via API. Please import manually:"
        echo "1. Go to http://localhost:3000"
        echo "2. Click '+' → Import"
        echo "3. Upload file: $dashboard_file"
    fi
}

# Function to show access information
show_access_info() {
    echo ""
    print_success "Observability Stack Access Information"
    echo "=============================================="
    echo ""
    
    # Grafana
    echo "🌐 Grafana Dashboard:"
    echo "  URL: http://localhost:3000"
    echo "  Username: admin"
    echo "  Password: $ADMIN_PASSWORD"
    echo ""
    
    # Prometheus
    echo "📊 Prometheus:"
    echo "  URL: http://localhost:9090"
    echo ""
    
    # Alertmanager
    echo "🚨 Alertmanager:"
    echo "  URL: http://localhost:9093"
    echo ""
    
    echo "💡 Tips:"
    echo "  - Open the URLs in your browser"
    echo "  - Use Ctrl+C to stop port forwarding"
    echo "  - Run this script again to restart port forwarding"
    echo ""
}

# Function to create production-ready RL autoscaling dashboard
create_rl_autoscaling_dashboard() {
    print_status "Creating production-ready RL Autoscaling Dashboard..."
    
    # Create dashboard JSON configuration
    cat > /tmp/rl_autoscaling_dashboard.json << 'EOF'
{
  "dashboard": {
    "id": null,
    "title": "RL Adaptive Autoscaling - Production Monitor",
    "tags": ["kubernetes", "autoscaling", "reinforcement-learning", "production"],
    "style": "dark",
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Cluster Overview",
        "type": "stat",
        "targets": [
          {
            "expr": "count(kube_pod_info)",
            "legendFormat": "Total Pods"
          },
          {
            "expr": "count(kube_node_info)",
            "legendFormat": "Total Nodes"
          },
          {
            "expr": "sum(kube_deployment_status_replicas)",
            "legendFormat": "Total Replicas"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
        "fieldConfig": {
          "defaults": {
            "color": {"mode": "palette-classic"},
            "custom": {"displayMode": "list"},
            "mappings": []
          }
        }
      },
      {
        "id": 2,
        "title": "HPA Scaling Status",
        "type": "timeseries",
        "targets": [
          {
            "expr": "kube_horizontalpodautoscaler_status_current_replicas",
            "legendFormat": "{{namespace}}/{{horizontalpodautoscaler}} - Current"
          },
          {
            "expr": "kube_horizontalpodautoscaler_spec_target_metrics",
            "legendFormat": "{{namespace}}/{{horizontalpodautoscaler}} - Target"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
        "fieldConfig": {
          "defaults": {
            "color": {"mode": "palette-classic"},
            "custom": {"drawStyle": "line", "lineInterpolation": "linear"},
            "mappings": []
          }
        }
      },
      {
        "id": 3,
        "title": "CPU Utilization by Pod (RL Target)",
        "type": "timeseries",
        "targets": [
          {
            "expr": "rate(container_cpu_usage_seconds_total{container!=\"\", pod=~\".*nginx.*\"}[5m]) * 100",
            "legendFormat": "{{pod}} - CPU %"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
        "fieldConfig": {
          "defaults": {
            "color": {"mode": "palette-classic"},
            "custom": {"drawStyle": "line", "lineInterpolation": "linear"},
            "thresholds": {
              "steps": [
                {"color": "green", "value": null},
                {"color": "yellow", "value": 50},
                {"color": "red", "value": 70}
              ]
            },
            "unit": "percent"
          }
        }
      },
      {
        "id": 4,
        "title": "Memory Usage by Pod",
        "type": "timeseries",
        "targets": [
          {
            "expr": "container_memory_usage_bytes{container!=\"\", pod=~\".*nginx.*\"} / 1024 / 1024",
            "legendFormat": "{{pod}} - Memory MB"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
        "fieldConfig": {
          "defaults": {
            "color": {"mode": "palette-classic"},
            "custom": {"drawStyle": "line", "lineInterpolation": "linear"},
            "unit": "MB"
          }
        }
      },
      {
        "id": 5,
        "title": "RL Agent Performance Metrics",
        "type": "timeseries",
        "targets": [
          {
            "expr": "rate(http_requests_total{job=\"nginx\"}[5m])",
            "legendFormat": "Request Rate (req/s)"
          },
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{job=\"nginx\"}[5m]))",
            "legendFormat": "95th Percentile Latency"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 16},
        "fieldConfig": {
          "defaults": {
            "color": {"mode": "palette-classic"},
            "custom": {"drawStyle": "line", "lineInterpolation": "linear"},
            "unit": "short"
          }
        }
      },
      {
        "id": 6,
        "title": "Scaling Events Timeline",
        "type": "logs",
        "targets": [
          {
            "expr": "{job=\"kube-state-metrics\"} |= \"HorizontalPodAutoscaler\"",
            "legendFormat": "HPA Events"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 16},
        "fieldConfig": {
          "defaults": {
            "color": {"mode": "palette-classic"},
            "custom": {"displayMode": "list"}
          }
        }
      },
      {
        "id": 7,
        "title": "Resource Efficiency Score",
        "type": "gauge",
        "targets": [
          {
            "expr": "avg(rate(container_cpu_usage_seconds_total{container!=\"\"}[5m]) * 100)",
            "legendFormat": "Avg CPU Utilization"
          }
        ],
        "gridPos": {"h": 8, "w": 6, "x": 0, "y": 24},
        "fieldConfig": {
          "defaults": {
            "color": {"mode": "palette-classic"},
            "custom": {"displayMode": "gauge"},
            "max": 100,
            "min": 0,
            "unit": "percent"
          }
        }
      },
      {
        "id": 8,
        "title": "Cost Optimization Index",
        "type": "stat",
        "targets": [
          {
            "expr": "sum(kube_deployment_status_replicas) / count(kube_node_info)",
            "legendFormat": "Pods per Node"
          }
        ],
        "gridPos": {"h": 8, "w": 6, "x": 6, "y": 24},
        "fieldConfig": {
          "defaults": {
            "color": {"mode": "palette-classic"},
            "custom": {"displayMode": "single-stat"},
            "unit": "short"
          }
        }
      },
      {
        "id": 9,
        "title": "RL Agent Decision Quality",
        "type": "timeseries",
        "targets": [
          {
            "expr": "abs(kube_horizontalpodautoscaler_status_current_replicas - kube_horizontalpodautoscaler_spec_target_metrics)",
            "legendFormat": "Scaling Decision Error"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 24},
        "fieldConfig": {
          "defaults": {
            "color": {"mode": "palette-classic"},
            "custom": {"drawStyle": "line", "lineInterpolation": "linear"},
            "unit": "short"
          }
        }
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "30s"
  },
  "folderId": null,
  "overwrite": true
}
EOF

    print_success "RL Autoscaling Dashboard configuration created!"
    echo "Dashboard JSON saved to: /tmp/rl_autoscaling_dashboard.json"
    echo ""
}

# Function to create RL-specific Prometheus queries
create_rl_prometheus_queries() {
    print_status "Creating RL-specific Prometheus query configurations..."
    
    cat > /tmp/rl_prometheus_queries.yaml << 'EOF'
# RL Autoscaling Prometheus Queries Configuration

# Core HPA Metrics
hpa_current_replicas: kube_horizontalpodautoscaler_status_current_replicas
hpa_target_replicas: kube_horizontalpodautoscaler_spec_target_metrics
hpa_min_replicas: kube_horizontalpodautoscaler_spec_min_replicas
hpa_max_replicas: kube_horizontalpodautoscaler_spec_max_replicas

# Resource Utilization (RL State)
cpu_usage_per_pod: rate(container_cpu_usage_seconds_total{container!=""}[5m]) * 100
memory_usage_per_pod: container_memory_usage_bytes{container!=""} / 1024 / 1024
cpu_requests: kube_pod_container_resource_requests{resource="cpu"}
memory_requests: kube_pod_container_resource_requests{resource="memory"}

# Performance Metrics (RL Reward)
request_rate: rate(http_requests_total{job="nginx"}[5m])
response_time_95p: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{job="nginx"}[5m]))
error_rate: rate(http_requests_total{job="nginx", status=~"5.."}[5m]) / rate(http_requests_total{job="nginx"}[5m])

# Scaling Efficiency (RL Performance)
scaling_frequency: rate(kube_horizontalpodautoscaler_status_current_replicas[5m])
resource_efficiency: avg(rate(container_cpu_usage_seconds_total{container!=""}[5m]) * 100)
cost_performance_ratio: sum(kube_deployment_status_replicas) / count(kube_node_info)

# RL Agent Specific Metrics
decision_accuracy: abs(kube_horizontalpodautoscaler_status_current_replicas - kube_horizontalpodautoscaler_spec_target_metrics)
scaling_latency: time() - kube_horizontalpodautoscaler_status_last_scale_time
EOF

    print_success "RL Prometheus queries configuration created!"
    echo "Queries saved to: /tmp/rl_prometheus_queries.yaml"
    echo ""
}

# Function to show useful Grafana dashboards
show_grafana_dashboards() {
    echo "📋 Production-Ready RL Autoscaling Dashboards:"
    echo "  1. RL Adaptive Autoscaling Monitor (Custom - Auto-created)"
    echo "  2. Kubernetes Cluster Monitoring (ID: 315)"
    echo "  3. Node Exporter (ID: 1860)"
    echo "  4. Kubernetes Pod Monitoring (ID: 6417)"
    echo "  5. HPA Metrics (custom queries)"
    echo ""
    echo "🚀 RL-Specific Features:"
    echo "  - Real-time scaling decision tracking"
    echo "  - Resource efficiency scoring"
    echo "  - Cost optimization metrics"
    echo "  - RL agent performance analysis"
    echo "  - Production-ready alerting"
    echo ""
    echo "To import additional dashboards:"
    echo "  1. Go to http://localhost:3000"
    echo "  2. Click '+' → Import"
    echo "  3. Enter the dashboard ID or upload JSON"
    echo ""
}

# Function to create production alerting rules for RL autoscaling
create_rl_alerting_rules() {
    print_status "Creating production alerting rules for RL autoscaling..."
    
    cat > /tmp/rl_alerting_rules.yaml << 'EOF'
groups:
  - name: rl-autoscaling-alerts
    rules:
      # High CPU utilization alert
      - alert: HighCPUUtilization
        expr: avg(rate(container_cpu_usage_seconds_total{container!=""}[5m]) * 100) > 80
        for: 2m
        labels:
          severity: warning
          component: rl-autoscaling
        annotations:
          summary: "High CPU utilization detected"
          description: "Average CPU utilization is {{ $value }}% for the last 5 minutes"

      # HPA scaling failure alert
      - alert: HPAScalingFailure
        expr: kube_horizontalpodautoscaler_status_condition{condition="AbleToScale", status="False"} > 0
        for: 1m
        labels:
          severity: critical
          component: rl-autoscaling
        annotations:
          summary: "HPA scaling failure detected"
          description: "HPA {{ $labels.horizontalpodautoscaler }} in namespace {{ $labels.namespace }} is unable to scale"

      # Resource efficiency alert
      - alert: LowResourceEfficiency
        expr: avg(rate(container_cpu_usage_seconds_total{container!=""}[5m]) * 100) < 20
        for: 10m
        labels:
          severity: warning
          component: rl-autoscaling
        annotations:
          summary: "Low resource efficiency detected"
          description: "Average CPU utilization is only {{ $value }}% - consider reducing replicas"

      # RL agent decision quality alert
      - alert: PoorRLDecisionQuality
        expr: abs(kube_horizontalpodautoscaler_status_current_replicas - kube_horizontalpodautoscaler_spec_target_metrics) > 2
        for: 5m
        labels:
          severity: warning
          component: rl-autoscaling
        annotations:
          summary: "Poor RL agent decision quality"
          description: "RL agent decision error is {{ $value }} replicas"

      # High latency alert
      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{job="nginx"}[5m])) > 0.5
        for: 2m
        labels:
          severity: warning
          component: rl-autoscaling
        annotations:
          summary: "High response latency detected"
          description: "95th percentile latency is {{ $value }}s"

      # High error rate alert
      - alert: HighErrorRate
        expr: rate(http_requests_total{job="nginx", status=~"5.."}[5m]) / rate(http_requests_total{job="nginx"}[5m]) > 0.05
        for: 2m
        labels:
          severity: critical
          component: rl-autoscaling
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }}"
EOF

    print_success "RL Alerting rules configuration created!"
    echo "Alerting rules saved to: /tmp/rl_alerting_rules.yaml"
    echo ""
}

# Function to show production-ready RL queries
show_rl_production_queries() {
    echo "🔍 Production-Ready RL Autoscaling Queries:"
    echo ""
    echo "📊 Core HPA Metrics:"
    echo "  Current Replicas: kube_horizontalpodautoscaler_status_current_replicas"
    echo "  Target Replicas: kube_horizontalpodautoscaler_spec_target_metrics"
    echo "  Scaling Events: rate(kube_horizontalpodautoscaler_status_current_replicas[5m])"
    echo ""
    echo "🎯 RL State Metrics:"
    echo "  CPU Usage: rate(container_cpu_usage_seconds_total{container!=\"\"}[5m]) * 100"
    echo "  Memory Usage: container_memory_usage_bytes{container!=\"\"} / 1024 / 1024"
    echo "  Resource Efficiency: avg(rate(container_cpu_usage_seconds_total{container!=\"\"}[5m]) * 100)"
    echo ""
    echo "🏆 RL Performance Metrics:"
    echo "  Request Rate: rate(http_requests_total{job=\"nginx\"}[5m])"
    echo "  Response Time: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{job=\"nginx\"}[5m]))"
    echo "  Error Rate: rate(http_requests_total{job=\"nginx\", status=~\"5..\"}[5m]) / rate(http_requests_total{job=\"nginx\"}[5m])"
    echo ""
    echo "💰 Cost Optimization:"
    echo "  Pods per Node: sum(kube_deployment_status_replicas) / count(kube_node_info)"
    echo "  Decision Quality: abs(kube_horizontalpodautoscaler_status_current_replicas - kube_horizontalpodautoscaler_spec_target_metrics)"
    echo ""
}

# Function to show RL autoscaling reporting guide
show_rl_reporting_guide() {
    echo "📋 RL Autoscaling Production Reporting Guide"
    echo "============================================"
    echo ""
    echo "🎯 Executive Summary Metrics:"
    echo "  • Resource Efficiency Score: CPU utilization optimization"
    echo "  • Cost Optimization Index: Pods per node ratio"
    echo "  • Scaling Decision Quality: HPA accuracy"
    echo "  • Performance Impact: Response time and throughput"
    echo ""
    echo "📊 Technical Deep Dive:"
    echo "  • RL Agent Performance: Decision accuracy over time"
    echo "  • Scaling Patterns: Frequency and magnitude analysis"
    echo "  • Resource Utilization: CPU/Memory efficiency trends"
    echo "  • Cost Analysis: Infrastructure cost per request"
    echo ""
    echo "🚨 Alerting & SLA Monitoring:"
    echo "  • High CPU utilization (>80%)"
    echo "  • HPA scaling failures"
    echo "  • Poor RL decision quality (>2 replica error)"
    echo "  • High latency (>500ms 95th percentile)"
    echo "  • High error rate (>5%)"
    echo ""
    echo "📈 Business Impact Metrics:"
    echo "  • Cost savings from optimized scaling"
    echo "  • Performance improvements vs traditional HPA"
    echo "  • Reliability improvements (uptime, error rates)"
    echo "  • Resource utilization optimization"
    echo ""
    echo "🔧 Production Recommendations:"
    echo "  • Monitor RL agent decision quality continuously"
    echo "  • Set up automated alerting for scaling anomalies"
    echo "  • Regular cost optimization reviews"
    echo "  • Performance baseline establishment"
    echo "  • A/B testing with traditional HPA"
    echo ""
}

# Function to show Prometheus queries for HPA
show_hpa_queries() {
    echo "🔍 Useful Prometheus Queries for HPA:"
    echo ""
    echo "CPU Usage per Pod:"
    echo "  rate(container_cpu_usage_seconds_total{container!=\"\"}[5m])"
    echo ""
    echo "Memory Usage per Pod:"
    echo "  container_memory_usage_bytes{container!=\"\"}"
    echo ""
    echo "HPA Current Replicas:"
    echo "  kube_horizontalpodautoscaler_status_current_replicas"
    echo ""
    echo "HPA Target Replicas:"
    echo "  kube_horizontalpodautoscaler_spec_target_metrics"
    echo ""
    echo "Pod Count by Deployment:"
    echo "  kube_deployment_status_replicas"
    echo ""
}

# Main execution
print_status "MicroK8s Observability Access Script"
echo "=========================================="

# Check if observability is enabled
check_observability

# Get credentials
get_grafana_credentials

# Ask user what they want to access
echo ""
print_status "Choose what to access:"
echo "1. Grafana only (quick access)"
echo "2. All observability services (Grafana + Prometheus + Alertmanager)"
echo "3. Exit"
echo ""

read -p "Enter choice (1-3): " choice

case $choice in
    1)
        print_status "Starting Grafana only with RL dashboard..."
        
        # Create production configurations
        create_rl_autoscaling_dashboard
        create_rl_prometheus_queries
        create_rl_alerting_rules
        
        if start_port_forward "kube-prom-stack-grafana" 3000 80; then
            echo ""
            print_success "Grafana Access Information"
            echo "=========================="
            echo "URL: http://localhost:3000"
            echo "Username: admin"
            echo "Password: $ADMIN_PASSWORD"
            echo ""
            
            # Import dashboard
            import_dashboard_via_api "/tmp/rl_autoscaling_dashboard.json"
            
            echo ""
            echo "Press Ctrl+C to stop port forwarding..."
            
            # Wait for interrupt
            trap 'echo ""; print_status "Stopping Grafana port forward..."; pkill -f "kubectl port-forward.*grafana"; exit' INT
            wait
        fi
        ;;
    2)
        print_status "Starting all observability services..."
        
        # Start port forwarding for all services
        if start_port_forward "kube-prom-stack-grafana" 3000 80; then
            GRAFANA_PID=$!
        fi

        if start_port_forward "kube-prom-stack-kube-prome-prometheus" 9090 9090; then
            PROMETHEUS_PID=$!
        fi

        if start_port_forward "kube-prom-stack-kube-prome-alertmanager" 9093 9093; then
            ALERTMANAGER_PID=$!
        fi

        # Show access information
        show_access_info

        # Create production-ready RL autoscaling configurations
print_status "Setting up production-ready RL autoscaling monitoring..."
create_rl_autoscaling_dashboard
create_rl_prometheus_queries
create_rl_alerting_rules

# Show useful information
show_grafana_dashboards
show_rl_production_queries
show_hpa_queries
show_rl_reporting_guide

        # Wait for user input
        echo "Press Enter to stop port forwarding and exit..."
        read

        # Cleanup
        print_status "Stopping port forwarding..."
        pkill -f "kubectl port-forward.*kube-prom-stack-grafana" 2>/dev/null || true
        pkill -f "kubectl port-forward.*kube-prom-stack-kube-prome-prometheus" 2>/dev/null || true
        pkill -f "kubectl port-forward.*kube-prom-stack-kube-prome-alertmanager" 2>/dev/null || true

        print_success "Port forwarding stopped. Observability services are no longer accessible."
        ;;
    3)
        print_status "Exiting..."
        exit 0
        ;;
    *)
        print_error "Invalid choice. Exiting..."
        exit 1
        ;;
esac 