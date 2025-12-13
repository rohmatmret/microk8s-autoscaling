#!/bin/bash

# Complete Publication Workflow Script - Docker Version
# This script automates the entire process from running tests to deploying monitoring with Docker

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
GRAFANA_PORT="${GRAFANA_PORT:-3000}"
PROMETHEUS_PORT="${PROMETHEUS_PORT:-9090}"
METRICS_PORT="${METRICS_PORT:-8080}"

# Container names
GRAFANA_CONTAINER="autoscaling-grafana"
PROMETHEUS_CONTAINER="autoscaling-prometheus"
METRICS_CONTAINER="autoscaling-metrics"

# Print functions
print_header() {
    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================================${NC}"
}

print_step() {
    echo -e "${GREEN}[STEP $1]${NC} $2"
}

print_info() {
    echo -e "${CYAN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Function to check if Docker is running
check_docker() {
    print_info "Checking Docker availability..."

    if ! command -v docker >/dev/null 2>&1; then
        print_error "Docker is not installed. Please install Docker first."
        print_info "Install from: https://docs.docker.com/get-docker/"
        exit 1
    fi

    if ! docker info >/dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker first."
        exit 1
    fi

    print_success "Docker is available and running"
    print_info "Docker version: $(docker --version)"
}

# Function to check for existing containers
check_existing_containers() {
    print_info "Checking for existing containers..."

    local existing_containers=""
    for container in "$GRAFANA_CONTAINER" "$PROMETHEUS_CONTAINER" "$METRICS_CONTAINER"; do
        if docker ps -a --format '{{.Names}}' | grep -q "^${container}$"; then
            existing_containers="$existing_containers $container"
        fi
    done

    if [ -n "$existing_containers" ]; then
        print_warning "Found existing containers:$existing_containers"
        print_info "Stopping and removing existing containers..."

        for container in $existing_containers; do
            docker stop "$container" >/dev/null 2>&1 || true
            docker rm "$container" >/dev/null 2>&1 || true
        done

        print_success "Existing containers cleaned up"
    fi
}

# Function to create Docker network
create_docker_network() {
    print_info "Creating Docker network for monitoring..."

    if docker network ls | grep -q "autoscaling-network"; then
        print_info "Network 'autoscaling-network' already exists"
    else
        docker network create autoscaling-network
        print_success "Created Docker network 'autoscaling-network'"
    fi
}

# Function to start metrics service
start_metrics_service() {
    print_info "Starting metrics service..."

    # Find the latest metrics file
    local latest_metrics=""
    if [ -d "$PROJECT_ROOT/metrics" ]; then
        latest_metrics=$(find "$PROJECT_ROOT/metrics" -name "autoscaler_metrics_*.prom" -type f -exec stat -f "%m %N" {} \; 2>/dev/null | sort -k 1nr | head -1 | cut -d' ' -f2- || echo "")
    fi

    # Create metrics server Python script
    cat > /tmp/metrics_server.py << 'EOF'
#!/usr/bin/env python3
import http.server
import socketserver
import os
import time
import signal
import sys
from datetime import datetime

class MetricsHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/metrics':
            self.send_response(200)
            self.send_header('Content-type', 'text/plain; version=0.0.4')
            self.end_headers()

            # Always read the file fresh - don't keep it open
            metrics_content = self.read_metrics_file()
            self.wfile.write(metrics_content.encode())

        elif self.path == '/reload':
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write("Metrics reloaded\n".encode())

        elif self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write("OK\n".encode())
        else:
            self.send_response(404)
            self.end_headers()

    def read_metrics_file(self):
        # Try to read the main metrics file first
        for metrics_file in ['/data/metrics.prom.tmp', '/data/metrics.prom']:
            if os.path.exists(metrics_file):
                try:
                    # Use 'rb' mode to avoid text encoding issues with large files
                    with open(metrics_file, 'r', encoding='utf-8', errors='replace') as f:
                        content = f.read()
                    return content
                except (IOError, OSError) as e:
                    # File might be being written, try the next one
                    continue
                except Exception as e:
                    return f"# Error reading metrics: {e}\n"

        # Return sample metrics if no real data available
        return self.get_sample_metrics()

    def get_sample_metrics(self):
        timestamp = int(time.time())
        return f'''# HELP autoscaler_info Autoscaler information
# TYPE autoscaler_info gauge
autoscaler_info{{version="1.0",cluster="docker"}} 1 {timestamp}

# HELP autoscaler_cpu_utilization CPU utilization by agent
# TYPE autoscaler_cpu_utilization gauge
autoscaler_cpu_utilization{{agent="hybrid_dqn_ppo",scenario="baseline"}} 0.65 {timestamp}
autoscaler_cpu_utilization{{agent="ppo",scenario="baseline"}} 0.52 {timestamp}
autoscaler_cpu_utilization{{agent="rule_based",scenario="baseline"}} 0.58 {timestamp}

# HELP autoscaler_response_time_seconds Response time in seconds
# TYPE autoscaler_response_time_seconds gauge
autoscaler_response_time_seconds{{agent="hybrid_dqn_ppo",scenario="baseline"}} 0.145 {timestamp}
autoscaler_response_time_seconds{{agent="ppo",scenario="baseline"}} 0.158 {timestamp}
autoscaler_response_time_seconds{{agent="rule_based",scenario="baseline"}} 0.172 {timestamp}

# HELP autoscaler_pod_count Current number of pods
# TYPE autoscaler_pod_count gauge
autoscaler_pod_count{{agent="hybrid_dqn_ppo",scenario="baseline"}} 5 {timestamp}
autoscaler_pod_count{{agent="ppo",scenario="baseline"}} 6 {timestamp}
autoscaler_pod_count{{agent="rule_based",scenario="baseline"}} 4 {timestamp}

# HELP autoscaler_resource_cost_dollars Resource cost in dollars
# TYPE autoscaler_resource_cost_dollars gauge
autoscaler_resource_cost_dollars{{agent="hybrid_dqn_ppo",scenario="baseline"}} 125.50 {timestamp}
autoscaler_resource_cost_dollars{{agent="ppo",scenario="baseline"}} 145.20 {timestamp}
autoscaler_resource_cost_dollars{{agent="rule_based",scenario="baseline"}} 135.80 {timestamp}
'''

if __name__ == "__main__":
    PORT = 8080
    Handler = MetricsHandler
    print(f"Starting metrics server on port {PORT}")
    print(f"Metrics endpoint: http://localhost:{PORT}/metrics")
    print(f"Health endpoint: http://localhost:{PORT}/health")

    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Serving metrics at port {PORT}")
        httpd.serve_forever()
EOF

    # Create metrics data file if we have real metrics
    if [ -n "$latest_metrics" ] && [ -f "$latest_metrics" ]; then
        print_info "Using real metrics data from: $latest_metrics"
        cp "$latest_metrics" /tmp/metrics.prom
    else
        print_warning "No metrics file found, will serve sample metrics"
        echo "# Sample metrics - run publication test to get real data" > /tmp/metrics.prom
    fi

    # Start metrics container
    docker run -d \
        --name "$METRICS_CONTAINER" \
        --network autoscaling-network \
        -p "$METRICS_PORT:8080" \
        -v /tmp/metrics_server.py:/app/server.py \
        -v /tmp/metrics.prom:/data/metrics.prom \
        -w /app \
        python:3.9-slim \
        python server.py

    # Wait for container to be ready
    sleep 3

    if docker ps --format '{{.Names}}' | grep -q "^${METRICS_CONTAINER}$"; then
        print_success "Metrics service started successfully"
        print_info "Metrics available at: http://localhost:$METRICS_PORT/metrics"
    else
        print_error "Failed to start metrics service"
        docker logs "$METRICS_CONTAINER" 2>/dev/null || true
        return 1
    fi
}

# Function to start Prometheus
start_prometheus() {
    print_info "Starting Prometheus..."

    # Create Prometheus configuration
    cat > /tmp/prometheus.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'autoscaling-metrics'
    static_configs:
      - targets: ['$METRICS_CONTAINER:8080']
    scrape_interval: 10s
    metrics_path: '/metrics'
EOF

    # Start Prometheus container
    docker run -d \
        --name "$PROMETHEUS_CONTAINER" \
        --network autoscaling-network \
        -p "$PROMETHEUS_PORT:9090" \
        -v /tmp/prometheus.yml:/etc/prometheus/prometheus.yml \
        prom/prometheus:latest \
        --config.file=/etc/prometheus/prometheus.yml \
        --storage.tsdb.path=/prometheus \
        --web.console.libraries=/etc/prometheus/console_libraries \
        --web.console.templates=/etc/prometheus/consoles \
        --storage.tsdb.retention.time=200h \
        --web.enable-lifecycle

    # Wait for container to be ready
    sleep 5

    if docker ps --format '{{.Names}}' | grep -q "^${PROMETHEUS_CONTAINER}$"; then
        print_success "Prometheus started successfully"
        print_info "Prometheus available at: http://localhost:$PROMETHEUS_PORT"
    else
        print_error "Failed to start Prometheus"
        docker logs "$PROMETHEUS_CONTAINER" 2>/dev/null || true
        return 1
    fi
}

# Function to start Grafana
start_grafana() {
    print_info "Starting Grafana..."

    # Create Grafana data directory
    mkdir -p /tmp/grafana-data
    chmod 777 /tmp/grafana-data

    # Start Grafana container
    docker run -d \
        --name "$GRAFANA_CONTAINER" \
        --network autoscaling-network \
        -p "$GRAFANA_PORT:3000" \
        -v /tmp/grafana-data:/var/lib/grafana \
        -e GF_SECURITY_ADMIN_PASSWORD=admin \
        -e GF_USERS_ALLOW_SIGN_UP=false \
        -e GF_INSTALL_PLUGINS=grafana-clock-panel,grafana-simple-json-datasource \
        grafana/grafana:latest

    # Wait for Grafana to be ready
    print_info "Waiting for Grafana to start..."
    local max_attempts=30
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        if curl -s http://localhost:$GRAFANA_PORT/api/health >/dev/null 2>&1; then
            break
        fi
        print_info "Attempt $attempt/$max_attempts: Waiting for Grafana..."
        sleep 3
        ((attempt++))
    done

    if [ $attempt -le $max_attempts ]; then
        print_success "Grafana started successfully"
        print_info "Grafana available at: http://localhost:$GRAFANA_PORT"
    else
        print_warning "Grafana may still be starting up"
        print_info "Check manually at: http://localhost:$GRAFANA_PORT"
    fi
}

# Function to configure Grafana data source
configure_grafana_datasource() {
    print_info "Configuring Prometheus data source in Grafana..."
    sleep 10  # Give Grafana more time to be ready

    # Add Prometheus data source
    cat > /tmp/datasource.json << EOF
{
  "name": "Prometheus",
  "type": "prometheus",
  "url": "http://$PROMETHEUS_CONTAINER:9090",
  "access": "proxy",
  "isDefault": true,
  "basicAuth": false,
  "jsonData": {
    "httpMethod": "POST"
  }
}
EOF

    # Configure data source
    if curl -X POST \
        -H "Content-Type: application/json" \
        -u admin:admin \
        -d @/tmp/datasource.json \
        "http://localhost:$GRAFANA_PORT/api/datasources" >/dev/null 2>&1; then
        print_success "Prometheus data source configured"
    else
        print_warning "Data source configuration may have failed (might already exist)"
    fi
}

# Function to import dashboard
import_dashboard() {
    print_info "Importing autoscaling dashboard to Grafana..."

    # Look for dashboard in multiple locations
    local dashboard_file=""

    # Priority 1: Project root (grafana_autoscaling_dashboard.json)
    if [ -f "$PROJECT_ROOT/grafana_autoscaling_dashboard.json" ]; then
        dashboard_file="$PROJECT_ROOT/grafana_autoscaling_dashboard.json"
        print_info "Found research dashboard: $dashboard_file"
    # Priority 2: metrics directory (old pattern)
    elif [ -f "$PROJECT_ROOT/metrics/grafana_dashboard_*.json" ]; then
        dashboard_file=$(find "$PROJECT_ROOT/metrics" -name "grafana_dashboard_*.json" -type f 2>/dev/null | head -1)
        print_info "Found dashboard in metrics: $dashboard_file"
    fi

    if [ -n "$dashboard_file" ] && [ -f "$dashboard_file" ]; then
        # Check if dashboard JSON already has correct structure
        if cat "$dashboard_file" | jq -e '.dashboard' >/dev/null 2>&1; then
            # Already wrapped, use as-is
            print_info "Dashboard JSON already wrapped, importing directly..."
            cp "$dashboard_file" /tmp/dashboard_import.json
        else
            # Need to wrap
            print_info "Wrapping dashboard JSON for import..."
            cat > /tmp/dashboard_import.json << EOF
{
  "dashboard": $(cat "$dashboard_file"),
  "folderId": 0,
  "overwrite": true
}
EOF
        fi

        # Import dashboard with verbose output
        print_info "Importing dashboard to Grafana..."
        local response=$(curl -s -X POST \
            -H "Content-Type: application/json" \
            -u admin:admin \
            -d @/tmp/dashboard_import.json \
            "http://localhost:$GRAFANA_PORT/api/dashboards/db")

        local exit_code=$?

        if [ $exit_code -eq 0 ]; then
            # Check response for success
            if echo "$response" | grep -q '"status":"success"'; then
                local dashboard_uid=$(echo "$response" | jq -r '.uid' 2>/dev/null)
                local dashboard_url=$(echo "$response" | jq -r '.url' 2>/dev/null)
                print_success "Dashboard imported successfully!"
                if [ -n "$dashboard_uid" ] && [ "$dashboard_uid" != "null" ]; then
                    print_info "Dashboard UID: $dashboard_uid"
                    print_info "Dashboard URL: http://localhost:$GRAFANA_PORT$dashboard_url"
                fi
            else
                print_warning "Dashboard import response: $response"
                # Try to extract error message
                local error_msg=$(echo "$response" | jq -r '.message' 2>/dev/null)
                if [ -n "$error_msg" ] && [ "$error_msg" != "null" ]; then
                    print_warning "Error: $error_msg"
                fi
            fi
        else
            print_error "Failed to import dashboard (curl error: $exit_code)"
        fi
    else
        # Create a basic dashboard if none exists
        print_warning "No dashboard file found in project root or metrics directory"
        print_info "Creating basic dashboard..."
        create_basic_dashboard
    fi
}

# Function to create a basic dashboard
create_basic_dashboard() {
    cat > /tmp/basic_dashboard.json << 'EOF'
{
  "dashboard": {
    "id": null,
    "title": "Autoscaling Performance Monitor",
    "tags": ["autoscaling", "performance"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "CPU Utilization by Agent",
        "type": "timeseries",
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
        "targets": [
          {
            "expr": "autoscaler_cpu_utilization",
            "legendFormat": "{{agent}} - {{scenario}}"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "color": {"mode": "palette-classic"},
            "unit": "percentunit",
            "min": 0,
            "max": 1
          }
        }
      },
      {
        "id": 2,
        "title": "Response Time Comparison",
        "type": "timeseries",
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
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
        "title": "Pod Count Over Time",
        "type": "timeseries",
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
        "targets": [
          {
            "expr": "autoscaler_pod_count",
            "legendFormat": "{{agent}} - {{scenario}}"
          }
        ]
      },
      {
        "id": 4,
        "title": "Resource Cost",
        "type": "stat",
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
        "targets": [
          {
            "expr": "autoscaler_resource_cost_dollars",
            "legendFormat": "{{agent}}"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "currencyUSD"
          }
        }
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "10s"
  },
  "folderId": 0,
  "overwrite": true
}
EOF

    curl -X POST \
        -H "Content-Type: application/json" \
        -u admin:admin \
        -d @/tmp/basic_dashboard.json \
        "http://localhost:$GRAFANA_PORT/api/dashboards/db" >/dev/null 2>&1

    print_success "Basic dashboard created"
}

# Function to update metrics with real data
update_metrics_data() {
    print_info "Updating metrics with latest data..."

    # Find the latest metrics file
    local latest_metrics=""
    if [ -d "$PROJECT_ROOT/metrics" ]; then
        latest_metrics=$(find "$PROJECT_ROOT/metrics" -name "autoscaler_metrics_*.prom" -type f -exec stat -f "%m %N" {} \; 2>/dev/null | sort -k 1nr | head -1 | cut -d' ' -f2- || echo "")
    fi

    if [ -n "$latest_metrics" ] && [ -f "$latest_metrics" ]; then
        print_info "Updating with real metrics from: $latest_metrics"

        # New strategy: Copy to .tmp file and let the server prefer it
        if docker cp "$latest_metrics" "$METRICS_CONTAINER:/data/metrics.prom.tmp"; then
            print_info "New metrics file copied as temporary file"

            # Signal the server to reload (optional endpoint)
            curl -s "http://localhost:$METRICS_PORT/reload" >/dev/null 2>&1 || true

            # Wait a moment then try to replace the main file
            sleep 1
            if docker exec "$METRICS_CONTAINER" sh -c "cp /data/metrics.prom.tmp /data/metrics.prom" 2>/dev/null; then
                print_success "Metrics data updated successfully"
                # Clean up temp file
                docker exec "$METRICS_CONTAINER" rm -f /data/metrics.prom.tmp 2>/dev/null || true
            else
                print_info "Main file replacement failed, but metrics server will use temporary file"
                print_success "Metrics data available via temporary file"
            fi
        else
            print_error "Failed to copy metrics file to container"
        fi
    else
        print_warning "No new metrics data found"
    fi
}

# Function to show access information
show_access_info() {
    print_header "SERVICE ACCESS INFORMATION"

    echo -e "${GREEN}🌐 Access URLs:${NC}"
    echo -e "  📊 Grafana Home: ${CYAN}http://localhost:$GRAFANA_PORT${NC}"
    echo -e "  📊 Research Dashboard: ${CYAN}http://localhost:$GRAFANA_PORT/d/autoscaling-research${NC}"
    echo -e "  📈 Prometheus Metrics: ${CYAN}http://localhost:$PROMETHEUS_PORT${NC}"
    echo -e "  🔧 Metrics Service: ${CYAN}http://localhost:$METRICS_PORT/metrics${NC}"
    echo ""
    echo -e "${GREEN}📧 Grafana Login:${NC}"
    echo -e "  Username: ${YELLOW}admin${NC}"
    echo -e "  Password: ${YELLOW}admin${NC}"
    echo ""
    echo -e "${GREEN}📋 Dashboard Details:${NC}"
    echo -e "  Dashboard UID: ${YELLOW}autoscaling-research${NC}"
    echo -e "  Dashboard Title: ${YELLOW}Autoscaling Research Dashboard - Hybrid DQN-PPO vs K8s HPA${NC}"
    echo -e "  Panels: ${YELLOW}24 panels in 6 sections${NC}"
    echo ""
    echo -e "${GREEN}🔍 Docker Containers:${NC}"
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
}

# Function to show container status
show_status() {
    print_header "MONITORING STACK STATUS"

    echo -e "${GREEN}📊 Container Status:${NC}"
    for container in "$GRAFANA_CONTAINER" "$PROMETHEUS_CONTAINER" "$METRICS_CONTAINER"; do
        if docker ps --format '{{.Names}}' | grep -q "^${container}$"; then
            status=$(docker ps --format '{{.Status}}' --filter "name=${container}")
            echo -e "  ✅ ${container}: ${status}"
        else
            echo -e "  ❌ ${container}: Not running"
        fi
    done

    echo ""
    echo -e "${GREEN}🔗 Network Status:${NC}"
    if docker network ls | grep -q "autoscaling-network"; then
        echo -e "  ✅ autoscaling-network: Active"
    else
        echo -e "  ❌ autoscaling-network: Not found"
    fi

    echo ""
    show_access_info
}

# Function to cleanup containers and network
cleanup_all() {
    print_warning "Cleaning up Docker monitoring stack..."

    # Stop and remove containers
    for container in "$GRAFANA_CONTAINER" "$PROMETHEUS_CONTAINER" "$METRICS_CONTAINER"; do
        if docker ps -a --format '{{.Names}}' | grep -q "^${container}$"; then
            print_info "Removing container: $container"
            docker stop "$container" >/dev/null 2>&1 || true
            docker rm "$container" >/dev/null 2>&1 || true
        fi
    done

    # Remove network
    if docker network ls | grep -q "autoscaling-network"; then
        docker network rm autoscaling-network >/dev/null 2>&1 || true
        print_info "Removed network: autoscaling-network"
    fi

    # Clean up temporary files
    rm -f /tmp/prometheus.yml /tmp/datasource.json /tmp/dashboard_import.json
    rm -f /tmp/basic_dashboard.json /tmp/metrics_server.py /tmp/metrics.prom
    rm -rf /tmp/grafana-data

    print_success "Cleanup completed"
}

# Main workflow function
main() {
    print_header "COMPLETE PUBLICATION WORKFLOW - DOCKER"
    echo -e "${PURPLE}Starting complete autoscaling publication workflow with Docker monitoring${NC}"
    echo ""

    # Step 1: Check Docker
    print_step "1" "Checking Docker availability"
    check_docker

    # Step 2: Clean existing containers
    print_step "2" "Cleaning existing containers"
    check_existing_containers

    # Step 3: Run publication test
    print_step "3" "Running publication-quality performance test"
    cd "$PROJECT_ROOT"
    export PUBLICATION_MODE=true
    export STATISTICAL_VALIDATION=true
    export REAL_TIME_MONITORING=true

    # Enable realistic virtual time simulation for production-like metrics
    # Each simulation step = 1-5 minutes of virtual time (random variance)
    export SIMULATION_TIME_STEP_MINUTES="${SIMULATION_TIME_STEP_MINUTES:-1}"
    export SIMULATION_TIME_STEP_VARIANCE="${SIMULATION_TIME_STEP_VARIANCE:-4}"

    print_info "Virtual time simulation enabled: ${SIMULATION_TIME_STEP_MINUTES}±${SIMULATION_TIME_STEP_VARIANCE} minutes per step"
    ./scripts/run-performance-test.sh publication

    # Step 4: Create Docker network
    print_step "4" "Creating Docker network"
    create_docker_network

    # Step 5: Start metrics service
    print_step "5" "Starting metrics service"
    start_metrics_service

    # Step 6: Start Prometheus
    print_step "6" "Starting Prometheus"
    start_prometheus

    # Step 7: Start Grafana
    print_step "7" "Starting Grafana"
    start_grafana

    # Step 8: Update metrics with real data
    print_step "8" "Updating metrics with test results"
    update_metrics_data

    # Step 9: Configure Grafana
    print_step "9" "Configuring Grafana data source"
    configure_grafana_datasource

    # Step 10: Import dashboard
    print_step "10" "Importing dashboard"
    import_dashboard

    # Step 11: Show access info
    print_step "11" "Displaying access information"
    show_access_info

    print_header "WORKFLOW COMPLETED SUCCESSFULLY"
    echo -e "${GREEN}✅ Complete Docker workflow finished!${NC}"
    echo ""
    echo -e "${CYAN}🎯 Next Steps:${NC}"
    echo -e "1. 🌐 Open Research Dashboard: ${YELLOW}http://localhost:$GRAFANA_PORT/d/autoscaling-research${NC}"
    echo -e "2. 📊 Login with admin/admin credentials (first time only)"
    echo -e "3. 🔍 Explore 24 panels across 6 research sections"
    echo -e "4. 🎛️  Use filters: Select agent (DQN-PPO/HPA) and scenario"
    echo -e "5. 📈 View raw metrics in Prometheus: ${YELLOW}http://localhost:$PROMETHEUS_PORT${NC}"
    echo -e "6. 📚 Check publication data in: ${YELLOW}publication_data/${NC}"
    echo ""
    echo -e "${PURPLE}📖 Documentation:${NC}"
    echo -e "  Dashboard Guide: ${YELLOW}GRAFANA_DASHBOARD_GUIDE.md${NC}"
    echo -e "  PromQL Reference: ${YELLOW}PROMQL_QUICK_REFERENCE.md${NC}"
    echo -e "  Deployment Guide: ${YELLOW}DASHBOARD_DEPLOYMENT_GUIDE.md${NC}"
    echo ""
    echo -e "${GREEN}🔧 Management Commands:${NC}"
    echo -e "  Check status: ${YELLOW}$0 --status${NC}"
    echo -e "  Update metrics: ${YELLOW}$0 --update-metrics${NC}"
    echo -e "  Cleanup: ${YELLOW}$0 --cleanup${NC}"
}

# Help function
show_help() {
    echo "Complete Publication Workflow Script - Docker Version"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help            Show this help message"
    echo "  --cleanup             Stop and remove all containers"
    echo "  --status              Show current monitoring stack status"
    echo "  --update-metrics      Update metrics with latest data"
    echo "  --start-only          Only start containers (skip publication test)"
    echo ""
    echo "Environment Variables:"
    echo "  GRAFANA_PORT         Grafana port (default: 3000)"
    echo "  PROMETHEUS_PORT      Prometheus port (default: 9090)"
    echo "  METRICS_PORT         Metrics service port (default: 8080)"
    echo ""
    echo "Examples:"
    echo "  $0                   Run complete workflow"
    echo "  $0 --status          Check container status"
    echo "  $0 --cleanup         Remove all containers"
    echo "  $0 --start-only      Start monitoring without running tests"
    echo ""
    echo "Docker Containers:"
    echo "  - $GRAFANA_CONTAINER: Grafana dashboard"
    echo "  - $PROMETHEUS_CONTAINER: Prometheus metrics"
    echo "  - $METRICS_CONTAINER: Autoscaling metrics server"
}

# Parse command line arguments
case "${1:-run}" in
    "--help"|"-h")
        show_help
        exit 0
        ;;
    "--cleanup")
        cleanup_all
        exit 0
        ;;
    "--status")
        show_status
        exit 0
        ;;
    "--update-metrics")
        check_docker
        update_metrics_data
        exit 0
        ;;
    "--start-only")
        check_docker
        check_existing_containers
        create_docker_network
        start_metrics_service
        start_prometheus
        start_grafana
        configure_grafana_datasource
        import_dashboard
        show_access_info
        exit 0
        ;;
    "run"|"")
        main
        ;;
    *)
        print_error "Unknown option: $1"
        show_help
        exit 1
        ;;
esac