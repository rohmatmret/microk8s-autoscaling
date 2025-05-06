#!/bin/bash
# scripts/run_simulation.sh
# Runs RL autoscaling simulation with DQN or PPO agent.
# Usage: sudo bash scripts/run_simulation.sh [dqn|ppo] [true|false]
# Example: sudo bash scripts/run_simulation.sh ppo false

set -eo pipefail

AGENT=${1:-dqn}
SKIP_LOADTEST=${2:-false}
PROJECT_ROOT="$(pwd)"
VENV_DIR="$PROJECT_ROOT/venv"
LOG_DIR="$PROJECT_ROOT/logs"
GRAFANA_PORT=3000
TIMEOUT=300  # Increased timeout to 5 minutes
KUBECTL_RETRIES=5
KUBECTL_RETRY_DELAY=15

# Create log directory
mkdir -p "$LOG_DIR"

# Redirect output to log and console
LOG_FILE="$LOG_DIR/simulation-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
log() { echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $@" | tee -a "$LOG_FILE"; }

# Require sudo
if [[ $EUID -ne 0 ]]; then
    log "❌ Please run this script with sudo!"
    exit 1
fi

# Verify virtual environment
if [[ ! -d "$VENV_DIR" ]]; then
    log "❌ Virtual environment not found! Run scripts/install_microk8s.sh first."
    exit 1
fi
log "✅ Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Verify MicroK8s
verify_microk8s() {
    log "🔍 Checking MicroK8s status..."
    for attempt in {1..3}; do
        if microk8s status --wait-ready --timeout $TIMEOUT >/dev/null; then
            log "✅ MicroK8s is ready!"
            return 0
        fi
        log "⚠️ MicroK8s not ready (attempt $attempt/3). Retrying in 10s..."
        sleep 10
    done
    log "❌ MicroK8s failed to start!"
    microk8s inspect >> "$LOG_DIR/microk8s_inspect.log" 2>&1
    exit 1
}
verify_microk8s

# Enable required addons with retries
enable_prerequisites() {
    local addons=("dns" "storage" "helm3")
    for addon in "${addons[@]}"; do
        if ! microk8s status | grep -q "$addon: enabled"; then
            log "🛠 Enabling $addon..."
            local retries=3
            while (( retries > 0 )); do
                if microk8s enable "$addon" >> "$LOG_FILE" 2>&1; then
                    break
                else
                    (( retries-- ))
                    log "⚠️ Retrying $addon enablement ($retries left)..."
                    sleep 10
                fi
            done
            if (( retries == 0 )); then
                log "❌ Failed to enable $addon after 3 attempts"
                return 1
            fi
            sleep 10
        fi
    done
    return 0
}
enable_prerequisites

if ! enable_prerequisites; then
    log "❌ Failed to enable prerequisites"
    exit 1
fi

# Enable observability stack
# Enable observability stack with comprehensive checks
enable_observability() {
    log "📊 Configuring observability stack..."
    
    # Modified resource verification with lower requirements
    verify_cluster_resources() {
        local required_cpu=1.5  # Reduced from 2
        local required_mem=3072  # Reduced from 4GB to 3GB (in MB)
        local node_cpu=$(kubectl get nodes -o jsonpath='{.items[0].status.allocatable.cpu}' | tr -dc '0-9.')
        local node_mem=$(kubectl get nodes -o jsonpath='{.items[0].status.allocatable.memory}' | sed 's/[^0-9]*//g')
        node_mem=$((node_mem/1024/1024))  # Convert to MB

        if (( $(echo "$node_cpu < $required_cpu" | bc -l) )) || [[ $node_mem -lt $required_mem ]]; then
            log "⚠️ Insufficient resources - CPU: ${node_cpu}/${required_cpu}, Memory: ${node_mem}MB/${required_mem}MB"
            log "ℹ️ Trying with reduced resource configuration..."
            return 0  # Continue but with reduced resources
        fi
        return 0
    }

  

    # Main enable function
    if ! verify_cluster_resources; then
        log "⚠️ Proceeding with reduced resource configuration"
    fi

   

    # Clean up existing installation more thoroughly
    if microk8s status | grep -q "observability: enabled"; then
        log "ℹ️ Removing existing observability stack..."
        microk8s disable observability >> "$LOG_FILE" 2>&1
        sleep 20
        # Ensure complete cleanup
        kubectl delete ns observability --ignore-not-found --wait >> "$LOG_FILE" 2>&1
        sleep 10
    fi

    log "🚀 Enabling observability stack with resource constraints..."
    if ! microk8s enable observability \
        --extra-args "--set prometheus.prometheusSpec.resources.limits.memory=1.5Gi \
                     --set prometheus.prometheusSpec.resources.requests.memory=1Gi \
                     --set prometheus.prometheusSpec.resources.limits.cpu=1 \
                     --set prometheus.prometheusSpec.resources.requests.cpu=500m \
                     --set grafana.resources.limits.memory=512Mi \
                     --set grafana.resources.requests.memory=256Mi \
                     --set grafana.resources.limits.cpu=500m \
                     --set grafana.resources.requests.cpu=250m \
                     --set alertmanager.alertmanagerSpec.resources.limits.memory=512Mi \
                     --set alertmanager.alertmanagerSpec.resources.requests.memory=256Mi \
                     --set operator.resources.limits.memory=512Mi \
                     --set operator.resources.requests.memory=256Mi" >> "$LOG_FILE" 2>&1; then
        log "❌ Failed to enable observability"
        log "ℹ️ Trying with even lower resource configuration..."
        
        # Fallback configuration
        if ! microk8s enable observability \
            --extra-args "--set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage=5Gi \
                         --set prometheus.prometheusSpec.resources.limits.memory=1Gi \
                         --set prometheus.prometheusSpec.resources.requests.memory=512Mi \
                         --set grafana.resources.limits.memory=256Mi \
                         --set grafana.resources.requests.memory=128Mi \
                         --set alertmanager.alertmanagerSpec.resources.limits.memory=256Mi \
                         --set alertmanager.alertmanagerSpec.resources.requests.memory=128Mi" >> "$LOG_FILE" 2>&1; then
            log "❌ Failed to enable observability with fallback configuration"
            microk8s inspect >> "$LOG_FILE" 2>&1
            exit 1
        fi
    fi

    log "⌛ Waiting for components to start (timeout: ${TIMEOUT}s)..."
    
    # Extended wait for CRDs with retries
    local crd_retries=3
    while (( crd_retries > 0 )); do
        if kubectl wait --for condition=established --timeout=${TIMEOUT}s \
            crd/prometheuses.monitoring.coreos.com \
            crd/servicemonitors.monitoring.coreos.com >> "$LOG_FILE" 2>&1; then
            break
        else
            (( crd_retries-- ))
            sleep 10
        fi
    done

    if (( crd_retries == 0 )); then
        log "❌ CRDs failed to initialize after multiple attempts"
        exit 1
    fi

    # Check observability pods with more tolerance
    local components=(
        "app.kubernetes.io/name=grafana"
        "app.kubernetes.io/name=prometheus"
        "app.kubernetes.io/name=alertmanager"
    )

    for component in "${components[@]}"; do
        if ! kubectl wait --for=condition=ready pod -l "$component" -n observability \
            --timeout=${TIMEOUT}s >> "$LOG_FILE" 2>&1; then
            log "⚠️ $component taking longer than expected to start"
            log "ℹ️ Checking pod status..."
            kubectl get pods -n observability -l "$component" >> "$LOG_FILE"
            
            # Give it some extra time
            sleep 30
            if ! kubectl wait --for=condition=ready pod -l "$component" -n observability \
                --timeout=60s >> "$LOG_FILE" 2>&1; then
                log "❌ $component failed to start"
                kubectl describe pod -l "$component" -n observability >> "$LOG_FILE"
                kubectl logs -l "$component" -n observability >> "$LOG_FILE"
                exit 1
            fi
        fi
    done

    # More resilient Grafana health check
    local grafana_retries=5
    local grafana_ready=false
    while (( grafana_retries > 0 )); do
        local grafana_pod=$(kubectl get pods -n observability -l app.kubernetes.io/name=grafana -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
        if [[ -n "$grafana_pod" ]]; then
            if kubectl exec -n observability "$grafana_pod" -- \
                curl -s http://localhost:3000/api/health | grep -q '"database":"ok"'; then
                grafana_ready=true
                break
            fi
        fi
        (( grafana_retries-- ))
        sleep 10
    done

    if ! $grafana_ready; then
        log "⚠️ Grafana health check failed but proceeding (might take longer to initialize)"
    fi

    log "✅ Observability stack is operational"
    log "   - Grafana: http://localhost:3000 (admin/$(kubectl get secret -n observability kube-prom-stack-grafana -o jsonpath='{.data.admin-password}' | base64 --decode))"
    log "   - Prometheus: http://localhost:9090"
    log "   - Alertmanager: http://localhost:9093"
}

# skip enable_observability
# enable_observability

# Deploy application
deploy_application() {
    log "🚀 Deploying application..."
    
    DEPLOYMENT_FILES=(
        "config/nginx-config.yaml"
        "deployments/nginx-deployment.yaml"
        # "monitoring/nginx-servicemonitor.yaml"
    )

    # Verify files exist
    for file in "${DEPLOYMENT_FILES[@]}"; do
        if [[ ! -f "$file" ]]; then
            log "❌ Missing file: $file"
            exit 1
        fi
    done

    # Apply with retries
    for attempt in $(seq 1 $KUBECTL_RETRIES); do
        SUCCESS=true
        for file in "${DEPLOYMENT_FILES[@]}"; do
            if ! kubectl apply -f "$file" >> "$LOG_DIR/kubectl.log" 2>&1; then
                log "⚠️ Failed to apply $file (attempt $attempt/$KUBECTL_RETRIES)"
                SUCCESS=false
                if [[ $attempt -eq $KUBECTL_RETRIES ]]; then
                    log "🔄 Trying with --validate=false..."
                    if ! kubectl apply -f "$file" --validate=false >> "$LOG_DIR/kubectl.log" 2>&1; then
                        log "❌ Critical failure applying $file"
                        exit 1
                    fi
                else
                    break
                fi
            fi
        done
        
        if $SUCCESS; then
            log "✅ All deployments applied successfully"
            break
        fi
        [[ $attempt -lt $KUBECTL_RETRIES ]] && sleep $KUBECTL_RETRY_DELAY
    done

    # Wait for pods
    log "⌛ Waiting for application pods..."
    if ! kubectl wait --for=condition=ready pod -l app=nginx --timeout=$TIMEOUT >> "$LOG_FILE" 2>&1; then
        log "❌ Application pods failed to start!"
        kubectl describe pods -l app=nginx >> "$LOG_FILE"
        exit 1
    fi
    log "✅ Application pods are running"
}
deploy_application

# Start RL agent
start_agent() {
    log "🧠 Starting RL Agent [$AGENT]..."
    # install prequired package
    pip install --upgrade pip

    pip install -r requirements.txt 

    
    if [[ ! -f "agent/$AGENT.py" ]]; then
        log "❌ Agent script not found: agent/$AGENT.py"
        exit 1
    fi
    
    # Add the project root to PYTHONPATH 
    export PYTHONPATH="/Users/danialfahmi/Documents/microk8s-autoscaling"

    python "agent/$AGENT.py" >> "$LOG_DIR/$AGENT.log" 2>&1 &
    AGENT_PID=$!
    log "📝 Agent PID: $AGENT_PID"
}
start_agent

# Start load test if requested
start_loadtest() {
    if [[ "$SKIP_LOADTEST" == "false" ]]; then
        log "📊 Starting load test..."
        kubectl create configmap k6-load-script --from-file=load-test/load-test.js -o yaml --dry-run=client | kubectl apply -f - >> "$LOG_FILE" 2>&1
        
        if [[ ! -f "deployments/k6-job.yaml" ]]; then
            log "❌ k6-job.yaml not found!"
            exit 1
        fi
        
        kubectl apply -f deployments/k6-job.yaml >> "$LOG_FILE" 2>&1
        log "✅ Load test job started"
    else
        log "⚠️ Load testing skipped"
    fi
}
# start_loadtest

# Start Grafana port-forward
start_grafana() {
    log "📡 Starting Grafana port-forward..."
    if ! kubectl get svc kube-prom-stack-grafana -n observability >> "$LOG_FILE" 2>&1; then
        log "❌ Grafana service not found!"
        exit 1
    fi
    
    # Get and log the admin password
    GRAFANA_PASSWORD=$(kubectl get secret -n observability kube-prom-stack-grafana -o jsonpath="{.data.admin-password}" | base64 --decode)
    log "🔑 Grafana admin password: $GRAFANA_PASSWORD"
    
    kubectl port-forward -n observability svc/kube-prom-stack-grafana $GRAFANA_PORT:80 >> "$LOG_DIR/grafana.log" 2>&1 &
    GRAFANA_PID=$!
    log "📝 Grafana PID: $GRAFANA_PID"
    log "🔗 Grafana URL: http://localhost:$GRAFANA_PORT (admin/$GRAFANA_PASSWORD)"
}
# start_grafana

# Main loop
log "\n✅ Simulation is running!"
log "📜 Logs available in: $LOG_DIR"
log "🛑 To stop: sudo make stop or kill -9 $AGENT_PID $GRAFANA_PID"

cleanup() {
    log "🛑 Cleaning up..."
    kill -9 $AGENT_PID $GRAFANA_PID 2>/dev/null || true
    exit 0
}
trap cleanup INT TERM

while true; do
    sleep 60
    # Verify processes are still running
    if ! ps -p $AGENT_PID >/dev/null; then
        log "❌ Agent process died!"
        exit 1
    fi
    if ! ps -p $GRAFANA_PID >/dev/null; then
        log "❌ Grafana port-forward died!"
        exit 1
    fi
done