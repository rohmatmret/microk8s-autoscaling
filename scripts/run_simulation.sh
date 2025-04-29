#!/bin/bash
# scripts/run_simulation.sh
# Runs RL autoscaling simulation with DQN or PPO agent.
# Usage: sudo bash scripts/run_simulation.sh [dqn|ppo] [true|false]
# Example: sudo bash scripts/run_simulation.sh ppo false

set -e  # Exit on error

AGENT=${1:-dqn}
SKIP_LOADTEST=${2:-false}
PROJECT_ROOT="$(pwd)"
VENV_DIR="$PROJECT_ROOT/venv"
LOG_DIR="$PROJECT_ROOT/logs"
GRAFANA_PORT=3000
TIMEOUT=120  # Increased timeout for pod readiness
KUBECTL_RETRIES=3
KUBECTL_RETRY_DELAY=10

# Create log directory
mkdir -p "$LOG_DIR"

# Redirect output to log and console
LOG_FILE="$LOG_DIR/simulation.log"
exec 3>&1 1>>"$LOG_FILE" 2>&1
log() { echo "$@" >&3; echo "$@" >> "$LOG_FILE"; }

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
log "🔍 Checking MicroK8s status..."
for attempt in {1..3}; do
    if microk8s status --wait-ready --timeout 300 >/dev/null; then
        log "✅ MicroK8s is ready!"
        break
    fi
    log "⚠️ MicroK8s not ready (attempt $attempt/3). Retrying in 10s..."
    sleep 10
    if [[ $attempt -eq 3 ]]; then
        log "❌ MicroK8s failed to start! Check logs in $LOG_FILE"
        microk8s inspect >> "$LOG_DIR/microk8s_inspect.log" 2>&1
        exit 1
    fi
done

# Apply deployment with retries
log "🚀 Deploying to MicroK8s..."
for attempt in $(seq 1 $KUBECTL_RETRIES); do
    if kubectl apply -f deployments/nginx-deployment.yaml >> "$LOG_DIR/kubectl.log" 2>&1; then
        log "✅ Deployment applied!"
        break
    fi
    log "⚠️ Deployment failed (attempt $attempt/$KUBECTL_RETRIES). Retrying in $KUBECTL_RETRY_DELAY seconds..."
    sleep $KUBECTL_RETRY_DELAY
    if [[ $attempt -eq $KUBECTL_RETRIES ]]; then
        log "❌ Deployment failed after $KUBECTL_RETRIES attempts! Trying with --validate=false..."

          if !kubectl apply -f config/nginx-config.yaml --validate=false >> "$LOG_DIR/kubectl.log" 2>&1; then
            log "❌ Deployment ConfigMap still failed! Check $LOG_DIR/kubectl.log and $LOG_DIR/microk8s_inspect.log"
            microk8s inspect >> "$LOG_DIR/microk8s_inspect.log" 2>&1
            exit 1 
        fi
        log "✅ Succes Deployment ConfigMap"
        
        if ! kubectl apply -f deployments/nginx-deployment.yaml --validate=false >> "$LOG_DIR/kubectl.log" 2>&1; then
            log "❌ Deployment still failed! Check $LOG_DIR/kubectl.log and $LOG_DIR/microk8s_inspect.log"
            microk8s inspect >> "$LOG_DIR/microk8s_inspect.log" 2>&1
            exit 1
        fi

        log "✅ Deployment applied with --validate=false"

      

        if !kubectl apply -f monitoring/nginx-servicemonitor.yaml --validate=false >> "$LOG_DIR/kubectl.log" 2>&1; then
            log "❌ Deployment ServiceMonitor still failed! Check $LOG_DIR/kubectl.log and $LOG_DIR/microk8s_inspect.log"
            microk8s inspect >> "$LOG_DIR/microk8s_inspect.log" 2>&1
            exit 1
        fi
        log "✅ Succes Deployment ServiceMonitor"

    fi
done

# Wait for pods to be Running
log "⌛ Waiting for pods to become Running..."
start_time=$(date +%s)
while [[ $(( $(date +%s) - start_time )) -lt $TIMEOUT ]]; do
    pods=$(kubectl get pods -n default -o jsonpath='{range .items[*]}{.status.phase}{"\n"}{end}' 2>/dev/null)
    if echo "$pods" | grep -v -E "Pending|ContainerCreating|Error|CrashLoopBackOff" | grep -q "Running"; then
        log "✅ Pods are running!"
        break
    fi
    sleep 2
done
if [[ $(( $(date +%s) - start_time )) -ge $TIMEOUT ]]; then
    log "❌ Timeout waiting for pods to be Running!"
    kubectl get pods -n default >> "$LOG_DIR/kubectl.log" 2>&1
    inspect >> "$LOG_DIR/microk8s_inspect.log" 2>&1
    exit 1
fi

# Start RL agent
log "🧠 Launching Reinforcement Learning Agent [$AGENT]..."
if [[ "$AGENT" != "dqn" && "$AGENT" != "ppo" ]]; then
    log "❌ Invalid agent: $AGENT. Use 'dqn' or 'ppo'."
    exit 1
fi
if [[ ! -f "agent/$AGENT.py" ]]; then
    log "❌ Agent script agent/$AGENT.py not found!"
    exit 1
fi
python agent/"$AGENT".py >> "$LOG_DIR/$AGENT.log" 2>&1 &
AGENT_PID=$!
log "📝 Agent PID: $AGENT_PID"


# Start port forwarding for Grafana
# log "📡 Starting Grafana port-forward..."
# if !kubectl get svc -n monitoring grafana >/dev/null 2>&1; then
#     log "❌ Grafana service not found in monitoring namespace!"
#     exit 1
# fi

# Optionally start load test
if [[ "$SKIP_LOADTEST" == "false" ]]; then
    log "📊 Running load test with k6..."
    
    # Create/update ConfigMap with k6 script
    kubectl create configmap k6-load-script --from-file=load-test/load-test.js

    log "Apply Job k6"
    # Apply k6 Job YAML (make sure deployments/k6-job.yaml exists)
    if [[ ! -f "deployments/k6-job.yaml" ]]; then
        log "❌ k6-job.yaml not found in deployments/"
        exit 1
    fi

    kubectl apply -f deployments/k6-job.yaml >> "$LOG_DIR/kubectl.log" 2>&1
    log "✅ k6 load test job applied!"

    # k6 run load-test/loadtest.js >> "$LOG_DIR/k6.log" 2>&1 &
    # K6_PID=$!
    # log "📝 k6 PID: $K6_PID"
else
    log "⚠️ Load testing skipped."
fi

# Check if port is in use
# if lsof -i :"$GRAFANA_PORT" >/dev/null; then
#     log "❌ Port $GRAFANA_PORT is already in use! Choose another port or close the process."
#     exit 1
# fi

kubectl port-forward -n observability svc/grafana "$GRAFANA_PORT:$GRAFANA_PORT" >> "$LOG_DIR/grafana.log" 2>&1 &
GRAFANA_PID=$!
log "📝 Grafana PID: $GRAFANA_PID"

log "\n✅ Simulation is running!"
log "🔗 Access Grafana at: http://localhost:$GRAFANA_PORT"
log "📜 Logs available in: $LOG_DIR"
log "🛑 To stop, run: sudo make stop or kill $AGENT_PID ${K6_PID:-} $GRAFANA_PID 2>/dev/null"

# Trap Ctrl+C to clean up
trap 'log "🛑 Stopping simulation..."; kill $AGENT_PID ${K6_PID:-} $GRAFANA_PID 2>/dev/null; exit 0' INT

# Keep script running to maintain background processes
wait