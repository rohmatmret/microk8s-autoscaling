#!/bin/bash
# Dedicated script for Hybrid DQN-PPO simulation
# This script creates a separate environment to avoid conflicts with HPA simulation

set -e  # Exit on any error

echo "🤖 Starting Hybrid DQN-PPO Simulation..."

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

# Configuration
PROJECT_ROOT="$(pwd)"
VENV_DIR="$PROJECT_ROOT/venv"
LOG_DIR="$PROJECT_ROOT/logs/hybrid"
NAMESPACE="hybrid-sim"
DEPLOYMENT_NAME="nginx-hybrid"
HPA_NAME="nginx-hybrid-hpa"
SERVICE_NAME="nginx-hybrid"
CONFIG_NAME="nginx-hybrid-config"

# Create log directory
mkdir -p "$LOG_DIR"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
print_status "Checking prerequisites..."

if ! command_exists microk8s; then
    print_error "MicroK8s is not installed. Please install it first."
    exit 1
fi

if ! command_exists kubectl; then
    print_error "kubectl is not installed. Please install it first."
    exit 1
fi

if ! command_exists python3; then
    print_error "Python 3 is not installed. Please install it first."
    exit 1
fi

# Check virtual environment
if [[ ! -d "$VENV_DIR" ]]; then
    print_error "Virtual environment not found! Run: python -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

print_success "Prerequisites check passed!"

# Activate virtual environment
print_status "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Check MicroK8s status
print_status "Checking MicroK8s status..."
if ! microk8s status --wait-ready --timeout 300; then
    print_error "MicroK8s is not ready"
    exit 1
fi

print_success "MicroK8s is ready!"

# Create dedicated namespace
print_status "Creating dedicated namespace: $NAMESPACE..."
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Clean up any existing hybrid resources
print_status "Cleaning up existing hybrid resources..."
kubectl delete deployment $DEPLOYMENT_NAME -n $NAMESPACE --ignore-not-found=true
kubectl delete hpa $HPA_NAME -n $NAMESPACE --ignore-not-found=true  
kubectl delete service $SERVICE_NAME -n $NAMESPACE --ignore-not-found=true
kubectl delete configmap $CONFIG_NAME -n $NAMESPACE --ignore-not-found=true
sleep 5

# Create hybrid-specific configuration
print_status "Creating hybrid-specific nginx configuration..."
cat > "$LOG_DIR/nginx-hybrid-config.yaml" << 'EOF'
apiVersion: v1
kind: ConfigMap
metadata:
  name: nginx-hybrid-config
  namespace: hybrid-sim
data:
  nginx.conf: |
    user  nginx;
    worker_processes  auto;

    error_log  /var/log/nginx/error.log notice;
    pid        /var/run/nginx.pid;

    events {
        worker_connections  1024;
    }

    http {
        include       /etc/nginx/mime.types;
        default_type  application/octet-stream;

        log_format  main  '$remote_addr - $remote_user [$time_local] "$request" '
                          '$status $body_bytes_sent "$http_referer" '
                          '"$http_user_agent" "$http_x_forwarded_for"';

        access_log  /var/log/nginx/access.log  main;

        sendfile        on;
        keepalive_timeout  65;

        server {
            listen       80;
            server_name  localhost;

            location / {
                root   /usr/share/nginx/html;
                index  index.html index.htm;
                # Add some CPU load for testing
                add_header X-Hybrid-Agent "DQN-PPO";
            }

            location /nginx_status {
                stub_status on;
                access_log off;
                allow 127.0.0.1;
                deny all;
            }

            location /health {
                return 200 "Hybrid Agent Ready\n";
                add_header Content-Type text/plain;
            }

            error_page   500 502 503 504  /50x.html;
            location = /50x.html {
                root   /usr/share/nginx/html;
            }
        }
    }
EOF

# Create hybrid-specific deployment
print_status "Creating hybrid-specific deployment..."
cat > "$LOG_DIR/nginx-hybrid-deployment.yaml" << 'EOF'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-hybrid
  namespace: hybrid-sim
  labels:
    app: nginx-hybrid
    simulation: hybrid-dqn-ppo
spec:
  replicas: 1
  selector:
    matchLabels:
      app: nginx-hybrid
  template:
    metadata:
      labels:
        app: nginx-hybrid
        simulation: hybrid-dqn-ppo
    spec:
      containers:
        - name: nginx
          image: nginx:latest
          ports:
            - containerPort: 80
          volumeMounts:
            - name: nginx-config-volume
              mountPath: /etc/nginx/nginx.conf
              subPath: nginx.conf
          resources:
            requests:
              cpu: "50m"      # Lower for testing
              memory: "64Mi"
            limits:
              cpu: "200m"     # Allow spikes
              memory: "128Mi"
        - name: nginx-exporter
          image: nginx/nginx-prometheus-exporter:latest
          args:
            - "-nginx.scrape-uri=http://127.0.0.1/nginx_status"
          ports:
            - containerPort: 9113
          resources:
            requests:
              cpu: "5m"
              memory: "32Mi"
            limits:
              cpu: "50m"
              memory: "64Mi"
      volumes:
        - name: nginx-config-volume
          configMap:
            name: nginx-hybrid-config
---
apiVersion: v1
kind: Service
metadata:
  name: nginx-hybrid
  namespace: hybrid-sim
  labels:
    app: nginx-hybrid
    simulation: hybrid-dqn-ppo
spec:
  selector:
    app: nginx-hybrid
  ports:
    - name: web
      port: 80
      targetPort: 80
    - name: metrics
      port: 9113
      targetPort: 9113
  type: ClusterIP
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: nginx-hybrid-hpa
  namespace: hybrid-sim
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: nginx-hybrid
  minReplicas: 1
  maxReplicas: 8
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 15  # Lower threshold for faster scaling
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 30  # Faster scale down for testing
      policies:
      - type: Pods
        value: 1
        periodSeconds: 10
    scaleUp:
      stabilizationWindowSeconds: 15  # Faster scale up  
      policies:
      - type: Pods
        value: 2
        periodSeconds: 10
EOF

# Apply hybrid configuration
print_status "Applying hybrid configuration..."
kubectl apply -f "$LOG_DIR/nginx-hybrid-config.yaml"
sleep 2

print_status "Applying hybrid deployment..."
kubectl apply -f "$LOG_DIR/nginx-hybrid-deployment.yaml"
sleep 5

# Wait for deployment
print_status "Waiting for hybrid deployment to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/$DEPLOYMENT_NAME -n $NAMESPACE

# Expose service for testing
print_status "Exposing service for load testing..."
kubectl patch svc $SERVICE_NAME -n $NAMESPACE -p '{"spec":{"type":"NodePort"}}'
sleep 2

# Get service information
NODE_IP=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}')
NODE_PORT=$(kubectl get svc $SERVICE_NAME -n $NAMESPACE -o jsonpath='{.spec.ports[0].nodePort}')
CLUSTER_IP=$(kubectl get svc $SERVICE_NAME -n $NAMESPACE -o jsonpath='{.spec.clusterIP}')

print_success "🌐 Hybrid simulation environment ready!"
echo ""
echo "📊 Service Information:"
echo "   Namespace: $NAMESPACE"
echo "   Deployment: $DEPLOYMENT_NAME"
echo "   Service: $SERVICE_NAME"
echo "   Cluster IP: $CLUSTER_IP"
echo "   External URL: http://$NODE_IP:$NODE_PORT"
echo ""

# Test connectivity
print_status "Testing connectivity..."
if curl -s --connect-timeout 5 http://$NODE_IP:$NODE_PORT/health >/dev/null; then
    print_success "✅ Hybrid service is accessible!"
else
    print_warning "⚠️  Service might not be ready yet. Try again in a few moments."
fi

# Start hybrid agent training
print_status "🧠 Starting Hybrid DQN-PPO Agent..."

# Set environment variables for the hybrid agent
export DEPLOYMENT_NAME="nginx-hybrid"
export NAMESPACE="hybrid-sim" 
export SERVICE_NAME="nginx-hybrid"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Check if hybrid training script exists
if [[ -f "$PROJECT_ROOT/train_hybrid.py" ]]; then
    print_status "Starting hybrid training..."
    echo "Training logs will be saved to: $LOG_DIR/hybrid_training.log"
    
    # Run in background and capture PID
    python "$PROJECT_ROOT/train_hybrid.py" > "$LOG_DIR/hybrid_training.log" 2>&1 &
    HYBRID_PID=$!
    
    print_success "🤖 Hybrid agent started (PID: $HYBRID_PID)"
    echo "Monitor training: tail -f $LOG_DIR/hybrid_training.log"
elif [[ -f "$PROJECT_ROOT/agent/hybrid_dqn_ppo.py" ]]; then
    print_status "Starting hybrid agent directly..."
    echo "Training logs will be saved to: $LOG_DIR/hybrid_agent.log"
    
    python "$PROJECT_ROOT/agent/hybrid_dqn_ppo.py" > "$LOG_DIR/hybrid_agent.log" 2>&1 &
    HYBRID_PID=$!
    
    print_success "🤖 Hybrid agent started (PID: $HYBRID_PID)"
    echo "Monitor training: tail -f $LOG_DIR/hybrid_agent.log"
else
    print_error "Hybrid training script not found!"
    print_error "Expected: train_hybrid.py or agent/hybrid_dqn_ppo.py"
    exit 1
fi

echo ""
print_status "🔧 Useful commands for monitoring:"
echo "   View pods: kubectl get pods -n $NAMESPACE -l app=nginx-hybrid"
echo "   View HPA: kubectl get hpa $HPA_NAME -n $NAMESPACE -w"
echo "   View logs: kubectl logs -n $NAMESPACE -l app=nginx-hybrid -f"
echo "   Load test: k6 run load-test/load-test-flexible.js -e TARGET_URL=http://$NODE_IP:$NODE_PORT"
echo ""

print_status "🧪 Load testing examples:"
echo "   # Gradual load increase"
echo "   k6 run load-test/load-test-flexible.js -e TARGET_URL=http://$NODE_IP:$NODE_PORT"
echo ""
echo "   # Quick spike test"  
echo "   for i in {1..100}; do curl -s http://$NODE_IP:$NODE_PORT >/dev/null & done"
echo ""

# Start monitoring in background
print_status "📊 Starting monitoring..."
(
    while true; do
        echo "=== $(date) ===" >> "$LOG_DIR/monitoring.log"
        echo "Pods:" >> "$LOG_DIR/monitoring.log"
        kubectl get pods -n $NAMESPACE -l app=nginx-hybrid -o wide >> "$LOG_DIR/monitoring.log" 2>&1
        echo "" >> "$LOG_DIR/monitoring.log"
        echo "HPA Status:" >> "$LOG_DIR/monitoring.log"
        kubectl get hpa $HPA_NAME -n $NAMESPACE >> "$LOG_DIR/monitoring.log" 2>&1
        echo "" >> "$LOG_DIR/monitoring.log"
        echo "Resource Usage:" >> "$LOG_DIR/monitoring.log"
        kubectl top pods -n $NAMESPACE 2>/dev/null >> "$LOG_DIR/monitoring.log" || echo "Metrics not ready" >> "$LOG_DIR/monitoring.log"
        echo "========================================" >> "$LOG_DIR/monitoring.log"
        sleep 30
    done
) &
MONITORING_PID=$!

# Cleanup function
cleanup() {
    print_status "🛑 Cleaning up hybrid simulation..."
    kill -9 $HYBRID_PID $MONITORING_PID 2>/dev/null || true
    
    print_status "Removing hybrid resources..."
    kubectl delete namespace $NAMESPACE --ignore-not-found=true
    
    print_success "Cleanup completed!"
    exit 0
}
trap cleanup INT TERM

print_success "🎉 Hybrid DQN-PPO simulation is running!"
print_status "📜 Logs available in: $LOG_DIR"
print_status "🛑 To stop: Press Ctrl+C or kill $HYBRID_PID"

echo ""
print_status "Simulation will run until manually stopped..."
print_status "Monitor the hybrid agent training and scaling decisions in real-time!"

# Keep script running
while true; do
    sleep 60
    # Check if processes are still running
    if ! ps -p $HYBRID_PID >/dev/null 2>&1; then
        print_error "❌ Hybrid agent process died!"
        cat "$LOG_DIR/hybrid_training.log" | tail -20
        break
    fi
done