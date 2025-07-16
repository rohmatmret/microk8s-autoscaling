#!/bin/bash

# HPA Simulation Script for MicroK8s
# This script sets up and runs a complete HPA simulation with nginx and load testing

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

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to wait for deployment to be ready
wait_for_deployment() {
    local deployment_name=$1
    local namespace=${2:-default}
    print_status "Waiting for deployment $deployment_name to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/$deployment_name -n $namespace
    print_success "Deployment $deployment_name is ready!"
}

# Function to get node IP
get_node_ip() {
    kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}'
}

# Function to get service port
get_service_port() {
    local service_name=$1
    local namespace=${2:-default}
    kubectl get service $service_name -n $namespace -o jsonpath='{.spec.ports[0].nodePort}'
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

print_success "Prerequisites check passed!"

# Start MicroK8s if not running
print_status "Checking MicroK8s status..."
if microk8s status | grep -q "microk8s is running"; then
    print_status "MicroK8s is already running."
else
    print_status "Starting MicroK8s..."
    microk8s start
    print_success "MicroK8s started successfully!"
fi

# Show MicroK8s status
print_status "MicroK8s status:"
microk8s status

# Enable required addons
print_status "Enabling required MicroK8s addons..."

# Enable Ingress
print_status "Enabling Ingress..."
microk8s enable ingress
print_success "Ingress enabled."

# Enable MetalLB to provide a public IP address for the Ingress controller.
# The best solution isn't one or the other; you actually need both to work together.
# They solve two different problems:
#
# MetalLB's Job (The Street Address): MetalLB gives your cluster a public IP address.
# It gives the Ingress controller a reachable, external entry point.
#
# Ingress's Job (The Receptionist): The Ingress controller and your Ingress rule
# act like a receptionist. It sits at that public IP address and directs incoming
# traffic to the correct internal service (nginx) based on the rules you define.
#
# Think of it like an apartment building:
# - MetalLB provides the building's public street address so people can find it.
# - The Ingress controller is the front desk that takes your request and directs you
#   to the correct apartment (nginx service).
print_status "Enabling MetalLB..."
# Determine a suitable IP range for MetalLB from the node's IP.
NODE_IP_FOR_METALLB=$(get_node_ip)
if [ -z "$NODE_IP_FOR_METALLB" ]; then
    print_warning "Could not determine node IP for MetalLB configuration. You may need to configure MetalLB manually."
    # Fallback to a default range, which may or may not work.
    METALLB_RANGE="192.168.0.150-192.168.0.200"
else
    # This creates a range like 192.168.1.240-192.168.1.250 from an IP like 192.168.1.100
    IP_PREFIX=$(echo "$NODE_IP_FOR_METALLB" | cut -d. -f1-3)
    METALLB_RANGE="$IP_PREFIX.240-$IP_PREFIX.250"
fi
print_status "Attempting to configure MetalLB with IP range: $METALLB_RANGE"
microk8s enable metallb:"$METALLB_RANGE"
print_success "MetalLB enabled."

# microk8s enable metrics-server
# microk8s enable monitoring

print_success "Addons enabled successfully!"

# Wait for addons to be ready
print_status "Waiting for addons to be ready..."
sleep 30

# # Configure kubectl to use MicroK8s
# print_status "Configuring kubectl for MicroK8s..."
# microk8s kubectl config view --raw > ~/.kube/config
# chmod 600 ~/.kube/config

# Verify cluster is ready
print_status "Verifying cluster status..."
kubectl cluster-info
kubectl get nodes

# Create namespace if it doesn't exist
print_status "Creating namespace..."
kubectl create namespace ingress --dry-run=client -o yaml | kubectl apply -f -

# Apply nginx configuration
print_status "Applying nginx configuration..."
kubectl apply -f config/nginx-config.yaml

# Apply nginx deployment
print_status "Applying nginx deployment..."
kubectl apply -f deployments/nginx-deployment.yaml

# Wait for nginx deployment
wait_for_deployment "nginx"

# Apply HPA
print_status "Applying Horizontal Pod Autoscaler..."
kubectl apply -f deployments/nginx-hpa.yaml

# Apply ingress
print_status "Applying ingress configuration..."
kubectl apply -f deployments/ingres.yaml

# Apply ingress controller service
print_status "Applying ingress controller service..."
kubectl apply -f deployments/nginx-controller-service.yaml

# Note: K6 load testing removed - testing will be done externally
print_status "K6 load testing removed - use external tools for load testing"

# Get cluster information
NODE_IP=$(get_node_ip)
print_status "Cluster Node IP: $NODE_IP"

# Wait for ingress to be ready
print_status "Waiting for ingress to be ready..."
sleep 10

# Get ingress external IP/port
print_status "Getting ingress external access information..."
INGRESS_IP=$(kubectl get ingress nginx-ingress -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")
INGRESS_HOST=$(kubectl get ingress nginx-ingress -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || echo "")

if [ -n "$INGRESS_IP" ]; then
    EXTERNAL_URL="http://$INGRESS_IP"
elif [ -n "$INGRESS_HOST" ]; then
    EXTERNAL_URL="http://$INGRESS_HOST"
else
    # Fallback to node IP and port
    NODE_PORT=$(get_service_port "nginx-ingress-controller" "ingress")
    if [ -n "$NODE_PORT" ]; then
        EXTERNAL_URL="http://$NODE_IP:$NODE_PORT"
    else
        EXTERNAL_URL="http://$NODE_IP"
    fi
fi

print_success "External access URL: $EXTERNAL_URL"

# Show current status
print_status "Current deployment status:"
kubectl get pods -o wide
kubectl get services
kubectl get hpa
kubectl get ingress

kubectl get svc -n ingress


# Start monitoring in background
print_status "Starting monitoring..."
(
    while true; do
        echo "=== $(date) ==="
        echo "Pods:"
        kubectl get pods -o wide
        echo ""
        echo "HPA Status:"
        kubectl get hpa
        echo ""
        echo "Service Status:"
        kubectl get services
        echo ""
        echo "Resource Usage:"
        kubectl top pods 2>/dev/null || echo "Metrics server not ready yet"
        echo ""
        echo "========================================"
        sleep 30
    done
) > logs/hpa_monitoring.log 2>&1 &
MONITORING_PID=$!

# Wait a bit for everything to stabilize
print_status "Waiting for system to stabilize..."
sleep 30

# Show current status and provide external testing instructions
print_status "Deployment is ready for external testing!"
print_status "Current deployment status:"
kubectl get pods -o wide
kubectl get hpa
kubectl get services
kubectl get ingress

print_success "External access URL: $EXTERNAL_URL"

echo ""
print_status "To test HPA scaling from outside the cluster:"
echo "1. Use the external URL: $EXTERNAL_URL"
echo "2. Generate load using tools like:"
echo "   - k6 (recommended): ./scripts/external_load_test.sh $EXTERNAL_URL 300 50"
echo "   - curl in a loop: for i in {1..1000}; do curl $EXTERNAL_URL; done"
echo "   - Apache Bench: ab -n 10000 -c 100 $EXTERNAL_URL"
echo "   - wrk: wrk -t12 -c400 -d30s $EXTERNAL_URL"
echo ""
echo "3. Monitor scaling in real-time:"
echo "   kubectl get hpa -w"
echo "   kubectl top pods"
echo ""
echo "4. Press Ctrl+C to stop monitoring and continue..."

# Wait for user to test externally
read -p "Press Enter when you want to stop monitoring and show final results..."

# Stop monitoring
kill $MONITORING_PID 2>/dev/null || true

# Show final results
print_status "Final deployment status:"
kubectl get pods -o wide
kubectl get hpa
kubectl top pods

print_success "HPA simulation completed successfully!"
print_status "Monitoring logs saved to: logs/hpa_monitoring.log"
print_status "External access URL: $EXTERNAL_URL"

# Show useful commands
echo ""
print_status "Useful commands for further investigation:"
echo "  kubectl get hpa -w                    # Watch HPA in real-time"
echo "  kubectl top pods                      # View resource usage"
echo "  kubectl logs -f deployment/nginx      # View nginx logs"
echo "  kubectl describe hpa nginx-hpa        # Detailed HPA information"
echo "  kubectl get events --sort-by=.metadata.creationTimestamp  # View events"
echo ""
print_status "To clean up the deployment, run:"
echo "  kubectl delete -f deployments/nginx-hpa.yaml"
echo "  kubectl delete -f deployments/ingres.yaml"
echo "  kubectl delete -f deployments/nginx-controller-service.yaml"
echo "  kubectl delete deployment nginx"
echo "  kubectl delete service nginx"
echo "  kubectl delete configmap nginx-config"
echo "  kubectl delete configmap k6-load-script"
echo "  kubectl delete job k6-loadtest" 