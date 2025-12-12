#!/bin/bash
# Complete deployment script for MicroK8s Autoscaling Demo

set -e  # Exit on any error

echo "🚀 Starting MicroK8s Autoscaling Demo Deployment..."

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

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    print_error "kubectl is not installed or not in PATH"
    exit 1
fi

# Check if MicroK8s is running
if ! microk8s status --wait-ready; then
    print_error "MicroK8s is not running"
    exit 1
fi

print_status "Deploying nginx configuration..."
kubectl apply -f config/nginx-config.yaml
sleep 2

print_status "Deploying nginx application..."
kubectl apply -f deployments/nginx-deployment.yaml
sleep 5

print_status "Waiting for nginx deployment to be ready..."
kubectl rollout status deployment/nginx-deployment --timeout=300s

print_status "Deploying HPA (Horizontal Pod Autoscaler)..."
kubectl apply -f deployments/nginx-hpa.yaml

print_status "Checking deployment status..."
kubectl get deployment nginx-deployment
kubectl get hpa nginx-hpa
kubectl get svc nginx

# Get service endpoint
print_status "Getting service information..."
CLUSTER_IP=$(kubectl get svc nginx -o jsonpath='{.spec.clusterIP}')
NODE_PORT=$(kubectl get svc nginx -o jsonpath='{.spec.ports[0].nodePort}' 2>/dev/null || echo "ClusterIP")

print_success "✅ Deployment complete!"
echo ""
echo "📊 Service Information:"
echo "   Cluster IP: $CLUSTER_IP"
echo "   Port: 80"
if [ "$NODE_PORT" != "ClusterIP" ]; then
    echo "   NodePort: $NODE_PORT"
fi
echo ""

# Expose service for load testing
print_status "Creating NodePort service for external access..."
kubectl patch svc nginx -p '{"spec":{"type":"NodePort"}}'
sleep 2

NODE_PORT=$(kubectl get svc nginx -o jsonpath='{.spec.ports[0].nodePort}')
NODE_IP=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}')

print_success "🌐 External Access:"
echo "   URL: http://$NODE_IP:$NODE_PORT"
echo ""

print_status "Testing connectivity..."
if curl -s --connect-timeout 5 http://$NODE_IP:$NODE_PORT >/dev/null; then
    print_success "✅ Service is accessible!"
else
    print_warning "⚠️  Service might not be ready yet. Try again in a few moments."
fi

echo ""
print_status "🔧 Useful commands:"
echo "   View pods: kubectl get pods -l app=nginx"
echo "   View HPA: kubectl get hpa nginx-hpa"
echo "   View logs: kubectl logs -l app=nginx -f"
echo "   Port forward: kubectl port-forward svc/nginx 8080:80"
echo ""

print_status "🧪 To run load test:"
echo "   k6 run load-test/load-test-flexible.js -e TARGET_URL=http://$NODE_IP:$NODE_PORT"
echo ""

print_success "🎉 Ready for RL agent testing!"