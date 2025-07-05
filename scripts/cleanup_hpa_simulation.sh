#!/bin/bash

# Cleanup Script for HPA Simulation
# This script removes all resources created during the HPA simulation

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

print_status "Starting cleanup of HPA simulation resources..."

# Stop any running monitoring processes
print_status "Stopping monitoring processes..."
pkill -f "kubectl get hpa -w" 2>/dev/null || true
pkill -f "kubectl top pods" 2>/dev/null || true

# Delete resources in reverse order of creation
print_status "Deleting ConfigMaps..."
kubectl delete configmap nginx-config --ignore-not-found=true

print_status "Deleting HPA..."
kubectl delete -f deployments/nginx-hpa.yaml --ignore-not-found=true

print_status "Deleting ingress..."
kubectl delete -f deployments/ingres.yaml --ignore-not-found=true

print_status "Deleting ingress controller service..."
kubectl delete -f deployments/nginx-controller-service.yaml --ignore-not-found=true

print_status "Deleting nginx deployment and service..."
kubectl delete deployment nginx --ignore-not-found=true
kubectl delete service nginx --ignore-not-found=true

print_status "Deleting ingress namespace..."
kubectl delete namespace ingress --ignore-not-found=true

# Wait for resources to be fully deleted
print_status "Waiting for resources to be fully deleted..."
sleep 10

# Verify cleanup
print_status "Verifying cleanup..."
echo "Remaining pods:"
kubectl get pods --all-namespaces

echo "Remaining services:"
kubectl get services --all-namespaces

echo "Remaining HPA:"
kubectl get hpa --all-namespaces

echo "Remaining ConfigMaps:"
kubectl get configmaps --all-namespaces

print_success "Cleanup completed successfully!"

print_status "To completely reset MicroK8s, you can run:"
echo "  microk8s reset"
echo "  microk8s start" 