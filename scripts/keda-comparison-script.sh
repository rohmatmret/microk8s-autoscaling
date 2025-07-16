#!/bin/bash

# KEDA vs HPA Comparison Script
# This script helps you test and compare HPA and KEDA autoscaling
# Uses the same configuration as run_hpa_simulation.sh for consistency

set -e

echo "🚀 Starting KEDA vs HPA Comparison Test"
echo "========================================"

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

# Check if KEDA is installed
check_keda() {
    print_status "Checking KEDA installation..."
    if kubectl get pods -n keda 2>/dev/null | grep -q "keda-operator"; then
        print_success "KEDA is installed and running"
        return 0
    else
        print_error "KEDA is not installed. Please install KEDA first."
        print_status "You can install KEDA using: helm repo add kedacore https://kedacore.github.io/charts && helm install keda kedacore/keda --namespace keda --create-namespace"
        return 1
    fi
}

# Deploy the application (same as HPA simulation)
deploy_app() {
    print_status "Deploying nginx application with same config as HPA simulation..."
    
    # Create namespace if it doesn't exist
    print_status "Creating namespace..."
    kubectl create namespace ingress --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply nginx configuration
    print_status "Applying nginx configuration..."
    kubectl apply -f ../config/nginx-config.yaml
    
    # Apply nginx deployment
    print_status "Applying nginx deployment..."
    kubectl apply -f ../deployments/nginx-deployment.yaml
    
    # Wait for nginx deployment
    wait_for_deployment "nginx"
    
    # Apply ingress
    print_status "Applying ingress configuration..."
    kubectl apply -f ../deployments/ingres.yaml
    
    # Apply ingress controller service
    print_status "Applying ingress controller service..."
    kubectl apply -f ../deployments/nginx-controller-service.yaml
    
    print_success "Application deployed successfully with same config as HPA simulation"
}

# Test HPA
test_hpa() {
    print_status "Testing HPA autoscaling..."
    
    # Clean up any existing KEDA objects first
    print_status "Cleaning up any existing KEDA objects..."
    kubectl delete scaledobject nginx-keda-simple --ignore-not-found=true
    kubectl delete scaledobject nginx-keda-cpu --ignore-not-found=true
    kubectl delete scaledobject nginx-keda-memory --ignore-not-found=true
    kubectl delete scaledobject nginx-keda-advanced --ignore-not-found=true
    sleep 5
    
    # Apply HPA
    kubectl apply -f ../deployments/nginx-hpa.yaml
    
    # Wait for HPA to be ready
    sleep 10
    
    # Get HPA status
    print_status "HPA Status:"
    kubectl get hpa nginx-hpa -o wide
    
    # Monitor HPA for 2 minutes (macOS compatible)
    print_status "Monitoring HPA for 2 minutes..."
    if command -v timeout >/dev/null 2>&1; then
        timeout 120s kubectl get hpa nginx-hpa -w || true
    else
        # macOS fallback - use gtimeout if available, otherwise just watch for 120s
        if command -v gtimeout >/dev/null 2>&1; then
            gtimeout 120s kubectl get hpa nginx-hpa -w || true
        else
            print_warning "timeout command not available, monitoring for 120 seconds manually..."
            kubectl get hpa nginx-hpa -w &
            HPA_PID=$!
            sleep 120
            kill $HPA_PID 2>/dev/null || true
        fi
    fi
    
    print_success "HPA test completed"
}

# Test KEDA
test_keda() {
    print_status "Testing KEDA autoscaling..."
    
    # Clean up HPA first since KEDA and HPA cannot manage the same deployment
    print_status "Removing HPA to allow KEDA to manage the deployment..."
    kubectl delete hpa nginx-hpa --ignore-not-found=true
    sleep 5
    
    # Clean up any existing KEDA objects to avoid conflicts
    print_status "Cleaning up any existing KEDA objects..."
    kubectl delete scaledobject nginx-keda-simple --ignore-not-found=true
    kubectl delete scaledobject nginx-keda-cpu --ignore-not-found=true
    kubectl delete scaledobject nginx-keda-memory --ignore-not-found=true
    kubectl delete scaledobject nginx-keda-advanced --ignore-not-found=true
    sleep 5
    
    # Apply KEDA configurations
    kubectl apply -f ../deployments/nginx-keda-simple.yaml
    
    # Wait for KEDA objects to be ready
    sleep 15
    
    # Get KEDA ScaledObjects status
    print_status "KEDA ScaledObjects Status:"
    kubectl get scaledobject -o wide
    
    # Monitor KEDA for 2 minutes (macOS compatible)
    print_status "Monitoring KEDA for 2 minutes..."
    if command -v timeout >/dev/null 2>&1; then
        timeout 120s kubectl get scaledobject -w || true
    else
        # macOS fallback - use gtimeout if available, otherwise just watch for 120s
        if command -v gtimeout >/dev/null 2>&1; then
            gtimeout 120s kubectl get scaledobject -w || true
        else
            print_warning "timeout command not available, monitoring for 120 seconds manually..."
            kubectl get scaledobject -w &
            KEDA_PID=$!
            sleep 120
            kill $KEDA_PID 2>/dev/null || true
        fi
    fi
    
    print_success "KEDA test completed"
}

# Test both autoscalers sequentially
test_both_autoscalers() {
    print_status "Testing both HPA and KEDA autoscalers sequentially..."
    
    print_status "=== PHASE 1: Testing HPA ==="
    test_hpa
    
    print_status "Waiting 30 seconds before testing KEDA..."
    sleep 30
    
    print_status "=== PHASE 2: Testing KEDA ==="
    test_keda
    
    print_success "Both autoscalers tested successfully"
}

# Generate load for testing (same as HPA simulation)
generate_load() {
    print_status "Generating load for autoscaling test..."
    
    # Get cluster information (same as HPA simulation)
    NODE_IP=$(get_node_ip)
    print_status "Cluster Node IP: $NODE_IP"
    
    # Get ingress external IP/port (same as HPA simulation)
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
    
    print_status "External access URL: $EXTERNAL_URL"
    print_status "Generating load against $EXTERNAL_URL..."
    
    # Generate load using curl in background (same pattern as HPA simulation)
    for i in {1..5}; do
        (
            while true; do
                curl -s "$EXTERNAL_URL" > /dev/null
                sleep 0.1
            done
        ) &
        LOAD_PIDS[$i]=$!
    done
    
    print_status "Load generation started. Press Ctrl+C to stop..."
    print_status "External access URL: $EXTERNAL_URL"
    
    # Wait for user to stop
    trap 'cleanup_load' INT
    wait
}

# Cleanup load generation
cleanup_load() {
    print_status "Stopping load generation..."
    for pid in "${LOAD_PIDS[@]}"; do
        kill $pid 2>/dev/null || true
    done
    print_success "Load generation stopped"
}

# Compare results
compare_results() {
    print_status "Comparing HPA vs KEDA results..."
    
    echo ""
    echo "📊 COMPARISON RESULTS"
    echo "===================="
    
    # Show current status
    print_status "Current deployment status:"
    kubectl get pods -o wide
    kubectl get services
    kubectl get hpa --ignore-not-found=true
    kubectl get scaledobject --ignore-not-found=true
    kubectl get ingress
    
    echo ""
    echo "🔴 HPA Status:"
    if kubectl get hpa nginx-hpa 2>/dev/null; then
        echo "HPA is currently active"
        kubectl get hpa nginx-hpa -o yaml | grep -A 10 "currentMetrics:" || echo "No current metrics available"
    else
        echo "HPA is not currently active"
    fi
    
    echo ""
    echo "🔵 KEDA Status:"
    if kubectl get scaledobject 2>/dev/null | grep -q "nginx-keda"; then
        echo "KEDA is currently active"
        kubectl get scaledobject -o yaml | grep -A 10 "currentMetrics:" || echo "No current metrics available"
    else
        echo "KEDA is not currently active"
    fi
    
    echo ""
    echo "📈 Pod Count Comparison:"
    echo "Total Pods: $(kubectl get pods -l app=nginx --no-headers | wc -l)"
    echo "HPA Target: nginx deployment"
    echo "KEDA Target: nginx deployment"
    
    echo ""
    echo "⚡ Scaling Speed Comparison:"
    echo "HPA typically scales based on CPU/memory metrics"
    echo "KEDA can scale based on multiple triggers including custom metrics"
    echo ""
    echo "⚠️  Note: HPA and KEDA cannot manage the same deployment simultaneously"
    echo "   They were tested sequentially for fair comparison"
    
    # Show resource usage
    echo ""
    echo "💻 Resource Usage:"
    kubectl top pods 2>/dev/null || echo "Metrics server not ready yet"
}

# Cleanup function (same as HPA simulation)
cleanup() {
    print_status "Cleaning up test resources..."
    
    # Stop load generation if running
    cleanup_load
    
    # Delete KEDA objects by name to avoid conflicts
    kubectl delete scaledobject nginx-keda-simple --ignore-not-found=true
    kubectl delete scaledobject nginx-keda-cpu --ignore-not-found=true
    kubectl delete scaledobject nginx-keda-memory --ignore-not-found=true
    kubectl delete scaledobject nginx-keda-advanced --ignore-not-found=true
    
    # Delete HPA
    kubectl delete hpa nginx-hpa --ignore-not-found=true
    
    # Delete application (same as HPA simulation cleanup)
    kubectl delete -f ../deployments/ingres.yaml --ignore-not-found=true
    kubectl delete -f ../deployments/nginx-controller-service.yaml --ignore-not-found=true
    kubectl delete -f ../deployments/nginx-deployment.yaml --ignore-not-found=true
    kubectl delete -f ../config/nginx-config.yaml --ignore-not-found=true
    
    print_success "Cleanup completed"
}

# Show external access information
show_external_access() {
    print_status "Getting external access information..."
    
    NODE_IP=$(get_node_ip)
    INGRESS_IP=$(kubectl get ingress nginx-ingress -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")
    INGRESS_HOST=$(kubectl get ingress nginx-ingress -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || echo "")
    
    if [ -n "$INGRESS_IP" ]; then
        EXTERNAL_URL="http://$INGRESS_IP"
    elif [ -n "$INGRESS_HOST" ]; then
        EXTERNAL_URL="http://$INGRESS_HOST"
    else
        NODE_PORT=$(get_service_port "nginx-ingress-controller" "ingress")
        if [ -n "$NODE_PORT" ]; then
            EXTERNAL_URL="http://$NODE_IP:$NODE_PORT"
        else
            EXTERNAL_URL="http://$NODE_IP"
        fi
    fi
    
    print_success "External access URL: $EXTERNAL_URL"
    
    echo ""
    print_status "To test autoscaling from outside the cluster:"
    echo "1. Use the external URL: $EXTERNAL_URL"
    echo "2. Generate load using tools like:"
    echo "   - curl in a loop: for i in {1..1000}; do curl $EXTERNAL_URL; done"
    echo "   - Apache Bench: ab -n 10000 -c 100 $EXTERNAL_URL"
    echo "   - wrk: wrk -t12 -c400 -d30s $EXTERNAL_URL"
    echo ""
    echo "3. Monitor scaling in real-time:"
    echo "   kubectl get hpa -w"
    echo "   kubectl get scaledobject -w"
    echo "   kubectl top pods"
}

# Main execution
main() {
    case "${1:-}" in
        "deploy")
            check_keda && deploy_app
            ;;
        "test-hpa")
            test_hpa
            ;;
        "test-keda")
            test_keda
            ;;
        "test-both")
            test_both_autoscalers
            ;;
        "load")
            generate_load
            ;;
        "compare")
            compare_results
            ;;
        "cleanup")
            cleanup
            ;;
        "external-access")
            show_external_access
            ;;
        "full-test")
            check_keda
            deploy_app
            test_both_autoscalers
            show_external_access
            print_status "Ready for external load testing. Press Enter when done..."
            read -p ""
            compare_results
            ;;
        *)
            echo "Usage: $0 {deploy|test-hpa|test-keda|test-both|load|compare|cleanup|external-access|full-test}"
            echo ""
            echo "Commands:"
            echo "  deploy         - Deploy the nginx application (same as HPA simulation)"
            echo "  test-hpa       - Test HPA autoscaling (removes KEDA first)"
            echo "  test-keda      - Test KEDA autoscaling (removes HPA first)"
            echo "  test-both      - Test both autoscalers sequentially"
            echo "  load           - Generate load for testing"
            echo "  compare        - Compare HPA vs KEDA results"
            echo "  cleanup        - Clean up all test resources"
            echo "  external-access - Show external access information"
            echo "  full-test      - Run complete comparison test"
            echo ""
            echo "Note: HPA and KEDA cannot manage the same deployment simultaneously."
            echo "      They are tested sequentially for fair comparison."
            exit 1
            ;;
    esac
}

# Run main function
main "$@" 