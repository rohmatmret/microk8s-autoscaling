#!/bin/bash

# Production Nginx Deployment Script
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check if kubectl is available
    if ! command -v kubectl &> /dev/null; then
        print_error "kubectl is not installed"
        exit 1
    fi
    
    # Check if cluster is accessible
    if ! kubectl cluster-info &> /dev/null; then
        print_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    print_success "Prerequisites check passed"
}

# Deploy production nginx
deploy_production() {
    print_status "Deploying production nginx..."
    
    # Apply configurations in order
    kubectl apply -f config/nginx-production-config.yaml
    kubectl apply -f deployments/nginx-production-deployment.yaml
    kubectl apply -f deployments/nginx-production-hpa.yaml
    kubectl apply -f deployments/nginx-production-ingress.yaml
    
    # Wait for deployment
    print_status "Waiting for deployment to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/nginx-production
    
    print_success "Production nginx deployed successfully"
}

# Deploy monitoring
deploy_monitoring() {
    print_status "Deploying monitoring..."
    
    kubectl apply -f monitoring/nginx-production-servicemonitor.yaml
    kubectl apply -f monitoring/nginx-production-alerts.yaml
    
    print_success "Monitoring deployed successfully"
}

# Verify deployment
verify_deployment() {
    print_status "Verifying deployment..."
    
    # Check pods
    kubectl get pods -l app=nginx-production -o wide
    
    # Check services
    kubectl get svc -l app=nginx-production
    
    # Check HPA
    kubectl get hpa nginx-production-hpa
    
    # Check ingress
    kubectl get ingress nginx-production-ingress
    
    print_success "Deployment verification completed"
}

# Show access information
show_access_info() {
    print_status "Production access information:"
    
    NODE_IP=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}')
    
    echo ""
    echo "�� External Access:"
    echo "  URL: http://$NODE_IP"
    echo "  Host: nginx-production.local (add to /etc/hosts)"
    echo ""
    echo "📊 Monitoring:"
    echo "  Metrics: http://$NODE_IP:9113/metrics"
    echo "  Health: http://$NODE_IP/health"
    echo "  Status: http://$NODE_IP/nginx_status"
    echo ""
    echo "🔧 Management:"
    echo "  kubectl get pods -l app=nginx-production"
    echo "  kubectl get hpa nginx-production-hpa"
    echo "  kubectl logs -f deployment/nginx-production"
}

# Main execution
main() {
    case "${1:-}" in
        "deploy")
            check_prerequisites
            deploy_production
            deploy_monitoring
            verify_deployment
            show_access_info
            ;;
        "verify")
            verify_deployment
            show_access_info
            ;;
        "monitoring")
            deploy_monitoring
            ;;
        *)
            echo "Usage: $0 {deploy|verify|monitoring}"
            echo ""
            echo "Commands:"
            echo "  deploy     - Deploy complete production setup"
            echo "  verify     - Verify deployment status"
            echo "  monitoring - Deploy monitoring only"
            exit 1
            ;;
    esac
}

main "$@" 