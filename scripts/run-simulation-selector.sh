#!/bin/bash
# Simulation Selector Script
# Choose which autoscaling simulation to run

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║              MicroK8s Autoscaling Simulations               ║"
    echo "║                     Simulation Selector                     ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_option() {
    echo -e "${BLUE}[$1]${NC} $2"
}

print_description() {
    echo -e "    ${YELLOW}→${NC} $1"
}

print_warning() {
    echo -e "${RED}⚠️  WARNING:${NC} $1"
}

print_header

echo ""
echo "Choose your autoscaling simulation:"
echo ""

print_option "1" "Traditional HPA Simulation"
print_description "Uses standard Kubernetes HPA with CPU thresholds"
print_description "Deployment: nginx-deployment (default namespace)"
print_description "Script: ./scripts/run_hpa_simulation.sh"
print_description "Use case: Baseline comparison, traditional autoscaling"

echo ""

print_option "2" "Hybrid DQN-PPO RL Simulation" 
print_description "Advanced RL agent combining DQN + PPO algorithms"
print_description "Deployment: nginx-hybrid (hybrid-sim namespace)"
print_description "Script: ./scripts/run-hybrid-simulation.sh"
print_description "Use case: ML-based autoscaling research"

echo ""

print_option "3" "Individual RL Agent Testing"
print_description "Test single DQN or PPO agents in simulation mode"
print_description "No Kubernetes deployment needed"
print_description "Scripts: python agent/dqn.py --simulate, python agent/ppo.py --simulate"
print_description "Use case: Agent development and testing"

echo ""

print_option "4" "Production Deployment"
print_description "Production-ready deployment with enhanced security"
print_description "Deployment: nginx-production (default namespace)"
print_description "Script: ./scripts/deploy-production.sh"
print_description "Use case: Production environment setup"

echo ""

print_option "5" "KEDA vs HPA Comparison"
print_description "Compare KEDA and HPA scaling behaviors side-by-side"
print_description "Script: ./scripts/keda-comparison-script.sh"
print_description "Use case: Event-driven vs metric-based autoscaling comparison"

echo ""

print_warning "DO NOT run multiple simulations simultaneously - they will conflict!"
print_warning "Always clean up previous simulations before starting new ones."

echo ""
echo -e "${GREEN}What would you like to do?${NC}"
echo ""
print_option "c" "Clean up all deployments first"
print_option "h" "Show help and detailed information"
print_option "q" "Quit"

echo ""
read -p "Enter your choice (1-5, c, h, q): " choice

case $choice in
    1)
        echo -e "${GREEN}Starting Traditional HPA Simulation...${NC}"
        echo ""
        print_warning "This will deploy nginx-deployment in default namespace"
        read -p "Continue? (y/N): " confirm
        if [[ $confirm =~ ^[Yy]$ ]]; then
            ./scripts/run_hpa_simulation.sh
        else
            echo "Cancelled."
        fi
        ;;
    2)
        echo -e "${GREEN}Starting Hybrid DQN-PPO Simulation...${NC}"
        echo ""
        print_warning "This will deploy nginx-hybrid in hybrid-sim namespace"
        print_warning "Make sure you have the hybrid training script ready"
        read -p "Continue? (y/N): " confirm
        if [[ $confirm =~ ^[Yy]$ ]]; then
            ./scripts/run-hybrid-simulation.sh
        else
            echo "Cancelled."
        fi
        ;;
    3)
        echo -e "${GREEN}Individual RL Agent Testing${NC}"
        echo ""
        echo "Choose an agent to test:"
        echo "  [a] DQN Agent (python agent/dqn.py --simulate)"
        echo "  [b] PPO Agent (python agent/ppo.py --simulate)" 
        echo "  [c] Hybrid Agent (python train_hybrid.py)"
        echo ""
        read -p "Enter choice (a/b/c): " agent_choice
        
        case $agent_choice in
            a)
                echo "Starting DQN simulation..."
                python agent/dqn.py --simulate --timesteps 50000 --eval-episodes 50
                ;;
            b)
                echo "Starting PPO simulation..."
                python agent/ppo.py --simulate --timesteps 50000 --eval-episodes 50
                ;;
            c)
                echo "Starting Hybrid simulation..."
                python train_hybrid.py
                ;;
            *)
                echo "Invalid choice."
                ;;
        esac
        ;;
    4)
        echo -e "${GREEN}Starting Production Deployment...${NC}"
        echo ""
        print_warning "This will deploy nginx-production with production settings"
        read -p "Continue? (y/N): " confirm
        if [[ $confirm =~ ^[Yy]$ ]]; then
            ./scripts/deploy-production.sh
        else
            echo "Cancelled."
        fi
        ;;
    5)
        echo -e "${GREEN}Starting KEDA vs HPA Comparison...${NC}"
        echo ""
        read -p "Continue? (y/N): " confirm
        if [[ $confirm =~ ^[Yy]$ ]]; then
            ./scripts/keda-comparison-script.sh
        else
            echo "Cancelled."
        fi
        ;;
    c)
        echo -e "${YELLOW}Cleaning up all deployments...${NC}"
        echo ""
        echo "This will remove:"
        echo "  - All nginx deployments in all namespaces"
        echo "  - All HPA resources"
        echo "  - All ConfigMaps"
        echo "  - hybrid-sim namespace"
        echo ""
        read -p "Are you sure? (y/N): " confirm
        if [[ $confirm =~ ^[Yy]$ ]]; then
            echo "Cleaning up..."
            
            # Default namespace cleanup
            kubectl delete deployment nginx nginx-deployment nginx-production --ignore-not-found=true
            kubectl delete hpa nginx-hpa nginx-production-hpa --ignore-not-found=true
            kubectl delete svc nginx nginx-production --ignore-not-found=true
            kubectl delete configmap nginx-config nginx-production-config --ignore-not-found=true
            
            # Hybrid namespace cleanup  
            kubectl delete namespace hybrid-sim --ignore-not-found=true
            
            # KEDA cleanup
            kubectl delete scaledobject --all --ignore-not-found=true
            
            echo -e "${GREEN}✅ Cleanup completed!${NC}"
        else
            echo "Cleanup cancelled."
        fi
        ;;
    h)
        echo -e "${CYAN}=== Detailed Help ===${NC}"
        echo ""
        echo -e "${GREEN}1. Traditional HPA:${NC}"
        echo "   • Standard Kubernetes autoscaling"
        echo "   • CPU threshold: 20%"
        echo "   • Scale range: 1-5 pods"
        echo "   • Good for: Baseline performance testing"
        echo ""
        echo -e "${GREEN}2. Hybrid DQN-PPO:${NC}"
        echo "   • Machine Learning based autoscaling"
        echo "   • Combines DQN (decision) + PPO (optimization)"
        echo "   • Adaptive learning from cluster behavior"
        echo "   • Good for: Research and advanced scenarios"
        echo ""
        echo -e "${GREEN}3. Individual Agents:${NC}"
        echo "   • Test agents without Kubernetes"
        echo "   • Simulation environment only"
        echo "   • Good for: Development and debugging"
        echo ""
        echo -e "${GREEN}4. Production:${NC}"
        echo "   • Security hardened deployment"
        echo "   • Production-ready configurations"
        echo "   • Good for: Real-world deployment"
        echo ""
        echo -e "${GREEN}5. KEDA Comparison:${NC}"
        echo "   • Compare event-driven vs metric-driven"
        echo "   • Side-by-side analysis"
        echo "   • Good for: Understanding different approaches"
        echo ""
        ;;
    q)
        echo "Goodbye!"
        exit 0
        ;;
    *)
        echo -e "${RED}Invalid choice. Please run the script again.${NC}"
        exit 1
        ;;
esac