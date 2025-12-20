#!/bin/bash
# Paperspace training launcher script
# Usage: ./scripts/paperspace_train.sh [dqn|ppo|hybrid|optimize] [timesteps]

set -e

AGENT_TYPE=${1:-hybrid}
TIMESTEPS=${2:-100000}

echo "========================================"
echo "Paperspace Training Launcher"
echo "Agent: $AGENT_TYPE | Steps: $TIMESTEPS"
echo "========================================"

# Setup environment if not already done
if [ ! -d "models" ]; then
    echo "Running initial setup..."
    bash scripts/paperspace_setup.sh
fi

# Set Weights & Biases to online mode for Paperspace
export WANDB_MODE=online
export WANDB_PROJECT=microk8s-autoscaling-paperspace

# GPU check
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# Start training based on agent type
case $AGENT_TYPE in
    dqn)
        echo "Training DQN Agent..."
        python agent/dqn.py \
            --simulate \
            --timesteps $TIMESTEPS \
            --eval-episodes 50 \
            --save-freq 5000
        ;;

    ppo)
        echo "Training PPO Agent..."
        python agent/ppo.py \
            --simulate \
            --timesteps $TIMESTEPS \
            --eval-episodes 50 \
            --save-freq 5000
        ;;

    hybrid)
        echo "Training Hybrid DQN-PPO Agent..."
        python train_hybrid.py \
            --config config/paperspace_training.yaml \
            --mock \
            --steps $TIMESTEPS \
            --complex-traffic
        ;;

    optimize)
        echo "Running Bayesian Hyperparameter Optimization..."
        python train_hybrid.py \
            --config config/paperspace_training.yaml \
            --mock \
            --optimize \
            --trials 20 \
            --steps $TIMESTEPS
        ;;

    *)
        echo "Unknown agent type: $AGENT_TYPE"
        echo "Usage: $0 [dqn|ppo|hybrid|optimize] [timesteps]"
        exit 1
        ;;
esac

# Training complete notification
echo ""
echo "========================================"
echo "✅ Training Complete!"
echo "========================================"
echo "Models saved in: ./models/$AGENT_TYPE"
echo "Logs available in: ./logs"
echo "WandB dashboard: https://wandb.ai/your-entity/microk8s-autoscaling-paperspace"
echo ""
echo "Next steps:"
echo "1. Download models: gradient notebooks artifacts download --id <notebook-id>"
echo "2. Sync models to local: rsync or git-lfs"
echo "3. Run local evaluation: python examples/hybrid_traffic_simulation.py"
