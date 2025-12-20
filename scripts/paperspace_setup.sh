#!/bin/bash
# Paperspace environment setup script

set -e

echo "========================================"
echo "Paperspace MicroK8s RL Training Setup"
echo "========================================"

# Update system packages
echo "[1/6] Updating system packages..."
apt-get update -qq
apt-get install -y git vim htop tmux

# Install Python dependencies
echo "[2/6] Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Verify GPU availability
echo "[3/6] Verifying GPU..."
python -c "import torch; print(f'PyTorch GPU Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}')"

# Setup Weights & Biases
echo "[4/6] Configuring Weights & Biases..."
if [ -n "$WANDB_API_KEY" ]; then
    wandb login $WANDB_API_KEY
    echo "✅ WandB configured"
else
    echo "⚠️  WANDB_API_KEY not set. Run: wandb login"
fi

# Create necessary directories
echo "[5/6] Creating directories..."
mkdir -p models/dqn models/ppo models/hybrid
mkdir -p logs metrics test_results

# Verify environment
echo "[6/6] Environment verification..."
python -c "import stable_baselines3; import gymnasium; import wandb; print('✅ All dependencies installed')"

echo ""
echo "✅ Setup complete! Ready for training."
echo "========================================"
