#!/bin/bash
# Paperspace Notebook Setup Script
# Run this once when you first create your notebook

set -e

echo "🚀 Setting up Paperspace environment for MicroK8s RL Autoscaling..."

# Update system
echo "📦 Installing system dependencies..."
apt-get update && apt-get install -y git

# Install Python dependencies
echo "🐍 Installing Python packages..."
pip install --upgrade pip
pip install -r requirements_paperspace.txt

# Set up Weights & Biases (OFFLINE MODE)
echo "📊 Configuring Weights & Biases (Offline Mode)..."
echo "   WandB will save data locally without requiring login"
echo "   You can sync data later with: wandb sync ./wandb"

# Configure environment
echo "⚙️  Setting environment variables..."
source .env.paperspace
export MOCK_MODE=true
export COMPLEX_TRAFFIC=true
export USE_WANDB=true

echo "✅ Setup complete! Ready to train."
echo ""
echo "To start training, run:"
echo "  python agent/train_hybrid.py --timesteps 100000 --mock --complex-traffic"
