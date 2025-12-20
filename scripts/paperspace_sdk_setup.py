#!/usr/bin/env python3
"""
Paperspace Setup Script
Configures environment and provides setup instructions for Paperspace training
"""

import os
import shutil
from pathlib import Path

# Configuration
PAPERSPACE_API_KEY = "74df63b242e6ecbdedcd83b5c21269"
PROJECT_NAME = "microk8s-autoscaling"

def check_requirements():
    """Check if required files exist"""
    print("🔍 Checking project requirements...")

    required_files = [
        "agent/train_hybrid.py",
        "examples/hybrid_traffic_simulation.py",
    ]

    missing = []
    for file in required_files:
        if not Path(file).exists():
            missing.append(file)

    if missing:
        print(f"❌ Missing files: {', '.join(missing)}")
        return False

    print("✅ All required files present")
    return True

def create_paperspace_requirements():
    """Create requirements file optimized for Paperspace"""
    print("\n📦 Creating Paperspace requirements file...")

    requirements = """# Paperspace Training Requirements
# Core RL libraries
stable-baselines3>=2.0.0
gymnasium>=0.29.0
torch>=2.0.0

# Kubernetes client
kubernetes>=25.0.0

# Monitoring and logging
wandb>=0.15.0
tensorboard>=2.13.0

# Utilities
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
scikit-learn>=1.3.0

# Optuna for hyperparameter tuning
optuna>=3.3.0
"""

    with open("requirements_paperspace.txt", "w") as f:
        f.write(requirements)

    print("✅ Created requirements_paperspace.txt")
    return True

def save_config():
    """Save API key to environment file"""
    env_file = ".env.paperspace"
    print(f"\n💾 Saving configuration to {env_file}...")

    with open(env_file, 'w') as f:
        f.write(f"# Paperspace Configuration\n")
        f.write(f"export PAPERSPACE_API_KEY={PAPERSPACE_API_KEY}\n")
        f.write(f"export WANDB_PROJECT={PROJECT_NAME}\n")
        f.write(f"export WANDB_ENTITY=your-wandb-username\n")

    print("✅ Configuration saved!")
    print(f"   To use on Paperspace: source {env_file}")

def create_setup_script():
    """Create a setup script for Paperspace notebook"""
    print("\n📝 Creating Paperspace setup script...")

    script = """#!/bin/bash
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

# Set up Weights & Biases
echo "📊 Configuring Weights & Biases..."
# Run: wandb login <your-api-key>

# Configure environment
echo "⚙️  Setting environment variables..."
export MOCK_MODE=true
export COMPLEX_TRAFFIC=true
export USE_WANDB=true

echo "✅ Setup complete! Ready to train."
echo ""
echo "To start training, run:"
echo "  python agent/train_hybrid.py --timesteps 100000 --mock --complex-traffic"
"""

    with open("scripts/paperspace_notebook_setup.sh", "w") as f:
        f.write(script)

    os.chmod("scripts/paperspace_notebook_setup.sh", 0o755)
    print("✅ Created scripts/paperspace_notebook_setup.sh")

def print_instructions():
    """Print detailed setup instructions"""
    print("\n" + "=" * 70)
    print("  📚 PAPERSPACE TRAINING INSTRUCTIONS")
    print("=" * 70)
    print("""
IMPORTANT: The 'gradient' package is not the Paperspace CLI.
Modern Paperspace uses a web-based interface. Follow these steps:

STEP 1: Create Paperspace Account & Notebook
────────────────────────────────────────────
1. Go to: https://www.paperspace.com/
2. Sign up or log in with your credentials
3. Navigate to: Gradient → Notebooks
4. Click "Create Notebook"
5. Select Machine Type: "P5000" (16GB GPU, $0.78/hr)
6. Select Runtime: "PyTorch" or "Python 3"
7. Click "Start Notebook"

STEP 2: Upload Your Project
────────────────────────────────────────────
Option A - Git (Recommended):
  1. In notebook terminal: git clone https://github.com/yourusername/microk8s-autoscaling
  2. cd microk8s-autoscaling

Option B - Direct Upload:
  1. Zip your project: tar -czf project.tar.gz .
  2. Upload via Paperspace Files tab
  3. Extract in notebook: tar -xzf project.tar.gz

STEP 3: Setup Environment
────────────────────────────────────────────
In the Paperspace notebook terminal:
  1. bash scripts/paperspace_notebook_setup.sh
  2. wandb login  # Enter your W&B API key from https://wandb.ai/authorize

STEP 4: Start Training
────────────────────────────────────────────
Run one of these commands:

  # Train Hybrid DQN-PPO (100K steps, ~10 hours)
  python agent/train_hybrid.py --timesteps 100000 --mock --complex-traffic

  # Quick test (10K steps, ~1 hour)
  python agent/train_hybrid.py --timesteps 10000 --mock --complex-traffic

  # Train adaptive version
  python agent/train_adaptive_hybrid.py --timesteps 100000 --mock --complex-traffic

STEP 5: Monitor Progress
────────────────────────────────────────────
  • Weights & Biases: https://wandb.ai/
  • Check logs in notebook: tail -f logs/train_hybrid.log
  • TensorBoard: tensorboard --logdir ./logs/

STEP 6: Download Results
────────────────────────────────────────────
Option A - Web Interface:
  1. Navigate to Files tab
  2. Download models/ folder
  3. Download logs/ folder

Option B - Command Line (from local machine):
  # Get notebook SSH details from Paperspace
  scp -r paperspace:~/microk8s-autoscaling/models ./models_paperspace
  scp -r paperspace:~/microk8s-autoscaling/logs ./logs_paperspace

STEP 7: Test Locally
────────────────────────────────────────────
  python examples/hybrid_traffic_simulation.py \\
    --model-path ./models_paperspace/hybrid

""")
    print("=" * 70)

def main():
    print("=" * 70)
    print("  Paperspace Training Setup")
    print("=" * 70)

    # Check requirements
    if not check_requirements():
        print("\n⚠️  Please ensure all required files exist.")
        return

    # Create Paperspace-specific files
    create_paperspace_requirements()
    create_setup_script()
    save_config()

    # Print instructions
    print_instructions()

    print("\n✅ Setup files created! Follow the instructions above.")
    print(f"\n💡 Your API Key: {PAPERSPACE_API_KEY}")
    print("   (Keep this secure - don't commit to git!)\n")

if __name__ == "__main__":
    main()
