#!/bin/bash
# scripts/training.sh
# Runs RL autoscaling training with DQN or PPO agent.
# Usage: bash scripts/training.sh [dqn|ppo] [simulate|real] [timesteps] [eval_episodes]
# Example: bash scripts/training.sh dqn simulate 100000 100

set -eo pipefail

# Default parameters
AGENT=${1:-dqn}
ENV_MODE=${2:-simulate}
TIMESTEPS=${3:-100000}
EVAL_EPISODES=${4:-100}
PROJECT_ROOT="$(pwd)"
VENV_DIR="$PROJECT_ROOT/venv"
LOG_DIR="$PROJECT_ROOT/logs"

# Validate inputs
if [[ "$AGENT" != "dqn" && "$AGENT" != "ppo" ]]; then
    echo "❌ Invalid agent: $AGENT. Must be 'dqn' or 'ppo'."
    exit 1
fi
if [[ "$ENV_MODE" != "simulate" && "$ENV_MODE" != "real" ]]; then
    # echo "❌ Invalid environment mode: $ENV_MODE. Must be 'simulate' or 'real'."
    ENV_MODE="simulate"
    # exit 1
fi
if ! [[ "$TIMESTEPS" =~ ^[0-9]+$ ]] || [ "$TIMESTEPS" -le 0 ]; then
    echo "❌ Invalid timesteps: $TIMESTEPS. Must be a positive integer."
    exit 1
fi
if ! [[ "$EVAL_EPISODES" =~ ^[0-9]+$ ]] || [ "$EVAL_EPISODES" -le 0 ]; then
    echo "❌ Invalid eval_episodes: $EVAL_EPISODES. Must be a positive integer."
    exit 1
fi

# Create log directory
mkdir -p "$LOG_DIR"

# Redirect output to log and console
LOG_FILE="$LOG_DIR/training-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
log() { echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $@" | tee -a "$LOG_FILE"; }

# Check for virtual environment
if [ ! -d "$VENV_DIR" ]; then
    log "❌ Virtual environment not found at $VENV_DIR. Run 'make setup' first."
    exit 1
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"
log "✅ Virtual environment activated: $VENV_DIR"

# Check if agent script exists
if [[ ! -f "agent/$AGENT.py" ]]; then
    log "❌ Agent script not found: agent/$AGENT.py"
    exit 1
fi

# Set PYTHONPATH dynamically
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
log "📌 PYTHONPATH set to: $PYTHONPATH"

# Start agent training
log "🧠 Starting training with RL agent [$AGENT] in [$ENV_MODE] mode..."
if [[ "$ENV_MODE" == "simulate" ]]; then
    python "agent/$AGENT.py" --simulate --timesteps "$TIMESTEPS" --eval-episodes "$EVAL_EPISODES"
else
    python "agent/$AGENT.py" --timesteps "$TIMESTEPS" --eval-episodes "$EVAL_EPISODES"
fi

# Check if training was successful
if [ $? -eq 0 ]; then
    log "✅ Training completed successfully! Logs in $LOG_FILE"
else
    log "❌ Training failed. Check logs in $LOG_FILE"
    exit 1
fi

# Deactivate virtual environment
deactivate
log "✅ Virtual environment deactivated"