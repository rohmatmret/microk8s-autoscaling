#!/bin/bash
# scripts/install_microk8s.sh
# Installs MicroK8s, Python, k6, and dependencies for RL autoscaling simulation.
# Usage: sudo bash scripts/install_microk8s.sh

set -e  # Exit on error

# Ensure Bash version >= 4
if [[ ${BASH_VERSINFO[0]} -lt 4 ]]; then
    echo "❌ This script requires Bash 4 or higher. Current version: $BASH_VERSION"
    exit 1
fi

LOG_DIR="$(pwd)/logs"
LOG_FILE="$LOG_DIR/install.log"
mkdir -p "$LOG_DIR"

# Redirect output to log and console
exec 3>&1 1>>"$LOG_FILE" 2>&1
log() { echo "$@" >&3; echo "$@" >> "$LOG_FILE"; }

# Require sudo
if [[ $EUID -ne 0 ]]; then
    log "❌ Please run this script with sudo!"
    exit 1
fi

# Detect OS
OS=$(uname -s | tr '[:upper:]' '[:lower:]' || echo "unknown")
log "🔍 Detected OS: $OS"
case "$OS" in
    linux*)  PKG_MANAGER="snap" ;;
    darwin*) PKG_MANAGER="brew" ;;
    *)
        log "❌ Unsupported OS: $OS"
        exit 1
        ;;
esac
log "🔍 Package manager: $PKG_MANAGER"

# Install package manager dependencies
if [[ "$PKG_MANAGER" == "brew" ]]; then
    if ! command -v brew >/dev/null; then
        log "📌 Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
fi

# Install MicroK8s
if ! command -v microk8s >/dev/null; then
    log "📌 Installing MicroK8s..."
    if [[ "$PKG_MANAGER" == "snap" ]]; then
        snap install microk8s --classic || {
            log "❌ Failed to install MicroK8s via snap!"
            exit 1
        }
    elif [[ "$PKG_MANAGER" == "brew" ]]; then
        brew install ubuntu/microk8s/microk8s || {
            log "❌ Failed to install MicroK8s via brew!"
            exit 1
        }
        microk8s install --cpu 4 --mem 8 --disk 30 || {
            log "❌ Failed to initialize MicroK8s!"
            exit 1
        }
    fi
fi

# Wait for MicroK8s to be ready with retries
log "🔍 Waiting for MicroK8s to be ready..."
for attempt in {1..3}; do
    if microk8s status --wait-ready --timeout 300 >/dev/null; then
        log "✅ MicroK8s status ready!"
        break
    fi
    log "⚠️ MicroK8s not ready (attempt $attempt/3). Retrying in 10s..."
    sleep 10
    if [[ $attempt -eq 3 ]]; then
        log "❌ MicroK8s failed to start! Inspecting..."
        microk8s inspect >> "$LOG_DIR/microk8s_inspect.log" 2>&1
        exit 1
    fi
done

# Verify MicroK8s API server
log "🔍 Verifying MicroK8s API server..."
if ! microk8s kubectl get nodes >/dev/null 2>&1; then
    log "⚠️ API server not responding. Restarting MicroK8s..."
    microk8s stop
    microk8s start
    sleep 10
    if ! microk8s status --wait-ready --timeout 300 >/dev/null; then
        log "❌ MicroK8s API server failed! Check $LOG_DIR/microk8s_inspect.log"
        microk8s inspect >> "$LOG_DIR/microk8s_inspect.log" 2>&1
        exit 1
    fi
fi

# Enable MicroK8s add-ons
log "🔧 Enabling MicroK8s add-ons..."
for addon in dns storage ingress prometheus grafana metrics-server dashboard; do
    microk8s enable "$addon" >> "$LOG_FILE" 2>&1 || {
        log "⚠️ Warning: Failed to enable $addon (may already be enabled)"
    }
done
log "✅ MicroK8s add-ons enabled!"

# Install Python
PYTHON="python3"
if ! command -v "$PYTHON" >/dev/null; then
    log "📌 Installing Python..."
    if [[ "$PKG_MANAGER" == "snap" ]]; then
        apt update && apt install -y python3 python3-pip python3-venv || {
            log "❌ Failed to install Python!"
            exit 1
        }
    elif [[ "$PKG_MANAGER" == "brew" ]]; then
        brew install python || {
            log "❌ Failed to install Python!"
            exit 1
        }
    fi
fi
PYTHON_VERSION=$($PYTHON --version 2>&1)
log "✅ Python installed: $PYTHON_VERSION"

# Create virtual environment
VENV_DIR="$(pwd)/venv"
if [[ ! -d "$VENV_DIR" ]]; then
    log "📌 Creating virtual environment..."
    "$PYTHON" -m venv "$VENV_DIR" || {
        log "❌ Failed to create virtual environment!"
        exit 1
    }
fi
source "$VENV_DIR/bin/activate"
log "✅ Virtual environment activated!"

# Install Python dependencies
log "📦 Installing Python dependencies..."
pip install --upgrade pip >> "$LOG_FILE" 2>&1
pip install -r requirements.txt >> "$LOG_FILE" 2>&1 || {
    log "❌ Failed to install dependencies! Check $LOG_FILE"
    exit 1
}
log "✅ Dependencies installed!"

# Configure kubectl
KUBE_CONFIG="$HOME/.kube/config"
BACKUP_CONFIG="$HOME/.kube/config.backup.$(date +%s)"
log "📌 Generating kubeconfig..."
TEMP_CONFIG=$(mktemp)
if ! microk8s config > "$TEMP_CONFIG" 2>>"$LOG_FILE"; then
    log "❌ Failed to generate kubeconfig!"
    rm -f "$TEMP_CONFIG"
    exit 1
fi
# Validate kubeconfig
if ! grep -q "apiVersion: v1" "$TEMP_CONFIG" || ! grep -q "kind: Config" "$TEMP_CONFIG"; then
    log "❌ Generated kubeconfig is invalid! Check $LOG_FILE"
    cat "$TEMP_CONFIG" >> "$LOG_FILE"
    rm -f "$TEMP_CONFIG"
    exit 1
fi
if [[ -f "$KUBE_CONFIG" ]]; then
    log "📌 Backing up existing kubectl config to $BACKUP_CONFIG..."
    cp "$KUBE_CONFIG" "$BACKUP_CONFIG"
fi
log "📌 Configuring kubectl..."
mv "$TEMP_CONFIG" "$KUBE_CONFIG" || {
    log "❌ Failed to write kubeconfig!"
    exit 1
}
chmod 600 "$KUBE_CONFIG"
if ! kubectl get nodes >> "$LOG_FILE" 2>&1; then
    log "❌ Failed to verify kubectl! Check $LOG_FILE"
    microk8s inspect >> "$LOG_DIR/microk8s_inspect.log" 2>&1
    exit 1
fi
log "✅ kubectl configured!"

# Install k6
if ! command -v k6 >/dev/null; then
    log "📌 Installing k6..."
    if [[ "$PKG_MANAGER" == "snap" ]]; then
        snap install k6 || {
            log "❌ Failed to install k6 via snap!"
            exit 1
        }
    elif [[ "$PKG_MANAGER" == "brew" ]]; then
        brew install k6 || {
            log "❌ Failed to install k6 via brew!"
            exit 1
        }
    fi
fi
log "✅ k6 installed!"

log "\n✅ Installation completed successfully!"
log "📜 Logs available in: $LOG_FILE"
log "🚀 Run the simulation with: sudo bash scripts/run_simulation.sh [dqn|ppo] [true|false]"
log "🔗 Grafana will be available at: http://localhost:3000 after running the simulation"