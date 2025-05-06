#!/bin/bash
# scripts/setup_microk8s_multipass.sh
# Sets up Multipass, creates a VM, installs MicroK8s, and configures kubeconfig for autoscaling simulation.
# Usage: bash scripts/setup_microk8s_multipass.sh

set -e  # Exit on error

# Ensure Bash version >= 4
if [[ ${BASH_VERSINFO[0]} -lt 4 ]]; then
    echo "❌ This script requires Bash 4 or higher. Current version: $BASH_VERSION"
    exit 1
fi

# Setup logging
LOG_DIR="$(pwd)/logs"
LOG_FILE="$LOG_DIR/setup.log"
mkdir -p "$LOG_DIR"
exec 3>&1 1>>"$LOG_FILE" 2>&1
log() { echo "$@" >&3; echo "$@" >> "$LOG_FILE"; }

# Check if running on macOS
OS=$(uname -s | tr '[:upper:]' '[:lower:]')
if [[ "$OS" != "darwin" ]]; then
    log "❌ This script is designed for macOS only!"
    exit 1
fi
log "🔍 Detected OS: macOS"

# Install Homebrew if not installed
if ! command -v brew >/dev/null; then
    log "📌 Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    log "✅ Homebrew installed!"
fi

# Install Multipass if not installed
if ! command -v multipass >/dev/null; then
    log "📌 Installing Multipass..."
    brew install multipass
    if [[ $? -ne 0 ]]; then
        log "❌ Failed to install Multipass!"
        exit 1
    fi
    log "✅ Multipass installed, version: $(multipass version)"
else
    log "🔍 Multipass already installed, version: $(multipass version)"
fi

# Create VM with Multipass
VM_NAME="microk8s-vm"
log "🔍 Checking for VM $VM_NAME..."
if multipass info "$VM_NAME" >/dev/null 2>&1; then
    log "🔍 VM $VM_NAME already exists. Checking state..."
    VM_STATE=$(multipass info "$VM_NAME" --format yaml | grep -E "^State:" | awk '{print $2}')
    if [[ "$VM_STATE" == "Running" ]]; then
        log "✅ VM $VM_NAME is already running."
    elif [[ "$VM_STATE" == "Stopped" ]]; then
        log "📌 Starting VM $VM_NAME..."
        multipass start "$VM_NAME"
        if [[ $? -ne 0 ]]; then
            log "❌ Failed to start VM $VM_NAME!"
            exit 1
        fi
        # Wait for VM to be ready
        for attempt in {1..3}; do
            if multipass info "$VM_NAME" --format yaml | grep -q "State: Running"; then
                log "✅ VM $VM_NAME started successfully!"
                break
            fi
            log "⚠️ Waiting for VM to start (attempt $attempt/3)..."
            sleep 5
            if [[ $attempt -eq 3 ]]; then
                log "❌ VM $VM_NAME failed to start!"
                exit 1
            fi
        done
    else
        log "❌ VM $VM_NAME is in unexpected state: $VM_STATE"
        exit 1
    fi
else
    log "📌 Creating VM $VM_NAME..."
    multipass launch --name "$VM_NAME" --cpus 2 --mem 4G --disk 40G
    if [[ $? -ne 0 ]]; then
        log "❌ Failed to create VM $VM_NAME!"
        exit 1
    fi
    # Wait for VM to be ready
    for attempt in {1..3}; do
        if multipass info "$VM_NAME" --format yaml | grep -q "State: Running"; then
            log "✅ VM $VM_NAME created and running!"
            break
        fi
        log "⚠️ Waiting for VM to start (attempt $attempt/3)..."
        sleep 5
        if [[ $attempt -eq 3 ]]; then
            log "❌ VM $VM_NAME failed to start after creation!"
            exit 1
        fi
    done
fi

# Install MicroK8s in the VM
log "📌 Installing MicroK8s in VM $VM_NAME..."
multipass exec "$VM_NAME" -- /bin/bash -c "
    set -e
    sudo snap install microk8s --classic --channel=1.32
    sudo usermod -a -G microk8s ubuntu
    sudo chown -f -R ubuntu ~/.kube
    newgrp microk8s <<EOF
        microk8s status --wait-ready --timeout 300
    EOF
    microk8s enable dns storage metrics-server
"
if [[ $? -ne 0 ]]; then
    log "❌ Failed to install or configure MicroK8s in VM!"
    exit 1
fi
log "✅ MicroK8s installed and configured in VM!"

# Configure kubeconfig on host
KUBE_CONFIG="$HOME/.kube/config"
BACKUP_CONFIG="$HOME/.kube/config.backup.$(date +%s)"
TEMP_CONFIG=$(mktemp)
log "📌 Fetching kubeconfig from VM..."
multipass exec "$VM_NAME" -- /snap/bin/microk8s config > "$TEMP_CONFIG"
if [[ $? -ne 0 ]] || [[ ! -s "$TEMP_CONFIG" ]]; then
    log "❌ Failed to fetch kubeconfig!"
    rm -f "$TEMP_CONFIG"
    exit 1
fi

# Validate kubeconfig
if ! grep -q "apiVersion: v1" "$TEMP_CONFIG" || ! grep -q "kind: Config" "$TEMP_CONFIG"; then
    log "❌ Generated kubeconfig is invalid!"
    cat "$TEMP_CONFIG" >> "$LOG_FILE"
    rm -f "$TEMP_CONFIG"
    exit 1
fi

# Backup existing kubeconfig if it exists
if [[ -f "$KUBE_CONFIG" ]]; then
    log "📌 Backing up existing kubeconfig to $BACKUP_CONFIG..."
    cp "$KUBE_CONFIG" "$BACKUP_CONFIG"
fi

# Merge kubeconfig
log "📌 Merging kubeconfig..."
KUBECONFIG="$KUBE_CONFIG:$TEMP_CONFIG" kubectl config view --flatten > "$KUBE_CONFIG.new"
if [[ $? -ne 0 ]]; then
    log "❌ Failed to merge kubeconfig!"
    rm -f "$TEMP_CONFIG" "$KUBE_CONFIG.new"
    exit 1
fi
mv "$KUBE_CONFIG.new" "$KUBE_CONFIG"
chmod 600 "$KUBE_CONFIG"
rm -f "$TEMP_CONFIG"
log "✅ Kubeconfig configured!"

# Verify kubectl access
log "🔍 Verifying kubectl access..."
if ! kubectl get nodes >> "$LOG_FILE" 2>&1; then
    log "❌ Failed to verify kubectl! Check $LOG_FILE"
    multipass exec "$VM_NAME" -- /snap/bin/microk8s inspect >> "$LOG_DIR/microk8s_inspect.log" 2>&1
    exit 1
fi
log "✅ kubectl verified! Cluster is accessible."

log "\n✅ Setup completed successfully!"
log "📜 Logs available in: $LOG_FILE"
log "🔗 Run 'kubectl get nodes' to verify cluster."
log "🔗 Access VM with 'multipass shell $VM_NAME'."