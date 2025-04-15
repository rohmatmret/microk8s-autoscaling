# Makefile
# Manages setup and simulation for MicroK8s RL autoscaling project.

# Variables
VENV_DIR = venv
PYTHON = python3
MICROK8S = microk8s
AGENT ?= dqn
SKIP_LOADTEST ?= false
LOG_DIR = logs

# Default target
.PHONY: all
all: setup

# Check sudo (for targets requiring root)
.PHONY: check-sudo
check-sudo:
	@if [ "$$EUID" -ne 0 ]; then echo "❌ Please run with sudo (e.g., sudo make <target>)"; exit 1; fi

# Install everything using install_microk8s.sh
.PHONY: setup
setup: check-sudo
	@echo "🚀 Running full setup with install_microk8s.sh..."
	@bash scripts/install_microk8s.sh
	@echo "✅ Setup completed! Logs in $(LOG_DIR)/install.log"

# Verify MicroK8s status and resources
.PHONY: verify
verify: check-sudo
	@echo "🔍 Verifying MicroK8s status..."
	@$(MICROK8S) status --wait-ready --timeout 300
	@kubectl get nodes
	@kubectl get all -A
	@echo "✅ MicroK8s verified!"

# Run simulation
.PHONY: start-simulation
start-simulation: check-sudo
	@echo "🚀 Starting simulation with agent: $(AGENT), skip_loadtest: $(SKIP_LOADTEST)"
	@bash scripts/run_simulation.sh $(AGENT) $(SKIP_LOADTEST)
	@echo "✅ Simulation started! Logs in $(LOG_DIR)/simulation.log"

# Alias for start-simulation
.PHONY: start
start: start-simulation

# Stop simulation
.PHONY: stop
stop: check-sudo
	@echo "🛑 Stopping simulation..."
	@bash scripts/stop_simulation.sh
	@echo "✅ Simulation stopped! Logs in $(LOG_DIR)/stop.log"

# Restart simulation
.PHONY: restart
restart: stop start

# Clean up (remove venv, logs, and MicroK8s)
.PHONY: clean
clean: check-sudo
	@echo "🧹 Cleaning up..."
	@bash scripts/stop_simulation.sh || true
	@if [ -d "$(VENV_DIR)" ]; then rm -rf "$(VENV_DIR)"; fi
	@if [ -d "$(LOG_DIR)" ]; then rm -rf "$(LOG_DIR)"; fi
	@$(MICROK8S) stop || true
	@if command -v multipass >/dev/null; then multipass delete microk8s-vm --purge || true; fi
	@if command -v snap >/dev/null; then snap remove microk8s --purge || true; fi
	@if command -v brew >/dev/null; then brew uninstall microk8s || true; fi
	@echo "✅ Cleanup completed!"

# Help
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  setup           : Install MicroK8s, Python, k6, and dependencies"
	@echo "  verify          : Verify MicroK8s status and resources"
	@echo "  start-simulation: Run simulation (AGENT=dqn|ppo, SKIP_LOADTEST=true|false)"
	@echo "  start           : Alias for start-simulation"
	@echo "  stop            : Stop simulation"
	@echo "  restart         : Restart simulation"
	@echo "  clean           : Remove venv, logs, and MicroK8s"
	@echo "  help            : Show this help"
	@echo "Example: sudo make start-simulation AGENT=ppo SKIP_LOADTEST=false"