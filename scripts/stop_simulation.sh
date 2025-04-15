#!/bin/bash

echo "🛑 Stopping simulation processes..."

# Kill reinforcement learning agents
pkill -f dqn.py || true
pkill -f ppo.py || true

# Kill k6 load testing
pkill -f k6 || true

# Kill any active port-forwarding
pkill -f "port-forward" || true

echo "🛑 Stopping MicroK8s..."
echo "✅ All simulation processes have been stopped."