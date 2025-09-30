# MicroK8s RL Autoscaling - Hyperparameter Optimization Guide

This guide covers how to achieve convergence and optimal performance using Bayesian Optimization and hyperparameter tuning for both DQN and PPO agents.

## 🎯 Overview

Our autoscaling system supports three optimization approaches:
1. **DQN with Custom Bayesian Optimization** - Manual Gaussian Process implementation
2. **DQN with Optuna Optimization** - Advanced TPE (Tree-structured Parzen Estimator)
3. **PPO with Optuna Optimization** - Built-in hyperparameter optimization

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [DQN Optimization](#dqn-optimization)
- [PPO Optimization](#ppo-optimization)
- [Understanding Convergence](#understanding-convergence)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)

## 🚀 Quick Start

### Prerequisites
```bash
# Set Python path
export PYTHONPATH="/Users/danialfahmi/Documents/microk8s-autoscaling"

# Ensure dependencies are installed
pip install optuna wandb stable-baselines3 scikit-learn
```

### Quick Test Commands
```bash
# Test basic DQN training
python -m agent.dqn --simulate --timesteps 10000 --eval-episodes 5

# Test DQN optimization (short)
python -m agent.dqn_optimization --simulate --trials 5 --timesteps-per-trial 5000

# Test PPO optimization
python -m agent.ppo --simulate --timesteps 50000
```

## 🔬 DQN Optimization

### 💾 IMPORTANT: Persistent Storage (Resume Support)

**Your optimization progress is now automatically saved!**

All trial results are stored in a SQLite database (`optuna_dqn_studies.db`), so you can:
- **Resume interrupted runs** - Stop anytime, continue later
- **View past studies** - Check all optimization history
- **Continue adding trials** - Extend existing studies with more trials

#### Quick Examples:
```bash
# First run - creates study and database
python -m agent.dqn_optimization --simulate --trials 50 --study-name my_optimization

# Interrupted? Resume automatically (no prompt)
python -m agent.dqn_optimization --simulate --trials 50 --study-name my_optimization --resume

# List all saved studies
python -m agent.dqn_optimization --list-studies

# Add more trials to completed study
python -m agent.dqn_optimization --simulate --trials 20 --study-name my_optimization --resume
```

**Database location:** `optuna_dqn_studies.db` (created automatically in current directory)

**⚠️ Important:** Always use the **same `--study-name`** to resume a study. If you change the name, it creates a new study.

### 🚀 Smart Auto-Switching Mode

**The best of both worlds**: Start fast offline, finish with detailed online tracking!

**How it works:**
- Starts in offline mode (fast, no internet delays)
- Automatically switches to online mode for final trials (detailed tracking)
- Configurable threshold (default: last 2% of trials)

**Examples:**

```bash
# Default: 20 trials, auto-switch for final trial (98% complete)
python -m agent.dqn_optimization --simulate --wandb-mode offline

# 50 trials: offline for trials 0-48, online for trial 49
python -m agent.dqn_optimization --simulate --trials 50 --wandb-mode offline

# 100 trials: offline for trials 0-97, online for trials 98-99 (last 2%)
python -m agent.dqn_optimization --simulate --trials 100 --wandb-mode offline

# Custom switch point: last 5% of trials
python -m agent.dqn_optimization --simulate --trials 20 --wandb-mode offline --auto-switch-threshold 0.05

# Quick test: 5 trials, switch for final trial
python -m agent.dqn_optimization --simulate --trials 5 --wandb-mode offline

# Disable auto-switching (stay offline throughout)
python -m agent.dqn_optimization --simulate --trials 50 --wandb-mode offline --disable-auto-switch
```

**Auto-Switch Thresholds:**
- `--auto-switch-threshold 0.01` = Last 1% of trials
- `--auto-switch-threshold 0.02` = Last 2% of trials (default)
- `--auto-switch-threshold 0.05` = Last 5% of trials
- `--auto-switch-threshold 0.10` = Last 10% of trials

### Method 1: Optuna Bayesian Optimization with Persistence

**Features:**
- ✅ **Persistent SQLite storage** - Resume anytime
- TPE (Tree-structured Parzen Estimator) sampler
- Early stopping and convergence detection
- Real-time visualization with WandB
- Study management (list, resume, extend)

**Basic Usage:**
```bash
# First run - creates persistent study
python -m agent.dqn_optimization \
  --simulate \
  --trials 50 \
  --timesteps-per-trial 50000 \
  --study-name dqn_v1

# Interrupted at trial 23? Resume from trial 24
python -m agent.dqn_optimization \
  --simulate \
  --trials 50 \
  --timesteps-per-trial 50000 \
  --study-name dqn_v1 \
  --resume

# Quick test run
python -m agent.dqn_optimization \
  --simulate \
  --trials 10 \
  --timesteps-per-trial 10000 \
  --eval-episodes 5 \
  --study-name dqn_test

# Production run (longer) with custom database
python -m agent.dqn_optimization \
  --simulate \
  --trials 100 \
  --timesteps-per-trial 100000 \
  --eval-episodes 20 \
  --study-name dqn_production \
  --storage sqlite:///production_studies.db
```

**Study Management:**
```bash
# List all studies in database
python -m agent.dqn_optimization --list-studies

# Output example:
# 📚 Studies in sqlite:///optuna_dqn_studies.db:
#
#   📊 dqn_v1
#      Trials: 23
#      Best value: -156.42
#      Created: 2025-09-30 14:23:15
#
#   📊 dqn_test
#      Trials: 10
#      Best value: -178.91
#      Created: 2025-09-30 15:45:10

# Continue adding trials to completed study
python -m agent.dqn_optimization \
  --simulate \
  --trials 20 \
  --study-name dqn_v1 \
  --resume  # Runs trials 51-70
```

**What it optimizes:**
- `learning_rate`: 1e-5 to 1e-3 (log scale)
- `buffer_size`: 50,000 to 200,000
- `batch_size`: [32, 64, 128, 256]
- `gamma`: 0.95 to 0.999
- `tau`: 0.001 to 0.1
- `train_freq`: [1, 4, 8]
- `gradient_steps`: 1 to 4
- `learning_starts`: 1,000 to 10,000
- `target_update_interval`: 1,000 to 5,000
- `exploration_fraction`: 0.1 to 0.5
- `exploration_final_eps`: 0.01 to 0.2
- `max_grad_norm`: 5 to 20
- `net_arch_size`: [32, 64, 128]
- `net_arch_layers`: 1 to 3

**Output:**
- Best parameters saved to `best_dqn_params_YYYYMMDD_HHMMSS.json`
- Symlink created: `best_dqn_params_latest.json`
- Convergence plots and analysis

### Method 2: Using Optimized Parameters

**Option A: Load from file (recommended)**
```bash
python -m agent.dqn --simulate --load-optimized best_dqn_params_latest.json --timesteps 200000
```

**Option B: Manual parameters**
```bash
python -m agent.dqn --simulate \
  --learning-rate 0.000847 \
  --buffer-size 150000 \
  --batch-size 128 \
  --gamma 0.995 \
  --tau 0.008 \
  --train-freq 1 \
  --timesteps 200000
```

### Sample DQN Optimization Output
```
Best parameters found: {
  "learning_rate": 0.000847,
  "buffer_size": 150000,
  "batch_size": 128,
  "gamma": 0.995,
  "tau": 0.008,
  "train_freq": 1,
  "gradient_steps": 1,
  "learning_starts": 7500,
  "target_update_interval": 3000,
  "exploration_fraction": 0.35,
  "exploration_final_eps": 0.08,
  "max_grad_norm": 12,
  "net_arch_size": 128,
  "net_arch_layers": 3
}
Best value: 145.23
Converged: True
```

## 🎪 PPO Optimization

PPO already includes built-in Optuna optimization in the main script!

### Usage
```bash
# PPO with automatic optimization (20 trials)
python -m agent.ppo --simulate --timesteps 100000 --eval-episodes 50

# Quick test
python -m agent.ppo --simulate --timesteps 50000 --eval-episodes 20

# Production run
python -m agent.ppo --simulate --timesteps 200000 --eval-episodes 100
```

### What PPO Optimizes
Looking at `agent/hyperparameter_optimization.py`:
- `learning_rate`: 1e-5 to 1e-3 (log scale)
- `n_steps`: 1024 to 4096
- `batch_size`: 32 to 256
- `gamma`: 0.9 to 0.999
- `gae_lambda`: 0.9 to 0.99
- `clip_range`: 0.1 to 0.4
- `ent_coef`: 0.01 to 0.1
- `vf_coef`: 0.3 to 0.7

### PPO Optimization Process
1. **Optuna Study Creation** - TPE sampler for Bayesian optimization
2. **20 Trials** - Each trial trains for 10,000 timesteps
3. **Best Parameters Selection** - Highest average reward
4. **Final Training** - Uses optimized parameters for full training

## 📊 Understanding Convergence

### Convergence Indicators

**✅ Good Convergence:**
- Best value plateaus for 10+ consecutive trials
- Improvement rate < 1% for patience period
- Stable parameter values in recent trials
- Convergence plots show clear plateau

**❌ Poor Convergence:**
- Best value = -inf (trials failing)
- High variance in trial outcomes
- Parameters jumping randomly
- No clear improvement trend

### Monitoring Tools

**1. WandB Dashboard**
- Real-time optimization progress
- Parameter evolution plots
- Convergence metrics

**2. Convergence Monitor**
```python
from agent.convergence_monitor import ConvergenceMonitor

# Analyze optimization results
results = optimize_dqn_hyperparameters(...)
convergence_analysis = analyze_optimization_convergence(results)
```

**3. Manual Convergence Check**
```bash
# Check convergence plots
ls -la *.png | grep convergence

# View latest optimization logs
tail -f dqn.log
```

## 🔧 Troubleshooting

### Common Issues

**0. Optimization Restarts from Beginning**
```bash
# Problem: Running the same command restarts trials from 0

# ❌ WRONG - No study name, creates new study each time
python -m agent.dqn_optimization --simulate --trials 50

# ✅ CORRECT - Use consistent study name to resume
python -m agent.dqn_optimization --simulate --trials 50 --study-name my_study

# Then resume with --resume flag
python -m agent.dqn_optimization --simulate --trials 50 --study-name my_study --resume

# Check what studies exist
python -m agent.dqn_optimization --list-studies
```

**Solutions:**
- Always use `--study-name` to create persistent studies
- Use `--resume` flag to continue without prompts
- Database is saved as `optuna_dqn_studies.db` in current directory
- Don't delete the `.db` file if you want to resume

**1. Optimization Fails (Best value: -inf)**
```bash
# Symptoms
Best value: -inf
Converged: False

# Solutions
# A. Try shorter trials
python -m agent.dqn_optimization --simulate --trials 5 --timesteps-per-trial 5000

# B. Check basic training works
python -m agent.dqn --simulate --timesteps 10000

# C. Check logs for specific errors
cat dqn.log | grep ERROR
```

**2. Memory Issues**
```bash
# Reduce batch size and buffer size
python -m agent.dqn --simulate --batch-size 32 --buffer-size 50000

# Use fewer parallel environments
# Modify make_vec_env(n_envs=1) in code
```

**3. WandB Permission Issues**
```bash
# Fix wandb cache permissions
sudo chmod -R 755 /Users/danialfahmi/Library/Caches/wandb

# Or set custom cache directory
export WANDB_CACHE_DIR="$HOME/wandb_cache"
mkdir -p "$HOME/wandb_cache"
```

**4. Import Errors**
```bash
# Ensure PYTHONPATH is set
export PYTHONPATH="/Users/danialfahmi/Documents/microk8s-autoscaling"

# Use module syntax
python -m agent.dqn_optimization
# Instead of: python agent/dqn_optimization.py
```

### Debug Commands
```bash
# Test environment creation
python -c "from agent.environment_simulated import MicroK8sEnvSimulated; env = MicroK8sEnvSimulated(); print('Environment OK')"

# Test DQN agent creation
python -c "from agent.dqn import DQNAgent; from agent.environment_simulated import MicroK8sEnvSimulated; env = MicroK8sEnvSimulated(); agent = DQNAgent(env, env, True); print('Agent OK')"

# Check available GPU/CPU
python -c "import torch; print(f'Device: {torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")}')"
```

## 📈 Best Practices

### For DQN Optimization

**1. Start Small, Scale Up**
```bash
# Start with quick test
python -m agent.dqn_optimization --simulate --trials 5 --timesteps-per-trial 5000

# Then medium run
python -m agent.dqn_optimization --simulate --trials 20 --timesteps-per-trial 25000

# Finally full optimization
python -m agent.dqn_optimization --simulate --trials 50 --timesteps-per-trial 50000
```

**2. Use Appropriate Trial Counts**
- **Testing**: 5-10 trials
- **Development**: 20-30 trials
- **Production**: 50-100 trials

**3. Monitor Resources**
```bash
# Check CPU/Memory usage during optimization
htop

# Monitor disk space (logs can be large)
df -h
```

### For PPO Optimization

**1. PPO is Ready-to-Use**
```bash
# PPO already includes optimization, just run:
python -m agent.ppo --simulate --timesteps 100000
```

**2. Adjust Evaluation Episodes**
- **Simulated**: 50-100 episodes
- **Real environment**: 10-20 episodes

### Parameter Guidelines

**DQN Hyperparameters:**
- `learning_rate`: Start with 0.0005, typically optimal around 0.0003-0.001
- `batch_size`: 64 or 128 work well for most cases
- `gamma`: 0.99 is good default, can go up to 0.999 for longer episodes
- `buffer_size`: 100K-200K, larger if you have memory

**PPO Hyperparameters:**
- `learning_rate`: Often lower than DQN, around 1e-4 to 1e-3
- `n_steps`: 2048 is good default, can increase for longer episodes
- `batch_size`: Usually smaller than buffer, 64-256
- `clip_range`: 0.2 is standard, sometimes 0.1-0.3 works better

## 📊 Expected Results

### DQN Performance
- **Baseline (no optimization)**: ~80-120 average reward
- **Optimized**: ~140-180 average reward
- **Convergence time**: 20-50 trials (depending on timesteps)

### PPO Performance
- **Baseline**: ~90-130 average reward
- **Optimized**: ~150-200 average reward
- **Convergence time**: 15-25 trials

### Training Time Estimates
- **DQN optimization** (50 trials × 50K timesteps): 3-6 hours
- **PPO optimization** (20 trials × 10K timesteps): 1-2 hours
- **Final training** (200K timesteps): 30-60 minutes

## 🎯 Workflow Summary

### Complete DQN Optimization Workflow (with Resume Support)
```bash
# 1. Set environment
export PYTHONPATH="/Users/danialfahmi/Documents/microk8s-autoscaling"

# 2. Quick test
python -m agent.dqn --simulate --timesteps 10000

# 3. Run optimization with persistent study
python -m agent.dqn_optimization \
  --simulate \
  --trials 50 \
  --timesteps-per-trial 50000 \
  --study-name my_dqn_optimization

# 3a. If interrupted, resume from where you stopped
python -m agent.dqn_optimization \
  --simulate \
  --trials 50 \
  --timesteps-per-trial 50000 \
  --study-name my_dqn_optimization \
  --resume

# 3b. Check progress anytime
python -m agent.dqn_optimization --list-studies

# 4. Train with best parameters
python -m agent.dqn --simulate --load-optimized best_dqn_params_latest.json --timesteps 200000

# 5. Evaluate final model
python -m agent.dqn --simulate --load-optimized best_dqn_params_latest.json --timesteps 0 --eval-episodes 50
```

### Multi-Day Optimization Example
```bash
# Day 1: Start optimization (run 20 trials, then stop)
python -m agent.dqn_optimization --simulate --trials 50 --study-name week1_optimization
# ... Ctrl+C after 20 trials complete

# Day 2: Resume and complete remaining 30 trials
python -m agent.dqn_optimization --simulate --trials 50 --study-name week1_optimization --resume
# Continues from trial 21

# Day 3: Extend study with 20 more trials (70 total)
python -m agent.dqn_optimization --simulate --trials 20 --study-name week1_optimization --resume
# Runs trials 51-70

# Check final results
python -m agent.dqn_optimization --list-studies
```

### Complete PPO Optimization Workflow
```bash
# 1. Set environment
export PYTHONPATH="/Users/danialfahmi/Documents/microk8s-autoscaling"

# 2. Run PPO with built-in optimization
python -m agent.ppo --simulate --timesteps 100000 --eval-episodes 50

# That's it! PPO handles optimization automatically
```

## 🔍 Monitoring and Analysis

### Key Metrics to Watch
- **Objective Value**: Should increase over trials
- **Parameter Stability**: Values should converge
- **Training Stability**: No NaN or infinite values
- **Resource Utilization**: CPU should stay in 40-80% range
- **Scaling Actions**: Balanced distribution of up/down/no-change

### Files Generated
- `best_dqn_params_YYYYMMDD_HHMMSS.json` - Optimized parameters
- `best_dqn_params_latest.json` - Symlink to latest
- `convergence_analysis_YYYYMMDD_HHMMSS.png` - Convergence plots
- `optimization_convergence_YYYYMMDD_HHMMSS.png` - Parameter evolution
- `dqn.log` / `ppo.log` - Training logs
- `./wandb/` - WandB experiment tracking

## 🎉 Success Indicators

**You know optimization worked when:**
1. ✅ Best value > 140 (for autoscaling reward)
2. ✅ Convergence = True
3. ✅ Parameters file created successfully
4. ✅ Training with optimized params shows improved performance
5. ✅ CPU utilization stays in target range (40-80%)
6. ✅ Reasonable scaling action distribution

---

## 💡 Tips for Success

1. **Always test basic training before optimization**
2. **Start with short trials, increase gradually**
3. **Monitor system resources during long runs**
4. **Keep backups of good parameter files**
5. **Use WandB to track optimization progress**
6. **Compare results with baseline (non-optimized) training**
7. **✨ NEW: Always use `--study-name` for persistent studies**
8. **✨ NEW: Use `--resume` to continue interrupted optimizations**
9. **✨ NEW: Check `--list-studies` to see all saved progress**
10. **✨ NEW: Don't delete `optuna_dqn_studies.db` - it's your progress!**

## 📊 New CLI Flags Reference

| Flag | Description | Example |
|------|-------------|---------|
| `--study-name` | Name for persistent study (required for resume) | `--study-name my_optimization` |
| `--resume` | Resume existing study without prompt | `--resume` |
| `--list-studies` | List all studies in database | `--list-studies` |
| `--storage` | Custom SQLite database path | `--storage sqlite:///custom.db` |
| `--traffic-seed` | Fixed seed for reproducible traffic | `--traffic-seed 42` |

Happy optimizing! 🚀