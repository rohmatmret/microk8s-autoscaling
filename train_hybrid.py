#!/usr/bin/env python3
"""
Hybrid DQN-PPO Training Script for MicroK8s Autoscaling

This script trains a hybrid reinforcement learning agent that combines:
- DQN for discrete scaling actions (scale up/down/hold)
- PPO for reward function optimization

Usage:
    python train_hybrid.py --config config.yaml --steps 50000
"""

import argparse
import os
import sys
import logging
import yaml
from datetime import datetime
from typing import Dict, Any
import time

# Add agent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'agent'))

from agent.hybrid_dqn_ppo import HybridDQNPPOAgent, HybridConfig
from agent.hybrid_environment import HybridMicroK8sEnv
from agent.kubernetes_api import KubernetesAPI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("hybrid_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        raise

def create_default_config() -> Dict[str, Any]:
    """Create default configuration."""
    return {
        'environment': {
            'deployment_name': 'nginx-deployment',
            'namespace': 'default',
            'max_pods': 10,
            'scaling_delay': 10,
            'metrics_sync_interval': 10,
            'prometheus_url': 'http://localhost:9090'
        },
        'dqn': {
            'learning_rate': 0.0005,
            'buffer_size': 100000,
            'batch_size': 64,
            'gamma': 0.99,
            'tau': 0.1,
            'epsilon_start': 1.0,
            'epsilon_end': 0.07,
            'epsilon_decay': 0.995,
            'target_update_freq': 2000
        },
        'ppo': {
            'learning_rate': 0.0003,
            'n_steps': 2048,
            'batch_size': 64,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.01
        },
        'hybrid': {
            'reward_optimization_freq': 100,
            'batch_evaluation_steps': 50,
            'state_dim': 6,
            'action_dim': 3,
            'hidden_dim': 64
        },
        'training': {
            'total_steps': 50000,
            'eval_freq': 1000,
            'checkpoint_freq': 5000,
            'log_freq': 100
        }
    }

def save_config(config: Dict[str, Any], path: str):
    """Save configuration to YAML file."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        logger.info(f"Configuration saved to {path}")
    except Exception as e:
        logger.error(f"Failed to save config to {path}: {e}")

def create_hybrid_config(config_dict: Dict[str, Any]) -> HybridConfig:
    """Create HybridConfig from dictionary."""
    return HybridConfig(
        # DQN Configuration
        dqn_learning_rate=config_dict['dqn']['learning_rate'],
        dqn_buffer_size=config_dict['dqn']['buffer_size'],
        dqn_batch_size=config_dict['dqn']['batch_size'],
        dqn_gamma=config_dict['dqn']['gamma'],
        dqn_tau=config_dict['dqn']['tau'],
        dqn_epsilon_start=config_dict['dqn']['epsilon_start'],
        dqn_epsilon_end=config_dict['dqn']['epsilon_end'],
        dqn_epsilon_decay=config_dict['dqn']['epsilon_decay'],
        dqn_target_update_freq=config_dict['dqn']['target_update_freq'],
        
        # PPO Configuration
        ppo_learning_rate=config_dict['ppo']['learning_rate'],
        ppo_n_steps=config_dict['ppo']['n_steps'],
        ppo_batch_size=config_dict['ppo']['batch_size'],
        ppo_gamma=config_dict['ppo']['gamma'],
        ppo_gae_lambda=config_dict['ppo']['gae_lambda'],
        ppo_clip_range=config_dict['ppo']['clip_range'],
        ppo_ent_coef=config_dict['ppo']['ent_coef'],
        
        # Hybrid Configuration
        reward_optimization_freq=config_dict['hybrid']['reward_optimization_freq'],
        batch_evaluation_steps=config_dict['hybrid']['batch_evaluation_steps'],
        state_dim=config_dict['hybrid']['state_dim'],
        action_dim=config_dict['hybrid']['action_dim'],
        hidden_dim=config_dict['hybrid']['hidden_dim']
    )

def setup_environment(config: Dict[str, Any]) -> HybridMicroK8sEnv:
    """Setup the hybrid environment."""
    env_config = config['environment']
    
    # Create Kubernetes API
    k8s_api = KubernetesAPI(
        max_pods=env_config['max_pods'],
        namespace=env_config['namespace']
    )
    
    # Create hybrid environment
    env = HybridMicroK8sEnv(
        deployment_name=env_config['deployment_name'],
        namespace=env_config['namespace'],
        max_pods=env_config['max_pods'],
        scaling_delay=env_config['scaling_delay'],
        metrics_sync_interval=env_config['metrics_sync_interval'],
        prometheus_url=env_config['prometheus_url']
    )
    
    logger.info("Hybrid environment setup completed")
    return env

def train_hybrid_agent(config: Dict[str, Any], output_dir: str = "./models/hybrid", mock_mode: bool = False):
    """Train the hybrid DQN-PPO agent."""
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save configuration (with error handling)
        try:
            config_path = os.path.join(output_dir, "config.yaml")
            save_config(config, config_path)
        except Exception as e:
            logger.warning(f"Failed to save config to {output_dir}: {e}")
            # Try saving to current directory
            try:
                save_config(config, "config.yaml")
                logger.info("Config saved to current directory")
            except Exception as e2:
                logger.warning(f"Failed to save config even to current directory: {e2}")
        
        # Setup environment
        env = setup_environment(config)
        
        # Create hybrid config
        hybrid_config = create_hybrid_config(config)
        
        # Create Kubernetes API
        k8s_api = KubernetesAPI(
            max_pods=config['environment']['max_pods'],
            namespace=config['environment']['namespace']
        )
        
        # Create hybrid agent with mock mode if requested
        agent = HybridDQNPPOAgent(hybrid_config, k8s_api, mock_mode=mock_mode)
        
        # Start training
        logger.info("Starting hybrid agent training...")
        start_time = datetime.now()
        
        agent.train(total_steps=config['training']['total_steps'])
        
        # Save final models
        agent.save_models(output_dir)
        
        end_time = datetime.now()
        training_duration = end_time - start_time
        
        logger.info(f"Training completed in {training_duration}")
        logger.info(f"Models saved to {output_dir}")
        
        # Log final metrics
        final_metrics = {
            'training_duration': str(training_duration),
            'total_steps': config['training']['total_steps'],
            'final_episode_count': agent.episode_count,
            'model_path': output_dir,
            'mock_mode': mock_mode
        }
        
        try:
            with open(os.path.join(output_dir, "training_summary.yaml"), 'w') as f:
                yaml.dump(final_metrics, f, default_flow_style=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save training summary: {e}")
            # Try saving to current directory
            try:
                with open("training_summary.yaml", 'w') as f:
                    yaml.dump(final_metrics, f, default_flow_style=False, indent=2)
                logger.info("Training summary saved to current directory")
            except Exception as e2:
                logger.warning(f"Failed to save training summary even to current directory: {e2}")
        
        return agent
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

def evaluate_agent(agent: HybridDQNPPOAgent, env: HybridMicroK8sEnv, 
                  episodes: int = 10) -> Dict[str, float]:
    """Evaluate the trained agent."""
    logger.info(f"Evaluating agent over {episodes} episodes...")
    
    eval_rewards = []
    eval_metrics = {
        'avg_latency': [],
        'avg_throughput': [],
        'avg_cost': [],
        'avg_cpu': [],
        'avg_memory': []
    }
    
    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0.0
        episode_metrics = []
        
        for step in range(100):  # Max 100 steps per episode
            action = agent.get_scaling_decision(state)
            state, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            if 'custom_metrics' in info:
                episode_metrics.append(info['custom_metrics'])
            
            if terminated or truncated:
                break
        
        eval_rewards.append(episode_reward)
        
        # Calculate average metrics for this episode
        if episode_metrics:
            avg_metrics = {
                'latency': sum(m['latency'] for m in episode_metrics) / len(episode_metrics),
                'throughput': sum(m['throughput'] for m in episode_metrics) / len(episode_metrics),
                'cost': sum(m['cost'] for m in episode_metrics) / len(episode_metrics),
                'cpu': sum(m['cpu_utilization'] for m in episode_metrics) / len(episode_metrics),
                'memory': sum(m['memory_utilization'] for m in episode_metrics) / len(episode_metrics)
            }
            
            for key, value in avg_metrics.items():
                eval_metrics[f'avg_{key}'].append(value)
    
    # Calculate final evaluation metrics
    results = {
        'mean_reward': sum(eval_rewards) / len(eval_rewards),
        'std_reward': (sum((r - sum(eval_rewards) / len(eval_rewards)) ** 2 for r in eval_rewards) / len(eval_rewards)) ** 0.5,
        'min_reward': min(eval_rewards),
        'max_reward': max(eval_rewards)
    }
    
    # Add average metrics
    for key, values in eval_metrics.items():
        if values:
            results[key] = sum(values) / len(values)
    
    logger.info(f"Evaluation results: {results}")
    return results

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Hybrid DQN-PPO Agent for MicroK8s Autoscaling")
    parser.add_argument("--config", type=str, default="config/hybrid_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--output", type=str, default="./models/hybrid",
                       help="Output directory for models and logs")
    parser.add_argument("--steps", type=int, default=50000,
                       help="Total training steps")
    parser.add_argument("--eval", action="store_true",
                       help="Run evaluation after training")
    parser.add_argument("--create-config", action="store_true",
                       help="Create default configuration file")
    parser.add_argument("--mock", action="store_true",
                       help="Run in mock mode (no Kubernetes required)")
    
    args = parser.parse_args()
    
    try:
        # Create default config if requested
        if args.create_config:
            config = create_default_config()
            os.makedirs(os.path.dirname(args.config), exist_ok=True)
            save_config(config, args.config)
            logger.info(f"Default configuration created at {args.config}")
            return
        
        # Load configuration
        if os.path.exists(args.config):
            config = load_config(args.config)
        else:
            logger.warning(f"Config file {args.config} not found, using default configuration")
            config = create_default_config()
        
        # Override training steps if specified
        if args.steps != 50000:
            config['training']['total_steps'] = args.steps
        
        # Train agent
        agent = train_hybrid_agent(config, args.output, mock_mode=args.mock)
        
        # Evaluate if requested
        if args.eval:
            env = setup_environment(config)
            eval_results = evaluate_agent(agent, env)
            
            # Save evaluation results
            eval_path = os.path.join(args.output, "evaluation_results.yaml")
            with open(eval_path, 'w') as f:
                yaml.dump(eval_results, f, default_flow_style=False, indent=2)
            
            logger.info(f"Evaluation results saved to {eval_path}")
        
        logger.info("Training completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

# Example inference script
config = HybridConfig()
k8s_api = KubernetesAPI(max_pods=10, namespace="default")
agent = HybridDQNPPOAgent(config, k8s_api)
agent.load_models("./models/hybrid")

while True:
    state = agent.get_state()
    action = agent.get_scaling_decision(state)
    print("Scaling action:", action)
    # Optionally: agent.execute_action(action)
    time.sleep(10) 