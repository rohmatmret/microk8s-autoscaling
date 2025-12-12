"""
Training script for Adaptive Hybrid DQN-PPO with learnable CPU target.

This script demonstrates TRUE dynamic optimization where the agent learns
the optimal CPU target rather than using a fixed hardcoded value.

Usage:
    python agent/train_adaptive_hybrid.py --steps 50000 --mock
    python agent/train_adaptive_hybrid.py --steps 100000 --mock --complex-traffic
"""

import argparse
import logging
import yaml
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from agent.adaptive_hybrid_dqn_ppo import (
    AdaptiveHybridDQNPPOAgent,
    HybridConfig
)
from agent.kubernetes_api import KubernetesAPI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("adaptive_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/hybrid_config.yaml") -> HybridConfig:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Create HybridConfig with default values, then update
        config = HybridConfig()

        # Update DQN parameters
        if 'dqn' in config_dict:
            for key, value in config_dict['dqn'].items():
                setattr(config, f'dqn_{key}', value)

        # Update PPO parameters
        if 'ppo' in config_dict:
            for key, value in config_dict['ppo'].items():
                setattr(config, f'ppo_{key}', value)

        # Update hybrid parameters
        if 'hybrid' in config_dict:
            for key, value in config_dict['hybrid'].items():
                setattr(config, key, value)

        logger.info(f"Configuration loaded from {config_path}")
        return config

    except FileNotFoundError:
        logger.warning(f"Config file {config_path} not found, using defaults")
        return HybridConfig()
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return HybridConfig()


def main():
    parser = argparse.ArgumentParser(
        description="Train Adaptive Hybrid DQN-PPO with learnable CPU target"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/hybrid_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=50000,
        help='Number of training steps'
    )
    parser.add_argument(
        '--mock',
        action='store_true',
        help='Run in mock mode (simulation)'
    )
    parser.add_argument(
        '--complex-traffic',
        action='store_true',
        help='Use complex traffic patterns during training'
    )
    parser.add_argument(
        '--save-path',
        type=str,
        default='./models/adaptive_hybrid',
        help='Path to save trained models'
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Initialize Kubernetes API
    if args.mock:
        logger.info("Running in MOCK mode (simulation)")
        k8s_api = KubernetesAPI(
            namespace="default",
            max_pods=10,
            mock_mode=True
        )
    else:
        logger.info("Running in PRODUCTION mode (real cluster)")
        k8s_api = KubernetesAPI(
            namespace="default",
            max_pods=10,
            mock_mode=False
        )

    # Initialize adaptive agent
    logger.info("=" * 80)
    logger.info("ADAPTIVE HYBRID DQN-PPO AGENT")
    logger.info("=" * 80)
    logger.info("Key Feature: CPU target is LEARNED dynamically")
    logger.info("  - NOT hardcoded to 65%")
    logger.info("  - Adapts to traffic patterns")
    logger.info("  - Optimizes SLA vs cost trade-off")
    logger.info("  - Range: 50%-80% (learned)")
    logger.info("=" * 80)

    agent = AdaptiveHybridDQNPPOAgent(
        config=config,
        k8s_api=k8s_api,
        mock_mode=args.mock
    )

    # Log training configuration
    logger.info(f"Training Configuration:")
    logger.info(f"  Total steps: {args.steps:,}")
    logger.info(f"  Mock mode: {args.mock}")
    logger.info(f"  Complex traffic: {args.complex_traffic}")
    logger.info(f"  DQN learning rate: {config.dqn_learning_rate}")
    logger.info(f"  PPO learning rate: {config.ppo_learning_rate}")
    logger.info(f"  Batch size: {config.dqn_batch_size}")
    logger.info(f"  Save path: {args.save_path}")

    # Train agent
    try:
        logger.info("\nStarting training...")
        agent.train(total_steps=args.steps)

        # Get CPU target statistics
        target_stats = agent.get_cpu_target_statistics()
        logger.info("\n" + "=" * 80)
        logger.info("LEARNED CPU TARGET STATISTICS")
        logger.info("=" * 80)
        logger.info(f"  Mean:    {target_stats['mean']:.3f} ({target_stats['mean']*100:.1f}%)")
        logger.info(f"  Min:     {target_stats['min']:.3f} ({target_stats['min']*100:.1f}%)")
        logger.info(f"  Max:     {target_stats['max']:.3f} ({target_stats['max']*100:.1f}%)")
        logger.info(f"  Std Dev: {target_stats['std']:.3f}")
        logger.info(f"  Current: {target_stats['current']:.3f} ({target_stats['current']*100:.1f}%)")
        logger.info("=" * 80)

        if target_stats['std'] > 0.05:
            logger.info("✅ CPU target is DYNAMIC (high variance)")
            logger.info("   Agent adapts target based on traffic patterns")
        else:
            logger.warning("⚠️ CPU target is relatively STATIC (low variance)")
            logger.warning("   Consider training longer or adjusting learning rate")

        # Save models
        agent.save_models(args.save_path)
        logger.info(f"\n✅ Training complete! Models saved to {args.save_path}")

    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        agent.save_models(args.save_path)
        logger.info(f"Models saved to {args.save_path}")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
