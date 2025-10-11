"""Training script for Hybrid DQN-PPO with Lagrangian Constraints.

This example demonstrates how to train the Hybrid DQN-PPO agent with
hard SLA constraints using Lagrangian multipliers.

Usage:
    python examples/train_constrained_hybrid.py --steps 100000 --sla-limit 0.15

Expected results:
    - SLA violation rate: ≤ 15% (constrained)
    - Lambda converged: 2-4 (stable)
    - Safety interventions: 5-10%
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.hybrid_dqn_ppo import HybridDQNPPOAgent, HybridConfig
from agent.constrained_ppo import ConstrainedPPORewardOptimizer, ConstraintConfig
from agent.kubernetes_api import KubernetesAPI
import wandb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train Hybrid DQN-PPO with constraints")
    parser.add_argument("--steps", type=int, default=100000, help="Training steps")
    parser.add_argument("--sla-limit", type=float, default=0.15, help="Max SLA violation rate (0-1)")
    parser.add_argument("--sla-threshold", type=float, default=0.15, help="Latency threshold (seconds)")
    parser.add_argument("--lambda-lr", type=float, default=0.01, help="Lambda learning rate")
    parser.add_argument("--enable-safety", action="store_true", help="Enable safety layer")
    parser.add_argument("--mock", action="store_true", help="Use mock mode")
    args = parser.parse_args()

    # Initialize WandB
    wandb.init(
        project="microk8s-constrained-autoscaling",
        name=f"constrained_hybrid_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "algorithm": "Constrained Hybrid DQN-PPO",
            "sla_limit": args.sla_limit,
            "sla_threshold": args.sla_threshold,
            "lambda_lr": args.lambda_lr,
            "safety_layer": args.enable_safety,
            "total_steps": args.steps
        },
        mode="offline"
    )

    # Configure constraint settings
    constraint_config = ConstraintConfig(
        max_sla_violation_rate=args.sla_limit,
        sla_threshold_latency=args.sla_threshold,
        lambda_init=1.0,
        lambda_lr=args.lambda_lr,
        enable_safety_layer=args.enable_safety,
        safety_margin=0.1
    )

    # Configure hybrid agent
    hybrid_config = HybridConfig(
        state_dim=7,
        action_dim=3,
        hidden_dim=64
    )

    # Initialize Kubernetes API
    k8s_api = KubernetesAPI(namespace="default")

    # Initialize constrained optimizer
    constrained_optimizer = ConstrainedPPORewardOptimizer(
        state_dim=7,
        config=constraint_config
    )

    # Initialize hybrid agent
    agent = HybridDQNPPOAgent(
        config=hybrid_config,
        k8s_api=k8s_api,
        mock_mode=args.mock
    )

    logger.info("=" * 80)
    logger.info("CONSTRAINED HYBRID DQN-PPO TRAINING")
    logger.info("=" * 80)
    logger.info(f"Total steps: {args.steps}")
    logger.info(f"SLA constraint: ≤ {args.sla_limit * 100:.1f}% violations")
    logger.info(f"Latency threshold: {args.sla_threshold * 1000:.0f}ms")
    logger.info(f"Lambda learning rate: {args.lambda_lr}")
    logger.info(f"Safety layer: {'Enabled' if args.enable_safety else 'Disabled'}")
    logger.info("=" * 80)

    # Training loop
    for step in range(args.steps):
        # Get current state
        state = agent.get_state()

        # DQN selects action
        action = agent.dqn_agent.select_action(state, training=True)

        # Safety layer filtering (if enabled)
        if args.enable_safety:
            safe_action = constrained_optimizer.get_safe_action(
                state, action, None
            )
        else:
            safe_action = action

        # Execute action
        result = agent.execute_action(safe_action)
        next_state = agent.get_state()

        # Calculate base reward
        base_reward = agent.calculate_base_reward(state, next_state)

        # Apply Lagrangian constraint
        constrained_reward, sla_violation = constrained_optimizer.calculate_constrained_reward(
            base_reward=base_reward,
            state=state,
            metrics=result['metrics']
        )

        # Store in DQN replay buffer (use constrained reward)
        done = agent._is_episode_done(next_state)
        agent.dqn_agent.replay_buffer.push(
            state, safe_action, constrained_reward, next_state, done
        )

        # Update DQN
        if step % 10 == 0:
            dqn_loss = agent.dqn_agent.update(agent.config.dqn_batch_size)
            if dqn_loss > 0:
                wandb.log({"dqn/loss": dqn_loss, "train/step": step})

        # Log metrics every 100 steps
        if step % 100 == 0:
            diagnostics = constrained_optimizer.get_diagnostics()

            wandb.log({
                "train/step": step,
                "rewards/base": base_reward,
                "rewards/constrained": constrained_reward,
                "constraint/sla_violation_rate": diagnostics['constraint/sla_violation_rate'],
                "constraint/lambda": diagnostics['constraint/lambda_value'],
                "constraint/satisfied": diagnostics['constraint/satisfied'],
                "constraint/margin": diagnostics['constraint/margin'],
                "safety/intervention_rate": diagnostics['safety/intervention_rate'],
                "metrics/latency": diagnostics['metrics/avg_latency'],
                "actions/safety_modified": 1 if safe_action != action else 0
            })

            if step % 1000 == 0:
                logger.info(f"Step {step}/{args.steps}:")
                logger.info(f"  SLA Violation Rate: {diagnostics['constraint/sla_violation_rate']:.3f} "
                          f"(limit: {args.sla_limit:.3f})")
                logger.info(f"  Lambda: {diagnostics['constraint/lambda_value']:.3f}")
                logger.info(f"  Constraint: {'✅ Satisfied' if diagnostics['constraint/satisfied'] else '❌ Violated'}")
                logger.info(f"  Safety Interventions: {diagnostics['safety/intervention_rate']:.2%}")
                logger.info(f"  Avg Latency: {diagnostics['metrics/avg_latency'] * 1000:.1f}ms")

        # Periodic evaluation
        if step % 5000 == 0 and step > 0:
            agent.save_models("./models/constrained_hybrid")
            logger.info(f"Checkpoint saved at step {step}")

    # Final diagnostics
    final_diagnostics = constrained_optimizer.get_diagnostics()

    logger.info("=" * 80)
    logger.info("TRAINING COMPLETED")
    logger.info("=" * 80)
    logger.info(f"Final SLA Violation Rate: {final_diagnostics['constraint/sla_violation_rate']:.3f}")
    logger.info(f"Final Lambda: {final_diagnostics['constraint/lambda_value']:.3f}")
    logger.info(f"Constraint Satisfied: {'✅ Yes' if final_diagnostics['constraint/satisfied'] else '❌ No'}")
    logger.info(f"Safety Intervention Rate: {final_diagnostics['safety/intervention_rate']:.2%}")
    logger.info(f"Average Latency: {final_diagnostics['metrics/avg_latency'] * 1000:.1f}ms")

    # Save final model
    agent.save_models("./models/constrained_hybrid")
    logger.info("Final model saved to ./models/constrained_hybrid")

    wandb.finish()


if __name__ == "__main__":
    main()
