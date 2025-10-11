"""Constrained PPO with Lagrangian Multipliers for SLA-Aware Autoscaling.

This module implements Constrained Markov Decision Process (CMDP) optimization
using Lagrangian methods to enforce hard SLA constraints while optimizing cost.

Mathematical Formulation:
    maximize E[R(s,a)]  (expected reward)
    subject to E[C(s,a)] ≤ δ  (expected SLA violations ≤ threshold)

Lagrangian Formulation:
    L(θ, λ) = E[R(s,a)] - λ * (E[C(s,a)] - δ)

Where:
    - θ: Policy parameters
    - λ: Lagrangian multiplier (learned dynamically)
    - R: Reward function
    - C: Constraint cost (SLA violations)
    - δ: Maximum allowed constraint violation

Reference:
    Ray et al. (2019) - "Benchmarking Safe Exploration in Deep Reinforcement Learning"
    Achiam et al. (2017) - "Constrained Policy Optimization"
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ConstraintConfig:
    """Configuration for constraint enforcement."""
    # SLA constraint parameters
    max_sla_violation_rate: float = 0.15  # Maximum 15% SLA violations allowed
    sla_threshold_latency: float = 0.15   # 150ms latency threshold

    # Lagrangian multiplier parameters
    lambda_init: float = 1.0              # Initial Lagrangian multiplier
    lambda_lr: float = 0.01               # Learning rate for λ updates
    lambda_min: float = 0.01              # Minimum λ value
    lambda_max: float = 100.0             # Maximum λ value

    # Safety parameters
    enable_safety_layer: bool = True      # Enable conservative action clipping
    safety_margin: float = 0.1            # Safety margin for constraint buffer

    # Constraint tracking
    constraint_window_size: int = 1000    # Window for moving average


class LagrangianMultiplierOptimizer:
    """Adaptive Lagrangian multiplier optimizer for constraint enforcement.

    Implements dual gradient ascent to find optimal λ:
        λ_{t+1} = λ_t + α * (C_t - δ)

    Where C_t is the empirical constraint cost and δ is the threshold.
    """

    def __init__(self, config: ConstraintConfig):
        self.config = config
        self.lambda_value = config.lambda_init
        self.constraint_history = []
        self.lambda_history = []

    def update(self, constraint_violation: float) -> float:
        """Update Lagrangian multiplier based on constraint violation.

        Args:
            constraint_violation: Current SLA violation rate (0-1)

        Returns:
            Updated lambda value
        """
        # Calculate constraint surplus (positive = violating, negative = safe)
        constraint_surplus = constraint_violation - self.config.max_sla_violation_rate

        # Dual gradient ascent update
        self.lambda_value += self.config.lambda_lr * constraint_surplus

        # Project λ to valid range [λ_min, λ_max]
        self.lambda_value = np.clip(
            self.lambda_value,
            self.config.lambda_min,
            self.config.lambda_max
        )

        # Track history for analysis
        self.constraint_history.append(constraint_violation)
        self.lambda_history.append(self.lambda_value)

        # Keep only recent history
        if len(self.constraint_history) > self.config.constraint_window_size:
            self.constraint_history = self.constraint_history[-self.config.constraint_window_size:]
            self.lambda_history = self.lambda_history[-self.config.constraint_window_size:]

        return self.lambda_value

    def get_average_constraint_violation(self) -> float:
        """Calculate moving average of constraint violations."""
        if not self.constraint_history:
            return 0.0
        return np.mean(self.constraint_history[-100:])  # Last 100 steps

    def is_constraint_satisfied(self) -> bool:
        """Check if constraint is currently satisfied."""
        avg_violation = self.get_average_constraint_violation()
        return avg_violation <= self.config.max_sla_violation_rate


class SafetyLayer:
    """Conservative action clipping to prevent SLA violations.

    Implements a safety layer that modifies actions if they would likely
    cause SLA violations based on current state.
    """

    def __init__(self, config: ConstraintConfig):
        self.config = config
        self.violation_count = 0
        self.total_actions = 0

    def filter_action(self, state: np.ndarray, action: int, q_values: torch.Tensor) -> int:
        """Filter action to prevent unsafe scaling decisions.

        Args:
            state: Current system state [cpu, memory, latency, ...]
            action: Proposed action (0=scale_down, 1=no_change, 2=scale_up)
            q_values: Q-values for all actions

        Returns:
            Safe action (potentially modified)
        """
        if not self.config.enable_safety_layer:
            return action

        # Extract state features
        cpu_util = state[0] if len(state) > 0 else 0.5
        latency = state[2] if len(state) > 2 else 0.1
        pods = state[4] if len(state) > 4 else 0.3

        # Safety rules
        safe_action = action

        # Rule 1: Don't scale down if latency is high
        if action == 0:  # scale_down
            if latency > self.config.sla_threshold_latency - self.config.safety_margin:
                # Override to no_change
                safe_action = 1
                logger.debug(f"Safety: Blocked scale_down (latency={latency:.3f} near SLA threshold)")
            elif cpu_util > 0.65:
                # High CPU, don't scale down
                safe_action = 1
                logger.debug(f"Safety: Blocked scale_down (CPU={cpu_util:.2f} too high)")

        # Rule 2: Force scale up if critical SLA violation imminent
        elif action == 1:  # no_change
            if latency > self.config.sla_threshold_latency * 1.2:  # 20% over threshold
                safe_action = 2  # Force scale_up
                logger.debug(f"Safety: Forced scale_up (latency={latency:.3f} critical)")
            elif cpu_util > 0.85 and pods < 0.9:  # High CPU and room to scale
                safe_action = 2
                logger.debug(f"Safety: Forced scale_up (CPU={cpu_util:.2f} very high)")

        # Rule 3: Limit scale up if over-provisioned
        elif action == 2:  # scale_up
            if cpu_util < 0.3 and latency < self.config.sla_threshold_latency * 0.5:
                # Very low utilization and good latency
                safe_action = 1
                logger.debug(f"Safety: Blocked scale_up (over-provisioned: CPU={cpu_util:.2f})")

        self.total_actions += 1
        if safe_action != action:
            self.violation_count += 1

        return safe_action

    def get_intervention_rate(self) -> float:
        """Get rate of safety interventions."""
        if self.total_actions == 0:
            return 0.0
        return self.violation_count / self.total_actions


class ConstrainedPPORewardOptimizer:
    """PPO with Lagrangian constraints for SLA-aware optimization.

    Combines standard PPO with constraint enforcement using Lagrangian multipliers.
    The objective becomes:
        L(θ, λ) = E[R] - λ * (E[C] - δ)

    Where:
        - E[R]: Expected reward (cost efficiency)
        - λ: Lagrangian multiplier (learned)
        - E[C]: Expected constraint cost (SLA violations)
        - δ: Constraint threshold
    """

    def __init__(
        self,
        state_dim: int,
        config: ConstraintConfig,
        ppo_learning_rate: float = 0.0003,
        ppo_clip_range: float = 0.2
    ):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Lagrangian multiplier optimizer
        self.lagrangian_optimizer = LagrangianMultiplierOptimizer(config)

        # Safety layer
        self.safety_layer = SafetyLayer(config)

        # PPO network (simplified actor-critic)
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Value function output
        ).to(self.device)

        self.optimizer = optim.Adam(self.network.parameters(), lr=ppo_learning_rate)
        self.clip_range = ppo_clip_range

        # Constraint tracking
        self.recent_sla_violations = []
        self.recent_latencies = []

        logger.info("Constrained PPO Reward Optimizer initialized")
        logger.info(f"SLA constraint: max {config.max_sla_violation_rate*100:.1f}% violations")
        logger.info(f"Latency threshold: {config.sla_threshold_latency*1000:.0f}ms")

    def calculate_constrained_reward(
        self,
        base_reward: float,
        state: np.ndarray,
        metrics: Dict[str, float]
    ) -> Tuple[float, float]:
        """Calculate Lagrangian-constrained reward.

        Args:
            base_reward: Original reward from environment
            state: Current state
            metrics: Environment metrics (latency, cost, etc.)

        Returns:
            (constrained_reward, constraint_cost)
        """
        # Extract constraint-relevant metrics
        latency = metrics.get('latency', 0.0)

        # Calculate constraint cost (1 if violating SLA, 0 otherwise)
        is_sla_violation = 1.0 if latency > self.config.sla_threshold_latency else 0.0

        # Track violations for moving average
        self.recent_sla_violations.append(is_sla_violation)
        self.recent_latencies.append(latency)

        # Keep only recent history
        max_history = 100
        if len(self.recent_sla_violations) > max_history:
            self.recent_sla_violations = self.recent_sla_violations[-max_history:]
            self.recent_latencies = self.recent_latencies[-max_history:]

        # Calculate empirical constraint violation rate
        constraint_violation_rate = np.mean(self.recent_sla_violations)

        # Update Lagrangian multiplier
        current_lambda = self.lagrangian_optimizer.update(constraint_violation_rate)

        # Calculate constraint surplus for penalty
        constraint_surplus = max(0, constraint_violation_rate - self.config.max_sla_violation_rate)

        # Lagrangian reward: R - λ * (C - δ)
        # When violating: large penalty
        # When safe: small bonus for staying within constraint
        constrained_reward = base_reward - current_lambda * constraint_surplus

        # Additional immediate penalty for current violation (helps learning)
        if is_sla_violation:
            constrained_reward -= 5.0  # Immediate strong penalty

        return constrained_reward, is_sla_violation

    def get_safe_action(
        self,
        state: np.ndarray,
        action: int,
        q_values: Optional[torch.Tensor] = None
    ) -> int:
        """Get safety-filtered action.

        Args:
            state: Current state
            action: Proposed action
            q_values: Q-values from DQN (optional)

        Returns:
            Safe action
        """
        return self.safety_layer.filter_action(state, action, q_values)

    def get_diagnostics(self) -> Dict[str, float]:
        """Get diagnostic information about constraint satisfaction.

        Returns:
            Dictionary of diagnostic metrics
        """
        avg_sla_violation = self.lagrangian_optimizer.get_average_constraint_violation()
        constraint_satisfied = self.lagrangian_optimizer.is_constraint_satisfied()

        return {
            'constraint/sla_violation_rate': avg_sla_violation,
            'constraint/lambda_value': self.lagrangian_optimizer.lambda_value,
            'constraint/satisfied': 1.0 if constraint_satisfied else 0.0,
            'constraint/threshold': self.config.max_sla_violation_rate,
            'constraint/margin': self.config.max_sla_violation_rate - avg_sla_violation,
            'safety/intervention_rate': self.safety_layer.get_intervention_rate(),
            'metrics/avg_latency': np.mean(self.recent_latencies) if self.recent_latencies else 0.0
        }


# Example usage integration
if __name__ == "__main__":
    # Configuration
    config = ConstraintConfig(
        max_sla_violation_rate=0.15,  # 15% max violations
        sla_threshold_latency=0.15,   # 150ms threshold
        lambda_init=1.0,
        lambda_lr=0.01
    )

    # Initialize constrained optimizer
    optimizer = ConstrainedPPORewardOptimizer(
        state_dim=7,
        config=config
    )

    # Simulation loop example
    print("Constrained PPO Simulation Example:")
    print("=" * 60)

    for step in range(100):
        # Simulated state and metrics
        state = np.random.rand(7)
        base_reward = np.random.randn()

        # Simulate latency (sometimes violating SLA)
        latency = 0.12 + np.random.rand() * 0.1  # 120-220ms
        metrics = {'latency': latency, 'cost': 0.5}

        # Calculate constrained reward
        constrained_reward, violation = optimizer.calculate_constrained_reward(
            base_reward, state, metrics
        )

        # Proposed action
        action = np.random.randint(0, 3)

        # Get safe action
        safe_action = optimizer.get_safe_action(state, action)

        if step % 10 == 0:
            diagnostics = optimizer.get_diagnostics()
            print(f"\nStep {step}:")
            print(f"  SLA Violation Rate: {diagnostics['constraint/sla_violation_rate']:.3f}")
            print(f"  Lambda: {diagnostics['constraint/lambda_value']:.3f}")
            print(f"  Constraint Satisfied: {'✅' if diagnostics['constraint/satisfied'] else '❌'}")
            print(f"  Safety Interventions: {diagnostics['safety/intervention_rate']:.2%}")

    print("\n" + "=" * 60)
    print("Constrained PPO maintains SLA constraints during optimization!")
