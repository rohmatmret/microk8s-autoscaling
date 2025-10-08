import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Optional, Tuple, List
from collections import deque
import random
from datetime import datetime
import wandb
from dataclasses import dataclass
from agent.kubernetes_api import KubernetesAPI
from agent.bayesian_optimization import BayesianOptimizer
from agent.metrics_callback import AutoscalingMetricsCallback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("hybrid_agent.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

@dataclass
class HybridConfig:
    """Configuration for hybrid DQN-PPO agent.

    Optimized hyperparameters from Optuna optimization:
    - DQN: /best_dqn_params_latest.json (Best value: -4.0025, converged)
      Re-optimize: python -m agent.dqn_optimization --simulate --trials 20
    - PPO: /best_ppo_params_latest.json (Best value: -3.4400, converged)
      Re-optimize: python -m agent.ppo_optimization --simulate --trials 20

    To load optimized params dynamically:
      hybrid_agent = HybridDQNPPOAgent(config, k8s_api,
                                        load_optimized_params='best_dqn_params_latest.json')
    """
    # DQN Configuration - Using optimized parameters
    dqn_learning_rate: float = 0.0003320947521799842
    dqn_buffer_size: int = 140978
    dqn_batch_size: int = 256
    dqn_gamma: float = 0.9732108019691487
    dqn_tau: float = 0.0380978024125337
    dqn_epsilon_start: float = 1.0
    dqn_epsilon_end: float = 0.05  # Reduced from 0.1448 for less random exploration
    dqn_epsilon_decay: float = 0.995  # Faster decay from 0.999 for quicker convergence
    dqn_target_update_freq: int = 1049
    
    # PPO Configuration - Using optimized parameters
    ppo_learning_rate: float = 0.0001799454998195903
    ppo_n_steps: int = 3156
    ppo_batch_size: int = 256
    ppo_gamma: float = 0.9920186311406627
    ppo_gae_lambda: float = 0.9473714121408062
    ppo_clip_range: float = 0.19614066300807234
    ppo_ent_coef: float = 0.0020419888561576545
    ppo_vf_coef: float = 0.45570629287422937
    ppo_max_grad_norm: float = 0.9280948967769252
    ppo_n_epochs: int = 11
    
    # Hybrid Configuration
    reward_optimization_freq: int = 100
    batch_evaluation_steps: int = 50
    state_dim: int = 7  # cpu, memory, latency, swap, nodes, load_mean, throughput
    action_dim: int = 3
    hidden_dim: int = 64

class QNetwork(nn.Module):
    """Q-Network for DQN agent."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

class PPONetwork(nn.Module):
    """Actor-Critic network for PPO reward optimizer."""
    
    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super(PPONetwork, self).__init__()
        # Actor (policy) network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # Output reward modulation factor
            nn.Tanh()  # Bound between -1 and 1
        )
        
        # Critic (value) network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        return self.actor(x), self.critic(x)

class ReplayBuffer:
    """Experience replay buffer for DQN."""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (np.array(state), np.array(action), np.array(reward), 
                np.array(next_state), np.array(done))
        
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """DQN agent for discrete scaling actions."""
    
    def __init__(self, config: HybridConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.q_network = QNetwork(config.state_dim, config.action_dim, config.hidden_dim).to(self.device)
        self.target_network = QNetwork(config.state_dim, config.action_dim, config.hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.dqn_learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(config.dqn_buffer_size)
        
        # Exploration
        self.epsilon = config.dqn_epsilon_start
        self.steps = 0
        
        logger.info("DQN Agent initialized")
        
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using enhanced epsilon-greedy policy with adaptive exploration."""
        # Enhanced exploration for dynamic behavior
        if training and random.random() < self.epsilon:
            # Adaptive exploration based on CPU utilization and latency
            if len(state) >= 4:
                cpu_util = state[0]
                latency = state[2] if len(state) > 2 else 0.0
                # Bias exploration towards scaling actions when metrics indicate need
                if cpu_util > 0.8 or latency > 0.18:  # High CPU or approaching SLA violation
                    return random.choices([0, 1, 2], weights=[0.8, 0.1, 0.1])[0]  # 80% scale-up
                elif cpu_util < 0.4 and latency < 0.15:  # Low CPU and good latency
                    return random.choices([0, 1, 2], weights=[0.1, 0.7, 0.2])[0]  # 70% scale-down
            return random.randint(0, self.config.action_dim - 1)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def update(self, batch_size: int) -> float:
        """Update Q-network using experience replay."""
        if len(self.replay_buffer) < batch_size:
            return 0.0
            
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.config.dqn_gamma * next_q_values * ~dones)
        
        # Loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        if self.steps % self.config.dqn_target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            
        # Decay epsilon
        self.epsilon = max(self.config.dqn_epsilon_end, 
                          self.epsilon * self.config.dqn_epsilon_decay)
        
        self.steps += 1
        return loss.item()
    
    def save(self, path: str):
        """Save DQN model."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, path)
        
    def load(self, path: str):
        """Load DQN model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']

class PPORewardOptimizer:
    """PPO agent for optimizing reward signals."""
    
    def __init__(self, config: HybridConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Network
        self.network = PPONetwork(config.state_dim, config.hidden_dim).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=config.ppo_learning_rate)
        
        # Bayesian optimizer for reward function tuning
        # Define parameter bounds for dynamic scaling optimization with aggressive latency focus
        param_bounds = {
            'latency_weight': (0.8, 3.0),  # Much higher priority on latency (increased significantly)
            'cpu_weight': (0.2, 0.8),     # Higher CPU sensitivity
            'memory_weight': (0.05, 0.3),
            'cost_weight': (0.05, 0.4),   # Higher cost awareness
            'throughput_weight': (0.1, 0.5),  # More throughput focus
            'latency_threshold': (0.1, 0.25),  # Much tighter latency bounds (reduced from 0.15-0.4)
            'cpu_threshold': (0.4, 0.7),       # Aligned with HPA thresholds (30%-70%)
            'cost_threshold': (0.5, 0.9),      # More cost-sensitive
            'scaling_reward': (0.1, 1.0),      # Reward for appropriate scaling
            'stability_penalty': (0.1, 0.5)    # Penalty for excessive changes
        }
        self.bayesian_optimizer = BayesianOptimizer(param_bounds)
        
        # Metrics tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.latency_history = []
        self.throughput_history = []
        self.cost_history = []
        self.queue_length_history = []
        
        logger.info("PPO Reward Optimizer initialized")
        
    def optimize_reward(self, state: np.ndarray, base_reward: float, 
                       metrics: Dict[str, float]) -> float:
        """Optimize reward using PPO and Bayesian optimization."""
        # Get reward modulation from PPO network
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            reward_modulation, _ = self.network(state_tensor)
            reward_modulation = reward_modulation.item()
        
        # Bayesian optimization for reward function parameters
        # Use the suggest method to get optimized parameters
        reward_params = self.bayesian_optimizer.suggest()
        
        # Calculate optimized reward
        optimized_reward = self._calculate_optimized_reward(
            base_reward, reward_modulation, metrics, reward_params
        )
        
        return optimized_reward
    
    def _calculate_optimized_reward(self, base_reward: float, modulation: float,
                                  metrics: Dict[str, float], params: Dict[str, float]) -> float:
        """Calculate optimized reward using multiple factors with dynamic scaling focus."""
        latency = metrics.get('latency', 0.0)
        throughput = metrics.get('throughput', 0.0)
        cost = metrics.get('cost', 0.0)
        queue_length = metrics.get('queue_length', 0.0)

        # Base reward with modulation
        reward = base_reward * (1 + modulation * 0.5)  # Reduce modulation impact

        # Enhanced latency penalty (much more aggressive)
        latency_threshold = params.get('latency_threshold', 0.2)  # Lower default threshold
        latency_weight = params.get('latency_weight', 1.5)  # Higher default weight

        # Progressive penalty system for latency violations
        if latency > latency_threshold:
            violation_severity = (latency - latency_threshold) / latency_threshold
            # Exponential penalty for increasing latency violations
            latency_penalty = latency_weight * violation_severity * (1.0 + violation_severity ** 2) * 5.0
            reward -= latency_penalty

        # Additional severe penalty for critical latency violations
        if latency > 0.5:  # Critical latency threshold
            reward -= latency_weight * 10.0  # Large fixed penalty

        # Throughput bonus (encourage high throughput)
        throughput_weight = params.get('throughput_weight', 0.3)
        if throughput > 0.6:
            reward += throughput_weight * throughput

        # Cost penalty (scaled by usage)
        cost_threshold = params.get('cost_threshold', 0.7)
        cost_weight = params.get('cost_weight', 0.2)
        if cost > cost_threshold:
            reward -= cost_weight * (cost - cost_threshold) * 1.5

        # Queue length penalty (avoid backlogs)
        if queue_length > 0.3:
            reward -= 0.5 * queue_length

        # Dynamic scaling rewards (new parameters)
        scaling_reward = params.get('scaling_reward', 0.5)
        if 'action_appropriateness' in metrics:
            reward += scaling_reward * metrics['action_appropriateness']

        # Stability penalty for excessive oscillation
        stability_penalty = params.get('stability_penalty', 0.2)
        if 'scaling_frequency' in metrics and metrics['scaling_frequency'] > 0.8:
            reward -= stability_penalty * metrics['scaling_frequency']

        return reward
    
    def update(self, states: List[np.ndarray], actions: List[int], 
               rewards: List[float], next_states: List[np.ndarray], 
               dones: List[bool]) -> float:
        """Update PPO network using collected experience."""
        if len(states) < self.config.ppo_batch_size:
            return 0.0
            
        # Convert to tensors
        states_tensor = torch.FloatTensor(states).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        
        # Forward pass
        reward_modulations, values = self.network(states_tensor)
        
        # Calculate loss (simplified PPO loss)
        loss = nn.MSELoss()(values.squeeze(), rewards_tensor)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def update_metrics(self, episode_reward: float, episode_length: int,
                      metrics: Dict[str, float]):
        """Update metrics for reward optimization."""
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.latency_history.append(metrics.get('latency', 0.0))
        self.throughput_history.append(metrics.get('throughput', 0.0))
        self.cost_history.append(metrics.get('cost', 0.0))
        self.queue_length_history.append(metrics.get('queue_length', 0.0))
        
        # Keep only recent history
        max_history = 1000
        if len(self.episode_rewards) > max_history:
            self.episode_rewards = self.episode_rewards[-max_history:]
            self.episode_lengths = self.episode_lengths[-max_history:]
            self.latency_history = self.latency_history[-max_history:]
            self.throughput_history = self.throughput_history[-max_history:]
            self.cost_history = self.cost_history[-max_history:]
            self.queue_length_history = self.queue_length_history[-max_history:]
    
    def save(self, path: str):
        """Save PPO model."""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'latency_history': self.latency_history,
            'throughput_history': self.throughput_history,
            'cost_history': self.cost_history,
            'queue_length_history': self.queue_length_history
        }, path)
        
    def load(self, path: str):
        """Load PPO model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.latency_history = checkpoint.get('latency_history', [])
        self.throughput_history = checkpoint.get('throughput_history', [])
        self.cost_history = checkpoint.get('cost_history', [])
        self.queue_length_history = checkpoint.get('queue_length_history', [])

class HybridDQNPPOAgent:
    """Hybrid DQN-PPO agent for adaptive autoscaling."""

    def __init__(self, config: HybridConfig, k8s_api: KubernetesAPI, mock_mode: bool = False,
                 load_optimized_params: str = None):
        """
        Initialize Hybrid DQN-PPO agent.

        Args:
            config: Hybrid configuration
            k8s_api: Kubernetes API instance
            mock_mode: Whether to run in mock mode
            load_optimized_params: Path to JSON file with optimized DQN parameters
        """
        # Load optimized parameters if provided
        if load_optimized_params:
            import json
            try:
                with open(load_optimized_params, 'r') as f:
                    optimized_params = json.load(f)
                logger.info(f"Loading optimized DQN parameters from {load_optimized_params}")

                # Update DQN config parameters
                for key, value in optimized_params.items():
                    dqn_key = f"dqn_{key}"
                    if hasattr(config, dqn_key):
                        setattr(config, dqn_key, value)
                        logger.info(f"Updated {dqn_key}: {value}")
            except Exception as e:
                logger.error(f"Failed to load optimized parameters: {e}")

        self.config = config
        self.k8s_api = k8s_api
        self.mock_mode = mock_mode

        # Initialize agents
        self.dqn_agent = DQNAgent(config)
        self.ppo_optimizer = PPORewardOptimizer(config)

        # Training state
        self.total_steps = 0
        self.episode_count = 0
        self.current_episode_reward = 0.0
        self.current_episode_length = 0

        # Mock mode pod tracking - FIX: Properly track current pods
        self.mock_current_pods = 1  # Track actual pod count in mock mode

        # Complex traffic simulation support
        self.use_complex_traffic = False
        self.current_scenario = None
        self.scenario_step = 0

        # Metrics tracking
        self.metrics_callback = AutoscalingMetricsCallback()

        # Enhanced throughput tracking
        self.throughput_history = []
        self.request_history = []
        self.response_time_history = []
        self.success_rate_history = []
        self.bandwidth_history = []
        self.step_timestamps = []

        # Performance metrics
        self.current_rps = 0.0
        self.peak_rps = 0.0
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0

        # Initialize WandB
        self._init_wandb()
        
        logger.info("Hybrid DQN-PPO Agent initialized")
        
    def _init_wandb(self):
        """Initialize WandB for experiment tracking."""
        # Skip if already initialized (e.g., by wrapper)
        if wandb.run is not None:
            logger.info("WandB already initialized, skipping agent initialization")
            return

        try:
            wandb.init(
                project="microk8s_hybrid_autoscaling",
                name=f"hybrid_dqn_ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config={
                    "algorithm": "Hybrid DQN-PPO",
                    "dqn_learning_rate": self.config.dqn_learning_rate,
                    "ppo_learning_rate": self.config.ppo_learning_rate,
                    "dqn_buffer_size": self.config.dqn_buffer_size,
                    "ppo_n_steps": self.config.ppo_n_steps,
                    "reward_optimization_freq": self.config.reward_optimization_freq,
                    "batch_evaluation_steps": self.config.batch_evaluation_steps
                },
                tags=["hybrid", "autoscaling", "DQN", "PPO"],
                sync_tensorboard=True,
                mode="offline",  # Offline mode - sync later with: wandb sync ./wandb/run-xxx
                reinit=True  # Allow reinitialization
            )
        except Exception as e:
            logger.warning(f"WandB initialization failed: {e}")
    
    def get_state(self) -> np.ndarray:
        """Get current cluster state with throughput metric."""
        if self.mock_mode:
            # FIX: Maintain consistent state based on current pod count
            # Calculate realistic state based on load and pod count
            current_pods = self.mock_current_pods

            # Determine load source
            if self.use_complex_traffic and self.current_scenario:
                # Use complex traffic pattern
                base_load = self._generate_complex_traffic(self.current_scenario, self.scenario_step)
            else:
                # Use simple oscillating load (original behavior)
                import time
                time_factor = (time.time() % 3600) / 3600.0  # 0-1 over an hour
                base_load = 500 + 300 * np.sin(time_factor * 2 * np.pi)  # Oscillating load

            # Calculate metrics based on pod count and load
            pod_capacity = 200  # RPS per pod
            cpu_util = min(0.95, base_load / (current_pods * pod_capacity))
            cpu_util = max(0.1, cpu_util)  # Keep in range

            memory_util = 0.3 + cpu_util * 0.4  # Memory correlates with CPU
            latency = 0.05 + max(0, (cpu_util - 0.7) * 0.4)  # Latency increases when overloaded
            throughput = min(base_load, current_pods * pod_capacity) / 1000.0  # Normalized

            return np.array([
                cpu_util,                       # CPU utilization (realistic)
                memory_util,                    # Memory utilization (correlated)
                latency,                        # Latency (increases with load)
                cpu_util * 0.1,                 # Swap (minimal when healthy)
                current_pods / 10.0,            # Normalized pod count
                base_load / 1000.0,             # Load mean (normalized)
                throughput                      # Throughput (RPS normalized)
            ], dtype=np.float32)
        
        try:
            cluster_state = self.k8s_api.get_cluster_state()

            # Calculate throughput from history
            throughput = 0.0
            if len(self.throughput_history) > 0:
                throughput = np.mean(self.throughput_history[-6:]) / 1000.0  # Normalize by max RPS
                throughput = min(throughput, 1.0)

            state = np.array([
                min(cluster_state["cpu"] / 100.0, 1.0),
                min(cluster_state["memory"] / 1e9, 1.0),
                min(cluster_state["latency"] / 0.2, 1.0),
                min(cluster_state.get("swap", 0) / 200e6, 1.0),
                min(cluster_state.get("nodes", 1) / 10, 1.0),
                0.5,        # Load mean (placeholder - could be calculated from load history)
                throughput  # Throughput (from throughput_history)
            ], dtype=np.float32)
            return state
        except Exception as e:
            logger.error(f"Failed to get cluster state: {e}")
            return np.zeros(self.config.state_dim, dtype=np.float32)
    
    def execute_action(self, action: int) -> Dict[str, Any]:
        """Execute scaling action and return metrics with enhanced throughput tracking."""
        if self.mock_mode:
            # Mock action execution for testing - FIX: Use tracked pod count
            current_replicas = self.mock_current_pods
            if action == 0:  # Scale up
                new_replicas = min(current_replicas + 1, self.k8s_api.max_pods)
            elif action == 1:  # Scale down
                new_replicas = max(current_replicas - 1, 1)
            else:  # No change
                new_replicas = current_replicas

            # FIX: Update our tracked pod count
            self.mock_current_pods = new_replicas

            # Enhanced mock metrics with realistic throughput simulation
            latency = np.random.uniform(0.05, 0.3)

            # Simulate realistic throughput based on pod count and load
            base_rps_per_pod = 25.0  # Base RPS capacity per pod
            load_factor = np.random.uniform(0.7, 1.3)  # Variable load

            # Calculate throughput with scaling effects
            if new_replicas > current_replicas:  # Scaling up
                # Temporary performance dip during scaling
                throughput_factor = 0.8
            elif new_replicas < current_replicas:  # Scaling down
                # Potential overload
                throughput_factor = 1.2 if new_replicas >= 2 else 1.5
            else:
                throughput_factor = 1.0

            current_rps = (new_replicas * base_rps_per_pod * load_factor) / throughput_factor

            # Apply latency penalty to throughput
            if latency > 0.2:  # High latency reduces effective throughput
                current_rps *= (0.5 - (latency - 0.2))

            # Ensure reasonable bounds
            current_rps = max(5.0, min(current_rps, new_replicas * 50.0))

            # Update peak RPS tracking
            self.peak_rps = max(self.peak_rps, current_rps)
            self.current_rps = current_rps

            # Simulate request success rate
            success_rate = 0.99
            if latency > 0.25:  # High latency correlates with failures
                success_rate *= (1.0 - (latency - 0.25) * 2)
            if new_replicas < 2 and current_rps > 40:  # Overload scenario
                success_rate *= 0.9

            success_rate = max(0.5, min(success_rate, 1.0))

            # Calculate bandwidth (approximate)
            avg_response_size_kb = 1.2  # Average response size
            bandwidth_mbps = (current_rps * avg_response_size_kb * 8) / 1000  # Convert to Mbps

            # Store historical data
            import time
            current_time = time.time()
            self.step_timestamps.append(current_time)
            self.throughput_history.append(current_rps)
            self.response_time_history.append(latency)
            self.success_rate_history.append(success_rate)
            self.bandwidth_history.append(bandwidth_mbps)

            # Keep only recent history (last 100 steps)
            max_history = 100
            if len(self.throughput_history) > max_history:
                self.step_timestamps = self.step_timestamps[-max_history:]
                self.throughput_history = self.throughput_history[-max_history:]
                self.response_time_history = self.response_time_history[-max_history:]
                self.success_rate_history = self.success_rate_history[-max_history:]
                self.bandwidth_history = self.bandwidth_history[-max_history:]

            # Enhanced metrics with throughput data
            metrics = {
                'latency': latency,
                'throughput': current_rps / 100.0,  # Normalized for compatibility
                'cost': new_replicas / self.k8s_api.max_pods,
                'queue_length': np.random.uniform(0.0, 0.2),
                # New throughput-specific metrics
                'requests_per_second': current_rps,
                'peak_rps': self.peak_rps,
                'success_rate': success_rate,
                'bandwidth_mbps': bandwidth_mbps,
                'response_time_p95': latency * np.random.uniform(1.5, 2.5),
                'response_time_p99': latency * np.random.uniform(2.5, 4.0),
                'total_requests': self.total_requests + current_rps,
                'successful_requests': self.successful_requests + (current_rps * success_rate),
                'failed_requests': self.failed_requests + (current_rps * (1 - success_rate))
            }

            # Update counters
            self.total_requests += current_rps
            self.successful_requests += (current_rps * success_rate)
            self.failed_requests += (current_rps * (1 - success_rate))

            return {
                'action': action,
                'replicas': new_replicas,
                'metrics': metrics,
                'cluster_state': {},
                'throughput_data': {
                    'current_rps': current_rps,
                    'peak_rps': self.peak_rps,
                    'avg_rps_1min': np.mean(self.throughput_history[-6:]) if len(self.throughput_history) >= 6 else current_rps,
                    'avg_rps_5min': np.mean(self.throughput_history[-30:]) if len(self.throughput_history) >= 30 else current_rps,
                    'success_rate': success_rate,
                    'bandwidth_mbps': bandwidth_mbps
                }
            }
        
        try:
            current_replicas = self.k8s_api._get_current_replicas("nginx-deployment")
            
            if action == 0:  # Scale up
                new_replicas = min(current_replicas + 1, self.k8s_api.max_pods)
                self.k8s_api.safe_scale("nginx-deployment", new_replicas)
            elif action == 1:  # Scale down
                new_replicas = max(current_replicas - 1, 1)
                self.k8s_api.safe_scale("nginx-deployment", new_replicas)
            # action == 2: No change
            
            # Wait for scaling to take effect
            import time
            time.sleep(5)
            
            # Get updated metrics
            cluster_state = self.k8s_api.get_cluster_state()
            metrics = {
                'latency': cluster_state.get('latency', 0.0) / 0.2,  # Normalized
                'throughput': 1.0 - cluster_state.get('cpu', 0.0) / 100.0,  # Inverse of CPU
                'cost': cluster_state.get('pods', 0.0) / self.k8s_api.max_pods,
                'queue_length': max(0.0, cluster_state.get('latency', 0.0) - 0.1) / 0.1
            }
            
            return {
                'action': action,
                'replicas': new_replicas,
                'metrics': metrics,
                'cluster_state': cluster_state
            }
            
        except Exception as e:
            logger.error(f"Failed to execute action: {e}")
            try:
                current_replicas = self.k8s_api._get_current_replicas("nginx-deployment")
            except:
                current_replicas = 1
            return {
                'action': action,
                'replicas': current_replicas,
                'metrics': {'latency': 1.0, 'throughput': 0.0, 'cost': 1.0, 'queue_length': 1.0},
                'cluster_state': {}
            }
    
    def calculate_base_reward(self, prev_state: np.ndarray, current_state: np.ndarray) -> float:
        """
        Calculate base reward with balanced SLA and efficiency focus.

        Key objectives (equal priority):
        1. Meet SLA targets (latency < 150ms)
        2. Optimize CPU utilization (target 60-70%)
        3. Minimize cost (avoid over-provisioning)
        """
        # Ensure we have the right number of state dimensions
        if len(current_state) >= 4:
            cpu, memory, latency, pods = current_state[:4]
        else:
            # Fallback for smaller state spaces
            cpu, memory, latency, pods = current_state[0], current_state[1], current_state[2], 0.0

        if len(prev_state) >= 4:
            prev_cpu, prev_memory, prev_latency, prev_pods = prev_state[:4]
        else:
            prev_cpu, prev_memory, prev_latency, prev_pods = prev_state[0], prev_state[1], prev_state[2], 0.0

        reward = 0.0

        # 1. SLA Performance (PRIMARY OBJECTIVE - maintain quality)
        # SLA threshold: 200ms (0.2s) per research report
        SLA_THRESHOLD = 0.2

        if latency < 0.10:  # Excellent (50% below SLA)
            reward += 20.0  # Very strong reward for excellent performance
        elif latency < 0.15:  # Good (25% below SLA)
            reward += 15.0  # Strong positive reinforcement
        elif latency < SLA_THRESHOLD:  # Within SLA (0.15-0.2s)
            reward += 8.0  # Good reward for meeting SLA
        elif latency < 0.25:  # Minor SLA violation (0.2-0.25s)
            # Gentle penalty for minor violations (help learning)
            violation_magnitude = (latency - SLA_THRESHOLD) / 0.05  # Normalize to 0-1
            reward -= 10.0 * violation_magnitude  # Linear penalty: 0 to -10
        else:  # Severe SLA violation (>0.25s)
            # Stronger penalty for severe violations
            violation_severity = (latency - 0.25) / 0.25
            reward -= 15.0 + 10.0 * violation_severity  # -15 to -25

        # 2. CPU Efficiency (TERTIARY priority - much less important than SLA)
        # Target: 60-70% CPU utilization for optimal efficiency
        cpu_target = 0.65  # Optimal target
        cpu_error = abs(cpu - cpu_target)

        if cpu_error < 0.05:  # 60-70% range - PERFECT!
            reward += 3.0  # Small reward - don't over-emphasize efficiency
        elif cpu_error < 0.15:  # 50-80% range - Good enough
            reward += 1.0  # Acceptable range
        elif cpu < 0.30:  # Very low CPU utilization (wasteful)
            reward -= 2.0  # Small penalty - waste is better than SLA violations
        elif cpu > 0.85:  # Very high CPU (danger zone for SLA)
            reward -= 12.0  # STRONG penalty - high risk of SLA violation

        # 3. Cost Efficiency (discourage wasteful over-provisioning only)
        # Allow more pods if needed for SLA compliance
        if pods > 0.9:  # Using >90% of max pods (extreme)
            reward -= 5.0  # Moderate penalty - we might need this for SLA
        elif pods > 0.6 and cpu < 0.40 and latency < 0.15:  # Clear over-provisioning
            # Only penalize if we have many pods, low CPU, AND good latency
            waste_penalty = (pods - 0.6) * 5.0
            reward -= waste_penalty  # Gentle penalty for waste

        # 4. Optimal State Bonus (encourage SLA compliance primarily)
        # Reward excellent SLA compliance regardless of efficiency
        if latency < 0.15:  # Excellent latency is most important
            if 0.50 <= cpu <= 0.75:  # Reasonable CPU range
                reward += 10.0  # Big bonus for great SLA + reasonable efficiency
            else:
                reward += 5.0  # Still good reward for excellent SLA alone

        # 5. Dynamic Scaling Rewards (encourage smart proactive decisions)
        pod_change = pods - prev_pods

        # Reward proactive scaling up when approaching limits (before SLA violation)
        if pod_change > 0:
            if latency > 0.18:  # Proactive scaling before SLA violation (0.2)
                reward += 5.0  # Strong reward for preventing violations
            elif latency > 0.15 or cpu > 0.70:  # Early proactive scaling
                reward += 3.0  # Good proactive behavior
            elif cpu > 0.75:  # Scaling when CPU is high
                reward += 2.0

        # Reward scaling down when safe (low CPU + good SLA)
        elif pod_change < 0 and cpu < 0.50 and latency < 0.15:
            reward += 3.0  # Encourage efficient scale-down

        # Penalize harmful scaling decisions
        elif pod_change > 0 and cpu < 0.40 and latency < 0.15:
            # Scaling up when not needed (wasteful)
            reward -= 4.0  # ✅ Stronger penalty (was -1.5)
        elif pod_change < 0 and (cpu > 0.70 or latency > 0.18):
            # Risky scale-down
            reward -= 6.0  # Strong penalty for dangerous decisions

        # 6. Stability Bonus (reward steady efficient state)
        if abs(pod_change) == 0:
            if 0.60 <= cpu <= 0.70 and latency < 0.20:
                reward += 1.0  # Bonus for maintaining optimal state
            elif 0.40 <= cpu <= 0.80 and latency < 0.25:
                reward += 0.3  # Small bonus for acceptable steady state

        # 7. Extreme State Penalties
        if pods < 0.1:  # Too few pods - risk of failure
            reward -= 5.0  # ✅ Stronger penalty (was -1.0)
        elif pods > 0.95:  # At maximum capacity - very inefficient
            reward -= 6.0  # ✅ Much stronger penalty (was -1.0)

        return reward
    
    def step(self) -> Dict[str, Any]:
        """Execute one step of the hybrid agent."""
        # Get current state
        current_state = self.get_state()

        # Select action using DQN
        action = self.dqn_agent.select_action(current_state, training=True)

        # Execute action
        result = self.execute_action(action)

        # Get next state
        next_state = self.get_state()

        # Calculate base reward
        base_reward = self.calculate_base_reward(current_state, next_state)

        # PPO optimizes reward for adaptive learning
        optimized_reward = self.ppo_optimizer.optimize_reward(
            current_state, base_reward, result['metrics']
        )
        # Handle case where optimize_reward returns tuple (reward, value)
        if isinstance(optimized_reward, tuple):
            optimized_reward = optimized_reward[0]

        # HYBRID APPROACH: Blend base and optimized rewards for stability with adaptation
        # 70% stable base reward + 30% PPO-optimized reward
        dqn_reward = 0.7 * base_reward + 0.3 * optimized_reward

        # Store experience in DQN replay buffer with blended reward
        done = self._is_episode_done(next_state)
        self.dqn_agent.replay_buffer.push(
            current_state, action, dqn_reward, next_state, done  # Use blended reward
        )

        # Update episode tracking with optimized reward (for monitoring)
        self.current_episode_reward += optimized_reward
        self.current_episode_length += 1
        self.total_steps += 1

        # Update PPO metrics
        self.ppo_optimizer.update_metrics(
            self.current_episode_reward, self.current_episode_length, result['metrics']
        )

        # Log all rewards for comparison
        self._log_metrics(action, base_reward, optimized_reward, dqn_reward, result['metrics'])
        
        # Check if episode is done
        if done:
            self._end_episode()
        
        # Periodic updates
        if self.total_steps % 10 == 0:  # Update DQN every 10 steps
            dqn_loss = self.dqn_agent.update(self.config.dqn_batch_size)
            if dqn_loss > 0 and wandb.run is not None:
                try:
                    wandb.log({"dqn/loss": dqn_loss, "train/global_step": self.total_steps})
                except Exception as e:
                    logger.warning(f"Failed to log DQN loss to wandb: {e}")

        if self.total_steps % self.config.reward_optimization_freq == 0:
            # Fix PPO update call - remove next_states and dones parameters
            ppo_loss = self.ppo_optimizer.update(
                [current_state], [action], [optimized_reward], [0.0], [0.0]  # dummy values, log_probs
            )
            if ppo_loss > 0 and wandb.run is not None:
                try:
                    wandb.log({"ppo/loss": ppo_loss, "train/global_step": self.total_steps})
                except Exception as e:
                    logger.warning(f"Failed to log PPO loss to wandb: {e}")
        
        return {
            'state': current_state,
            'action': action,
            'reward': dqn_reward,  # Return blended reward used for training
            'base_reward': base_reward,  # Track base reward
            'optimized_reward': optimized_reward,  # Track optimized reward
            'next_state': next_state,
            'done': done,
            'metrics': result['metrics']
        }
    
    def _is_episode_done(self, state: np.ndarray) -> bool:
        """Check if episode should end."""
        return state[2] > 0.95 or state[3] >= 0.95  # High latency or max pods
    
    def _end_episode(self):
        """Handle episode end."""
        self.episode_count += 1

        # Log episode metrics
        if wandb.run is not None:
            try:
                wandb.log({
                    "episode/reward": self.current_episode_reward,
                    "episode/length": self.current_episode_length,
                    "episode/count": self.episode_count,
                    "train/global_step": self.total_steps
                })
            except Exception as e:
                logger.warning(f"Failed to log episode metrics to wandb: {e}")

        # Reset episode tracking
        self.current_episode_reward = 0.0
        self.current_episode_length = 0

        logger.info(f"Episode {self.episode_count} completed")
    
    def _log_metrics(self, action: int, base_reward: float, optimized_reward: float, dqn_reward: float, metrics: Dict[str, float]):
        """Log metrics to WandB with enhanced throughput tracking."""
        if wandb.run is None:
            return  # Skip logging if wandb not initialized

        try:
            wandb_metrics = {
                "actions/scale_up": 1 if action == 0 else 0,
                "actions/scale_down": 1 if action == 1 else 0,
                "actions/no_change": 1 if action == 2 else 0,
                "rewards/base": base_reward,  # Base reward from calculate_base_reward
                "rewards/optimized": optimized_reward,  # PPO-optimized reward
                "rewards/dqn_training": dqn_reward,  # Blended reward used for DQN training
                "rewards/ppo_delta": optimized_reward - base_reward,  # PPO contribution
                "rewards/blend_factor": 0.3,  # Track blend ratio
                "metrics/latency": metrics['latency'],
                "metrics/throughput": metrics['throughput'],
                "metrics/cost": metrics['cost'],
                "metrics/queue_length": metrics['queue_length'],
                "train/global_step": self.total_steps
            }

            # Add enhanced throughput metrics if available
            if 'requests_per_second' in metrics:
                wandb_metrics.update({
                    "throughput/requests_per_second": metrics['requests_per_second'],
                    "throughput/peak_rps": metrics['peak_rps'],
                    "throughput/success_rate": metrics['success_rate'],
                    "throughput/bandwidth_mbps": metrics['bandwidth_mbps'],
                    "performance/response_time_p95": metrics.get('response_time_p95', 0),
                    "performance/response_time_p99": metrics.get('response_time_p99', 0)
                })

            wandb.log(wandb_metrics)
        except Exception as e:
            logger.warning(f"Failed to log metrics to wandb: {e}")

    def export_throughput_metrics(self, output_dir: str = "monitoring", test_id: str = None):
        """Export throughput metrics to CSV files for analysis."""
        if not self.throughput_history:
            logger.warning("No throughput data to export")
            return

        import csv
        import os
        from datetime import datetime

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Generate test ID if not provided
        if test_id is None:
            test_id = f"hybrid_dqn_ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Export throughput metrics
        throughput_file = os.path.join(output_dir, f"throughput_metrics_{test_id}.csv")

        with open(throughput_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "agent_name", "instantaneous_rps", "avg_rps_1min",
                "avg_rps_5min", "peak_rps", "throughput_efficiency",
                "request_success_rate", "bandwidth_utilization_mbps"
            ])

            # Calculate CPU utilization (mock for now)
            cpu_utilization = self.mock_current_pods / self.k8s_api.max_pods * 100

            for i, timestamp in enumerate(self.step_timestamps):
                if i < len(self.throughput_history):
                    current_rps = self.throughput_history[i]
                    success_rate = self.success_rate_history[i] if i < len(self.success_rate_history) else 0.95
                    bandwidth = self.bandwidth_history[i] if i < len(self.bandwidth_history) else 0

                    # Calculate moving averages
                    start_1min = max(0, i - 5)  # Last 6 points (1 minute)
                    start_5min = max(0, i - 29)  # Last 30 points (5 minutes)

                    avg_rps_1min = np.mean(self.throughput_history[start_1min:i+1])
                    avg_rps_5min = np.mean(self.throughput_history[start_5min:i+1])

                    # Throughput efficiency (RPS per CPU%)
                    throughput_efficiency = current_rps / cpu_utilization if cpu_utilization > 0 else 0

                    writer.writerow([
                        int(timestamp), "hybrid_dqn_ppo", f"{current_rps:.1f}",
                        f"{avg_rps_1min:.1f}", f"{avg_rps_5min:.1f}", f"{self.peak_rps:.1f}",
                        f"{throughput_efficiency:.3f}", f"{success_rate:.4f}", f"{bandwidth:.2f}"
                    ])

        # Export Prometheus-style metrics
        prometheus_file = os.path.join(output_dir, f"prometheus_metrics_{test_id}.csv")

        with open(prometheus_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "requests_per_second", "total_requests", "successful_requests",
                "failed_requests", "response_time_p50", "response_time_p95",
                "response_time_p99", "bytes_transferred", "connection_rate"
            ])

            cumulative_requests = 0
            for i, timestamp in enumerate(self.step_timestamps):
                if i < len(self.throughput_history):
                    current_rps = self.throughput_history[i]
                    success_rate = self.success_rate_history[i] if i < len(self.success_rate_history) else 0.95
                    response_time = self.response_time_history[i] if i < len(self.response_time_history) else 0.1

                    # Calculate cumulative metrics
                    cumulative_requests += current_rps
                    successful_requests = int(cumulative_requests * success_rate)
                    failed_requests = int(cumulative_requests * (1 - success_rate))

                    # Response time percentiles
                    response_time_p50 = response_time
                    response_time_p95 = response_time * 2.0
                    response_time_p99 = response_time * 3.5

                    # Bytes transferred (estimate)
                    avg_response_size = 1200  # bytes
                    bytes_transferred = int(current_rps * avg_response_size)

                    writer.writerow([
                        int(timestamp), f"{current_rps:.1f}", int(cumulative_requests),
                        successful_requests, failed_requests, f"{response_time_p50:.3f}",
                        f"{response_time_p95:.3f}", f"{response_time_p99:.3f}",
                        bytes_transferred, f"{current_rps:.1f}"
                    ])

        logger.info(f"Throughput metrics exported:")
        logger.info(f"  📊 Throughput data: {throughput_file}")
        logger.info(f"  🌐 Prometheus data: {prometheus_file}")

        return throughput_file, prometheus_file
    
    def _generate_complex_traffic(self, scenario: dict, step: int) -> float:
        """Generate traffic load matching test patterns."""
        pattern = scenario['pattern']
        base = scenario['base_load']
        max_load = scenario['max_load']
        duration = scenario['duration_steps']

        if pattern == 'steady':
            return base + np.random.uniform(-10, 10)
        elif pattern == 'gradual':
            progress = min(step / duration, 1.0)
            return base + (max_load - base) * progress + np.random.uniform(-20, 20)
        elif pattern == 'spike':
            if step < duration * 0.3:
                return base + np.random.uniform(-10, 10)
            elif step < duration * 0.6:
                return max_load + np.random.uniform(-50, 50)
            else:
                return base + np.random.uniform(-10, 10)
        elif pattern == 'daily':
            hour = (step / duration) * 24
            if 9 <= hour <= 17:
                peak_factor = np.sin((hour - 9) / 8 * np.pi)
                return base + (max_load - base) * peak_factor + np.random.uniform(-30, 30)
            else:
                return base + np.random.uniform(-10, 10)
        return base

    def train(self, total_steps: int = 50000, use_complex_traffic: bool = False):
        """Train the hybrid agent.

        Args:
            total_steps: Number of training steps
            use_complex_traffic: If True, cycles through complex traffic scenarios
        """
        self.use_complex_traffic = use_complex_traffic

        # Define complex traffic scenarios
        traffic_scenarios = [
            {'name': 'baseline_steady', 'base_load': 100, 'max_load': 150,
             'duration_steps': 100, 'pattern': 'steady'},
            {'name': 'gradual_ramp', 'base_load': 100, 'max_load': 500,
             'duration_steps': 200, 'pattern': 'gradual'},
            {'name': 'sudden_spike', 'base_load': 100, 'max_load': 800,
             'duration_steps': 150, 'pattern': 'spike'},
            {'name': 'daily_pattern', 'base_load': 100, 'max_load': 600,
             'duration_steps': 432, 'pattern': 'daily'}
        ] if use_complex_traffic else []

        scenario_idx = 0

        logger.info(f"Starting training for {total_steps} steps")
        if use_complex_traffic:
            logger.info(f"Using complex traffic patterns: {[s['name'] for s in traffic_scenarios]}")

        try:
            while self.total_steps < total_steps:
                # Setup scenario if using complex traffic
                if use_complex_traffic:
                    self.current_scenario = traffic_scenarios[scenario_idx]
                    self.scenario_step = self.total_steps % self.current_scenario['duration_steps']

                    # Cycle to next scenario when current one completes
                    if self.scenario_step == 0 and self.total_steps > 0:
                        scenario_idx = (scenario_idx + 1) % len(traffic_scenarios)
                        if self.scenario_step == 0:
                            logger.info(f"Switching to scenario: {traffic_scenarios[scenario_idx]['name']}")

                step_result = self.step()

                # Periodic evaluation
                if self.total_steps % 1000 == 0:
                    self._evaluate()

                # Periodic checkpointing
                if self.total_steps % 5000 == 0:
                    self.save_models()

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        finally:
            self.save_models()

            # Export throughput metrics for analysis
            try:
                self.export_throughput_metrics()
                logger.info("Throughput metrics exported successfully")
            except Exception as e:
                logger.warning(f"Failed to export throughput metrics: {e}")

            wandb.finish()
    
    def _evaluate(self):
        """Evaluate agent performance."""
        logger.info(f"Evaluating at step {self.total_steps}")

        # Run evaluation episodes
        eval_rewards = []
        eval_base_rewards = []
        for _ in range(5):
            episode_reward = 0.0
            episode_base_reward = 0.0
            state = self.get_state()

            for _ in range(100):  # Max 100 steps per eval episode
                action = self.dqn_agent.select_action(state, training=False)
                result = self.execute_action(action)
                next_state = self.get_state()

                base_reward = self.calculate_base_reward(state, next_state)
                optimized_reward = self.ppo_optimizer.optimize_reward(
                    state, base_reward, result['metrics']
                )

                episode_reward += optimized_reward
                episode_base_reward += base_reward
                state = next_state

                if self._is_episode_done(state):
                    break

            eval_rewards.append(episode_reward)
            eval_base_rewards.append(episode_base_reward)
        
        avg_reward = np.mean(eval_rewards)
        avg_base_reward = np.mean(eval_base_rewards)

        if wandb.run is not None:
            try:
                wandb.log({
                    "eval/mean_reward": avg_reward,
                    "eval/mean_base_reward": avg_base_reward,  # DQN's actual learning signal
                    "eval/std_reward": np.std(eval_rewards),
                    "eval/std_base_reward": np.std(eval_base_rewards),
                    "train/global_step": self.total_steps
                })
            except Exception as e:
                logger.warning(f"Failed to log evaluation results to wandb: {e}")

        logger.info(f"Evaluation: mean reward = {avg_reward:.2f}, mean base reward = {avg_base_reward:.2f}")
    
    def save_models(self, path: str = "./models/hybrid"):
        """Save both DQN and PPO models."""
        try:
            os.makedirs(path, exist_ok=True)
            
            self.dqn_agent.save(f"{path}/dqn_model.pth")
            self.ppo_optimizer.save(f"{path}/ppo_model.pth")
            
            logger.info(f"Models saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save models: {e}")
            # Try saving to current directory as fallback
            try:
                self.dqn_agent.save("dqn_model.pth")
                self.ppo_optimizer.save("ppo_model.pth")
                logger.info("Models saved to current directory as fallback")
            except Exception as e2:
                logger.error(f"Failed to save models even to current directory: {e2}")
    
    def load_models(self, path: str = "./models/hybrid"):
        """Load both DQN and PPO models."""
        try:
            self.dqn_agent.load(f"{path}/dqn_model.pth")
            self.ppo_optimizer.load(f"{path}/ppo_model.pth")
            logger.info(f"Models loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
    
    def get_scaling_decision(self, state: np.ndarray) -> int:
        """Get scaling decision for current state (for inference)."""
        return self.dqn_agent.select_action(state, training=False)

    def step_with_external_state(self, external_state: np.ndarray) -> Dict[str, Any]:
        """Execute one step using externally provided state (for test simulation)."""
        # Use external state instead of get_state()
        current_state = external_state

        # Select action using DQN
        action = self.dqn_agent.select_action(current_state, training=True)

        # Execute action
        result = self.execute_action(action)

        # For reward calculation, we'll use current state and assume next state changes based on action
        # This is a simplified version for testing
        base_reward = self.calculate_base_reward(current_state, current_state)  # Simplified

        # Optimize reward using PPO
        optimized_reward = self.ppo_optimizer.optimize_reward(
            current_state, base_reward, result['metrics']
        )
        # Handle case where optimize_reward returns tuple (reward, value)
        if isinstance(optimized_reward, tuple):
            optimized_reward = optimized_reward[0]

        return {
            'state': current_state,
            'action': action,
            'reward': optimized_reward,
            'new_pods': self.mock_current_pods,  # Return new pod count to test simulation
            'metrics': result['metrics']
        } 