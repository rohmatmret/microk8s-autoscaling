import optuna
import logging
from typing import Dict, Any
import wandb
import numpy as np
from datetime import datetime
from agent.dqn import DQNAgent
from agent.environment_simulated import MicroK8sEnvSimulated
from agent.environment import MicroK8sEnv

logger = logging.getLogger(__name__)

def objective_dqn(
    trial: optuna.Trial,
    env,
    eval_episodes: int = 20,
    timesteps_per_trial: int = 50000
) -> float:
    """
    Objective function for DQN hyperparameter optimization.

    Args:
        trial: Optuna trial object
        env: The environment to train on
        eval_episodes: Number of episodes for evaluation
        timesteps_per_trial: Number of timesteps to train per trial

    Returns:
        Average reward achieved
    """
    # Define DQN-specific hyperparameter search space
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "buffer_size": trial.suggest_int("buffer_size", 50000, 200000),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        "gamma": trial.suggest_float("gamma", 0.95, 0.999),
        "tau": trial.suggest_float("tau", 0.001, 0.1),
        "train_freq": trial.suggest_categorical("train_freq", [1, 4, 8]),
        "gradient_steps": trial.suggest_int("gradient_steps", 1, 4),
        "learning_starts": trial.suggest_int("learning_starts", 1000, 10000),
        "target_update_interval": trial.suggest_int("target_update_interval", 1000, 5000),
        "exploration_fraction": trial.suggest_float("exploration_fraction", 0.1, 0.5),
        "exploration_final_eps": trial.suggest_float("exploration_final_eps", 0.01, 0.2),
        "max_grad_norm": trial.suggest_int("max_grad_norm", 5, 20),
        "net_arch_size": trial.suggest_categorical("net_arch_size", [32, 64, 128]),
        "net_arch_layers": trial.suggest_int("net_arch_layers", 1, 3)
    }

    try:
        # Test environment first
        try:
            test_obs = env.reset()
            test_action = env.action_space.sample()
            env.step(test_action)
            logger.debug(f"Trial {trial.number}: Environment test passed")
        except Exception as env_error:
            logger.error(f"Trial {trial.number}: Environment test failed: {env_error}")
            return -1000.0

        # Create agent with optimized parameters
        logger.info(f"Trial {trial.number}: Creating DQN agent with params: {params}")
        agent = DQNAgent(
            env=env,
            environment=env,
            is_simulated=True,
            **params
        )
        logger.info(f"Trial {trial.number}: DQN agent created successfully")

        # Train the agent with error handling
        training_successful = False
        try:
            agent.train(
                total_timesteps=timesteps_per_trial,
                eval_episodes=eval_episodes
            )
            training_successful = True
            logger.info(f"Trial {trial.number}: Training completed successfully")
        except Exception as train_error:
            logger.warning(f"Training failed for trial {trial.number}: {train_error}")
            logger.warning(f"Trying with reduced timesteps...")
            try:
                agent.train(
                    total_timesteps=timesteps_per_trial // 4,
                    eval_episodes=max(1, eval_episodes // 2)
                )
                training_successful = True
                logger.info(f"Trial {trial.number}: Reduced training completed successfully")
            except Exception as second_train_error:
                logger.error(f"Second training attempt failed for trial {trial.number}: {second_train_error}")
                training_successful = False

        # Only evaluate if training was successful
        if training_successful:
            try:
                avg_reward = agent.evaluate(episodes=eval_episodes)
                logger.info(f"Trial {trial.number}: Evaluation successful, reward = {avg_reward:.4f}")
            except Exception as eval_error:
                logger.warning(f"Evaluation failed for trial {trial.number}: {eval_error}")
                try:
                    # Try with fewer episodes
                    avg_reward = agent.evaluate(episodes=max(1, eval_episodes // 2))
                    logger.info(f"Trial {trial.number}: Reduced evaluation successful, reward = {avg_reward:.4f}")
                except Exception as second_eval_error:
                    logger.error(f"Second evaluation attempt failed for trial {trial.number}: {second_eval_error}")
                    avg_reward = -1000.0
        else:
            logger.error(f"Trial {trial.number}: Skipping evaluation due to training failure")
            avg_reward = -1000.0

        # Validate reward value
        if avg_reward is None or np.isnan(avg_reward) or np.isinf(avg_reward):
            logger.warning(f"Invalid reward {avg_reward} for trial {trial.number}")
            avg_reward = -1000.0  # Penalty for invalid rewards

        # Convert to Python float for JSON serialization
        avg_reward = float(avg_reward)

        # Log to wandb
        wandb.log({
            "optuna/trial": trial.number,
            "optuna/objective_value": avg_reward,
            **{f"optuna/param_{k}": v for k, v in params.items()}
        })

        # Store additional metrics for convergence analysis (ensure JSON serializable)
        trial.set_user_attr("convergence_score", avg_reward)
        trial.set_user_attr("training_timesteps", int(timesteps_per_trial))

        logger.info(f"Trial {trial.number} completed with reward: {avg_reward:.4f}")
        return avg_reward

    except Exception as e:
        logger.error(f"Trial {trial.number} failed completely: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return -1000.0  # Return large negative value instead of -inf

def optimize_dqn_hyperparameters(
    simulate: bool = True,
    n_trials: int = 20,  # Reasonable default for full optimization
    eval_episodes: int = 20,
    timesteps_per_trial: int = 50000,
    study_name: str = None,
    wandb_mode: str = "offline",  # Default to offline mode
    auto_switch_threshold: float = 0.02,  # Switch to online for last 2% of trials
    disable_early_stop: bool = False,  # Disable early stopping
    traffic_seed: int = 42,  # Fixed seed for consistent traffic patterns across trials
    storage: str = None,  # SQLite storage URL for persistence
    resume: bool = True  # Whether to resume existing study
) -> Dict[str, Any]:
    """
    Optimize DQN hyperparameters using Optuna with convergence monitoring and persistence.

    Args:
        simulate: Whether to use simulated environment
        n_trials: Number of optimization trials
        eval_episodes: Number of episodes for evaluation
        timesteps_per_trial: Number of timesteps to train per trial
        study_name: Name of the Optuna study
        traffic_seed: Seed for traffic simulator (ensures consistent patterns across trials)
        storage: SQLite database URL (e.g., 'sqlite:///optuna_studies.db') for persistence
        resume: Resume existing study instead of creating new one

    Returns:
        Dictionary containing best hyperparameters and convergence info
    """
    # Use static study name by default for easy resumption
    if study_name is None:
        study_name = "dqn_optimization"

    # Setup persistent storage
    if storage is None:
        storage = "sqlite:///optuna_dqn_studies.db"

    logger.info(f"Using storage: {storage}")
    logger.info(f"Study name: {study_name}")

    # Initialize environment with fixed seed for reproducible traffic patterns
    env = MicroK8sEnvSimulated(seed=traffic_seed) if simulate else MicroK8sEnv()

    # Initialize wandb for optimization tracking
    wandb.init(
        project="dqn_hyperparameter_optimization",
        name=study_name,
        config={
            "algorithm": "DQN",
            "optimization_method": "Optuna",
            "n_trials": n_trials,
            "timesteps_per_trial": timesteps_per_trial,
            "eval_episodes": eval_episodes,
            "environment": "simulated" if simulate else "real",
            "traffic_seed": traffic_seed  # Log seed for reproducibility
        },
        mode=wandb_mode  # Support offline mode
    )

    # Create or load study with persistence and pruning
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,  # ✅ Add persistent storage
        direction="maximize",
        load_if_exists=True,  # ✅ Resume if exists
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
            interval_steps=5
        )
    )

    # Check if resuming existing study
    n_existing_trials = len(study.trials)
    if n_existing_trials > 0:
        logger.info(f"📊 Resuming study with {n_existing_trials} existing trials")
        logger.info(f"🎯 Current best value: {study.best_value:.4f}")
        logger.info(f"📋 Best params so far: {study.best_params}")

        # Save current best parameters before resuming
        try:
            import json
            best_params_file = f"best_dqn_params_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(best_params_file, 'w') as f:
                json.dump(study.best_params, f, indent=2)
            logger.info(f"💾 Saved current best parameters to {best_params_file}")

            # Update symlink
            import os
            latest_file = "best_dqn_params_latest.json"
            if os.path.exists(latest_file):
                os.remove(latest_file)
            os.symlink(best_params_file, latest_file)
            logger.info(f"🔗 Updated {latest_file} symlink")
        except Exception as e:
            logger.warning(f"Failed to save current best parameters: {e}")

        if resume:
            logger.info(f"▶️  Continuing optimization for {n_trials} more trials")
        else:
            logger.warning(f"⚠️  Study '{study_name}' already exists!")
            logger.warning(f"   Use --resume to continue, or change --study-name to start fresh")
            response = input("Continue with existing study? (y/n): ")
            if response.lower() != 'y':
                logger.info("Aborted by user")
                return {
                    "best_params": study.best_params,
                    "best_value": study.best_value,
                    "convergence_info": {"converged": False, "trials_to_convergence": n_existing_trials},
                    "study": study,
                    "params_file": best_params_file
                }
    else:
        logger.info(f"🆕 Starting new optimization study: {study_name}")

    # Set early stopping flag in study
    study.set_user_attr('disable_early_stop', disable_early_stop)

    # Convergence tracking - adjusted for fewer trials
    convergence_threshold = 0.05  # 5% improvement threshold (more lenient)
    patience = 3  # Number of trials without improvement (reduced from 10)
    no_improvement_count = 0
    best_value_history = []

    def callback(study, trial):
        nonlocal no_improvement_count, best_value_history

        current_best = study.best_value
        best_value_history.append(current_best)

        # Auto-switch to online mode for final trials OR when convergence detected
        online_threshold = max(1, int(n_trials * auto_switch_threshold))  # At least 1 trial or specified % of total
        trials_remaining = n_trials - (trial.number + 1)

        # Check if we need to switch to online mode
        should_switch = False
        switch_reason = ""

        if wandb_mode == "offline":
            # Switch if approaching final trials
            if trials_remaining <= online_threshold:
                should_switch = True
                switch_reason = f"final_{trials_remaining + 1}_trials"
                percentage_complete = ((trial.number + 1) / n_trials) * 100
                logger.info(f"🔄 Auto-switching to ONLINE mode for final {trials_remaining + 1} trials")
                logger.info(f"📊 Progress: {trial.number + 1}/{n_trials} trials ({percentage_complete:.1f}% complete)")

            # Also switch if convergence is detected (for better debugging)
            elif no_improvement_count >= patience - 1:  # One trial before stopping
                should_switch = True
                switch_reason = "convergence_detected"
                logger.info(f"🔄 Auto-switching to ONLINE mode due to convergence detection")
                logger.info(f"🔍 This will help debug why convergence occurred early")

        if should_switch:
            try:
                # Finish current offline run
                wandb.finish()

                # Reinitialize in online mode
                wandb.init(
                    project="dqn_hyperparameter_optimization",
                    name=f"{study_name}_{switch_reason}",
                    config={
                        "algorithm": "DQN",
                        "optimization_method": "Optuna",
                        "n_trials": n_trials,
                        "timesteps_per_trial": timesteps_per_trial,
                        "eval_episodes": eval_episodes,
                        "environment": "simulated" if simulate else "real",
                        "mode_switch": f"offline_to_online_at_trial_{trial.number + 1}",
                        "switch_reason": switch_reason
                    },
                    mode="online",
                    tags=["auto_switched", switch_reason],
                    resume="allow"
                )
                logger.info("✅ Successfully switched to online mode")
            except Exception as e:
                logger.warning(f"Failed to switch to online mode: {e}, continuing offline")

        # Check for convergence with early stopping for short runs
        # Only check convergence if we have valid (non-failed) trials
        valid_values = [v for v in best_value_history if v > -500]  # Filter out failed trials

        if len(best_value_history) >= patience and len(valid_values) >= 2:
            recent_improvement = (
                best_value_history[-1] - best_value_history[-patience]
            ) / abs(best_value_history[-patience]) if best_value_history[-patience] != 0 else 0

            # Only count as no improvement if we have valid trials
            if current_best > -500:  # Valid trial result
                if recent_improvement < convergence_threshold:
                    no_improvement_count += 1
                    logger.info(f"No significant improvement for {no_improvement_count} trials (valid results)")
                else:
                    no_improvement_count = 0
            else:
                # If current trial failed, don't count towards convergence
                logger.warning(f"Trial {trial.number} failed (value: {current_best}), not counting towards convergence")

        # If too many consecutive failures, that's a different kind of convergence issue
        recent_failures = sum(1 for v in best_value_history[-patience:] if v <= -500)
        if recent_failures >= patience:
            logger.error(f"🚨 {recent_failures} consecutive trial failures detected!")
            logger.error(f"💡 Switching to online mode for detailed debugging")
            if wandb_mode == "offline" and not should_switch:
                should_switch = True
                switch_reason = "consecutive_failures"
                # Force switch to online for debugging
                try:
                    wandb.finish()
                    wandb.init(
                        project="dqn_hyperparameter_optimization",
                        name=f"{study_name}_debug_failures",
                        config={"debug_mode": True, "consecutive_failures": recent_failures},
                        mode="online",
                        tags=["debug", "failures"]
                    )
                    logger.info("✅ Switched to online mode for failure debugging")
                except Exception as e:
                    logger.warning(f"Failed to switch for debugging: {e}")

        # Log convergence metrics (works for both online and offline)
        if wandb.run is not None:
            try:
                wandb.log({
                    "convergence/best_value": current_best,
                    "convergence/trial_number": trial.number,
                    "convergence/improvement_rate": recent_improvement if len(best_value_history) >= patience else 0,
                    "convergence/no_improvement_count": no_improvement_count,
                    "convergence/trials_remaining": trials_remaining,
                    "convergence/mode": "online" if wandb.run.mode != "offline" else "offline"
                })
            except Exception as e:
                logger.debug(f"Logging failed (normal in offline mode): {e}")

        # Early stopping only if truly converged with valid results (unless disabled)
        if not getattr(study.user_attrs, 'disable_early_stop', False):
            if no_improvement_count >= patience and len(valid_values) >= patience:
                logger.info(f"🛑 Stopping optimization after {trial.number + 1} trials (reason: true_convergence)")
                logger.info(f"✅ Converged with valid results: best = {max(valid_values):.4f}")
                study.stop()
            elif recent_failures >= patience and trial.number >= 2 * patience:
                logger.info(f"🛑 Stopping optimization after {trial.number + 1} trials (reason: consecutive_failures)")
                logger.error(f"❌ Unable to find working parameters after {recent_failures} failures")
                study.stop()

    # Run optimization with convergence monitoring
    try:
        study.optimize(
            lambda trial: objective_dqn(trial, env, eval_episodes, timesteps_per_trial),
            n_trials=n_trials,
            callbacks=[callback],
            show_progress_bar=True
        )
    except KeyboardInterrupt:
        logger.info("Optimization interrupted by user")

    # Get best parameters and convergence info
    best_params = study.best_params
    best_value = study.best_value
    n_completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])

    convergence_info = {
        "converged": no_improvement_count >= patience,
        "trials_to_convergence": n_completed_trials,
        "final_best_value": best_value,
        "improvement_history": best_value_history,
        "total_trials": len(study.trials)
    }

    # Log final results
    logger.info(f"Optimization completed:")
    logger.info(f"  Best parameters: {best_params}")
    logger.info(f"  Best value: {best_value:.4f}")
    logger.info(f"  Converged: {convergence_info['converged']}")
    logger.info(f"  Trials completed: {n_completed_trials}")

    # Final wandb logging (only if wandb is still active)
    if wandb.run is not None:
        try:
            wandb.log({
                "final/best_value": best_value,
                "final/best_params": best_params,
                "final/converged": convergence_info['converged'],
                "final/trials_completed": n_completed_trials
            })
        except Exception as e:
            logger.debug(f"Failed to log final metrics to wandb: {e}")

    # Save best parameters to JSON file for easy loading
    best_params_file = f"best_dqn_params_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    try:
        import json
        with open(best_params_file, 'w') as f:
            json.dump(best_params, f, indent=2)
        logger.info(f"Best parameters saved to {best_params_file}")

        # Also create a latest symlink
        import os
        latest_file = "best_dqn_params_latest.json"
        if os.path.exists(latest_file):
            os.remove(latest_file)
        os.symlink(best_params_file, latest_file)
        logger.info(f"Latest parameters available at {latest_file}")

    except Exception as e:
        logger.error(f"Failed to save best parameters: {e}")

    # Save optimization results
    results = {
        "best_params": best_params,
        "best_value": best_value,
        "convergence_info": convergence_info,
        "study": study,
        "params_file": best_params_file
    }

    wandb.finish()
    return results

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='DQN Hyperparameter Optimization')
    parser.add_argument('--simulate', action='store_true', help='Use simulated environment')
    parser.add_argument('--trials', type=int, default=20, help='Number of optimization trials (default: 20)')
    parser.add_argument('--eval-episodes', type=int, default=20, help='Evaluation episodes per trial')
    parser.add_argument('--timesteps-per-trial', type=int, default=10000, help='Training timesteps per trial (default: 10000)')
    parser.add_argument('--study-name', type=str, help='Optuna study name')
    parser.add_argument('--wandb-mode', type=str, default='offline', choices=['online', 'offline'], help='WandB mode (default: offline)')
    parser.add_argument('--auto-switch-threshold', type=float, default=0.02, help='Auto-switch to online for last X%% of trials (default: 0.02 = 2%%)')
    parser.add_argument('--disable-auto-switch', action='store_true', help='Disable automatic switching to online mode')
    parser.add_argument('--disable-early-stop', action='store_true', help='Disable early stopping (run all trials regardless of convergence)')
    parser.add_argument('--traffic-seed', type=int, default=42, help='Random seed for traffic simulator (default: 42)')
    parser.add_argument('--storage', type=str, default='sqlite:///optuna_dqn_studies.db', help='Optuna storage URL (default: sqlite:///optuna_dqn_studies.db)')
    parser.add_argument('--resume', action='store_true', help='Resume existing study without prompting')
    parser.add_argument('--list-studies', action='store_true', help='List all studies in the database and exit')

    args = parser.parse_args()

    # List studies mode
    if args.list_studies:
        import optuna
        import sys
        try:
            study_summaries = optuna.get_all_study_summaries(storage=args.storage)
            if not study_summaries:
                print(f"No studies found in {args.storage}")
            else:
                print(f"\n📚 Studies in {args.storage}:\n")
                for summary in study_summaries:
                    print(f"  📊 {summary.study_name}")
                    print(f"     Trials: {summary.n_trials}")
                    print(f"     Best value: {summary.best_trial.value if summary.best_trial else 'N/A'}")
                    print(f"     Created: {summary.datetime_start}")
                    print()
        except Exception as e:
            print(f"Error listing studies: {e}")
        sys.exit(0) 

    # Adjust auto-switch threshold if disabled
    auto_switch = 0.0 if args.disable_auto_switch else args.auto_switch_threshold

    results = optimize_dqn_hyperparameters(
        simulate=args.simulate,
        n_trials=args.trials,
        eval_episodes=args.eval_episodes,
        timesteps_per_trial=args.timesteps_per_trial,
        study_name=args.study_name,
        wandb_mode=args.wandb_mode,
        auto_switch_threshold=auto_switch,
        disable_early_stop=args.disable_early_stop,
        traffic_seed=args.traffic_seed,
        storage=args.storage,
        resume=args.resume
    )

    print(f"Best parameters found: {results['best_params']}")
    print(f"Best value: {results['best_value']:.4f}")
    print(f"Converged: {results['convergence_info']['converged']}")