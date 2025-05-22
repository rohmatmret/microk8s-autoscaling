import optuna
import logging
from typing import Dict, Any, Type, List
import wandb
import pandas as pd
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

def _get_trial_data(study: optuna.Study) -> Dict:
    """Extract trial data in a JSON-serializable format."""
    trials_data = {
        "number": [],
        "value": [],
        "state": [],
        "datetime_start": [],
        "datetime_complete": [],
        "duration": [],
        "params": []
    }
    
    for trial in study.trials:
        trials_data["number"].append(trial.number)
        trials_data["value"].append(trial.value if trial.value is not None else None)
        trials_data["state"].append(trial.state.name)
        trials_data["datetime_start"].append(
            trial.datetime_start.strftime('%Y-%m-%d %H:%M:%S') if trial.datetime_start else None
        )
        trials_data["datetime_complete"].append(
            trial.datetime_complete.strftime('%Y-%m-%d %H:%M:%S') if trial.datetime_complete else None
        )
        trials_data["duration"].append(
            (trial.datetime_complete - trial.datetime_start).total_seconds() 
            if trial.datetime_complete and trial.datetime_start else None
        )
        trials_data["params"].append(trial.params)
    
    return trials_data

def objective(
    trial: optuna.Trial,
    env,
    agent_class: Type,
    eval_episodes: int,
    timesteps_per_trial: int
) -> float:
    """
    Objective function for Optuna optimization.
    
    Args:
        trial: Optuna trial object
        env: The environment to train on
        agent_class: The agent class to instantiate
        eval_episodes: Number of episodes for evaluation
        timesteps_per_trial: Number of timesteps to train per trial
        
    Returns:
        Average reward achieved
    """
    # Define hyperparameter search space
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "n_steps": trial.suggest_int("n_steps", 1024, 4096),
        "batch_size": trial.suggest_int("batch_size", 32, 256),
        "gamma": trial.suggest_float("gamma", 0.9, 0.999),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.9, 0.99),
        "clip_range": trial.suggest_float("clip_range", 0.1, 0.4),
        "ent_coef": trial.suggest_float("ent_coef", 0.01, 0.1),
        "vf_coef": trial.suggest_float("vf_coef", 0.3, 0.7),
    }
    
    # Create and train agent with these parameters
    agent = agent_class(
        environment=env,
        learning_rate=params["learning_rate"],
        n_steps=params["n_steps"],
        batch_size=params["batch_size"],
        gamma=params["gamma"]
    )
    
    # Train the agent
    agent.train(
        total_timesteps=timesteps_per_trial,
        eval_episodes=eval_episodes
    )
    
    # Evaluate the agent
    avg_reward = agent.evaluate(episodes=eval_episodes)
    
    # Log to wandb
    wandb.log({
        "optuna/trial": trial.number,
        "optuna/objective_value": avg_reward,
        **{f"optuna/param_{k}": v for k, v in params.items()}
    })
    
    return avg_reward

def optimize_hyperparameters(
    env,
    agent_class: Type,
    n_trials: int = 20,
    eval_episodes: int = 10,
    timesteps_per_trial: int = 10000,
    study_name: str = "ppo_optimization"
) -> Dict[str, Any]:
    """
    Optimize hyperparameters using Optuna.
    
    Args:
        env: The environment to train on
        agent_class: The agent class to optimize
        n_trials: Number of optimization trials
        eval_episodes: Number of episodes for evaluation
        timesteps_per_trial: Number of timesteps to train per trial
        study_name: Name of the Optuna study
        
    Returns:
        Dictionary of best hyperparameters
    """
    # Create or load study
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        load_if_exists=True
    )
    
    # Run optimization
    study.optimize(
        lambda trial: objective(trial, env, agent_class, eval_episodes, timesteps_per_trial),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    # Get best parameters
    best_params = study.best_params
    best_value = study.best_value
    
    # Log final results
    logger.info(f"Best parameters found: {best_params}")
    logger.info(f"Best value achieved: {best_value:.2f}")
    
    # Log optimization history to wandb
    try:
        trials_data = _get_trial_data(study)
        
        wandb.log({
            "optuna/best_value": best_value,
            "optuna/best_params": best_params,
            "optuna/optimization_history": trials_data
        })
    except Exception as e:
        logger.error(f"Error logging optimization history to wandb: {str(e)}")
    
    return best_params 