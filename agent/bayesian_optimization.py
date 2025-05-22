import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm
from typing import Dict, List, Tuple, Any
import logging
import wandb

logger = logging.getLogger(__name__)

class BayesianOptimizer:
    def __init__(
        self,
        param_bounds: Dict[str, Tuple[float, float]],
        n_initial_points: int = 5,
        n_iterations: int = 20,
        acquisition_function: str = "ucb",
        kappa: float = 2.576,  # 99% confidence interval
        xi: float = 0.01,  # Exploration-exploitation trade-off
    ):
        """
        Initialize Bayesian Optimizer.
        
        Args:
            param_bounds: Dictionary of parameter names and their bounds (min, max)
            n_initial_points: Number of random points to evaluate before BO
            n_iterations: Number of BO iterations
            acquisition_function: Type of acquisition function ('ucb', 'ei', or 'pi')
            kappa: Exploration-exploitation trade-off parameter for UCB
            xi: Exploration-exploitation trade-off parameter for EI and PI
        """
        self.param_bounds = param_bounds
        self.n_initial_points = n_initial_points
        self.n_iterations = n_iterations
        self.acquisition_function = acquisition_function
        self.kappa = kappa
        self.xi = xi
        
        # Initialize Gaussian Process
        self.gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            normalize_y=True,
            n_restarts_optimizer=10,
            random_state=42
        )
        
        # Storage for observations
        self.X = []  # Parameter configurations
        self.y = []  # Objective values
        
        # Best observed value and configuration
        self.best_value = float('-inf')
        self.best_params = None
        
    def _normalize_params(self, params: Dict[str, float]) -> np.ndarray:
        """Normalize parameters to [0, 1] range."""
        normalized = []
        for param_name in self.param_bounds.keys():
            min_val, max_val = self.param_bounds[param_name]
            normalized.append((params[param_name] - min_val) / (max_val - min_val))
        return np.array(normalized)
    
    def _denormalize_params(self, normalized_params: np.ndarray) -> Dict[str, float]:
        """Convert normalized parameters back to original scale."""
        params = {}
        for i, (param_name, (min_val, max_val)) in enumerate(self.param_bounds.items()):
            params[param_name] = normalized_params[i] * (max_val - min_val) + min_val
        return params
    
    def _random_sample(self) -> Dict[str, float]:
        """Generate random parameter configuration."""
        params = {}
        for param_name, (min_val, max_val) in self.param_bounds.items():
            params[param_name] = np.random.uniform(min_val, max_val)
        return params
    
    def _acquisition_function(self, x: np.ndarray) -> float:
        """Calculate acquisition function value."""
        if len(self.X) == 0:
            return 0.0
            
        x = x.reshape(1, -1)
        mean, std = self.gp.predict(x, return_std=True)
        
        if self.acquisition_function == "ucb":
            return mean + self.kappa * std
        elif self.acquisition_function == "ei":
            # Expected Improvement
            best_f = max(self.y)
            z = (mean - best_f - self.xi) / std
            return (mean - best_f - self.xi) * norm.cdf(z) + std * norm.pdf(z)
        elif self.acquisition_function == "pi":
            # Probability of Improvement
            best_f = max(self.y)
            z = (mean - best_f - self.xi) / std
            return norm.cdf(z)
        else:
            raise ValueError(f"Unknown acquisition function: {self.acquisition_function}")
    
    def _optimize_acquisition(self) -> Dict[str, float]:
        """Find the next point to evaluate by optimizing the acquisition function."""
        best_acq = float('-inf')
        best_x = None
        
        # Random sampling for optimization
        n_samples = 1000
        for _ in range(n_samples):
            x = np.random.uniform(0, 1, size=len(self.param_bounds))
            acq = self._acquisition_function(x)
            
            if acq > best_acq:
                best_acq = acq
                best_x = x
                
        return self._denormalize_params(best_x)
    
    def suggest(self) -> Dict[str, float]:
        """Suggest next parameter configuration to evaluate."""
        if len(self.X) < self.n_initial_points:
            return self._random_sample()
        return self._optimize_acquisition()
    
    def update(self, params: Dict[str, float], value: float) -> None:
        """Update the model with new observation."""
        x = self._normalize_params(params)
        self.X.append(x)
        self.y.append(value)
        
        # Update best value and parameters
        if value > self.best_value:
            self.best_value = value
            self.best_params = params
            
        # Update GP model
        X = np.array(self.X)
        y = np.array(self.y)
        self.gp.fit(X, y)
        
        # Log to wandb
        wandb.log({
            "bo/iteration": len(self.X),
            "bo/objective_value": value,
            "bo/best_value": self.best_value,
            **{f"bo/param_{k}": v for k, v in params.items()}
        })
        
    def get_best_params(self) -> Tuple[Dict[str, float], float]:
        """Return the best parameters and their objective value."""
        return self.best_params, self.best_value 