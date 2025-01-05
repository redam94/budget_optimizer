"""Optimizers for models"""

# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/00_optimizer.ipynb.

# %% auto 0
__all__ = ['BaseOptimizer', 'ScipyBudgetOptimizer', 'OptunaBudgetOptimizer']

# %% ../nbs/00_optimizer.ipynb 5
import numpy as np
import xarray as xr
import pandas as pd
import scipy.optimize as opt
import optuna

from .utils.model_classes import BaseBudgetModel
from budget_optimizer.utils.model_helpers import (
  load_module,
  load_yaml,
  BudgetType, 
  AbstractModel
)

from pathlib import Path
from abc import ABC, abstractmethod

# %% ../nbs/00_optimizer.ipynb 6
class BaseOptimizer(ABC):
    """Optimizer wrapper for the pyswarms package"""
    _CONFIG_YAML = 'optimizer_config.yaml'
    _MODULE_FILE = "optimizer_config.py"
    
    def __init__(
        self, 
        model: BaseBudgetModel, # The model to optimize
        config_path: str|Path # Path to the configuration files
        ):
        
        self.model: BaseBudgetModel = model
        self._config_path: Path = Path(config_path) if isinstance(config_path, str) else config_path
        self.optimal_budget: BudgetType = None
        self.optimal_prediction: xr.DataArray = None
        self.optimal_contribution: xr.Dataset = None
        self.sol = None
        self._config = self._load_config()
        self._loss_fn = self._load_loss_fn()
        self._optimizer_array_to_budget = self._load_optimizer_array_to_budget()
        
    def _load_config(self):
        config = load_yaml(self._config_path / self._CONFIG_YAML)
        return config
    
    def reload_config(self):
        self._config = self._load_config()
        return self
    
    def _load_loss_fn(self):
        """Load the loss function from the config file"""
        module = load_module(self._MODULE_FILE.replace(".py", ""), self._config_path / self._MODULE_FILE)
        return module.loss_fn
    
    def _load_optimizer_array_to_budget(self):
        """Convert the optimizer array to a budget"""
        module = load_module(self._MODULE_FILE.replace(".py", ""), self._config_path / self._MODULE_FILE)
        return module.optimizer_array_to_budget
    
    def _optimizer_fn(self, x: np.ndarray):
        """Optimizer step"""
        budget = self._optimizer_array_to_budget(x)
        prediction = self.model.predict(budget)
        loss = self._loss_fn(prediction, **self._config['loss_fn_kwargs'])
        return loss
    
    @abstractmethod
    def optimize(
        self, 
        bounds: list[tuple[float, float]], # Bounds for the optimizer
        constraints: None|opt.LinearConstraint|tuple[float, float] = None, # Constraints for the optimizer
        **kwargs: dict # Additional arguments for the optimizer
        ):
        """Optimize the model"""
        raise NotImplementedError("This method should be implemented in the child class")

# %% ../nbs/00_optimizer.ipynb 7
class ScipyBudgetOptimizer(BaseOptimizer):
    """Optimizer wrapper for the pyswarms package"""
    
    def optimize(
        self, 
        bounds: list[tuple[float, float]], # Bounds for the optimizer
        constraints: None|opt.LinearConstraint, # Constraints for the optimizer
        init_pos: np.ndarray, # Initial position of the optimizer
        ):
        """Optimize the model"""
        import warnings
        warnings.filterwarnings("ignore")
        self.sol = opt.minimize(
            self._optimizer_fn, init_pos,
            method='trust-constr', 
            bounds=bounds, 
            constraints=constraints
            )
        if not self.sol.success:
            raise Exception(f"Optimization failed: {self.sol.message}")
        
        self.optimal_budget = self._optimizer_array_to_budget(self.sol.x)
        self.optimal_prediction = self.model.predict(self.optimal_budget) # The optimizer minimizes the cost, so we need to negate it
        self.optimal_contribution = self.model.contributions(self.optimal_budget)
        return self

# %% ../nbs/00_optimizer.ipynb 15
#| echo: false
#| echo: false
#| echo: false
from typing import Literal

# %% ../nbs/00_optimizer.ipynb 16
class OptunaBudgetOptimizer(BaseOptimizer):
    def __init__(
        self, 
        model: BaseBudgetModel, # The model to optimize
        config_path: str|Path, # Path to the configuration files
        objective_name: str = "loss", # Name of the objective
        direction: Literal["maximize", "minimize"] = "maximize", # Direction of the optimization
        sampler: optuna.samplers.BaseSampler = optuna.samplers.TPESampler, # Sampler for the optimization
        pruner: optuna.pruners.BasePruner|None = None, # Pruner for the optimization
        tol: float = 1e-3, # Tolerance for the constraints
        percent_out_tolerance: float = 0.1, # Percentage of the budget trials that can be outside the constraints
        sampler_kwargs: dict|None = None, # Additional arguments for the sampler
        pruner_kwargs: dict|None = None, # Additional arguments for the pruner
        ):
        super().__init__(model, config_path)
        self.objective_name = objective_name
        self.study = None
        self._direction = direction
        self.__tol = tol
        self.__percent_out_tolerance = percent_out_tolerance
        self.__sampler = sampler(constraints_func=self._constraints, **sampler_kwargs)
        self.__pruner = pruner(**pruner_kwargs) if not pruner is None else None
        
        
    def _constraints(self, trial):
        return trial.user_attrs["constraint"]
    
    def _optimizer_fn(self, bounds, constraints):
        def _opt_fn(trial):
            x = np.array([trial.suggest_float(name, *bound, step=.0001) for name, bound in bounds.items()])
            budget = self._optimizer_array_to_budget(x)
            total_budget = sum(v for v in budget.values())
            if constraints is None:
                trial.set_user_attr("constraint", (-1,))
            else:
                less_than_upper = ((total_budget - constraints[1])/constraints[1]) <= self.__tol
                greater_than_lower = ((total_budget - constraints[0])/constraints[0]) >= -self.__tol
                
                trial.set_user_attr("constraint", (-1,) if (less_than_upper and greater_than_lower) else (1,))
               
                trial.set_user_attr("test", (f"{total_budget:.2f}", constraints))
                trial.set_user_attr(
                    "tol", 
                    (
                        f"{((total_budget - constraints[1])/constraints[1]):.2%}",
                        f"{((total_budget - constraints[0])/constraints[0]):.2%}"
                    ))
                trial.set_user_attr("less or greater", (int(less_than_upper), int(greater_than_lower)))
                if (not (less_than_upper and greater_than_lower)) and np.random.uniform(0, 1) > self.__percent_out_tolerance:
                    raise optuna.exceptions.TrialPruned()
            trial.set_user_attr("budget", budget)
            prediction = self.model.predict(budget)
            
            
            loss = -self._loss_fn(prediction, **self._config['loss_fn_kwargs'])
            return loss
        
        return _opt_fn

    def optimize(
        self, 
        bounds: dict[str, tuple[float, float]], # Bounds for the optimizer
        constraints: None|tuple = None, # Constraints for the optimizer
        timeout: int = 60,
        n_trials: int = 100,
        storage: str|None = "sqlite:///db.sqlite3",
        study_name: str = "optimizer",
        n_jobs: int = 1
    ):
        """Optimize the model"""
            
        self.study = optuna.create_study(
            storage=storage,  # Specify the storage URL here.
            study_name=study_name,
            direction=self._direction,
            sampler=self.__sampler,
            pruner=self.__pruner)
        self.study.set_metric_names([self.objective_name])
        self.study.optimize(
            self._optimizer_fn(bounds, constraints), 
            n_trials=n_trials, 
            timeout=timeout,
            n_jobs=n_jobs)
        
        self.sol = self.study.best_trial
        self.optimal_budget = self._optimizer_array_to_budget(list(self.sol.params.values()))
        self.optimal_prediction = self.model.predict(self.optimal_budget)
        
