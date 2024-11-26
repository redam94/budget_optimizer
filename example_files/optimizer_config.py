import numpy as np
import xarray as xr
from budget_optimizer.utils.model_helpers import BudgetType, load_yaml
from pathlib import Path

# Define the optimizer configuration
CONFIG = load_yaml(Path(__file__).parent / "optimizer_config.yaml")


def loss_fn(x: xr.DataArray, start_date=None, end_date=None, dim="Period"):
    # x is a numpy array of shape (n_params,)
    # start_date and end_date are datetime objects
    # return a scalar loss
    x = x.sel({dim: slice(start_date, end_date)})
    return -np.sum(x)

def optimizer_array_to_budget(array: np.ndarray) -> BudgetType:
    initial_budget: BudgetType = CONFIG['initial_budget']
    budget: BudgetType = {}
        
    for i, key in enumerate(initial_budget.keys()):
        budget[key] = array[i]
    return budget