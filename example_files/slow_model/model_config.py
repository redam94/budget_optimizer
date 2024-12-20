import xarray as xr
from pathlib import Path
import numpy as np
from budget_optimizer.utils.model_helpers import AbstractModel, BudgetType
from time import sleep

INITIAL_BUDGET: BudgetType = dict(a=2., b=3.)

class SimpleModel(AbstractModel):
  """
  Simple model that just adds the two variables a and b.
  This can be as complex as you want as long as it has a predict method
  that takes an xarray Dataset and returns an xarray DataArray and 
  a contributions method that takes an xarray Dataset and returns an xarray Dataset.
  
  Ideally, the model should also have data that defines the initial data that the
  model was trained on. You can wrap cutom models or functions in a class like this.
  """
  def __init__(self, data: xr.Dataset = None):
    self.data = data
    
  def predict(self, x: xr.Dataset) -> xr.DataArray:
    x = x.copy()
    sleep(2) # Simulate a long computation
    x["prediction"] = np.exp(1 + .2*(x["a"]**2/(x["a"]**2 + np.exp(1)**2)) + .25*(x["b"]**4/(x["b"]**4 + np.exp(2)**4)))
    return x["prediction"]
  
  def contributions(self, x: xr.Dataset) -> xr.Dataset:
    return x

def budget_to_data(budget: BudgetType, model: AbstractModel) -> xr.Dataset:
    data = model.data.copy()
    for key, value in budget.items():
        data[key] = value/INITIAL_BUDGET[key]*data[key]
    return data
  
def model_loader(path: Path) -> AbstractModel:
    rng = np.random.default_rng(42)
    data_a = xr.DataArray(np.exp(1+rng.normal(0, .4, size=156)), dims='time', coords={"time": np.arange(1, 157)})
    data_b = xr.DataArray(np.exp(2+rng.normal(0, .2, size=156)), dims='time', coords={"time": np.arange(1, 157)})
    return SimpleModel(data = xr.Dataset({"a": data_a, "b": data_b}))

