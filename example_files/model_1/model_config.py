import xarray as xr
from pathlib import Path
import numpy as np
from budget_optimizer.utils.model_classes import _Model, Budget

class SimpleModel(_Model):
  def __init__(self, data: xr.Dataset = None):
    self.data = data
    
  def predict(self, x: xr.Dataset) -> xr.DataArray:
    x = x.copy()
    x["prediction"] = x["a"] + x["b"]
    return x["prediction"]
  
  def contributions(self, x: xr.Dataset) -> xr.Dataset:
    return x

def budget_to_data(budget: Budget, model: _Model) -> xr.Dataset:
    data = model.data.copy()
    for key, value in budget.items():
        data[key] = value*data[key]
    return data
  
def model_loader(path: Path) -> _Model:
    data_a = xr.DataArray(np.array([1, 2, 3]), dims='time', coords={"time": np.array([1, 2, 3])})
    data_b = xr.DataArray(np.array([4, 5, 6]), dims='time', coords={"time": np.array([1, 2, 3])})
    return SimpleModel(data = xr.Dataset({"a": data_a, "b": data_b}))