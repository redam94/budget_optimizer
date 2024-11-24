"""Model classes"""

# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/utils/00_model_classes.ipynb.

# %% auto 0
__all__ = ['Budget', 'BaseBudgetModel']

# %% ../../nbs/utils/00_model_classes.ipynb 4
from abc import ABC, abstractmethod
from pathlib import Path
from collections.abc import Sequence
from typing import (
  Callable, Generic, 
  TypeVar, Union, 
  Protocol, Dict,
  TypeAlias)
import xarray as xr
import types

# %% ../../nbs/utils/00_model_classes.ipynb 5
class _Model(ABC):
  
  @abstractmethod
  def predict(self, x: xr.Dataset) -> xr.Dataset:
    pass
  
  @abstractmethod
  def contributions(self, x: xr.Dataset) -> xr.Dataset:
    pass
  
  def __repr__(self) -> str:
    return f"Model"

# %% ../../nbs/utils/00_model_classes.ipynb 6
Budget = Union[Dict[str, float], xr.Dataset]

# %% ../../nbs/utils/00_model_classes.ipynb 7
class BaseBudgetModel(_Model):
    """
    Abstract class for all models
    """
    _FUNCTION_MODULE_NAME = "model_config.py"
    
    def __init__(
      self, 
      model_name: str, # Name used to identify the model
      model_kpi: str, # Key performance indicator output by the model predict
      model_path: str|Path, # Path to the model artifact
    ):
        self.model_name: str = model_name
        self.model_kpi: str = model_kpi
        self.model_path: Path = model_path if isinstance(model_path, Path) else Path(model_path)
        self._model = self._get_model_loader()(model_path)
        self._budget_to_data = self._get_budget_to_data()

    def _get_model_loader(self) -> Callable[str|Path, _Model]:
        """
        Get the function to load the model from the path
        """
        from importlib.util import spec_from_file_location, module_from_spec
        spec = spec_from_file_location(self._FUNCTION_MODULE_NAME.replace(".py", ""), self.model_path / self._FUNCTION_MODULE_NAME)
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.model_loader
    
    def _get_budget_to_data(self) -> Callable[Budget, xr.Dataset]:
        """
        Get the mapping from budget keys to data keys
        """
        from importlib.util import spec_from_file_location, module_from_spec
        spec = spec_from_file_location(self._FUNCTION_MODULE_NAME.replace(".py", ""), self.model_path / self._FUNCTION_MODULE_NAME)
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.budget_to_data
    
    def predict(
        self, 
        budget: Budget # Budget
        ) -> xr.DataArray: # Predicted target variable
        """
        Predict the target variable from the input data
        """
        data = self._budget_to_data(budget, self._model)
        return self._model.predict(data)
    
    
    def contributions(
        self, 
        budget: Budget # Budget
        ) -> xr.Dataset: # Contributions of the input data to the target variable
        """
        Get the contributions of the input data to the target variable
        """
        data = self._budget_to_data(budget, self._model)
        return self._model.contributions(data)
