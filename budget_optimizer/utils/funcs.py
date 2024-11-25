"""Useful functions for defining loss functions, metrics, model loading and data augmentation"""

# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/utils/01_helper_func.ipynb.

# %% auto 0
__all__ = ['load_module', 'AbstractModel']

# %% ../../nbs/utils/01_helper_func.ipynb 3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr

import importlib.util as import_utils
from pathlib import Path
from abc import ABC, abstractmethod

# %% ../../nbs/utils/01_helper_func.ipynb 4
def load_module(
  module_name: str, # The name of the module to load
  module_path: Path, # The path to the module
) -> object: # The loaded module
    "Load a module from a file path."
    spec = import_utils.spec_from_file_location(self._FUNCTION_MODULE_NAME.replace(".py", ""), self.model_path / self._FUNCTION_MODULE_NAME)
    module = import_utilsmodule_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# %% ../../nbs/utils/01_helper_func.ipynb 5
class AbstractModel(ABC):
  
  @abstractmethod
  def predict(self, x: xr.Dataset) -> xr.Dataset:
    pass
  
  @abstractmethod
  def contributions(self, x: xr.Dataset) -> xr.Dataset:
    pass
  
  def __repr__(self) -> str:
    return f"Model"