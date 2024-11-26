"""Useful functions/class/types for defining loss functions, metrics, model loading and data augmentation"""

# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/utils/01_model_helpers.ipynb.

# %% auto 0
__all__ = ['BudgetType', 'load_module', 'load_yaml', 'AbstractModel']

# %% ../../nbs/utils/01_model_helpers.ipynb 3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import yaml

import importlib.util as import_utils
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Union, List, Dict

# %% ../../nbs/utils/01_model_helpers.ipynb 5
def load_module(
  module_name: str, # The name of the module to load
  module_path: Path, # The path to the module
) -> object: # The loaded module
    "Load a module from a file path."
    spec = import_utils.spec_from_file_location(module_name, module_path)
    module = import_utils.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# %% ../../nbs/utils/01_model_helpers.ipynb 6
def load_yaml(
  file_path: Path, # The path to the YAML file
) -> Dict[str, Union[list[str], str]]: # The loaded module
    "Load a yaml file."
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

# %% ../../nbs/utils/01_model_helpers.ipynb 8
class AbstractModel(ABC):
  """An abstract class for models"""
  @abstractmethod
  def predict(self, x: xr.Dataset) -> xr.Dataset:
    pass
  
  @abstractmethod
  def contributions(self, x: xr.Dataset) -> xr.Dataset:
    pass
  
  def __repr__(self) -> str:
    return f"Model"

# %% ../../nbs/utils/01_model_helpers.ipynb 11
BudgetType = Union[Dict[str, float], xr.Dataset] # type alias for budget data
