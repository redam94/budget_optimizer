import numpy as np
import xarray as xr

def loss_fn(x: xr.DataArray, start_date=5, end_date=100, dim="Period"):
    # x is a numpy array of shape (n_params,)
    # start_date and end_date are datetime objects
    # return a scalar loss
    x = x.sel({dim: slice(start_date, end_date)})
    return -np.sum(x)