"""
Utility functions for pyLBA package.
"""

import numpy as np
import pytensor.tensor as pt
from typing import Union, Tuple


def broadcast_parameters(n: int, *args) -> Tuple[pt.TensorVariable, ...]:
    """
    Broadcast parameters to match the number of accumulators.
    
    Parameters
    ----------
    n : int
        Number of accumulators
    *args : array-like
        Parameters to broadcast
        
    Returns
    -------
    Tuple[pt.TensorVariable, ...]
        Broadcasted parameters
    """
    def _promote(x):
        x_var = np.array(x) if not hasattr(x, "ndim") else x
        if x_var.ndim == 0:
            return np.repeat(x_var, n)
        if x_var.ndim == 1:
            return x_var
        raise ValueError(f"Scalars or 1-D tensor of dim {n} only, got ndim={x_var.ndim}")
    
    return tuple(_promote(x.value) for x in args)


def validate_data(data):
    """
    Validate input data format.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data to validate
        
    Raises
    ------
    ValueError
        If data format is invalid
    """
    required_columns = ['rt', 'response']
    
    if not all(col in data.columns for col in required_columns):
        raise ValueError(f"Data must contain columns: {required_columns}")
    
    if (data['rt'] <= 0).any():
        raise ValueError("All reaction times must be positive")
    
    if (data['response'] < 0).any():
        raise ValueError("All responses must be non-negative")


def check_convergence(trace, r_hat_threshold: float = 1.1) -> bool:
    """
    Check MCMC convergence using R-hat statistic.
    
    Parameters
    ----------
    trace : InferenceData
        PyMC trace object
    r_hat_threshold : float, default=1.1
        R-hat threshold for convergence
        
    Returns
    -------
    bool
        True if converged, False otherwise
    """
    try:
        import arviz as az
        summary = az.summary(trace)
        return (summary['r_hat'] < r_hat_threshold).all()
    except ImportError:
        print("Warning: arviz not available, cannot check convergence")
        return True
