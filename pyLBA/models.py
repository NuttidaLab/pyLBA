"""
Linear Ballistic Accumulator (LBA) model implementation.
"""

import math
import numpy as np
import pandas as pd
import pytensor.tensor as pt
from typing import Optional, NamedTuple, Union, Dict, Any

from .core import AccumulatorModel, ModelParameters
from .utils import broadcast_parameters


class LBAParameters(ModelParameters):
    """
    Parameters for the Linear Ballistic Accumulator model.
    
    Parameters
    ----------
    A : float or array-like
        Start point variability (upper bound of uniform distribution)
    b : float or array-like
        Response threshold
    v : float or array-like
        Drift rate
    s : float or array-like
        Drift rate standard deviation
    tau : float or array-like
        Non-decision time
    """
    A: Union[float, np.ndarray]
    b: Union[float, np.ndarray]
    v: Union[float, np.ndarray]
    s: Union[float, np.ndarray]
    tau: Union[float, np.ndarray]
    
    def validate(self) -> bool:
        """Validate LBA parameter constraints."""
        # Convert to arrays for validation
        A = np.atleast_1d(self.A)
        b = np.atleast_1d(self.b)
        v = np.atleast_1d(self.v)
        s = np.atleast_1d(self.s)
        tau = np.atleast_1d(self.tau)
        
        # Basic positivity constraints
        if np.any(A <= 0):
            return False
        if np.any(b <= 0):
            return False
        if np.any(s <= 0):
            return False
        if np.any(tau < 0):
            return False
        
        # LBA-specific constraints
        if np.any(b <= A):
            return False
        
        return True


class LBAModel(AccumulatorModel):
    """
    Linear Ballistic Accumulator model.
    
    The LBA model assumes that evidence accumulates linearly towards
    response thresholds, with drift rates drawn from a normal distribution
    truncated at zero.
    
    References
    ----------
    Brown, S. D., & Heathcote, A. (2008). The simplest complete model of 
    choice response time: Linear ballistic accumulation. Cognitive Psychology, 
    57(3), 153-178.
    """
    
    def __init__(self):
        super().__init__("Linear Ballistic Accumulator")
    
    def pdf(self, rt: np.ndarray, response: np.ndarray, 
            parameters: LBAParameters, n_acc: Optional[int] = None) -> pt.TensorVariable:
        """
        Compute the probability density function for the LBA model.
        
        Parameters
        ----------
        rt : np.ndarray
            Response times
        response : np.ndarray
            Response choices (0-indexed)
        parameters : LBAParameters
            Model parameters
        n_acc : int, optional
            Number of accumulators (inferred from response if not provided)
            
        Returns
        -------
        pt.TensorVariable
            Probability density
        """
        if n_acc is None:
            n_acc = len(np.unique(response))
        
        # Broadcast parameters
        A, b, v, s, tau = broadcast_parameters(
            n_acc, parameters.A, parameters.b, parameters.v, parameters.s, parameters.tau
        )
        
        # Standard normal PDF and CDF
        normpdf = lambda x: pt.exp(-0.5 * x**2) / pt.sqrt(2 * math.pi)
        normcdf = lambda x: 0.5 * (1 + pt.erf(x / pt.sqrt(2)))
        
        def tpdf(t, A, b, v, s):
            """First-passage time probability density function for single accumulator."""
            g = (b - A - t * v) / (t * s)
            h = (b - t * v) / (t * s)
            
            # First-passage density formula
            pdf = (1/A) * (-v * normcdf(g) + s * normpdf(g) + v * normcdf(h) - s * normpdf(h))
            return pt.maximum(pdf, 1e-20)  # Numerical stability
        
        def tcdf(t, A, b, v, s):
            """First-passage time cumulative distribution function for single accumulator."""
            g = (b - A - t * v) / (t * s)
            h = (b - t * v) / (t * s)
            
            # CDF components
            p1 = ((b - A - t * v) / A) * normcdf(g)
            p2 = ((b - t * v) / A) * normcdf(h)
            p3 = ((t * s) / A) * normpdf(g)
            p4 = ((t * s) / A) * normpdf(h)
            
            cdf = 1 + p1 - p2 + p3 - p4
            return pt.clip(cdf, 1e-20, 1 - 1e-20)
        
        # First-passage densities for each accumulator
        f = pt.stack([tpdf(rt - tau[i], A[i], b[i], v[i], s[i]) for i in range(n_acc)], axis=0)
        F = pt.stack([tcdf(rt - tau[i], A[i], b[i], v[i], s[i]) for i in range(n_acc)], axis=0)
        p_zero = pt.prod(normcdf(-v / s))  # Probability that none ever finish
        
        # For each i: f_i * ∏_{j≠i} (1 − F_j)
        numer = pt.stack([
            f[i] * pt.prod(1 - pt.concatenate([F[:i], F[i+1:]]), axis=0)
            for i in range(n_acc)
        ], axis=0)
        
        # Select density + normalize
        pdf = numer[response, pt.arange(rt.shape[0])] / (1 - p_zero)
        
        # Zero out trials where rt ≤ τ for that accumulator
        t = rt - tau[response]
        return pt.switch(pt.gt(t, 0), pdf, 1e-20)
    
    def generate_data(self, n_trials: int, parameters: LBAParameters, 
                     n_acc: Optional[int] = None, seed: Optional[int] = None) -> pd.DataFrame:
        """
        Generate synthetic data from the LBA model.
        
        Parameters
        ----------
        n_trials : int
            Number of trials to simulate
        parameters : LBAParameters
            Model parameters
        n_acc : int, optional
            Number of accumulators (inferred from parameters if not provided)
        seed : int, optional
            Random seed for reproducibility
            
        Returns
        -------
        pd.DataFrame
            Simulated data with columns 'rt' and 'response'
        """
        rng = np.random.default_rng(seed)
        
        # Infer or validate n_acc
        v_raw = np.atleast_1d(parameters.v)
        n_acc_val = int(n_acc) if n_acc is not None else v_raw.size
        
        # Broadcast/validate parameters
        def _prep(param_val):
            arr = np.atleast_1d(param_val).astype(float)
            if arr.size == 1:
                return np.full(n_acc_val, arr.item(), dtype=float)
            if arr.size == n_acc_val:
                return arr
            raise ValueError(f"Parameter must be scalar or length={n_acc_val}, got size={arr.size}")
        
        A_arr = _prep(parameters.A)
        b_arr = _prep(parameters.b)
        v_arr = _prep(parameters.v)
        s_arr = _prep(parameters.s)
        tau_arr = _prep(parameters.tau)
        
        # Uniform start-points
        k = rng.random((n_acc_val, n_trials)) * A_arr[:, None]
        
        # Prepare full-shape loc/scale for drifts
        loc_mat = np.tile(v_arr[:, None], (1, n_trials))
        scale_mat = np.tile(s_arr[:, None], (1, n_trials))
        
        # Draw drifts truncated at zero
        d = rng.normal(loc=loc_mat, scale=scale_mat)
        mask = d <= 0
        while mask.any():
            d[mask] = rng.normal(
                loc=loc_mat[mask],
                scale=scale_mat[mask],
                size=mask.sum()
            )
            mask = d <= 0
        
        # Finish times
        T = tau_arr[:, None] + (b_arr[:, None] - k) / d
        
        # RT & response
        rt = T.min(axis=0)
        resp = T.argmin(axis=0)
        
        return pd.DataFrame({
            'rt': rt.astype('float32'),
            'response': resp.astype('int32')
        })
    
    def get_parameter_class(self) -> type:
        """Return the parameter class for the LBA model."""
        return LBAParameters
    
    def get_default_priors(self, n_acc: int) -> Dict[str, Any]:
        """
        Get default prior distributions for LBA parameters.
        
        Parameters
        ----------
        n_acc : int
            Number of accumulators
            
        Returns
        -------
        Dict[str, Any]
            Dictionary of prior distributions
        """
        import pymc as pm
        
        return {
            'A': pm.Uniform('A', lower=0, upper=10, shape=n_acc),
            'b': pm.Uniform('b', lower=0, upper=30, shape=n_acc),
            'v': pm.Uniform('v', lower=0, upper=10, shape=n_acc),
            's': pm.Uniform('s', lower=0, upper=10, shape=n_acc),
            'tau': pm.Uniform('tau', lower=0, upper=5, shape=n_acc)
        }
