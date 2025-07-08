"""
Linear Ballistic Accumulator (LBA) model implementation.
"""

from typing import Optional, Union, Dict, Any
from dataclasses import dataclass, fields
import math
import numpy as np
import pandas as pd
import pytensor.tensor as pt

from ..core.accumulator import AccumulatorModel
from ..core.parameters import ModelParameters
from ..utils import broadcast_parameters

@dataclass(frozen=True)
class LBAParameters(ModelParameters):
    A: Union[float, np.ndarray]
    b: Union[float, np.ndarray]
    v: Union[float, np.ndarray]
    s: Union[float, np.ndarray]
    tau: Union[float, np.ndarray]

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
    
    def pdf(self, 
            rt: np.ndarray, 
            response: np.ndarray, 
            parameters: LBAParameters, 
            n_acc: Optional[int] = None
        ) -> pt.TensorVariable:
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
    
    def get_parameter_class(self) -> type:
        """Return the parameter class for the LBA model."""
        return LBAParameters
    
    def get_default_priors(self, 
                           n_acc: int
        ) -> Dict[str, Any]:
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
