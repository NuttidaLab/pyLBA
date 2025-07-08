"""
Core abstract classes and data structures for accumulator models.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, NamedTuple, Union
import numpy as np
import pandas as pd
import pytensor.tensor as pt

class AccumulatorModel(ABC):
    """
    Abstract base class for accumulator models.
    
    All accumulator models should inherit from this class and implement
    the required abstract methods.
    """
    
    def __init__(self, name: str = "AccumulatorModel"):
        self.name = name
        self._fitted = False
        self._trace = None
        self._parameters = None
    
    @abstractmethod
    def pdf(self, rt: np.ndarray, response: np.ndarray, 
            parameters, **kwargs) -> pt.TensorVariable:
        """
        Compute the probability density function.
        
        Parameters
        ----------
        rt : np.ndarray
            Response times
        response : np.ndarray  
            Response choices (0-indexed)
        parameters : ModelParameters
            Model parameters
            
        Returns
        -------
        pt.TensorVariable
            Log probability density
        """
        pass
    
    @abstractmethod
    def get_parameter_class(self) -> type:
        """Return the parameter class for this model."""
        pass
    
    def fit_mcmc(self, data: pd.DataFrame, **kwargs) -> 'AccumulatorModel':
        """
        Fit model using MCMC.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data with columns 'rt' and 'response'
        **kwargs
            Additional arguments passed to the fitter
            
        Returns
        -------
        AccumulatorModel
            Fitted model instance
        """
        from ..fitting import MCMCFitter
        
        fitter = MCMCFitter(self)
        self._trace = fitter.fit(data, **kwargs)
        self._fitted = True
        return self
    
    def fit_em(self, data: pd.DataFrame, **kwargs) -> 'AccumulatorModel':
        """
        Fit model using Expectation-Maximization.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data with columns 'rt' and 'response'
        **kwargs
            Additional arguments passed to the fitter
            
        Returns
        -------
        AccumulatorModel
            Fitted model instance
        """
        from ..fitting import EMFitter
        
        fitter = EMFitter(self)
        self._parameters = fitter.fit(data, **kwargs)
        self._fitted = True
        return self
    
    def predict(self, n_trials: int, **kwargs) -> pd.DataFrame:
        """
        Generate predictions using fitted parameters.
        
        Parameters
        ----------
        n_trials : int
            Number of trials to simulate
        **kwargs
            Additional arguments
            
        Returns
        -------
        pd.DataFrame
            Predicted data
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if self._trace is not None:
            # Use posterior mean for predictions
            import arviz as az
            summary = az.summary(self._trace)
            param_dict = summary['mean'].to_dict()
            parameters = self.get_parameter_class().from_dict(param_dict)
        elif self._parameters is not None:
            parameters = self._parameters
        else:
            raise ValueError("No fitted parameters available")
        
        return self.generate_data(n_trials, parameters, **kwargs)
    
    @property
    def fitted(self) -> bool:
        """Check if model has been fitted."""
        return self._fitted
    
    @property
    def trace(self):
        """Get MCMC trace if available."""
        return self._trace
    
    @property
    def parameters(self):
        """Get fitted parameters if available."""
        return self._parameters

