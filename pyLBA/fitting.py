"""
Fitting methods for accumulator models.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from .core import AccumulatorModel, ModelParameters


class BaseFitter(ABC):
    """Base class for model fitting."""
    
    def __init__(self, model: AccumulatorModel):
        self.model = model
    
    @abstractmethod
    def fit(self, data: pd.DataFrame, **kwargs):
        """Fit the model to data."""
        pass


class MCMCFitter(BaseFitter):
    """MCMC fitting using PyMC."""
    
    def fit(self, data: pd.DataFrame, 
            draws: int = 1000,
            tune: int = 1000,
            chains: int = 4,
            cores: int = 4,
            progressbar: bool = True,
            priors: Optional[Dict[str, Any]] = None,
            **kwargs):
        """
        Fit model using MCMC sampling.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data with columns 'rt' and 'response'
        draws : int, default=1000
            Number of draws from posterior
        tune : int, default=1000
            Number of tuning steps
        chains : int, default=4
            Number of chains
        cores : int, default=4
            Number of cores to use
        progressbar : bool, default=True
            Show progress bar
        priors : dict, optional
            Prior distributions for parameters
        **kwargs
            Additional arguments passed to pm.sample
            
        Returns
        -------
        InferenceData
            PyMC trace object
        """
        rt = data['rt'].to_numpy()
        response = data['response'].to_numpy()
        
        # Infer number of accumulators
        n_acc = len(np.unique(response))
        
        with pm.Model() as graph:
            # Create parameter variables with priors
            param_vars = self._create_priors(n_acc, priors)
            
            # Create parameter object
            parameters = self.model.get_parameter_class()(**param_vars)
            
            # Calculate likelihood
            p = self.model.pdf(rt, response, parameters)
            logp = pt.sum(pt.log(p + 1e-12))
            
            pm.Potential('likelihood', logp)
            
            # Sample
            trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=cores,
                progressbar=progressbar,
                **kwargs
            )
            
        return trace
    
    def _create_priors(self, n_acc: int, priors: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create prior distributions for model parameters."""
        if priors is None:
            priors = {}
        
        # Check if model has default priors method
        if hasattr(self.model, 'get_default_priors'):
            default_priors = self.model.get_default_priors(n_acc)
        else:
            # Generic default priors
            param_class = self.model.get_parameter_class()
            param_fields = param_class._fields
            default_priors = {}
            
            for field in param_fields:
                default_priors[field] = pm.Uniform(field, lower=0, upper=50, shape=n_acc)
        
        # Override with user-specified priors
        param_vars = {}
        for field, default_prior in default_priors.items():
            if field in priors:
                param_vars[field] = priors[field]
            else:
                param_vars[field] = default_prior
        
        return param_vars


class EMFitter(BaseFitter):
    """Expectation-Maximization fitting."""
    
    def fit(self, data: pd.DataFrame,
            max_iter: int = 100,
            tol: float = 1e-6,
            init_params: Optional[ModelParameters] = None,
            **kwargs) -> ModelParameters:
        """
        Fit model using Expectation-Maximization.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data with columns 'rt' and 'response'
        max_iter : int, default=100
            Maximum number of iterations
        tol : float, default=1e-6
            Convergence tolerance
        init_params : ModelParameters, optional
            Initial parameter values
        **kwargs
            Additional arguments
            
        Returns
        -------
        ModelParameters
            Fitted parameters
        """
        # EM implementation would go here
        # For now, this is a placeholder
        raise NotImplementedError("EM fitting not yet implemented")
    
    def _e_step(self, data: pd.DataFrame, parameters: ModelParameters):
        """Expectation step."""
        pass
    
    def _m_step(self, data: pd.DataFrame, responsibilities):
        """Maximization step."""
        pass
