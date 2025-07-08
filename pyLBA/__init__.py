"""
pyLBA: A Python package for Linear Ballistic Accumulator and other accumulator models.

This package provides a unified interface for fitting and simulating various 
accumulator models used in cognitive psychology and neuroscience.
"""

from .models.linear_ballistic_accumulator import LBAModel, LBAParameters
from .core.accumulator import AccumulatorModel
from .core.parameters import Parameter, ModelParameters
from .fitting import MCMCFitter, EMFitter

__version__ = "0.1.0"
__author__ = "Rudramani Singha"
__email__ = "rgs2151@columbia.com"

__all__ = [
    "LBAModel",
    "LBAParameters",
    "AccumulatorModel", 
    "ModelParameters",
    "MCMCFitter",
    "EMFitter"
]
