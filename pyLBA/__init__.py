"""
pyLBA: A Python package for Linear Ballistic Accumulator and other accumulator models.

This package provides a unified interface for fitting and simulating various 
accumulator models used in cognitive psychology and neuroscience.
"""

from .models import LBAModel
from .core import AccumulatorModel, ModelParameters
from .fitting import MCMCFitter, EMFitter

__version__ = "0.1.0"
__author__ = "Rudramani Singha"
__email__ = "rgs2151@columbia.com"

__all__ = [
    "LBAModel",
    "AccumulatorModel", 
    "ModelParameters",
    "MCMCFitter",
    "EMFitter"
]
