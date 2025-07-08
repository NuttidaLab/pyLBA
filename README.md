# 🚀 pyLBA: Linear Ballistic Accumulator Models in Python

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](https://github.com/nuttidalab/pyLBA/actions)

A Python package for fitting and simulating Linear Ballistic Accumulator (LBA) models and other accumulator models commonly used in cognitive psychology and neuroscience.

## Features

- **Unified Interface**: Standardized API for different accumulator models
- **Flexible Fitting**: Both MCMC (via PyMC) and EM fitting methods
- **Parameter Management**: Type-safe parameter handling with named tuples
- **Data Generation**: Simulate synthetic data from fitted models
- **Extensible Design**: Easy to add new accumulator models

## Installation

### From PyPI (when available)

```bash
pip install pyLBA
```

### From Source

```bash
git clone https://github.com/nuttidalab/pyLBA.git
cd pyLBA
pip install -e . # Editable Installation
# or
pip install -e ".[dev]" # Development Installation
```

### Running Tests

```bash
pytest tests/
```

## Quick Start

### Basic Usage

```python
import pandas as pd
from pyLBA import LBAModel, LBAParameters

# Create model
model = LBAModel()

# Define parameters
params = LBAParameters(
    A=4.0,           # Start point variability
    b=[6, 10, 20],   # Response thresholds
    v=1.0,           # Drift rate
    s=1.0,           # Drift rate SD
    tau=0.0          # Non-decision time
)

# Generate synthetic data
data = model.generate_data(n_trials=500, parameters=params, n_acc=3)
print(data.head())
```

### Model Fitting

```python
# Fit model using MCMC
fitted_model = model.fit_mcmc(
    data=data,
    draws=1000,
    tune=1000,
    chains=4
)

# Check convergence
import arviz as az
print(az.summary(fitted_model.trace))

# Generate predictions
predictions = fitted_model.predict(n_trials=100)
```

### Custom Priors

```python
import pymc as pm

# Define custom priors
custom_priors = {
    'A': pm.Uniform('A', lower=0, upper=10, shape=3),
    'b': pm.Uniform('b', lower=5, upper=25, shape=3),
    'v': pm.Normal('v', mu=1, sigma=0.5, shape=3),
    's': pm.HalfNormal('s', sigma=2, shape=3),
    'tau': pm.Uniform('tau', lower=0, upper=1, shape=3)
}

# Fit with custom priors
fitted_model = model.fit_mcmc(
    data=data,
    priors=custom_priors,
    draws=1000,
    tune=1000
)
```

## Model Overview

### Linear Ballistic Accumulator (LBA)

The LBA model assumes that evidence accumulates linearly towards response thresholds, with drift rates drawn from a normal distribution truncated at zero.

**Parameters:**
- `A`: Start point variability (uniform distribution upper bound)
- `b`: Response threshold
- `v`: Drift rate (mean of truncated normal)
- `s`: Drift rate standard deviation
- `tau`: Non-decision time

**References:**
- Brown, S. D., & Heathcote, A. (2008). The simplest complete model of choice response time: Linear ballistic accumulation. *Cognitive Psychology*, 57(3), 153-178.

## Advanced Usage

### Parameter Validation

```python
# Parameters are automatically validated
params = LBAParameters(A=4, b=3, v=1, s=1, tau=0)  # Invalid: b <= A
print(params.validate())  # Returns False
```

### Extending with New Models

```python
from pyLBA.core import AccumulatorModel, ModelParameters
from typing import NamedTuple

class MyModelParameters(ModelParameters):
    param1: float
    param2: float

class MyModel(AccumulatorModel):
    def __init__(self):
        super().__init__("My Custom Model")
    
    def pdf(self, rt, response, parameters):
        # Implement your model's PDF
        pass
    
    def generate_data(self, n_trials, parameters, **kwargs):
        # Implement data generation
        pass
    
    def get_parameter_class(self):
        return MyModelParameters
```

## Data Format

Input data should be a pandas DataFrame with the following columns:
- `rt`: Response times (positive values)
- `response`: Response choices (0-indexed integers)

```python
data = pd.DataFrame({
    'rt': [0.5, 0.7, 0.4, 0.9, 0.6],
    'response': [0, 1, 0, 2, 1]
})
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## Acknowledgments

- Built on top of [PyMC](https://www.pymc.io/) for Bayesian modeling
