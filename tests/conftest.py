"""
Test configuration for pyLBA.
"""

import pytest
import numpy as np


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    import pandas as pd
    
    np.random.seed(42)
    n_trials = 100
    
    # Simulate some simple data
    rt = np.random.exponential(0.5, n_trials) + 0.2
    response = np.random.choice([0, 1, 2], n_trials, p=[0.3, 0.4, 0.3])
    
    return pd.DataFrame({
        'rt': rt,
        'response': response
    })


@pytest.fixture
def sample_parameters():
    """Sample LBA parameters for testing."""
    from pyLBA import LBAParameters
    
    return LBAParameters(
        A=4.0,
        b=[6.0, 10.0, 20.0],
        v=1.0,
        s=1.0,
        tau=0.0
    )


@pytest.fixture
def lba_model():
    """Create LBA model for testing."""
    from pyLBA import LBAModel
    
    return LBAModel()
