"""
Generic synthetic data generator for accumulator models.
"""
import numpy as np
import pandas as pd
from .utils import broadcast_parameters
from .models.linear_ballistic_accumulator import LBAParameters

def generate_LBA(n_trials: int, 
                 parameters: LBAParameters, 
                 n_acc: int = 1, 
                 seed: int = None
    ) -> pd.DataFrame:
    """
    Generate synthetic response time and choice data for an accumulator model.

    Parameters
    ----------
    n_trials : int
        Number of trials to simulate.
    parameters : LBAParameters
        Model parameters with fields A, b, v, s, tau.
    n_acc : int, optional
        Number of accumulators.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'rt' and 'response'.
    """
    rng = np.random.default_rng(seed)

    # Broadcast parameters to shape [n_acc]
    A_arr, b_arr, v_arr, s_arr, tau_arr = broadcast_parameters(
        n_acc, parameters.A, parameters.b, parameters.v, parameters.s, parameters.tau
    )
    # Uniform start-points
    k = rng.random((n_acc, n_trials)) * A_arr[:, None]

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

    # Extract RT and response
    rt = T.min(axis=0)
    resp = T.argmin(axis=0)

    return pd.DataFrame({
        'rt': rt.astype('float32'),
        'response': resp.astype('int32')
    })
