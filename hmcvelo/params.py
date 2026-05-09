"""
hcmvelo.params
--------------
Gene-specific rate parameter estimation at steady state.

At steady state (d/dt = 0), with βH = 1 fixed:

    Experiment 1 (read counts):
        2m = βM * c + βD * h

    Experiment 2 (proportions):
        2*xm = βM * (1 - xh - xm) + βD * xh

Both reduc to the linear regression form:
    A @ [βM, βD] = b
where A and b are built from the observed data across all cells for a gene.

scipy.optimize.least_squares with bounds βM >=0, βD >= 0.
"""

import numpy as np
from scipy.optimize import least_squares
from typing import Tuple


BETA_H_FIXED = 1.0


def _build_system(
    h_g: np.ndarray,
    m_g: np.ndarray,
    c_g: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the linear system A @ [βM, βD] = b for one gene.
 
    From steady-state constraint (Eq. 20-21):
        2m = βM * c + βD * h
    =>  A[:, 0] = c,  A[:, 1] = h,  b = 2m
 
    Parameters
    ----------
    h_g, m_g, c_g : (n_cells,) arrays for gene g
 
    Returns
    -------
    A : (n_valid_cells, 2)
    b : (n_valid_cells,)
    """
    valid = (h_g + m_g + c_g) > 0
    A = np.column_stack([c_g[valid], h_g[valid]])
    b = 2.0 * m_g[valid]
    return A,b


def _fit_gene(
    h_g: np.ndarray,
    m_g: np.ndarray,
    c_g: np.ndarray,
    min_cells: int = 3,
) -> np.ndarray:
    """
    Estimate [βM, βD] for one gene via constrained least squares.
 
    Parameters
    ----------
    h_g, m_g, c_g : (n_cells,) per-gene observations
    min_cells      : minimum valid cells required to attempt fit
 
    Returns
    -------
    params : array [βM, βD]  (zeros if fit fails or too few cells)
    """
    A, b = _build_system(h_g, m_g, c_g)

    if A.shape[0] < min_cells:
        return np.zeros(2)

    
