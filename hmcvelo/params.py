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


def least_squares_params(
    h: np.ndarray,
    m: np.ndarray,
    c: np.ndarray,
    min_cells: int = 3,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate gene-specific rate parameters [βM, βD] for all genes.
 
    βH is fixed to 1.0 by scale invariance (Section 2.6 of paper).
    Parameters are cell-independent, gene-specific (Eq. 23).
 
    Parameters
    ----------
    h, m, c : ndarray, shape (n_cells, n_genes)
        Observed 5hmC, 5mC, and unmodified cytosine.
        For Experiment 1: raw read counts.
        For Experiment 2: proportions (xh, xm, xc = 1 - xm - xh).
    min_cells : int
        Minimum number of non-zero cells to fit a gene.
    verbose : bool
        Print progress.
 
    Returns
    -------
    beta_M : ndarray, shape (n_genes,)
    beta_D : ndarray, shape (n_genes,)
    """
    n_cells, n_genes = h.shape
    beta_M = np.zeros(n_genes)
    beta_D = np.zeros(n_genes)
 
    null_count = 0
 
    for g in range(n_genes):
        params = _fit_gene(h[:, g], m[:, g], c[:, g], min_cells=min_cells)
        beta_M[g] = params[0]
        beta_D[g] = params[1]
        if params[0] == 0.0 and params[1] == 0.0:
            null_count += 1
 
    if verbose:
        fitted = n_genes - null_count
        print(f"[HMCVelo] Parameters estimated for {fitted}/{n_genes} genes "
              f"({null_count} null).")
 
    return beta_M, beta_D


def infer_cytosine_proportions(
    xm: np.ndarray,
    xh: np.ndarray,
) -> np.ndarray:
    """
    Infer unmodified cytosine proportion from conservation law (Eq. 9):
        xc = 1 - xm - xh
 
    Clips to [0, 1] to handle floating-point noise.
    """
    xc = 1.0 - xm - xh
    return np.clip(xc, 0.0, 1.0)