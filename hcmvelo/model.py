"""
hmcvelo.model
-------------

Core ODE for the methylation-demethylation cycle

State variables
---------------
m(t) : 5-methylcytosine (5mC)
h(t) : 5-hydroxymethylcytosine (5hmC)
c(t) : unmodified cytosine (C)

Conservation law: m + h + c = constant ==> dc/dt = -(dm/dt + dh/dt)

Rate parameters (after scale invariance βH = 1)
-----------------------------------------------

βM : rate of methylation (C --> 5mC, DNMT)
βH : rate of hydroxymethylation (5mC --> 5hmC, TET)
βD : rate of de-hydroxymethylation (5hmc --> C)
"""


import numpy as np
from scipy.integrate import odeint


def _ode(y, t, beta_M, beta_H, beta_D):
    """
    RHS of the ODE system.

    Parameters
    ----------
    y      : [m, h, c] array
    t      : time scalar
    beta_M : rate of methylation
    beta_H : rate of hydroxymethlation
    beta_D : rate of de-hydroxymethlation (must be 1.00 after fixing scale)

    Returns
    -------
    [dm/dt, dh/dt, dc/dt]
    """
    m, h, c = y
    dm = beta_M * c - beta_H * m
    dh = beta_H * m - beta_D * h
    dc = beta_D * h - beta_M * c
    return [dm, dh, dc]
    

def solve_model(
    m0: np.ndarray,
    h0: np.ndarray,
    c0: np.ndarray,
    beta_M: np.ndarray,
    beta_D: np.ndarray,
    beta_H: float = 1.0,
    dt: float = 1e-5
) -> np.ndarray:
    """
    Solves the HMCVelo ODE system for all cells x genes and returns
    the hydroxymethylation velocity.

    The velocity for gene g in cell i is estimated as:

        v[i, g] = (h(dt) - h0[i, g]) / dt

    Parameters
    ----------
    m0, h0, c0     : ndarray, shape (n_cells, n_genes)
        Observed initial conditions.
    beta_M, beta_D : ndarray, shape (n_genes, )
        Gene-specific rate parameters,
    beta_H         : float
        Hydroxymethylation rate, fixed to 1.0 by scale invariance.
    dt             : float
        Integration time-step. Paper shows dt=1e-5 is sufficient.

    Returns
    _______
    velocity       : ndarray, shape (n_cells, n_genes)
        Hydroxymethylation velocity per cell per gene.
    """
    n_cells, n_genes = m0.shape
    velocity = np.zeros((n_cells, n_genes), dtype=np.float64)
    t_span = [0.0, dt]

    for g in range(n_genes):
        bM = float(beta_M[g])
        bD = float(beta_D[g])

        if not (bM or bD): continue # skip null genes

        for i in range(n_cells):
            y0 = [float(m0[i, g]), float(h0[i, g]), float(c0[i, g])]

            if not any(y0): continue#skip zero-state cells

            sol = odeint(_ode, y0, t_span, args=(bM, beta_H, bD),
                        rtol=1e-15, atol=1e-15, full_output=False)

            h_final = sol[-1, 1] # h at time t = dt
            velocity[i, g] = (h_final - y0[1]) / dt

    return velocity


def solve_model_fast(
    m0: np.ndarray,
    h0: np.ndarray,
    c0: np.ndarray,
    beta_M: np.ndarray,
    beta_D: np.ndarray,
    beta_H: float = 1.0,
    dt: float = 1e-5,
) -> np.ndarray:
    """
    Vectorised analytic approximation for speed.
 
    For small dt the ODE is locally linear, so we can approximate:
 
        dh/dt ≈ βH * m0 - βD * h0     (equation 10 from paper)
 
    This is exact at steady state and a good approximation for small dt.
    Use this for large datasets; use solve_model for ground-truth validation.
 
    Returns
    -------
    velocity : ndarray, shape (n_cells, n_genes)
    """
    # Broadcast arrays over cells
    bM = beta_M[np.newaxis, :]
    bD = beta_D[np.newaxis, :]

    velocity = beta_H * m0 - bD * h0
    return velocity


def demethylation_velocity(
    m0: np.ndarray,
    h0: np.ndarray,
    c0: np.ndarray,
    beta_M: np.ndarray,
    beta_D: np.ndarray,
    beta_H: float = 1.0,
) -> np.ndarray:
    """
    Demethylation velocity (Eq. 11) — relevant for non-brain tissues
    where full demethylation (5hmC -> C) is the process of interest.
 
        d(t) = dh/dt + dc/dt = βH * m - βM * c
 
    Returns
    -------
    velocity : ndarray, shape (n_cells, n_genes)
    """
    bM = beta_M[np.newaxis, :]
    return beta_H * m0 - bM * c0

