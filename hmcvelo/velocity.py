"""
hmcvelo.velocity
"""

import numpy as np
from typing import Optional, Tuple
from .data import HMCData
from .params import least_squares_params, BETA_H_FIXED
from .model import solve_model, solve_model_fast, demethylation_velocity


class HMCVelo:
    """
    HMCVelo: Hydroxymethylation Velocity for single cells.

    Usage
    -----
    >>> hmc = HMCVelo(data)
    >>> hmc.fit()
    >>> hmc.compute_velocity()
    >>> v = hmc.velocity          # (n_cells, n_genes)

    Parameters
    ----------
    data     : HMCData
    beta_H   : float, hydroxymethylation rate (fixed to 1.0 by scale invariance)
    dt       : float, ODE integration time-step (paper: 1e-5)
    fast     : bool, use analytic approximation instead of odeint (faster, ~same results)
    min_cells: int, min cells with data to estimate a gene's parameters
    """

    def __init__(
        self,
        data: HMCData,
        beta_H: float = BETA_H_FIXED,
        dt: float = 1e-5,
        fast: bool = True,
        min_cells: int = 3,
    ):
        self.data = data
        self.beta_H = beta_H
        self.dt = dt
        self.fast = fast
        self.min_cells = min_cells

        # Set after fit()
        self.beta_M: Optional[np.ndarray] = None
        self.beta_D: Optional[np.ndarray] = None

        # Set after compute_velocity()
        self.velocity: Optional[np.ndarray] = None
        self._demethylation_velocity: Optional[np.ndarray] = None

    # Step 1: parameter estimation

    def fit(self, verbose: bool = True) -> "HMCVelo":
        """
        Estimate gene-specific [βM, βD] via constrained least squares.

        Sets self.beta_M and self.beta_D, both shape (n_genes,).
        βH is fixed to 1.0 by scale invariance.
        """
        if verbose:
            print(f"[HMCVelo] Fitting {self.data.n_genes} genes "
                  f"across {self.data.n_cells} cells...")

        self.beta_M, self.beta_D = least_squares_params(
            h=self.data.h,
            m=self.data.m,
            c=self.data.c,
            min_cells=self.min_cells,
            verbose=verbose,
        )
        return self

    # Step 2: velocity computation


    def compute_velocity(self, verbose: bool = True) -> "HMCVelo":
        """
        Compute hydroxymethylation velocity for all cells × genes.

        v[i, g] = dh_g/dt at cell i  =  βH * m[i,g] - βD[g] * h[i,g]
                                          (fast mode, Eq. 10)
        or numerically integrated (exact mode).

        Sets self.velocity, shape (n_cells, n_genes).
        """
        if self.beta_M is None:
            raise RuntimeError("Call .fit() before .compute_velocity()")

        if verbose:
            mode = "analytic approx" if self.fast else f"odeint (dt={self.dt})"
            print(f"[HMCVelo] Computing velocity ({mode})...")

        if self.fast:
            self.velocity = solve_model_fast(
                m0=self.data.m,
                h0=self.data.h,
                c0=self.data.c,
                beta_M=self.beta_M,
                beta_D=self.beta_D,
                beta_H=self.beta_H,
            )
        else:
            self.velocity = solve_model(
                m0=self.data.m,
                h0=self.data.h,
                c0=self.data.c,
                beta_M=self.beta_M,
                beta_D=self.beta_D,
                beta_H=self.beta_H,
                dt=self.dt,
            )

        if verbose:
            finite = np.isfinite(self.velocity).mean() * 100
            print(f"[HMCVelo] Done. {finite:.1f}% finite velocity values.")

        return self

    def compute_demethylation_velocity(self) -> np.ndarray:
        """
        Demethylation velocity (Eq. 11) for non-brain tissues.
        d(t) = dh/dt + dc/dt = βH * m - βM * c
        """
        if self.beta_M is None:
            raise RuntimeError("Call .fit() first")
        self._demethylation_velocity = demethylation_velocity(
            m0=self.data.m,
            h0=self.data.h,
            c0=self.data.c,
            beta_M=self.beta_M,
            beta_D=self.beta_D,
            beta_H=self.beta_H,
        )
        return self._demethylation_velocity

    # Step 3: inject into AnnData for scVelo compatibility

    def to_anndata(self):
        """
        Inject velocity and OMS into the HMCData's AnnData object
        so scVelo's embedding and plotting tools can be used directly.

        Adds:
          adata.layers['velocity']  <- HMC velocity
          adata.layers['spliced']   <- 5hmC  (OMS compat)
          adata.layers['unspliced'] <- 5mC
          adata.X                   <- OMS = 5hmC + 5mC  (embedding basis)

        Returns the modified AnnData.
        """
        if self.data.adata is None:
            raise ValueError("No AnnData found in HMCData. Use load_from_h5ad().")
        if self.velocity is None:
            raise RuntimeError("Call .compute_velocity() first")

        import scipy.sparse as sp

        adata = self.data.adata
        adata.layers["velocity"]  = self.velocity
        adata.layers["spliced"]   = self.data.h      # 5hmC
        adata.layers["unspliced"] = self.data.m      # 5mC
        adata.X = sp.csr_matrix(self.data.obs_methylation_state)

        return adata

    # run full pipeline in one call

    def run(self, verbose: bool = True) -> "HMCVelo":
        """Fit parameters and compute velocity in one call."""
        return self.fit(verbose=verbose).compute_velocity(verbose=verbose)

    # Summary

    def summary(self) -> dict:
        """Return a summary dict of the fitted model."""
        if self.velocity is None:
            return {"status": "not fitted"}

        nonzero_genes = int(((self.beta_M > 0) | (self.beta_D > 0)).sum())
        return {
            "n_cells": self.data.n_cells,
            "n_genes": self.data.n_genes,
            "nonzero_genes": nonzero_genes,
            "beta_H": self.beta_H,
            "dt": self.dt,
            "fast_mode": self.fast,
            "velocity_mean": float(np.nanmean(np.abs(self.velocity))),
            "velocity_finite_pct": float(np.isfinite(self.velocity).mean() * 100),
        }

    def __repr__(self):
        status = "fitted" if self.velocity is not None else "not fitted"
        return (f"HMCVelo({status}, n_cells={self.data.n_cells}, "
                f"n_genes={self.data.n_genes})")