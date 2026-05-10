from .velocity import HMCVelo
from .data import (
    HMCData,
    make_synthetic,
    load_from_h5ad,
    load_from_anndata,
    load_from_bismark_pair,
)
from .params import least_squares_params, BETA_H_FIXED, infer_cytosine_proportions
from .model import (
    solve_model,
    solve_model_fast,
    demethylation_velocity,
)

__version__ = "0.1.0"
__author__  = "Surya Sunkara"
__paper__   = "https://doi.org/10.64898/2026.04.20.719607"

__all__ = [
    # Main class
    "HMCVelo",
    # Data
    "HMCData",
    "make_synthetic",
    "load_from_h5ad",
    "load_from_anndata",
    "load_from_bismark_pair",
    "infer_cytosine_proportions",
    # Parameters
    "least_squares_params",
    "BETA_H_FIXED",
    # Model
    "solve_model",
    "solve_model_fast",
    "demethylation_velocity",
]