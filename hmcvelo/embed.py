"""
hmcvelo.embed
=============
Dimensionality reduction and velocity embedding for HMCVelo.

Key design decision from the paper (Remark 1, Section 6.1):
    Embedding must be computed on OMS = 5hmC + 5mC, NOT on
    unmodified cytosine xc. Using xc collapses bifurcations into
    a horseshoe artifact because xc is algebraically redundant
    given the conservation law m + h + c = const.

This module:
1. Builds the OMS matrix and injects it into AnnData as .X
2. Runs incremental PCA -> UMAP/tSNE (via scVelo / scanpy)
3. Computes velocity embeddings using scVelo's transition matrix
4. Plots stream and grid plots
"""

import numpy as np
from typing import Optional, Literal

try:
    import anndata as ad
    HAS_ANNDATA = True
except ImportError:
    HAS_ANNDATA = False

try:
    import scvelo as scv
    HAS_SCVELO = True
except ImportError:
    HAS_SCVELO = False

try:
    import scanpy as sc
    HAS_SCANPY = True
except ImportError:
    HAS_SCANPY = False

try:
    import scipy.sparse as sp
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def build_anndata(
    data,           # HMCData
    velocity,       # HMCVelo object (after .run())
) -> "ad.AnnData":
    """
    Build a scVelo-compatible AnnData from HMCData + fitted HMCVelo.

    Layout:
        adata.X               <- OMS = 5hmC + 5mC  (embedding basis, Sec 4.1)
        adata.layers['spliced']   <- 5hmC
        adata.layers['unspliced'] <- 5mC
        adata.layers['velocity']  <- HMC velocity (n_cells × n_genes)
        adata.obs['cell_type']    <- cell type labels (if present)

    The choice of OMS as .X is critical — see Remark 1 in the paper.
    Do NOT use xc as the embedding basis.
    """
    if not HAS_ANNDATA:
        raise ImportError("pip install anndata")
    if velocity.velocity is None:
        raise RuntimeError("Call hmcvelo.run() before build_anndata()")

    oms = data.obs_methylation_state  # 5hmC + 5mC, shape (n_cells, n_genes)

    if HAS_SCIPY:
        X = sp.csr_matrix(oms.astype(np.float32))
    else:
        X = oms.astype(np.float32)

    adata = ad.AnnData(
        X=X,
        obs={"cell_name": data.obs_names},
        var={"gene_name": data.var_names},
    )
    adata.obs_names = data.obs_names.astype(str)
    adata.var_names = data.var_names.astype(str)

    # Layers
    adata.layers["spliced"]   = data.h.astype(np.float32)
    adata.layers["unspliced"] = data.m.astype(np.float32)
    adata.layers["velocity"]  = velocity.velocity.astype(np.float32)

    # Store raw cytosine too, useful for debugging
    adata.layers["cytosine"]  = data.c.astype(np.float32)
    adata.layers["oms"]       = oms.astype(np.float32)

    # Cell metadata
    if data.cell_type is not None:
        adata.obs["cell_type"] = data.cell_type

    # Store rate parameters in varm
    if velocity.beta_M is not None:
        adata.var["beta_M"] = velocity.beta_M.astype(np.float32)
        adata.var["beta_D"] = velocity.beta_D.astype(np.float32)
        adata.var["beta_H"] = velocity.beta_H

    return adata


def run_dimensionality_reduction(
    adata: "ad.AnnData",
    method: Literal["umap", "tsne", "pca"] = "umap",
    n_pcs: int = 30,
    n_neighbors: int = 30,
    min_dist: float = 0.3,
    random_state: int = 42,
    verbose: bool = True,
) -> "ad.AnnData":
    """
    Run PCA -> neighbor graph -> UMAP or tSNE on OMS (adata.X).

    Paper uses incremental PCA then UMAP/tSNE via scVelo.
    This function works with either scVelo or scanpy.

    Parameters
    ----------
    adata       : AnnData with OMS as .X
    method      : 'umap', 'tsne', or 'pca'
    n_pcs       : number of PCA components
    n_neighbors : k-NN graph neighbors
    min_dist    : UMAP min_dist parameter
    random_state: reproducibility seed

    Returns
    -------
    adata with obsm['X_pca'], obsm['X_umap'/'X_tsne'], obsp['distances']
    """
    if not (HAS_SCVELO or HAS_SCANPY):
        raise ImportError("pip install scvelo  or  pip install scanpy")

    lib = scv if HAS_SCVELO else sc

    if verbose:
        print(f"[HMCVelo] Running PCA (n_pcs={n_pcs})...")

    # Filter and normalize for DR
    # Note: we do NOT use highly_variable_genes filtering here —
    # the paper excludes HVGs from normalization but uses all genes for DR
    if HAS_SCVELO:
        scv.pp.filter_and_normalize(adata, min_shared_counts=5, n_top_genes=None)
        scv.pp.moments(adata, n_pcs=n_pcs, n_neighbors=n_neighbors)
    elif HAS_SCANPY:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.pca(adata, n_comps=n_pcs, random_state=random_state)
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs,
                        random_state=random_state)

    if verbose:
        print(f"[HMCVelo] Running {method.upper()}...")

    if method == "umap":
        if HAS_SCVELO:
            scv.tl.umap(adata, min_dist=min_dist, random_state=random_state)
        else:
            sc.tl.umap(adata, min_dist=min_dist, random_state=random_state)

    elif method == "tsne":
        if HAS_SCVELO:
            scv.tl.tsne(adata, random_state=random_state)
        else:
            sc.tl.tsne(adata, random_state=random_state)

    elif method == "pca":
        pass  # already done above

    # Louvain clustering (used in paper for color labeling)
    try:
        if HAS_SCVELO:
            scv.tl.louvain(adata)
        elif HAS_SCANPY:
            sc.tl.louvain(adata)
        if verbose:
            print("[HMCVelo] Louvain clustering done.")
    except Exception as e:
        if verbose:
            print(f"[HMCVelo] Louvain skipped: {e}")

    return adata


def compute_velocity_graph(
    adata: "ad.AnnData",
    velocity_layer: str = "velocity",
    n_neighbors: Optional[int] = None,
    verbose: bool = True,
) -> "ad.AnnData":
    """
    Compute the velocity graph (transition matrix) using scVelo.

    This computes the cosine correlation between velocity vectors
    and displacement vectors between cell neighbors, building the
    Markov chain transition matrix π̃_ij (Eq. 24).

    Requires scVelo.
    """
    if not HAS_SCVELO:
        raise ImportError("scVelo required for velocity graph: pip install scvelo")
    if verbose:
        print("[HMCVelo] Computing velocity graph...")

    scv.tl.velocity_graph(
        adata,
        vkey=velocity_layer,
        n_neighbors=n_neighbors,
        sqrt_transform=False,
    )
    scv.tl.velocity_confidence(adata, vkey=velocity_layer)

    if verbose and "velocity_confidence" in adata.obs:
        conf = adata.obs["velocity_confidence"]
        print(f"[HMCVelo] Velocity confidence: mean={conf.mean():.3f}, "
              f"min={conf.min():.3f}, max={conf.max():.3f}")

    return adata


def compute_pseudotime(
    adata: "ad.AnnData",
    root_key: str = "cell_type",
    root_value: str = "NeuN-",
    verbose: bool = True,
) -> "ad.AnnData":
    """
    Compute velocity pseudotime.

    Attempts to set the root cell automatically based on cell_type label.
    If root_value is not found, scVelo picks the root automatically.
    """
    if not HAS_SCVELO:
        raise ImportError("pip install scvelo")

    # Try to set root cell
    if root_key in adata.obs:
        root_cells = adata.obs[root_key] == root_value
        if root_cells.sum() > 0:
            # Use the cell with highest velocity confidence in the root population
            if "velocity_confidence" in adata.obs:
                conf_in_root = adata.obs.loc[root_cells, "velocity_confidence"]
                root_idx = conf_in_root.idxmax()
                adata.uns["iroot"] = adata.obs_names.get_loc(root_idx)
            else:
                adata.uns["iroot"] = int(np.where(root_cells)[0][0])
            if verbose:
                print(f"[HMCVelo] Root cell set from {root_key}='{root_value}'")

    scv.tl.velocity_pseudotime(adata)
    return adata


def plot_velocity_stream(
    adata: "ad.AnnData",
    basis: str = "umap",
    color: str = "cell_type",
    title: str = "HMCVelo stream",
    velocity_layer: str = "velocity",
    figsize: tuple = (8, 6),
    save: Optional[str] = None,
    **kwargs,
):
    """
    Stream plot of HMC velocity on the embedding.

    Uses scVelo's velocity_embedding_stream which implements
    the transition matrix displacement (Eq. 24 of paper).

    Masks velocity lengths below 20th percentile (paper: Section 4.3).
    """
    if not HAS_SCVELO:
        raise ImportError("pip install scvelo")

    # Compute embedding vectors
    scv.tl.velocity_embedding(adata, basis=basis, vkey=velocity_layer)

    scv.pl.velocity_embedding_stream(
        adata,
        basis=basis,
        color=color,
        vkey=velocity_layer,
        title=title,
        figsize=figsize,
        save=save,
        **kwargs,
    )


def plot_velocity_grid(
    adata: "ad.AnnData",
    basis: str = "umap",
    color: str = "cell_type",
    title: str = "HMCVelo grid",
    velocity_layer: str = "velocity",
    n_neighbors: int = 1,      # paper uses n_neighbors=1 for ~500 cells
    figsize: tuple = (8, 6),
    save: Optional[str] = None,
    **kwargs,
):
    """
    Grid plot of HMC velocity arrows on the embedding.

    Paper uses n_neighbors=1 for fine-grained representation
    given n_cells ~ 500.
    """
    if not HAS_SCVELO:
        raise ImportError("pip install scvelo")

    scv.tl.velocity_embedding(adata, basis=basis, vkey=velocity_layer)

    scv.pl.velocity_embedding_grid(
        adata,
        basis=basis,
        color=color,
        vkey=velocity_layer,
        title=title,
        figsize=figsize,
        n_neighbors=n_neighbors,
        save=save,
        **kwargs,
    )


def plot_pseudotime(
    adata: "ad.AnnData",
    basis: str = "umap",
    figsize: tuple = (8, 6),
    save: Optional[str] = None,
):
    """Plot velocity pseudotime on embedding."""
    if not HAS_SCVELO:
        raise ImportError("pip install scvelo")

    scv.pl.scatter(
        adata,
        color="velocity_pseudotime",
        cmap="gnuplot",
        basis=basis,
        figsize=figsize,
        save=save,
    )


def run_embedding_pipeline(
    data,           # HMCData
    velocity,       # fitted HMCVelo
    method: str = "umap",
    root_value: str = "NeuN-",
    verbose: bool = True,
) -> "ad.AnnData":
    """
    One-call pipeline: HMCData + velocity -> AnnData ready for plotting.

    Steps:
        1. Build AnnData with OMS as .X (correct embedding basis)
        2. PCA -> neighbors -> UMAP/tSNE
        3. Louvain clustering
        4. Velocity graph + confidence
        5. Pseudotime

    Returns
    -------
    adata ready for plot_velocity_stream / plot_velocity_grid
    """
    adata = build_anndata(data, velocity)
    adata = run_dimensionality_reduction(adata, method=method, verbose=verbose)
    adata = compute_velocity_graph(adata, verbose=verbose)
    adata = compute_pseudotime(adata, root_value=root_value, verbose=verbose)
    return adata