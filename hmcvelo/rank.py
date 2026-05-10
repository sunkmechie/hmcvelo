"""
hmcvelo.rank
------------
Feature (gene/bin) ranking by differential velocity dynamics.

Method from paper (Section 4.4):
    Uses Welch t-test to find features per cell type showing
    velocity dynamics differentially regulated vs other cell types.
    Minimum correlation threshold: 0.3.
"""

import numpy as np
from typing import Optional
from scipy import stats

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import scvelo as scv
    HAS_SCVELO = True
except ImportError:
    HAS_SCVELO = False


def rank_velocity_genes(
    velocity: np.ndarray,
    cell_types: np.ndarray,
    var_names: np.ndarray,
    min_corr: float = 0.3,
    top_n: int = 50,
    verbose: bool = True,
) -> dict:
    """
    Rank genes by differential velocity dynamics across cell types.

    For each cell type, performs a one-vs-rest Welch t-test on the
    velocity values across genes. Returns a ranked list per cell type.

    Parameters
    ----------
    velocity   : (n_cells, n_genes) velocity array
    cell_types : (n_cells,) cell type label array
    var_names  : (n_genes,) gene/bin names
    min_corr   : minimum mean velocity magnitude to include a gene
    top_n      : number of top genes to return per cell type
    verbose    : print summary

    Returns
    -------
    dict mapping cell_type -> DataFrame with columns:
        gene, t_stat, p_value, mean_velocity, rank
    """
    unique_types = np.unique(cell_types)
    results = {}

    for ct in unique_types:
        mask_ct  = cell_types == ct
        mask_rest = ~mask_ct

        v_ct   = velocity[mask_ct]    # (n_ct, n_genes)
        v_rest = velocity[mask_rest]  # (n_rest, n_genes)

        # Welch t-test per gene
        t_stats = np.zeros(len(var_names))
        p_vals  = np.zeros(len(var_names))

        for g in range(len(var_names)):
            a = v_ct[:, g]
            b = v_rest[:, g]
            # Skip genes with no variance
            if np.std(a) < 1e-10 and np.std(b) < 1e-10:
                t_stats[g] = 0.0
                p_vals[g]  = 1.0
                continue
            try:
                t, p = stats.ttest_ind(a, b, equal_var=False)
                t_stats[g] = t if np.isfinite(t) else 0.0
                p_vals[g]  = p if np.isfinite(p) else 1.0
            except Exception:
                t_stats[g] = 0.0
                p_vals[g]  = 1.0

        mean_vel = np.abs(v_ct).mean(axis=0)

        # Apply minimum velocity filter (analogous to min_corr threshold)
        vel_threshold = np.quantile(mean_vel[mean_vel > 0], min_corr) if (mean_vel > 0).any() else 0.0
        keep = mean_vel >= vel_threshold

        # Rank by |t_stat| among kept genes
        ranked_idx = np.argsort(-np.abs(t_stats * keep.astype(float)))[:top_n]

        if HAS_PANDAS:
            import pandas as pd
            df = pd.DataFrame({
                "gene":         var_names[ranked_idx],
                "t_stat":       t_stats[ranked_idx],
                "p_value":      p_vals[ranked_idx],
                "mean_velocity": mean_vel[ranked_idx],
                "rank":         np.arange(1, len(ranked_idx) + 1),
            })
            results[ct] = df
        else:
            results[ct] = {
                "gene":          var_names[ranked_idx],
                "t_stat":        t_stats[ranked_idx],
                "p_value":       p_vals[ranked_idx],
                "mean_velocity": mean_vel[ranked_idx],
            }

        if verbose:
            top5 = ", ".join(var_names[ranked_idx[:5]])
            print(f"[HMCVelo] {ct}: top genes = {top5}")

    return results


def rank_velocity_genes_scvelo(
    adata,
    groupby: str = "cell_type",
    min_corr: float = 0.3,
    top_n: int = 50,
) -> dict:
    """
    Rank velocity genes using scVelo's built-in method (preferred).

    Wraps scvelo.tl.rank_velocity_genes which uses the same
    Welch t-test approach described in the paper.

    Parameters
    ----------
    adata   : AnnData with velocity layer and groupby column in .obs
    groupby : obs column to group by (default: 'cell_type')
    min_corr: minimum correlation threshold (paper uses 0.3)
    top_n   : genes to return per group

    Returns
    -------
    dict mapping group -> list of top gene names
    """
    if not HAS_SCVELO:
        raise ImportError("pip install scvelo")

    scv.tl.rank_velocity_genes(adata, groupby=groupby, min_corr=min_corr)

    results = {}
    if HAS_PANDAS:
        import pandas as pd
        df = scv.DataFrame(adata.uns["rank_velocity_genes"]["names"])
        for col in df.columns:
            results[col] = df[col].tolist()[:top_n]

    return results


def plot_velocity_genes(
    adata,
    genes: list,
    groupby: str = "louvain",
    n_cols: int = 3,
    figsize_per_gene: tuple = (4, 3),
    save: Optional[str] = None,
):
    """
    Plot phase portraits (5hmC vs 5mC) + velocity magnitude for top genes.

    Reproduces Figures 9-11 in the paper: each gene shows
    (a) phase portrait colored by cluster
    (b) velocity magnitude on embedding
    (c) expression on embedding
    """
    if not HAS_SCVELO:
        raise ImportError("pip install scvelo")

    import matplotlib.pyplot as plt

    n_genes = len(genes)
    n_rows  = int(np.ceil(n_genes / n_cols))
    fig_w   = figsize_per_gene[0] * n_cols * 3   # 3 panels per gene
    fig_h   = figsize_per_gene[1] * n_rows

    scv.pl.velocity(
        adata,
        genes,
        ncols=n_cols,
        figsize=(fig_w, fig_h),
        color=groupby,
        save=save,
    )


def velocity_confidence_table(
    adata,
    groupby: str = "cell_type",
) -> "pd.DataFrame":
    """
    Reproduce Table 2 from the paper: velocity confidence by cell type.

    Returns a DataFrame with rows = [velocity_length, velocity_confidence]
    and columns = cell types.
    """
    if not HAS_PANDAS:
        raise ImportError("pip install pandas")

    import pandas as pd

    if "velocity_confidence" not in adata.obs:
        raise RuntimeError("Run compute_velocity_graph() first.")

    groups = adata.obs[groupby].unique()
    rows = {"velocity_length": {}, "velocity_confidence": {}}

    for g in groups:
        mask = adata.obs[groupby] == g
        if "velocity_length" in adata.obs:
            rows["velocity_length"][g] = adata.obs.loc[mask, "velocity_length"].mean()
        rows["velocity_confidence"][g] = adata.obs.loc[mask, "velocity_confidence"].mean()

    df = pd.DataFrame(rows).T
    return df


def annotate_bins_to_genes(
    ranked: dict,
    bin_to_gene_map: Optional[dict] = None,
) -> dict:
    """
    Convert 100-Kb bin IDs to gene names.

    The paper maps bins to RefSeq gene IDs using UCSC Table Browser
    (one-to-one mapping, Section 5.2). This function applies a
    precomputed mapping dict.

    Parameters
    ----------
    ranked          : output of rank_velocity_genes()
    bin_to_gene_map : dict mapping bin_id -> gene_name
                      If None, returns bin IDs unchanged.

    Returns
    -------
    ranked dict with gene names instead of bin IDs
    """
    if bin_to_gene_map is None:
        return ranked

    annotated = {}
    for ct, data in ranked.items():
        if HAS_PANDAS:
            import pandas as pd
            df = data.copy()
            df["gene_name"] = df["gene"].map(
                lambda b: bin_to_gene_map.get(b, b)
            )
            annotated[ct] = df
        else:
            genes = [bin_to_gene_map.get(b, b) for b in data["gene"]]
            annotated[ct] = {**data, "gene_name": np.array(genes)}

    return annotated