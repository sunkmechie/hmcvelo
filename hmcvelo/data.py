"""
hmcvelo.data
------------
Data loading and preprocessing for both HMCVelo experiments.

Experiment 1 : raw read counts from .bismark.cov files
               5hmC from bisulfite-assisted ACE-seq (GSE236784)
               5mC  = bisulfite-seq (GSE236789) - ACE-seq  [subtraction]

Experiment 2 : normalized proportions from .h5ad (AnnData / Seurat)
               xh = layer '5hmC'
               xm = layer '5mC'
               xc = 1 - xm - xh  [inferred by conservation law]
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union
from dataclasses import dataclass

try:
    import anndata as ad
    HAS_ANNDATA = True
except ImportError:
    HAS_ANNDATA = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


@dataclass
class HMCData:
    """
    Holds the three cytosine state arrays needed by HMCVelo.

    All arrays are shape (n_cells, n_genes).
    For Experiment 1: absolute read counts.
    For Experiment 2: proportions in [0, 1], summing to 1 per site.

    Attributes
    ----------
    h        : 5-hydroxymethylcytosine (5hmC)
    m        : 5-methylcytosine (5mC)
    c        : unmodified cytosine
    obs_names: cell identifiers
    var_names: gene / bin identifiers
    cell_type: optional cell-type label per cell
    adata    : original AnnData if loaded from h5ad
    """
    h: np.ndarray
    m: np.ndarray
    c: np.ndarray
    obs_names: np.ndarray
    var_names: np.ndarray
    cell_type: Optional[np.ndarray] = None
    adata: Optional[object] = None   

    @property
    def n_cells(self) -> int:
        return self.h.shape[0]

    @property
    def n_genes(self) -> int:
        return self.h.shape[1]

    @property
    def obs_methylation_state(self) -> np.ndarray:
        """OMS = 5hmC + 5mC — used as embedding input (Section 4.1)."""
        return self.h + self.m

    def __repr__(self):
        return (f"HMCData(n_cells={self.n_cells}, n_genes={self.n_genes}, "
                f"cell_types={np.unique(self.cell_type) if self.cell_type is not None else None})")



# Experiment 2: AnnData / h5ad


def load_from_anndata(
    adata,
    layer_hmC: str = "5hmC",
    layer_mC:  str = "5mC",
    celltype_key: Optional[str] = "CellType",
) -> HMCData:
    """
    Load HMCData from an AnnData object (Experiment 2).

    Expects two layers with proportions for 5hmC and 5mC.
    Unmodified cytosine is inferred via conservation law:
        xc = 1 - xm - xh

    Parameters
    ----------
    adata        : AnnData object (from h5ad or converted Seurat)
    layer_hmC    : name of the 5hmC proportion layer
    layer_mC     : name of the 5mC proportion layer
    celltype_key : obs column with cell-type labels (or None)

    Returns
    -------
    HMCData
    """
    if not HAS_ANNDATA:
        raise ImportError("anndata is required: pip install anndata")

    # Extract proportions
    xh = _to_dense(adata.layers[layer_hmC])
    xm = _to_dense(adata.layers[layer_mC])
    xc = np.clip(1.0 - xm - xh, 0.0, 1.0)

    cell_type = None
    if celltype_key and celltype_key in adata.obs:
        cell_type = adata.obs[celltype_key].to_numpy().astype(str)

    return HMCData(
        h=xh.astype(np.float64),
        m=xm.astype(np.float64),
        c=xc.astype(np.float64),
        obs_names=adata.obs_names.to_numpy(),
        var_names=adata.var_names.to_numpy(),
        cell_type=cell_type,
        adata=adata,
    )


def load_from_h5ad(
    path: Union[str, Path],
    layer_hmC: str = "5hmC",
    layer_mC:  str = "5mC",
    celltype_key: Optional[str] = "CellType",
) -> HMCData:
    """Load HMCData from an .h5ad file (Experiment 2)."""
    if not HAS_ANNDATA:
        raise ImportError("anndata is required: pip install anndata")
    adata = ad.read_h5ad(str(path))
    return load_from_anndata(adata, layer_hmC, layer_mC, celltype_key)


# Experiment 1: bismark.cov files

def load_from_bismark_pair(
    ace_seq_files:  list,
    bisulfite_files: list,
    bin_size_kb: int = 100,
    min_coverage: int = 1,
    celltype_labels: Optional[list] = None,
) -> HMCData:
    """
    Load HMCData from paired bismark.cov files (Experiment 1).

    ACE-seq gives 5hmC directly (subtraction-free).
    Bisulfite-seq gives 5mC + 5hmC.
    5mC = bisulfite - ACE-seq.
    Unmodified C is inferred via conservation law.

    Parameters
    ----------
    ace_seq_files    : list of .bismark.cov paths for ACE-seq (5hmC), one per cell
    bisulfite_files  : list of .bismark.cov paths for BS-seq,  one per cell
    bin_size_kb      : genomic bin size in kb (paper uses 100)
    min_coverage     : minimum read coverage to keep a site
    celltype_labels  : optional per-cell type labels

    Returns
    -------
    HMCData
    """
    if not HAS_PANDAS:
        raise ImportError("pandas is required: pip install pandas")
    if len(ace_seq_files) != len(bisulfite_files):
        raise ValueError("ace_seq_files and bisulfite_files must be the same length")

    import pandas as pd

    n_cells = len(ace_seq_files)
    bin_size_bp = bin_size_kb * 1000

    # ---- first pass: collect all bin IDs across all cells ----
    all_bins = set()
    for f in ace_seq_files:
        df = _read_bismark(f, min_coverage)
        bins = _assign_bins(df, bin_size_bp)
        all_bins.update(bins["bin_id"].unique())

    bin_ids = sorted(all_bins)
    bin_index = {b: i for i, b in enumerate(bin_ids)}
    n_genes = len(bin_ids)

    h_mat = np.zeros((n_cells, n_genes), dtype=np.float64)
    m_mat = np.zeros((n_cells, n_genes), dtype=np.float64)

    # ---- second pass: fill matrices ----
    for cell_i, (ace_f, bs_f) in enumerate(zip(ace_seq_files, bisulfite_files)):
        ace_df = _read_bismark(ace_f, min_coverage)
        bs_df  = _read_bismark(bs_f,  min_coverage)

        ace_bins = _assign_bins(ace_df, bin_size_bp)
        bs_bins  = _assign_bins(bs_df,  bin_size_bp)

        # 5hmC per bin (ACE-seq)
        ace_agg = ace_bins.groupby("bin_id")["methylated"].sum()
        for bin_id, val in ace_agg.items():
            if bin_id in bin_index:
                h_mat[cell_i, bin_index[bin_id]] = val

        # 5mC per bin = BS - ACE, clip to 0
        bs_agg  = bs_bins.groupby("bin_id")["methylated"].sum()
        for bin_id, val in bs_agg.items():
            if bin_id in bin_index:
                ace_val = h_mat[cell_i, bin_index[bin_id]]
                m_mat[cell_i, bin_index[bin_id]] = max(0.0, val - ace_val)

    # Counts-per-cell normalization (Episcanpy-style)
    h_mat = _normalize_cpc(h_mat)
    m_mat = _normalize_cpc(m_mat)

    # Unmodified C from conservation law
    total = h_mat + m_mat
    # For read counts: c is inferred as the complement within the total coverage
    # We use a pragmatic estimate: c = max(0, total_coverage - total_modified)
    # Without sequencing depth info per bin, approximate as:
    c_mat = np.clip(total.max(axis=0, keepdims=True) - total, 0.0, None)

    obs_names = np.array([Path(f).stem for f in ace_seq_files])
    var_names = np.array(bin_ids)
    cell_type = np.array(celltype_labels) if celltype_labels else None

    return HMCData(
        h=h_mat,
        m=m_mat,
        c=c_mat,
        obs_names=obs_names,
        var_names=var_names,
        cell_type=cell_type,
    )


def make_synthetic(
    n_cells: int = 100,
    n_genes: int = 200,
    n_celltypes: int = 4,
    seed: int = 42,
) -> HMCData:
    """
    Generate synthetic HMCData for testing and demos.

    Simulates a bifurcating trajectory: NeuN- -> Ex / Inh,
    with NeuN+ as terminal state, matching the paper's setup.

    Each cell type has a different steady-state balance between
    5hmC, 5mC, and unmodified C, reflecting differentiation.
    """
    rng = np.random.default_rng(seed)
    celltype_names = ["NeuN-", "NeuN+", "Ex", "Inh"]
    labels_per_type = n_cells // n_celltypes

    # Steady-state proportions per cell type (xh, xm, xc)
    ss = {
        "NeuN-": (0.10, 0.50, 0.40),   # immature: low 5hmC
        "NeuN+": (0.25, 0.40, 0.35),   # mature: higher 5hmC
        "Ex":    (0.30, 0.35, 0.35),   # excitatory: high 5hmC
        "Inh":   (0.20, 0.45, 0.35),   # inhibitory: moderate
    }

    all_h, all_m, all_c, all_ct = [], [], [], []

    for ct in celltype_names:
        xh_mu, xm_mu, xc_mu = ss[ct]
        # Add gene-level and cell-level noise
        gene_factor = rng.gamma(2.0, 0.5, size=n_genes)
        for _ in range(labels_per_type):
            noise = rng.normal(0, 0.03, size=n_genes)
            xh = np.clip(xh_mu * gene_factor + noise, 0, 1)
            xm = np.clip(xm_mu * gene_factor + noise, 0, 1)
            # re-normalize to simplex
            total = xh + xm + 1e-6
            scale = rng.uniform(0.8, 1.0)
            xh = xh / total * scale
            xm = xm / total * scale
            xc = np.clip(1.0 - xh - xm, 0, 1)
            all_h.append(xh)
            all_m.append(xm)
            all_c.append(xc)
            all_ct.append(ct)

    h = np.array(all_h)
    m = np.array(all_m)
    c = np.array(all_c)

    return HMCData(
        h=h.astype(np.float64),
        m=m.astype(np.float64),
        c=c.astype(np.float64),
        obs_names=np.array([f"cell_{i}" for i in range(len(all_ct))]),
        var_names=np.array([f"bin_{g}" for g in range(n_genes)]),
        cell_type=np.array(all_ct),
    )


def _to_dense(x) -> np.ndarray:
    """Convert sparse or dense matrix to a dense float64 ndarray."""
    if hasattr(x, "toarray"):
        return x.toarray().astype(np.float64)
    return np.asarray(x, dtype=np.float64)


def _normalize_cpc(mat: np.ndarray) -> np.ndarray:
    """Counts-per-cell normalization: divide each cell by its total count."""
    totals = mat.sum(axis=1, keepdims=True)
    totals = np.where(totals == 0, 1.0, totals)
    return mat / totals * 1e4   # scale to 10k like scanpy default


def _read_bismark(path: Union[str, Path], min_coverage: int = 1):
    """Read a bismark.cov file into a DataFrame."""
    import pandas as pd
    df = pd.read_csv(
        str(path), sep="\t", header=None,
        names=["chrom", "start", "end", "pct_meth", "methylated", "unmethylated"],
    )
    coverage = df["methylated"] + df["unmethylated"]
    return df[coverage >= min_coverage].copy()


def _assign_bins(df, bin_size_bp: int):
    """Assign each CpG to a 100-Kb genomic bin."""
    import pandas as pd
    df = df.copy()
    df["bin_start"] = (df["start"] // bin_size_bp) * bin_size_bp
    df["bin_id"] = df["chrom"] + ":" + df["bin_start"].astype(str)
    return df