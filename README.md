# HMCVelo

**An unofficial implementation of HMCVelo** — a deterministic velocity model for DNA hydroxymethylation dynamics in single cells.

Based on: *Mishra, P. "HMCVelo: A Deterministic Model for Hydroxymethylation Velocity in Single Cells." bioRxiv (2026).* [`doi:10.64898/2026.04.20.719607`](https://doi.org/10.64898/2026.04.20.719607)

> Official repo (forthcoming): https://github.com/prmshr/HMCVelo

---

## What is HMCVelo?

RNA velocity infers cellular trajectories from spliced/unspliced mRNA ratios. HMCVelo does the same for **DNA methylation** — but methylation is cyclic (C → 5mC → 5hmC → C), not unidirectional like splicing. This requires a fundamentally different model.

HMCVelo models the methylation–demethylation cycle as three coupled ODEs:

```
dm/dt = βM·c  −  βH·m        (methylation,       C  → 5mC)
dh/dt = βH·m  −  βD·h        (hydroxymethylation, 5mC → 5hmC)
dc/dt = βD·h  −  βM·c        (demethylation,      5hmC → C)
```

With βH = 1 fixed by scale invariance, gene-specific parameters [βM, βD] are estimated at steady state via constrained least-squares. Velocity is `v = dh/dt` per cell per gene.

Applied to murine cortical neurons (n=519), HMCVelo achieves velocity confidence >0.89 across all cell types — vs <0.45 for RNA velocity repurposed on the same data.

---

## Installation

```bash
git clone https://github.com/sunkmechie/HMCVelo
cd HMCVelo
uv pip install -e ".[embed]"
```

**Core only** (no plotting):
```bash
uv pip install -e .
```

**With GPU acceleration:**
```bash
uv pip install -e ".[embed,gpu]"
```

---

## Quick Start

### Synthetic data (no download needed)

```python
from hmcvelo import HMCVelo
from hmcvelo.data import make_synthetic

# Generate synthetic bifurcating trajectory
# mimics the paper's NeuN- -> Ex/Inh setup
data = make_synthetic(n_cells=200, n_genes=500, seed=42)

# Fit and compute velocity
model = HMCVelo(data).run()
print(model.summary())

# Feature ranking
from hmcvelo.rank import rank_velocity_genes
ranked = rank_velocity_genes(model.velocity, data.cell_type, data.var_names)
```

### Real data (Experiment 2, h5ad)

```python
from hmcvelo.data import load_from_h5ad
from hmcvelo import HMCVelo
from hmcvelo.embed import run_embedding_pipeline

data = load_from_h5ad(
    "path/to/data.h5ad",
    layer_hmC="5hmC",
    layer_mC="5mC",
    celltype_key="CellType",
)

model = HMCVelo(data).run()

# Full embedding pipeline -> scVelo-compatible AnnData
adata = run_embedding_pipeline(data, model, method="umap")

# Plots
from hmcvelo.embed import plot_velocity_stream, plot_velocity_grid
plot_velocity_stream(adata, color="cell_type")
plot_velocity_grid(adata, color="louvain")
```

### Real data (Experiment 1, bismark.cov files)

```python
from glob import glob
from hmcvelo.data import load_from_bismark_pair
from hmcvelo import HMCVelo

# After downloading GSE236784 (ACE-seq) and GSE236789 (bisulfite-seq)
# Files must be matched by cell barcode order
ace_files = sorted(glob("data/ace_seq/*.cov.gz"))
bs_files  = sorted(glob("data/bisulfite/*.cov.gz"))

data = load_from_bismark_pair(
    ace_seq_files=ace_files,
    bisulfite_files=bs_files,
    bin_size_kb=100,           # paper uses 100-Kb bins
    celltype_labels=[...],     # from GEO sample metadata
)

model = HMCVelo(data).run()
```

---

## Data

### Experiment 1 — publicly available

GEO SuperSeries: [`GSE236798`](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE236798)

| SubSeries | Content | Size |
|-----------|---------|------|
| [`GSE236784`](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE236784) | ACE-seq → 5hmC (565 cells) | ~40 GB |
| [`GSE236789`](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE236789) | Bisulfite-seq → 5mC+5hmC (519 cells) | ~60 GB |

```bash
mkdir -p data/ace_seq data/bisulfite

wget -P data/raw/ https://ftp.ncbi.nlm.nih.gov/geo/series/GSE236nnn/GSE236784/suppl/GSE236784_RAW.tar
wget -P data/raw/ https://ftp.ncbi.nlm.nih.gov/geo/series/GSE236nnn/GSE236789/suppl/GSE236789_RAW.tar

tar -xf data/raw/GSE236784_RAW.tar -C data/ace_seq/
tar -xf data/raw/GSE236789_RAW.tar -C data/bisulfite/
```

### Experiment 2 — not yet public

The normalized `.h5ad` proportion data is Wu lab in-house. Check the [official repo](https://github.com/prmshr/HMCVelo) or contact the authors when it is released.


---

## Project Structure

```
HMCVelo/
├── hmcvelo/
│   ├── __init__.py     # public API
│   ├── data.py         # data loading (bismark.cov, h5ad, synthetic)
│   ├── params.py       # least-squares parameter estimation
│   ├── model.py        # ODE system + numerical/analytic solver
│   ├── velocity.py     # HMCVelo pipeline class
│   ├── embed.py        # dimensionality reduction + velocity embedding
│   └── rank.py         # feature ranking via Welch t-test
├── clifford/
│   └── hmcvelo_ga.py   # Clifford algebra reformulation (experimental)
├── notebooks/
│   └── demo.ipynb      # end-to-end demo on synthetic data
├── tests/
│   └── test_core.py    # full test suite
└── pyproject.toml
```

---

## Key Implementation Notes

### Embedding basis (Remark 1)
The paper proves that embedding must use **OMS = 5hmC + 5mC**, not unmodified cytosine xc. Since m+h+c=const, xc is algebraically redundant and collapses bifurcations into a horseshoe artifact. This implementation enforces OMS as `adata.X`.

### Scale invariance
βH is fixed to 1.0, reducing parameter space from 3 to 2 free parameters per gene. This affects velocity magnitude but not direction.

### Fast vs exact mode
```python
# Fast (default): analytic approximation v = βH*m - βD*h
# Equivalent to odeint at small dt, runs in seconds
HMCVelo(data, fast=True)

# Exact: scipy odeint, dt=1e-5
# Ground truth, slower for large datasets
HMCVelo(data, fast=False, dt=1e-5)
```

### Demethylation velocity (non-brain tissues)
```python
# For tissues where full demethylation is more relevant than 5hmC alone
d = model.compute_demethylation_velocity()   # d = βH*m - βM*c
```

---

## Tests

```bash
uv pip install -e ".[dev]"
uv run pytest tests/ -v
```

---

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{mishra2026hmcvelo,
  title   = {HMCVelo: A Deterministic Model for Hydroxymethylation Velocity in Single Cells},
  author  = {Mishra, Paramita},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.64898/2026.04.20.719607}
}
```

And the Joint-snhmC-seq data paper:

```bibtex
@article{fabyanic2024joint,
  title   = {Joint single-cell profiling resolves 5mC and 5hmC and reveals their distinct gene regulatory effects},
  author  = {Fabyanic, Evan B and Hu, Peng and Qiu, Qi and others},
  journal = {Nature Biotechnology},
  volume  = {42},
  pages   = {960--974},
  year    = {2024},
  doi     = {10.1038/s41587-023-01909-2}
}
```

---

## License

MIT