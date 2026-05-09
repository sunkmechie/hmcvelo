# HMCVelo — Hydroxymethylation Velocity for Single Cells

An **unofficial, open‑source implementation** of *HMCVelo* (Parmita Mishra, 2026).
Plus an experimental **Clifford (geometric) algebra reformulation**.  


**Paper**: [bioRxiv 10.64898/2026.04.20.719607](https://www.biorxiv.org/content/10.64898/2026.04.20.719607v1)  
**Status**: Incomplete


## What makes HMCVelo different?

- **First velocity model on DNA methylation**
- Models the full **cyclic methylation‑demethylation pathway**: `C ⇄ 5mC ⇄ 5hmC`.
- **Scale invariance** reduces 3 kinetic parameters to 2 free parameters per gene, solved with constrained least squares.
- **Mathematically rigorous**: proves that embedding on unmodified cytosine collapses trajectory bifurcations, a theorem with implications for all cyclic biochemical systems.
- Velocity confidence **>0.89** on real cortical neurons, while RNA velocity fails (<0.45).

This repo provides both a reference implementation and an experimental *Clifford algebra formulation*.


## Project structure


```

hmcvelo-package/
├── hmcvelo/              # Standard HMCVelo (faithful to paper)
│   ├── __init__.py
│   ├── model.py          # ODE system + numerical integration
│   ├── params.py         # Steady‑state least‑squares estimation
│   ├── velocity.py       # Velocity vector computation per cell/gene
│   ├── embed.py          # Dimensionality reduction & velocity embedding (scVelo‑compatible)
│   ├── data.py           # Synthetic data generation & preprocessing
│   └── rank.py           # Velocity gene ranking (Welch t‑test)
├── clifford/             # Experimental: Clifford (geometric) algebra reformulation
│   └── hmcvelo_ga.py     # GA‑based state representation, rotor evolution, analytical solutions
├── notebooks/
│   └── demo.ipynb        # Full demonstration on synthetic data
├── tests/
│   └── test_core.py      # Unit tests for core functions
├── HMCVelo_paper.md      # Full paper in markdown (for reference)
├── pyproject.toml
├── README.md
└── uv.lock

```



## Installation


```bash
git clone [https://github.com/your-org/hmcvelo.git](https://github.com/your-org/hmcvelo.git)
cd hmcvelo
uv sync

```

Core dependencies: `numpy`, `scipy`, `anndata`, `scvelo`, `scanpy`, `matplotlib`.



Check `notebooks/demo.ipynb` for a step‑by‑step walkthrough, including how to validate velocity confidence and compare with RNA velocity.



## Reference


```bibtex
@article{Mishra2026HMCVelo,
  title={HMCVelo: A Deterministic Model for Hydroxymethylation Velocity in Single Cells},
  author={Mishra, Paramita},
  journal={bioRxiv},
  year={2026},
  doi={10.64898/2026.04.20.719607}
}

```



## Contributing


Feel free to open issues, draft PRs, or just brainstorm in the Discussions tab.


## License

MIT
