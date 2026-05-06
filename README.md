# Sparse Autodiff Hessians Artifact

Anonymous code for the sparse Hessian recovery experiments reported in the paper. The repository contains:

- `colpackpy/`: Python bindings and PyTorch/NumPy recovery utilities for ColPack coloring.
- `matgl/src/`: MatGL source with the Hessian sparsity-pattern and Hessian-vector-product changes used by the experiments.
- `matgl/examples/`: paper experiment and plotting scripts.
- `data/`: small molecular/geometric inputs used by the benchmark scripts.
- `matgl/examples/sparse_hessian_results/`: retained result summaries and reference Hessians used for paper figure regeneration.


## Setup

Use Python 3.10 on Linux. The ColPack extension requires CMake and a C++17 compiler.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel cmake pybind11

# PyTorch/DGL stack used by the experiments.
python -m pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/repo.html

# Local packages and experiment dependencies.
python -m pip install -e ./matgl
python -m pip install -e ./colpackpy
python -m pip install ase scipy matplotlib pandas
```


## Run Instructions

Run commands from the repository root.

Validate the recovery utilities:

```bash
python -m unittest discover colpackpy/tests
```


Run the main random-TensorNet benchmarks. These are GPU-oriented and can be expensive for the larger systems:

```bash
python matgl/examples/sparse_hessian_benchmark.py --alkanes 
python matgl/examples/sparse_hessian_benchmark.py --water-cluster 
python matgl/examples/sparse_hessian_benchmark.py --abeta40 
```

Reanalyze the retained ORCA reference Hessian without rerunning quantum chemistry:

```bash
python matgl/examples/analyze_orca_hessian.py \
  --hessian matgl/examples/sparse_hessian_results/cutoff_experiments_chingolin_orca_pbe/pbe_orca_hessian_chingolin.hess
```

Run optional cutoff sweeps with fresh reference Hessians:

```bash
python matgl/examples/sparse_hessian_cutoff_xtb.py \
  --structure data/Chignolin/Coordinates_9999.xyz \
  --cutoffs 5,8,11,14

python matgl/examples/sparse_hessian_cutoff_dft.py \
  --structure data/Chignolin/Coordinates_9999.xyz \
  --backend pyscf \
  --basis sto-3g \
  --cutoffs 5,8,11
```

All experiment outputs are written under `matgl/examples/sparse_hessian_results/` by default.

## Anonymization Notes

This artifact is prepared for anonymous review. Local absolute paths, personal filenames, editor settings, generated binaries, and full ORCA output logs have been removed. Retained result metadata uses repository-relative paths.
