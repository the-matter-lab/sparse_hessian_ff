# Colour me shocked: Exact Molecular Hessians from local MLIPs in O(N) time using sparse differentiation

Official code for:

Luca Thiede\*, Andreas Burger\*, and Alán Aspuru-Guzik.
*Colour me shocked: Exact Molecular Hessians from local MLIPs in O(N) time using sparse differentiation.*
40th Conference on Neural Information Processing Systems (NeurIPS 2026).

\*Equal contribution.

```bibtex
@inproceedings{thiede2026colourmeshocked,
  title     = {Colour me shocked: Exact Molecular Hessians from local {MLIP}s in {O(N)} time using sparse differentiation},
  author    = {Thiede, Luca and Burger, Andreas and Aspuru-Guzik, Al{\'a}n},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
  note      = {40th Conference on Neural Information Processing Systems (NeurIPS 2026). Equal contribution: Thiede and Burger.}
}
```

The repository contains:

- `colpackpy/`: Python bindings and PyTorch/NumPy recovery utilities for ColPack coloring.
- `matgl/src/`: MatGL source with the Hessian sparsity-pattern and Hessian-vector-product changes used by the experiments.
- `matgl/examples/`: TensorNet, CHGNet, and ORCA experiment scripts.
- `mace-off/examples/`: MACE-OFF23 benchmark and sparsity-pattern check. Model weights are downloaded separately.
- `data/`: molecular structures used by the benchmark scripts.
- `matgl/examples/sparse_hessian_results/`: retained result summaries and reference Hessians used for figure regeneration.


## Setup

Use Python 3.10 on Linux. The ColPack extension requires CMake and a C++17 compiler. The TensorNet and CHGNet benchmarks require an NVIDIA GPU and the CUDA 12.1 PyTorch build below.

```bash
conda create -n sparse-hessian python=3.10 -y
conda activate sparse-hessian

# Build toolchain for colpackpy (CMake + compiler + pybind11).
conda install -c conda-forge cmake pybind11 cxx-compiler -y

python -m pip install --upgrade pip setuptools wheel

# PyTorch/DGL stack used by the experiments.
python -m pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/repo.html

# Local packages and experiment dependencies.
python -m pip install -e ./matgl
python -m pip install -e ./colpackpy
python -m pip install ase scipy matplotlib pandas
```

The MACE-OFF23 scripts also need the MACE code and the medium checkpoint. The weights are not in this repository. They are distributed by [ACEsuit/mace-off](https://github.com/ACEsuit/mace-off) under the Academic Software Licence (academic, non-commercial use).

```bash
python -m pip install mace-torch

mkdir -p mace-off/mace_off23
curl -L -o mace-off/mace_off23/MACE-OFF23_medium.model \
  https://github.com/ACEsuit/mace-off/raw/main/mace_off23/MACE-OFF23_medium.model
```


## Run instructions

Run commands from the repository root. Experiment outputs are written under `matgl/examples/sparse_hessian_results/` or `mace-off/examples/sparse_hessian_results/`.

Validate the recovery utilities:

```bash
python -m unittest discover colpackpy/tests
```

TensorNet benchmarks on alkanes, water clusters, and Aβ40. These are the main scaling experiments in the paper. The larger systems are expensive and require a CUDA GPU.

```bash
python matgl/examples/sparse_hessian_benchmark.py --alkanes
python matgl/examples/sparse_hessian_benchmark.py --water-cluster
python matgl/examples/sparse_hessian_benchmark.py --abeta40
```

CHGNet cutoff sweep on Chignolin. This regenerates the cutoff-versus-eigenvalue-error plots for the model whose receptive field covers the protein.

```bash
python matgl/examples/sparse_hessian_cutoff.py
```

ORCA PBE cutoff plots on the same Chignolin structure. The script emulates Hessian-vector products from the stored ORCA Hessian.

```bash
python matgl/examples/analyze_orca_hessian.py --hessian matgl/examples/sparse_hessian_results/cutoff_experiments_chingolin_orca_pbe/pbe_orca_hessian_chingolin.hess
```

MACE-OFF23 (medium) on the C300 alkane. This is the comparison reported in the paper: the same sparsity pattern as the two-layer TensorNet, with a much more expensive Hessian-vector product. The script uses CUDA when it is available and otherwise runs on CPU. `--water-cluster` runs the same comparison on the 964-molecule water cluster.

```bash
python mace-off/examples/sparse_hessian_benchmark.py --alkanes
python mace-off/examples/sparse_hessian_benchmark.py --water-cluster
```

Check that the graph sparsity pattern matches the MACE Hessian on `alkane_C10.xyz`, and that the acyclic recovery on `alkane_C300.xyz` stays inside that pattern:

```bash
python mace-off/examples/check_alkane_acyclic_sparsity.py
```


## Data

`data/Abeta40_PED00531/` contains five Aβ40 conformers taken from Protein Ensemble Database entry [PED00531](https://proteinensemble.org/entries/PED00531) (ensemble e001). Selection details are in `data/Abeta40_PED00531/README.txt`. Cite that entry when you use these structures, and follow the database terms.

Alkane, water-cluster, and Chignolin geometries used by the scripts are under `data/Alkanes`, `data/Water Cluster`, `data/water_cluster_all`, and `data/Chignolin`.


## Third-party licenses

- MatGL (`matgl/`, including the checkpoints in `matgl/pretrained_models/`) is BSD-3. See `matgl/LICENSE`.
- ColPack (`colpackpy/ColPack/`) is BSD-3. See `colpackpy/ColPack/LICENSE`.
- MACE-OFF23 weights, downloaded separately, are under the [Academic Software Licence](https://github.com/gabor1/ASL).
