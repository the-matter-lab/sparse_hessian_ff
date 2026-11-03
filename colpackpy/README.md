# ColPackPy: Python Bindings for ColPack Graph Coloring Library

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![NumPy 1.26.4](https://img.shields.io/badge/NumPy-1.26.4-blue.svg)](https://numpy.org/)
[![PyTorch 2.4.0](https://img.shields.io/badge/PyTorch-2.4.0-blue.svg)](https://pytorch.org/)

Python bindings for the [ColPack](https://github.com/CSCsw/ColPack) graph coloring library, enabling efficient sparse matrix recovery for Hessian matrices and general symmetric matrices.

`ColPack/` in this directory is a vendored copy of [CSCsw/ColPack](https://github.com/CSCsw/ColPack). The official ColPack project is that repository. This package is not the official ColPack code and is not maintained by the ColPack authors. ColPack is BSD-3, copyright (c) 2018, Alex Pothen. See `ColPack/LICENSE`. Please cite Gebremedhin, Nguyen, Patwary, and Pothen, *ColPack: Graph coloring software for derivative computation and beyond*, ACM Transactions on Mathematical Software, 40(1), 2013.

The typical use case is for sparse automatic differentiation: Autodiff lets use probe a Hessian using Hessian vector products (HVP). Given we know the sparsity pattern of a Hessian (or Jacobian, although the current release does not support this), using the coloring algorithms in this library we can construct our HVPs in such a way, that we extract the maximal information. A corresponding recovery algorithm then reconstructs the Hessian (or Jacobian).

The recovery algorithms support numpy and torch and can therefore be differentiated. Sparse tensors are supported for large scale matrices.

## Overview

ColPackPy provides Python interfaces to ColPack's graph coloring algorithms, as well as recovery algrotihms written in numpy and pytorh. Additionally, small wrapper to calculate Hessians of Pytorch models given the Hessians sparsity pattern as a sparse tensor:

- **Graph Coloring**: Distance-2, star, and acyclic coloring algorithms
- **Matrix Recovery**: Efficient compression and recovery of sparse Hessian matrices
- **Convinience wrapper for PyTorch Hessian**: Given a PyTorch model and sparsity pattern, get the sparse Hessian

## Installation

### From Source

```bash
# Clone or navigate to the repository
cd colpackpy

# Install in development mode
pip install -e .
```

### Prerequisites

- **Python 3.10+**
- **CMake** (build system generator; ColPack is compiled from source during installation)
- **C++ Compiler** (GCC 11.2.0 or compatible; we do not ship prebuild binaries)
- **PyBind11** (automatically installed)

### Optional Dependencies

- **PyTorch** (for autodiff of recovery)

## Tested Environment

This package has been tested on:

| Component | Version |
|-----------|---------|
| **Python** | 3.10.18 |
| **NumPy** | 1.26.4 |
| **PyTorch** | 2.4.0+cu121 |

**System**: Linux x86_64, GCC 11.2.0

## Usage

See the `examples/` directory for usage examples:

- **`toy_matrix_recovery_example.py`** - Basic matrix compression and recovery
- **`gradient_example.py`** - Gradient computation with sparse Hessians
- **`toy_nn_example.py`** - Neural network Hessian computation


