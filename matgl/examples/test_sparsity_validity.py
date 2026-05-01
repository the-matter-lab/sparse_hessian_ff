#!/usr/bin/env python3
"""Compare star / acyclic sparse Hessians to the naive dense reference on Abeta40 (middle).

Uses the same random TensorNet setup as ``sparse_hessian_benchmark`` (0 and 1 interaction
blocks, 5 Å cutoff). Reports max absolute elementwise error and max relative elementwise
error with a fixed epsilon in the denominator.

Example::

    conda run -n sparse_hessian_ff python matgl/examples/test_sparsity_validity.py --device auto
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import torch
from ase.io import read

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "matgl" / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

import matgl  # noqa: E402
from sparse_hessian_benchmark import (  # noqa: E402
    HessianBenchmark,
    _build_model_specs,
    compute_sparse_hessian,
)

DEFAULT_XYZ = REPO_ROOT / "data" / "Abeta40_PED00531" / "PED00531_model_0853_middle.xyz"
REL_EPS_DEFAULT = 1e-5


def _set_default_device(device: str) -> str:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)
    return device


def _hessian_error_metrics(
    reference: np.ndarray, approx: np.ndarray, rel_eps: float
) -> tuple[float, float]:
    """Return (max |Δ|, max |Δ| / (|ref| + rel_eps))."""
    ref = np.asarray(reference, dtype=np.float64)
    app = np.asarray(approx, dtype=np.float64)
    if ref.shape != app.shape:
        raise ValueError(f"Shape mismatch: ref {ref.shape} vs approx {app.shape}")
    diff = app - ref
    max_abs = float(np.max(np.abs(diff)))
    denom = np.abs(ref) + rel_eps
    max_rel = float(np.max(np.abs(diff) / denom))
    return max_abs, max_rel


def _run_method(bench: HessianBenchmark, atoms, method: str) -> np.ndarray:
    bench._configure_method(method)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    bench.calc.calculate(atoms, properties=["hessian"])
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    h = bench.calc.results["hessian"]
    out = np.array(h, dtype=np.float64, copy=True)
    bench._release_method_memory()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--xyz",
        type=Path,
        default=DEFAULT_XYZ,
        help=f"Structure file (default: {DEFAULT_XYZ})",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0],
        help="Random TensorNet seeds (default: 0)",
    )
    parser.add_argument(
        "--coloring-ordering",
        type=str,
        default="SMALLEST_LAST",
        help="ColPack vertex ordering for star/acyclic (default: SMALLEST_LAST)",
    )
    parser.add_argument(
        "--rel-eps",
        type=float,
        default=REL_EPS_DEFAULT,
        help=f"Epsilon in relative error denominator (default: {REL_EPS_DEFAULT})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=("cuda", "cpu", "auto"),
        help="PyTorch default device (default: cuda)",
    )
    args = parser.parse_args()

    if compute_sparse_hessian is None:
        raise SystemExit(
            "colpackpy is required for star/acyclic Hessians. "
            "Install colpackpy or use an environment where it is available."
        )

    xyz = args.xyz.resolve()
    if not xyz.is_file():
        raise SystemExit(f"Not a file: {xyz}")

    device = _set_default_device(args.device)
    print(f"Device: {device}")
    print(f"Structure: {xyz}")
    print(f"Relative error epsilon: {args.rel_eps}")
    print(f"Coloring ordering: {args.coloring_ordering}")
    print()

    atoms = read(xyz)

    for seed in args.seeds:
        print(f"--- random seed {seed} ---")
        for spec in _build_model_specs(seed):
            tag = spec["tag"]
            print(f"  model: {tag} (nblocks={spec['nblocks']})")
            bench = HessianBenchmark(
                model_spec=spec,
                test_over_conformations=False,
                coloring_ordering=args.coloring_ordering.upper(),
                collect_breakdown=False,
            )
            bench.ensure_calculator(atoms)

            h_naive = _run_method(bench, atoms, "naive")

            for method in ("star", "acyclic"):
                h_sparse = _run_method(bench, atoms, method)
                max_abs, max_rel = _hessian_error_metrics(
                    h_naive, h_sparse, args.rel_eps
                )
                print(
                    f"    {method:8s}  max|Δ|={max_abs:.6e}  "
                    f"max|Δ|/(|ref|+eps)={max_rel:.6e}"
                )
        print()


if __name__ == "__main__":
    main()
