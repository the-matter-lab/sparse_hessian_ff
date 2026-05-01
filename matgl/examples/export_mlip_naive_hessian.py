"""Compute the MLIP naive Hessian once and save a dense .npy for reanalyze_dft_hessian.py.

``sparse_hessian_cutoff_real.py`` uses the same settings; new runs also write
``full_naive_hessian_<timestamp>.npy`` automatically. Use this script to generate
that file without rerunning the full cutoff sweep (e.g. after an older run that
only saved ``cutoff_debug_data_*.npz`` without the matrix).
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch

torch.set_default_device("cuda")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matgl  # noqa: E402
from ase.io import read  # noqa: E402
from matgl.ext.ase import PESCalculator  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--structure",
        type=Path,
        default=REPO_ROOT / "data" / "Chignolin" / "Coordinates_9999.xyz",
        help="ASE geometry (default: same Chignolin as sparse_hessian_cutoff_real.py).",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for the dense Hessian .npy (e.g. .../full_naive_hessian_<stamp>.npy).",
    )
    p.add_argument(
        "--model-path",
        type=Path,
        default=REPO_ROOT
        / "matgl"
        / "pretrained_models"
        / "CHGNet-MatPES-r2SCAN-2025.2.10-2.7M-PES",
        help="CHGNet pretrained directory (matches sparse_hessian_cutoff_real.py).",
    )
    p.add_argument("--hessian-delta", type=float, default=5e-3, help="Matches cutoff_real naive.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pot = matgl.load_model(str(args.model_path))
    pot.calc_hessian = True
    pot.return_seed_matrix = True
    pot.hessian_delta = float(args.hessian_delta)
    pot.hessian_algorithm = "naive"
    pot.hessian_sparsity_cutoff = None

    atoms = read(str(args.structure.resolve()))
    calc = PESCalculator(pot)
    t0 = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    calc.calculate(atoms, properties=["hessian"])
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    H = np.asarray(calc.results["hessian"], dtype=float)
    print(f"Hessian shape {H.shape}, wall {time.perf_counter() - t0:.2f}s")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, H)
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
