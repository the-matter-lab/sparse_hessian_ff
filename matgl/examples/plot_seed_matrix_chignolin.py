#!/usr/bin/env python3
"""Build and plot a Hessian seed matrix for a Chignolin structure without any HVPs.

This script only:
1. loads a pretrained model,
2. constructs the graph / Hessian sparsity pattern,
3. asks ColPack for the seed matrix,
4. saves the seed matrix and a visualization.

It does not compute forces, Hessians, or Hessian-vector products.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "matgl" / "src"))
sys.path.insert(0, str(REPO_ROOT / "colpackpy" / "src"))

import matgl
from colpackpy.coloring import get_seeds_from_pattern
from matgl.ext.ase import Atoms2Graph
from matgl.graph.compute import (
    create_sparsity_pattern_from_cutoff,
    create_sparsity_pattern_from_graph,
    num_graph_hops_for_hessian_sparsity,
)


DEFAULT_MODEL = REPO_ROOT / "matgl" / "pretrained_models" / "TensorNet-ANI-1x-Subset-PES"
DEFAULT_XYZ = REPO_ROOT / "data" / "Chignolin" / "Coordinates_9999.xyz"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "sparse_hessian_results" / "chignolin_experiments" / "seed_matrices"

LABEL_FS = 18
TICK_FS = 16


def infer_num_layers(model) -> int:
    """Match the current default graph-based sparsity logic in matgl.apps.pes."""
    if hasattr(model, "layers"):
        return len(model.layers)
    return 1


def save_seed_matrix_plot(seed_matrix: np.ndarray, png_path: Path, x_ticks: np.ndarray) -> None:
    n_rows, n_cols = seed_matrix.shape
    cell_size_in = 0.03
    fig_width = max(3.0, cell_size_in * n_cols)
    fig_height = max(6.0, cell_size_in * n_rows)
    plt.figure(figsize=(fig_width, fig_height))
    plt.imshow(seed_matrix, cmap="Greys", aspect="equal", interpolation="nearest")
    plt.xlabel("Seed Vector Index", fontsize=LABEL_FS)
    plt.ylabel("Cartesian DOF Index", fontsize=LABEL_FS)
    x_ticks = x_ticks[x_ticks < n_cols]
    plt.xticks(x_ticks, fontsize=TICK_FS)
    plt.yticks(fontsize=TICK_FS)
    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xyz", type=Path, default=DEFAULT_XYZ, help="Input Chignolin xyz file")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Pretrained model directory")
    parser.add_argument(
        "--algorithm",
        choices=("star", "acyclic", "distance_two"),
        default="acyclic",
        help="ColPack coloring algorithm used to build the seed matrix",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=None,
        help="Optional geometric cutoff for the Hessian sparsity pattern. If omitted, use graph-based sparsity.",
    )
    parser.add_argument(
        "--ordering",
        default="SMALLEST_LAST",
        help="ColPack ordering strategy (default: SMALLEST_LAST)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for seed matrix outputs",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Seed matrix generator (no HVPs)")
    print("=" * 80)
    print(f"Structure: {args.xyz}")
    print(f"Model:     {args.model}")
    print(f"Algorithm: {args.algorithm}")

    atoms = read(str(args.xyz))
    potential = matgl.load_model(str(args.model))

    graph_converter = Atoms2Graph(potential.model.element_types, potential.model.cutoff)
    graph, _, _ = graph_converter.get_graph(atoms)

    if args.cutoff is None:
        num_layers = infer_num_layers(potential.model)
        sparsity_hops = num_graph_hops_for_hessian_sparsity(potential.model)
        print(f"Using graph-based sparsity with {sparsity_hops} hops ({num_layers} interaction layers + embedding if any)")
        sparsity_pattern, n = create_sparsity_pattern_from_graph(graph, sparsity_hops)
        pattern_tag = f"graph_hops_{sparsity_hops}"
    else:
        print(f"Using geometric cutoff-based sparsity with cutoff={args.cutoff:.2f} A")
        sparsity_pattern, n = create_sparsity_pattern_from_cutoff(graph, args.cutoff)
        pattern_tag = f"cutoff_{args.cutoff:g}A"

    print(f"Number of atoms: {len(atoms)}")
    print(f"Degrees of freedom: {n}")
    print(f"Sparsity entries: {len(sparsity_pattern)}")
    print("Building seed matrix from sparsity pattern...")

    seed_matrix, vertex_colors, _ = get_seeds_from_pattern(
        sparsity_pattern,
        n,
        args.ordering.upper(),
        args.algorithm.upper(),
    )
    seed_matrix = np.asarray(seed_matrix, dtype=float)
    vertex_colors = np.asarray(vertex_colors, dtype=int)

    print(f"Seed matrix shape: {seed_matrix.shape}")
    print(f"Number of colors / HVPs implied: {seed_matrix.shape[1]}")

    stem = args.xyz.stem
    base = f"{stem}_{args.algorithm}_{pattern_tag}"

    npz_path = args.output_dir / f"seed_matrix_{base}.npz"
    np.savez_compressed(
        npz_path,
        seed_matrix=seed_matrix,
        vertex_colors=vertex_colors,
        sparsity_pattern=np.asarray(sparsity_pattern, dtype=int),
        n=np.asarray([n], dtype=int),
        num_atoms=np.asarray([len(atoms)], dtype=int),
        algorithm=np.asarray([args.algorithm], dtype=object),
        ordering=np.asarray([args.ordering.upper()], dtype=object),
        model_path=np.asarray([str(args.model)], dtype=object),
        xyz_path=np.asarray([str(args.xyz)], dtype=object),
    )

    png_path = args.output_dir / f"seed_matrix_{base}.png"
    save_seed_matrix_plot(seed_matrix, png_path, np.array([0, 40, 80, 120, 160], dtype=int))

    naive_seed_matrix = np.eye(n, dtype=float)
    naive_base = f"{stem}_naive_identity"
    naive_npz_path = args.output_dir / f"seed_matrix_{naive_base}.npz"
    np.savez_compressed(
        naive_npz_path,
        seed_matrix=naive_seed_matrix,
        n=np.asarray([n], dtype=int),
        num_atoms=np.asarray([len(atoms)], dtype=int),
        algorithm=np.asarray(["naive"], dtype=object),
        model_path=np.asarray([str(args.model)], dtype=object),
        xyz_path=np.asarray([str(args.xyz)], dtype=object),
    )
    naive_png_path = args.output_dir / f"seed_matrix_{naive_base}.png"
    save_seed_matrix_plot(naive_seed_matrix, naive_png_path, np.array([0, 100, 200, 300, 400], dtype=int))

    print(f"Saved matrix data to: {npz_path}")
    print(f"Saved matrix plot to: {png_path}")
    print(f"Saved naive matrix data to: {naive_npz_path}")
    print(f"Saved naive matrix plot to: {naive_png_path}")


if __name__ == "__main__":
    main()
