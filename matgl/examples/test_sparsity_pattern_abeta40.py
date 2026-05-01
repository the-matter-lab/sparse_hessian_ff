#!/usr/bin/env python3
"""Diagnose Abeta40 sparsity patterns for naive and recovered TensorNet Hessians.

Runs the two endpoint Abeta40 conformations (most compact / most extended) with
the same random PyG TensorNet setup as ``sparse_hessian_benchmark``:

* 0 interaction blocks + tensor embedding (1 effective graph hop),
* 1 interaction block + tensor embedding (2 effective graph hops).

For each geometry/model pair, the script computes dense Hessians using naive,
star, and acyclic methods. It writes sparsity plots for exact nonzeros
(``H != 0``), tolerance-filtered nonzeros (``abs(H) > threshold``), tiny
nonzeros (``0 < abs(H) <= threshold``), plus the graph-hop sparsity pattern from
``create_sparsity_pattern_from_graph``. It also prints the maximum absolute
elementwise error between naive and the recovered Hessians.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from ase.io import read
from matplotlib.colors import ListedColormap

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "matgl" / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from matgl.graph.compute import create_sparsity_pattern_from_graph, num_graph_hops_for_hessian_sparsity  # noqa: E402
from sparse_hessian_benchmark import HessianBenchmark, compute_sparse_hessian  # noqa: E402

DEFAULT_GEOMETRIES = {
    "most_compact": REPO_ROOT / "data" / "Abeta40_PED00531" / "PED00531_model_0105_most_compact.xyz",
    "most_extended": REPO_ROOT / "data" / "Abeta40_PED00531" / "PED00531_model_0894_most_extended.xyz",
}
DEFAULT_OUTPUT_DIR = (
    SCRIPT_DIR / "sparse_hessian_results" / "abeta40_experiments" / "sparsity_pattern_recovery_checks"
)
DEFAULT_ORDERING = "SMALLEST_LAST"
PATTERN_CMAP = ListedColormap(["white", "black"])


def _set_default_device(device: str) -> str:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)
    return device


def _sync_if_needed(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def _clear_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _model_specs(seed: int) -> list[dict[str, object]]:
    return [
        {
            "tag": "tensornet_0interaction_5A",
            "display_name": "Random TensorNet (embedding only, 1 effective graph hop, 5 A)",
            "kind": "random_tensornet",
            "nblocks": 0,
            "cutoff": 5.0,
            "seed": seed,
        },
        {
            "tag": "tensornet_1interaction_5A",
            "display_name": "Random TensorNet (1 interaction + embedding, 2 effective graph hops, 5 A)",
            "kind": "random_tensornet",
            "nblocks": 1,
            "cutoff": 5.0,
            "seed": seed,
        },
    ]


def _run_method(bench: HessianBenchmark, atoms, method: str, device: str) -> np.ndarray:
    bench._configure_method(method)
    _sync_if_needed(device)
    assert bench.calc is not None
    bench.calc.calculate(atoms, properties=["hessian"])
    _sync_if_needed(device)
    hessian = np.array(bench.calc.results["hessian"], dtype=np.float64, copy=True)
    bench._release_method_memory()
    _clear_memory()
    return hessian


def _edge_list_to_dense_pattern(edges: np.ndarray, n: int) -> np.ndarray:
    pattern = np.zeros((n, n), dtype=bool)
    if len(edges) > 0:
        pattern[edges[:, 0], edges[:, 1]] = True
    return pattern


def _save_pattern_plot(pattern: np.ndarray, path: Path, title: str) -> None:
    plt.figure(figsize=(10, 10))
    plt.imshow(pattern.astype(np.uint8), cmap=PATTERN_CMAP, interpolation="nearest", aspect="equal")
    plt.title(title)
    plt.xlabel("Cartesian DOF")
    plt.ylabel("Cartesian DOF")
    plt.tight_layout()
    plt.savefig(path, dpi=250, bbox_inches="tight")
    plt.close()


def _patterns_from_hessian(hessian: np.ndarray, threshold: float) -> dict[str, np.ndarray]:
    abs_hessian = np.abs(hessian)
    exact = hessian != 0.0
    above_threshold = abs_hessian > threshold
    tiny_nonzero = exact & ~above_threshold
    return {
        "exact_nonzero": exact,
        "abs_gt_threshold": above_threshold,
        "tiny_nonzero_abs_le_threshold": tiny_nonzero,
    }


def _pattern_stats(pattern: np.ndarray, ndof: int) -> dict[str, object]:
    nnz = int(np.count_nonzero(pattern))
    return {
        "nonzeros": nnz,
        "fraction": float(nnz / (ndof * ndof)),
        "symmetric": bool(np.array_equal(pattern, pattern.T)),
    }


def _max_abs_error(reference: np.ndarray, candidate: np.ndarray) -> float:
    if reference.shape != candidate.shape:
        raise ValueError(f"Shape mismatch: reference {reference.shape}, candidate {candidate.shape}")
    return float(np.max(np.abs(candidate - reference)))


def _run_case(
    atoms,
    geometry_name: str,
    spec: dict[str, object],
    args: argparse.Namespace,
    device: str,
    timestamp: str,
) -> dict[str, object]:
    model_tag = str(spec["tag"])
    print(f"\n=== {geometry_name} | {model_tag} | ordering={args.coloring_ordering} ===", flush=True)

    bench = HessianBenchmark(
        model_spec=spec,
        test_over_conformations=False,
        coloring_ordering=args.coloring_ordering,
        collect_breakdown=False,
    )
    bench.ensure_calculator(atoms)
    assert bench.model is not None and bench._graph_converter is not None

    graph, _lat, _state = bench._graph_converter.get_graph(atoms)
    sparsity_hops = num_graph_hops_for_hessian_sparsity(bench.model)
    graph_edges, graph_n = create_sparsity_pattern_from_graph(graph, sparsity_hops)
    graph_pattern = _edge_list_to_dense_pattern(np.asarray(graph_edges, dtype=int), graph_n)

    stem = f"{geometry_name}_{model_tag}_{args.coloring_ordering.lower()}_{timestamp}"
    _save_pattern_plot(
        graph_pattern,
        args.output_dir / f"{stem}_graph_hops{int(sparsity_hops)}.png",
        f"{geometry_name} {model_tag}: graph-hop pattern ({sparsity_hops} hops)",
    )

    hessians: dict[str, np.ndarray] = {}
    method_summaries: dict[str, object] = {}
    for method in ("naive", "star", "acyclic"):
        print(f"  computing {method} Hessian...", flush=True)
        hessian = _run_method(bench, atoms, method, device)
        if hessian.shape != (graph_n, graph_n):
            raise RuntimeError(f"{method} Hessian shape {hessian.shape} does not match graph pattern {(graph_n, graph_n)}")
        hessians[method] = hessian

        patterns = _patterns_from_hessian(hessian, args.threshold)
        method_summary = {
            "hessian_shape": list(hessian.shape),
            "max_abs_hessian_element": float(np.max(np.abs(hessian))),
            "patterns": {},
        }
        for pattern_name, pattern in patterns.items():
            plot_path = args.output_dir / f"{stem}_{method}_{pattern_name}.png"
            _save_pattern_plot(
                pattern,
                plot_path,
                f"{geometry_name} {model_tag}: {method} {pattern_name}",
            )
            method_summary["patterns"][pattern_name] = _pattern_stats(pattern, graph_n)
        method_summaries[method] = method_summary

    errors = {
        "star_max_abs_error_vs_naive": _max_abs_error(hessians["naive"], hessians["star"]),
        "acyclic_max_abs_error_vs_naive": _max_abs_error(hessians["naive"], hessians["acyclic"]),
    }
    print(f"  star    max |H_recovered - H_naive| = {errors['star_max_abs_error_vs_naive']:.6e}", flush=True)
    print(f"  acyclic max |H_recovered - H_naive| = {errors['acyclic_max_abs_error_vs_naive']:.6e}", flush=True)

    npz_payload = {
        "graph_pattern": graph_pattern.astype(np.uint8),
        "sparsity_hops": np.asarray([sparsity_hops], dtype=int),
        "num_atoms": np.asarray([len(atoms)], dtype=int),
        "ndof": np.asarray([graph_n], dtype=int),
    }
    for method, hessian in hessians.items():
        for pattern_name, pattern in _patterns_from_hessian(hessian, args.threshold).items():
            npz_payload[f"{method}_{pattern_name}"] = pattern.astype(np.uint8)
        if args.save_hessians:
            npz_payload[f"{method}_hessian"] = hessian.astype(np.float32)

    npz_path = args.output_dir / f"{stem}_patterns.npz"
    np.savez_compressed(npz_path, **npz_payload)

    del hessians
    bench._release_method_memory()
    _clear_memory()

    return {
        "geometry": geometry_name,
        "model_tag": model_tag,
        "nblocks": int(spec["nblocks"]),
        "cutoff_angstrom": float(spec["cutoff"]),
        "seed": int(spec["seed"]),
        "coloring_ordering": args.coloring_ordering,
        "num_atoms": len(atoms),
        "ndof": graph_n,
        "sparsity_graph_hops": int(sparsity_hops),
        "threshold": args.threshold,
        "graph_pattern": _pattern_stats(graph_pattern, graph_n),
        "methods": method_summaries,
        "errors": errors,
        "archive": str(npz_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda", help="Torch device (default: cuda)")
    parser.add_argument("--random-seed", type=int, default=0, help="Random TensorNet seed (default: 0)")
    parser.add_argument(
        "--coloring-ordering",
        default=DEFAULT_ORDERING,
        help=f"ColPack ordering for star/acyclic recovery (default: {DEFAULT_ORDERING})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1e-10,
        help="Tolerance for abs(H) > threshold sparsity plots (default: 1e-10)",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument(
        "--save-hessians",
        action="store_true",
        help="Also store dense Hessians in the npz archives (large files).",
    )
    args = parser.parse_args()
    args.coloring_ordering = args.coloring_ordering.upper()

    if compute_sparse_hessian is None:
        raise SystemExit("colpackpy is required for star/acyclic recovery but could not be imported.")

    for name, path in DEFAULT_GEOMETRIES.items():
        if not path.is_file():
            raise SystemExit(f"Missing {name} geometry: {path}")

    device = _set_default_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("Abeta40 sparsity-pattern recovery diagnostic")
    print(f"Device:            {device}")
    print(f"Seed:              {args.random_seed}")
    print(f"Coloring ordering: {args.coloring_ordering}")
    print(f"Threshold:         {args.threshold:.3e}")
    print(f"Output directory:  {args.output_dir}")

    summaries: list[dict[str, object]] = []
    for geometry_name, xyz_path in DEFAULT_GEOMETRIES.items():
        atoms = read(xyz_path)
        print(f"\nLoaded {geometry_name}: {xyz_path.name} ({len(atoms)} atoms)", flush=True)
        for spec in _model_specs(args.random_seed):
            summaries.append(_run_case(atoms, geometry_name, spec, args, device, timestamp))
        del atoms
        _clear_memory()

    summary_path = args.output_dir / f"abeta40_sparsity_recovery_summary_{timestamp}.json"
    summary_path.write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nWrote summary: {summary_path}")


if __name__ == "__main__":
    main()
