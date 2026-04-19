import argparse
import json
from pathlib import Path

import matgl
import torch

from profile_sparse_hessian_breakdown import (
    COLORING_ORDERINGS,
    load_structure,
    log,
    profile_sparse_algorithm,
    resolve_xyz_path,
)

NON_DISTANCE_TWO_ORDERINGS = [
    ordering for ordering in COLORING_ORDERINGS if not ordering.startswith("DISTANCE_TWO_")
]
SPARSE_ALGORITHMS = ("star", "acyclic")


def print_table(results: list[dict]) -> None:
    print()
    print("=" * 140)
    print("Sparse Hessian ordering sweep")
    print("=" * 140)
    header = (
        f"{'method':<10} {'ordering':<24} {'colors':>8} {'hvp':>8} "
        f"{'color_s':>10} {'hvp_s':>10} {'recover_s':>10} {'total_s':>10} {'compress':>10}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        timings = result["timings_s"]
        print(
            f"{result['method']:<10} {result['ordering']:<24} "
            f"{result['num_colors']:>8d} {result['hvp_count']:>8d} "
            f"{timings['coloring_s']:>10.4f} {timings['hvp_loop_s']:>10.4f} "
            f"{timings['recovery_s']:>10.4f} {timings['total_wall_s']:>10.4f} "
            f"{result['compression_ratio']:>10.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare non-distance-two ColPack orderings for sparse Hessian star and acyclic algorithms."
    )
    parser.add_argument(
        "--water-size",
        choices=["237", "573"],
        default="237",
        help="Convenience selector for data/Water Cluster/<size>.xyz when --xyz is not provided.",
    )
    parser.add_argument(
        "--xyz",
        type=Path,
        default=None,
        help="Optional explicit xyz path. Overrides --water-size.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("../pretrained_models/TensorNet-ANI-1x-Subset-PES"),
        help="Model path, relative to this script.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch default device (default: cuda).",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=None,
        help="Optional Hessian sparsity cutoff. If omitted, uses graph-based sparsity like the benchmark.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional JSON output path for the full sweep results.",
    )
    args = parser.parse_args()

    torch.set_default_device(args.device)

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent
    xyz_path = resolve_xyz_path(args, repo_root)
    model_path = (script_dir / args.model).resolve() if not args.model.is_absolute() else args.model

    if not xyz_path.is_file():
        raise SystemExit(f"Not a file: {xyz_path}")

    log(f"Using xyz: {xyz_path}")
    log(f"Using model: {model_path}")
    if args.cutoff is None:
        log("Sparsity mode: graph-based")
    else:
        log(f"Sparsity mode: cutoff-based ({args.cutoff:.3f} A)")
    log(f"Orderings: {', '.join(NON_DISTANCE_TWO_ORDERINGS)}")

    atoms, _ = load_structure(xyz_path)
    log(f"Loaded structure with {len(atoms)} atoms")

    log("Loading model...")
    potential = matgl.load_model(str(model_path))
    potential.calc_hessian = True
    potential.calc_forces = True
    potential.return_seed_matrix = False
    potential.hessian_return_type = "dense"
    potential.hessian_sparsity_cutoff = args.cutoff
    log("Model loaded")

    results = []
    for algorithm in SPARSE_ALGORITHMS:
        for ordering in NON_DISTANCE_TWO_ORDERINGS:
            log(f"Profiling sparse algorithm={algorithm} ordering={ordering}")
            results.append(
                profile_sparse_algorithm(
                    potential,
                    atoms,
                    algorithm,
                    cutoff=args.cutoff,
                    ordering=ordering,
                )
            )

    print_table(results)

    if args.json_out is not None:
        json_path = args.json_out
        if not json_path.is_absolute():
            json_path = (script_dir / json_path).resolve()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "xyz_path": str(xyz_path),
            "model_path": str(model_path),
            "device": args.device,
            "cutoff": args.cutoff,
            "algorithms": list(SPARSE_ALGORITHMS),
            "orderings": NON_DISTANCE_TWO_ORDERINGS,
            "results": results,
        }
        json_path.write_text(json.dumps(payload, indent=2))
        log(f"Saved JSON summary to {json_path}")


if __name__ == "__main__":
    main()
