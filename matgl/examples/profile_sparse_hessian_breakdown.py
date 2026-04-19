import argparse
import gc
import json
import time
from pathlib import Path

import matgl
import numpy as np
import torch
from ase.io import read
from colpackpy.coloring import get_seeds_from_pattern  # type: ignore[reportMissingImports]
from colpackpy.recovery import recover_matrix  # type: ignore[reportMissingImports]
from matgl.ext.ase import Atoms2Graph
from matgl.graph.compute import (
    create_sparsity_pattern_from_cutoff,
    create_sparsity_pattern_from_graph,
    num_graph_hops_for_hessian_sparsity,
)
from torch.autograd import grad

COLORING_ORDERINGS = [
    "NATURAL",
    "LARGEST_FIRST",
    "SMALLEST_LAST",
    "INCIDENCE_DEGREE",
    "DISTANCE_TWO_LARGEST_FIRST",
    "DISTANCE_TWO_SMALLEST_LAST",
    "DISTANCE_TWO_INCIDENCE_DEGREE",
    "DYNAMIC_LARGEST_FIRST",
    "RANDOM",
]


def log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def sync_if_needed() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_call(label: str, fn):
    sync_if_needed()
    t0 = time.perf_counter()
    result = fn()
    sync_if_needed()
    elapsed = time.perf_counter() - t0
    log(f"{label}: {elapsed:.4f}s")
    return result, elapsed


def release_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def add_timing_accounting(summary: dict, tolerance_fraction: float = 0.05) -> dict:
    timings = summary["timings_s"]
    component_sum = sum(value for key, value in timings.items() if key != "total_wall_s")
    total_wall = timings["total_wall_s"]
    abs_diff = abs(component_sum - total_wall)
    rel_diff = abs_diff / total_wall if total_wall > 0 else 0.0
    summary["timing_accounting"] = {
        "component_sum_s": component_sum,
        "total_wall_s": total_wall,
        "abs_diff_s": abs_diff,
        "rel_diff_fraction": rel_diff,
        "within_5_percent": rel_diff <= tolerance_fraction,
    }
    return summary


def load_structure(xyz_path: Path):
    atoms, elapsed = timed_call(f"Load XYZ {xyz_path.name}", lambda: read(xyz_path))
    return atoms, elapsed


def build_graph_inputs(atoms, potential):
    converter = Atoms2Graph(potential.model.element_types, potential.model.cutoff)
    return converter.get_graph(atoms)


def prepare_graph_for_forward(g, lat):
    st = lat.new_zeros([g.batch_size, 3, 3])
    lattice = lat @ (torch.eye(3, device=lat.device) + st)
    g.edata["lattice"] = torch.repeat_interleave(lattice, g.batch_num_edges(), dim=0)
    g.edata["pbc_offshift"] = (g.edata["pbc_offset"].unsqueeze(dim=-1) * g.edata["lattice"]).sum(dim=1)
    g.ndata["pos"] = (
        g.ndata["frac_coords"].unsqueeze(dim=-1) * torch.repeat_interleave(lattice, g.batch_num_nodes(), dim=0)
    ).sum(dim=1)
    g.ndata["pos"].requires_grad_(True)
    return lattice


def compute_total_energies(potential, g, state_attr, l_g=None):
    total_energies = potential.model(g=g, state_attr=state_attr, l_g=l_g)
    total_energies = potential.data_std * total_energies + potential.data_mean
    if potential.calc_repuls:
        total_energies += potential.repuls(potential.model.element_types, g)
    if potential.element_refs is not None:
        total_energies += torch.squeeze(potential.element_refs(g))
    return total_energies


def profile_naive(potential, atoms, state_attr_override=None):
    timings = {}
    method_start = time.perf_counter()

    (graph, lat, state_attr_default), timings["graph_build_s"] = timed_call(
        "Graph conversion", lambda: build_graph_inputs(atoms, potential)
    )
    state_attr = state_attr_override if state_attr_override is not None else state_attr_default

    _, timings["graph_prepare_s"] = timed_call("Graph tensor preparation", lambda: prepare_graph_for_forward(graph, lat))

    total_energies, timings["forward_s"] = timed_call(
        "Model forward", lambda: compute_total_energies(potential, graph, state_attr)
    )

    grads, timings["force_grad_s"] = timed_call(
        "Force gradient",
        lambda: grad(
            total_energies,
            [graph.ndata["pos"]],
            grad_outputs=torch.ones_like(total_energies),
            create_graph=True,
            retain_graph=True,
        ),
    )

    r = grads[0].view(-1)
    n = r.size(0)

    def build_hessian():
        hessian = total_energies.new_zeros((n, n))
        for i in range(n):
            tmp = grad([r[i]], graph.ndata["pos"], retain_graph=i < n - 1)[0]
            if tmp is not None:
                hessian[i] = tmp.view(-1)
        return hessian

    hessian, timings["naive_hessian_s"] = timed_call("Naive Hessian construction", build_hessian)
    timings["total_wall_s"] = time.perf_counter() - method_start

    summary = {
        "method": "naive",
        "n_atoms": len(atoms),
        "ndof": int(n),
        "hvp_count": int(n),
        "hessian_shape": list(hessian.shape),
        "timings_s": timings,
    }

    del hessian, r, grads, total_energies, graph, lat
    release_memory()
    return add_timing_accounting(summary)


def profile_sparse_algorithm(
    potential,
    atoms,
    algorithm: str,
    state_attr_override=None,
    cutoff=None,
    ordering: str = "SMALLEST_LAST",
):
    timings = {}
    method_start = time.perf_counter()
    ordering = ordering.upper()

    (graph, lat, state_attr_default), timings["graph_build_s"] = timed_call(
        "Graph conversion", lambda: build_graph_inputs(atoms, potential)
    )
    state_attr = state_attr_override if state_attr_override is not None else state_attr_default

    _, timings["graph_prepare_s"] = timed_call("Graph tensor preparation", lambda: prepare_graph_for_forward(graph, lat))

    total_energies, timings["forward_s"] = timed_call(
        "Model forward", lambda: compute_total_energies(potential, graph, state_attr)
    )

    grads, timings["force_grad_s"] = timed_call(
        "Force gradient",
        lambda: grad(
            total_energies,
            [graph.ndata["pos"]],
            grad_outputs=torch.ones_like(total_energies),
            create_graph=True,
            retain_graph=True,
        ),
    )

    if cutoff is None:
        sparsity_hops = num_graph_hops_for_hessian_sparsity(potential.model)
        (sparsity_pattern, n), timings["sparsity_s"] = timed_call(
            "Sparsity pattern construction",
            lambda: create_sparsity_pattern_from_graph(graph, sparsity_hops),
        )
    else:
        (sparsity_pattern, n), timings["sparsity_s"] = timed_call(
            "Sparsity pattern construction",
            lambda: create_sparsity_pattern_from_cutoff(graph, cutoff),
        )

    (seed_matrix_np, colors_np, comps), timings["coloring_s"] = timed_call(
        f"{algorithm} coloring ({ordering})",
        lambda: get_seeds_from_pattern(sparsity_pattern, n, ordering, algorithm.upper()),
    )

    seed_matrix, timings["seed_tensor_s"] = timed_call(
        "Seed matrix to torch", lambda: torch.tensor(seed_matrix_np, dtype=graph.ndata["pos"].dtype, device=graph.ndata["pos"].device)
    )

    Y = torch.zeros(n, seed_matrix.shape[1], dtype=graph.ndata["pos"].dtype, device=graph.ndata["pos"].device)
    r = grads[0].view(-1)

    grad_seed_products, timings["seed_dot_s"] = timed_call(
        "Seed/gradient dot products", lambda: torch.mv(seed_matrix.T, r)
    )

    def hvp_loop():
        for i in range(seed_matrix.shape[1]):
            hvp_result = grad(
                grad_seed_products[i],
                graph.ndata["pos"],
                retain_graph=(i < seed_matrix.shape[1] - 1),
            )[0]
            if hvp_result is not None:
                Y[:, i] = hvp_result.view(-1)

    _, timings["hvp_loop_s"] = timed_call(f"{algorithm} compressed HVP loop", hvp_loop)

    recovered, timings["recovery_s"] = timed_call(
        f"{algorithm} recovery",
        lambda: recover_matrix(
            recover_from=algorithm.upper(),
            edges=sparsity_pattern,
            n=n,
            Y=Y,
            vertex_colors=colors_np,
            acyclic_components=comps,
            reconstruct_sparsely=False,
            verify=False,
        ),
    )

    hessian, timings["dense_tensor_wrap_s"] = timed_call(
        "Recovered Hessian to torch",
        lambda: torch.tensor(recovered, dtype=graph.ndata["pos"].dtype, device=graph.ndata["pos"].device),
    )

    timings["total_wall_s"] = time.perf_counter() - method_start

    summary = {
        "method": algorithm,
        "ordering": ordering,
        "n_atoms": len(atoms),
        "ndof": int(n),
        "num_colors": int(seed_matrix.shape[1]),
        "hvp_count": int(seed_matrix.shape[1]),
        "compression_ratio": float(n / seed_matrix.shape[1]),
        "num_sparse_edges": int(len(sparsity_pattern)),
        "hessian_shape": list(hessian.shape),
        "timings_s": timings,
    }

    del hessian, recovered, Y, grad_seed_products, seed_matrix, grads, total_energies, graph, lat, r
    release_memory()
    return add_timing_accounting(summary)


def resolve_xyz_path(args, repo_root: Path) -> Path:
    if args.xyz is not None:
        return args.xyz.resolve()
    return repo_root / "data" / "Water Cluster" / f"{args.water_size}.xyz"


def print_summary(results: list[dict]) -> None:
    print()
    print("=" * 88)
    print("Breakdown summary")
    print("=" * 88)
    for result in results:
        print()
        print(f"Method: {result['method']}")
        if "ordering" in result:
            print(f"  ordering: {result['ordering']}")
        print(f"  n_atoms: {result['n_atoms']}")
        print(f"  ndof: {result['ndof']}")
        print(f"  hvp_count: {result['hvp_count']}")
        if "num_colors" in result:
            print(f"  num_colors: {result['num_colors']}")
            print(f"  compression_ratio: {result['compression_ratio']:.3f}")
            print(f"  num_sparse_edges: {result['num_sparse_edges']}")
        for key, value in result["timings_s"].items():
            print(f"  {key}: {value:.4f}s")
        accounting = result["timing_accounting"]
        print(f"  component_sum_s: {accounting['component_sum_s']:.4f}s")
        print(f"  total_wall_s: {accounting['total_wall_s']:.4f}s")
        print(f"  abs_diff_s: {accounting['abs_diff_s']:.4f}s")
        print(f"  rel_diff_fraction: {accounting['rel_diff_fraction']:.4%}")
        print(f"  within_5_percent: {accounting['within_5_percent']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Break down sparse Hessian timing into coloring, HVP loop, and recovery."
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
        "--ordering",
        choices=COLORING_ORDERINGS,
        default="SMALLEST_LAST",
        help="ColPack vertex ordering heuristic for sparse coloring (default: SMALLEST_LAST).",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional JSON output path for the timing summary.",
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
    log(f"Coloring ordering: {args.ordering}")

    atoms, _ = load_structure(xyz_path)
    log(f"Loaded structure with {len(atoms)} atoms")

    log("Loading model...")
    potential = matgl.load_model(str(model_path))
    potential.calc_hessian = True
    potential.calc_forces = True
    potential.return_seed_matrix = False
    potential.hessian_return_type = "dense"
    potential.hessian_sparsity_cutoff = args.cutoff
    potential.hessian_coloring_ordering = args.ordering
    log("Model loaded")

    results = []

    log("Profiling naive baseline...")
    results.append(profile_naive(potential, atoms))

    for algorithm in ("star", "acyclic"):
        log(f"Profiling sparse algorithm: {algorithm}")
        results.append(
            profile_sparse_algorithm(
                potential,
                atoms,
                algorithm,
                cutoff=args.cutoff,
                ordering=args.ordering,
            )
        )

    print_summary(results)

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
            "ordering": args.ordering,
            "results": results,
        }
        json_path.write_text(json.dumps(payload, indent=2))
        log(f"Saved JSON summary to {json_path}")


if __name__ == "__main__":
    main()
