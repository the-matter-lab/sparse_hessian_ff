#!/usr/bin/env python3
"""Compare Hessian sparsity patterns against graph-based proxies.

This script computes:
1. a naive Hessian and its block-level sparsity pattern,
2. the graph-based sparsity pattern from ``create_sparsity_pattern_from_graph``.

For each pattern it saves a plot, a difference plot against the naive reference,
and a compressed ``.npz`` archive with the matrices and summary statistics.

TensorNet only: PyG (default) or DGL via ``Potential``, randomly initialized.
"""

from __future__ import annotations

import argparse
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

import matgl
from matgl.apps.pes import Potential
from matgl.ext.ase import Atoms2Graph, Atoms2GraphPYG, PESCalculator
from matgl.graph.compute import create_sparsity_pattern_from_graph, num_graph_hops_for_hessian_sparsity
from matgl.models import TensorNet
from matgl.models._tensornet_pyg import TensorNet as TensorNetPYG
DEFAULT_XYZ_CHIGNOLIN = REPO_ROOT / "data" / "Chignolin" / "Coordinates_9999.xyz"
DEFAULT_XYZ_ABETA40 = REPO_ROOT / "data" / "Abeta40_PED00531" / "PED00531_model_0853_middle.xyz"
DEFAULT_OUTPUT_CHIGNOLIN = SCRIPT_DIR / "sparse_hessian_results" / "chignolin_experiments" / "sparsity_pattern_checks"
DEFAULT_OUTPUT_ABETA40 = SCRIPT_DIR / "sparse_hessian_results" / "abeta40_experiments" / "sparsity_pattern_checks"

PATTERN_CMAP = ListedColormap(["white", "black"])
DIFF_CMAP = ListedColormap(["white", "black", "#d62728", "#1f77b4"])


def infer_num_layers(model) -> int:
    """Number of TensorNet interaction blocks (``len(model.layers)``)."""
    return len(model.layers)


def set_default_device(device: str) -> str:
    """Set the default PyTorch device."""
    resolved = device
    if device == "auto":
        resolved = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(resolved)
    return resolved


def build_random_tensornet_potential(atoms, nblocks: int, seed: int, cutoff: float = 5.0) -> Potential:
    """Build a randomly initialized DGL TensorNet potential for sparsity tests."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    element_types = tuple(sorted(set(atoms.get_chemical_symbols())))
    model = TensorNet(
        element_types=element_types,
        units=64,
        nblocks=nblocks,
        num_rbf=3,
        max_n=3,
        max_l=3,
        rbf_type="SphericalBessel",
        use_smooth=True,
        activation_type="swish",
        cutoff=cutoff,
        equivariance_invariance_group="O(3)",
        width=0.5,
        readout_type="weighted_atom",
        task_type="regression",
        niters_set2set=3,
        nlayers_set2set=3,
        field="node_feat",
        is_intensive=False,
        ntargets=1,
    )
    return Potential(
        model=model,
        data_mean=0.0,
        data_std=1.0,
        calc_forces=True,
        calc_stresses=False,
        calc_hessian=True,
    )


def build_random_tensornet_pyg(atoms, nblocks: int, seed: int, cutoff: float = 5.0) -> TensorNetPYG:
    """Random PyG TensorNet (same hyperparameters as ``build_random_tensornet_potential`` where applicable)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    element_types = tuple(sorted(set(atoms.get_chemical_symbols())))
    return TensorNetPYG(
        element_types=element_types,
        units=64,
        nblocks=nblocks,
        num_rbf=3,
        max_n=3,
        max_l=3,
        rbf_type="SphericalBessel",
        use_smooth=True,
        activation_type="swish",
        cutoff=cutoff,
        equivariance_invariance_group="O(3)",
        width=0.5,
        readout_type="weighted_atom",
        task_type="regression",
        niters_set2set=3,
        nlayers_set2set=3,
        field="node_feat",
        is_intensive=False,
        ntargets=1,
    )


def naive_hessian_pyg(model: TensorNetPYG, graph, lat: torch.Tensor) -> np.ndarray:
    """Dense naive Hessian ∂²E/∂x∂xᵀ for PyG TensorNet (matches ``Potential`` naive path)."""
    device = next(model.parameters()).device
    dtype = matgl.float_th
    # PyG ``Data.to()`` does not accept ``dtype=``; move tensors explicitly.
    graph = graph.to(device)
    graph.frac_coords = graph.frac_coords.to(device=device, dtype=dtype)
    graph.pbc_offset = graph.pbc_offset.to(device=device, dtype=dtype)
    lat = lat.to(device=device, dtype=dtype)
    graph.pbc_offshift = torch.matmul(graph.pbc_offset, lat[0])
    pos = torch.matmul(graph.frac_coords, lat[0]).detach().clone().requires_grad_(True)

    graph.pos = pos
    energy = model(graph, None)
    if energy.dim() > 0:
        energy = energy.sum()

    r = torch.autograd.grad(energy, pos, create_graph=True)[0].view(-1)
    s = r.numel()
    hessian = pos.new_zeros((s, s))
    for iatom in range(s):
        tmp = torch.autograd.grad(r[iatom], pos, retain_graph=iatom < s - 1)[0]
        if tmp is not None:
            hessian[iatom] = tmp.view(-1)
    return hessian.detach().cpu().numpy()


def edge_list_to_dense_pattern(edges: np.ndarray, n: int) -> np.ndarray:
    """Convert an edge list to a dense boolean matrix."""
    pattern = np.zeros((n, n), dtype=bool)
    if len(edges) > 0:
        pattern[edges[:, 0], edges[:, 1]] = True
    return pattern


def hessian_to_atom_block_pattern(hessian: np.ndarray, atol: float) -> np.ndarray:
    """Collapse a Cartesian Hessian into an atom-level 3x3 block sparsity pattern."""
    ndof = hessian.shape[0]
    if hessian.shape != (ndof, ndof):
        raise ValueError(f"Hessian must be square, got {hessian.shape}")
    if ndof % 3 != 0:
        raise ValueError(f"Hessian dimension must be divisible by 3, got {ndof}")

    num_atoms = ndof // 3
    blocks = hessian.reshape(num_atoms, 3, num_atoms, 3).transpose(0, 2, 1, 3)
    atom_pattern = np.any(blocks != 0.0, axis=(2, 3))
    atom_pattern |= atom_pattern.T
    return atom_pattern


def expand_atom_pattern(atom_pattern: np.ndarray) -> np.ndarray:
    """Expand an atom-level sparsity pattern to Cartesian coordinates."""
    return np.kron(atom_pattern.astype(np.uint8), np.ones((3, 3), dtype=np.uint8)).astype(bool)


def compare_patterns(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float | int]:
    """Compute overlap and error counts against a reference pattern."""
    missing = np.count_nonzero(reference & ~candidate)
    extra = np.count_nonzero(candidate & ~reference)
    overlap = np.count_nonzero(reference & candidate)
    union = np.count_nonzero(reference | candidate)
    return {
        "reference_nonzeros": int(np.count_nonzero(reference)),
        "candidate_nonzeros": int(np.count_nonzero(candidate)),
        "missing_nonzeros": int(missing),
        "extra_nonzeros": int(extra),
        "overlap_nonzeros": int(overlap),
        "union_nonzeros": int(union),
        "jaccard": float(overlap / union) if union else 1.0,
    }


def _atom_adjacency_no_self_loops(graph, num_atoms: int):
    """Binary undirected adjacency (no self-loops) as a dense numpy matrix."""
    if hasattr(graph, "edge_index"):
        ei = graph.edge_index
        if isinstance(ei, torch.Tensor):
            ei = ei.detach().cpu().numpy()
        src = ei[0].astype(int)
        dst = ei[1].astype(int)
    else:
        s, d = graph.edges()
        src = s.detach().cpu().numpy().astype(int)
        dst = d.detach().cpu().numpy().astype(int)
    adj = np.zeros((num_atoms, num_atoms), dtype=np.uint8)
    mask = src != dst
    adj[src[mask], dst[mask]] = 1
    adj = adj | adj.T
    return adj


def graph_distance_diagnostic(
    graph,
    naive_atom_pattern: np.ndarray,
    graph_atom_pattern: np.ndarray,
) -> dict[str, object]:
    """Bin missing/extra atom-level disagreements by shortest-path graph distance."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path

    num_atoms = naive_atom_pattern.shape[0]
    adj = _atom_adjacency_no_self_loops(graph, num_atoms)
    dist = shortest_path(csr_matrix(adj), directed=False, unweighted=True)
    dist[np.isinf(dist)] = -1
    dist = dist.astype(int)

    missing_mask = naive_atom_pattern & ~graph_atom_pattern
    extra_mask = graph_atom_pattern & ~naive_atom_pattern

    def _bin(mask: np.ndarray) -> dict[str, int]:
        rows, cols = np.where(mask)
        if len(rows) == 0:
            return {}
        values = dist[rows, cols]
        unique, counts = np.unique(values, return_counts=True)
        return {(f"dist={int(v)}" if v >= 0 else "disconnected"): int(c) for v, c in zip(unique, counts)}

    return {
        "num_atoms": int(num_atoms),
        "missing_atom_pairs": int(missing_mask.sum()),
        "extra_atom_pairs": int(extra_mask.sum()),
        "missing_by_distance": _bin(missing_mask),
        "extra_by_distance": _bin(extra_mask),
    }


def difference_image(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Encode pattern differences for plotting."""
    diff = np.zeros(reference.shape, dtype=np.uint8)
    diff[reference & candidate] = 1
    diff[reference & ~candidate] = 2
    diff[candidate & ~reference] = 3
    return diff


def save_pattern_plot(pattern: np.ndarray, path: Path, title: str) -> None:
    """Save a binary sparsity plot."""
    plt.figure(figsize=(8, 8))
    plt.imshow(pattern.astype(np.uint8), cmap=PATTERN_CMAP, interpolation="nearest", aspect="equal")
    plt.title(title)
    plt.xlabel("Cartesian DOF")
    plt.ylabel("Cartesian DOF")
    plt.tight_layout()
    plt.savefig(path, dpi=250, bbox_inches="tight")
    plt.close()


def save_difference_plot(reference: np.ndarray, candidate: np.ndarray, path: Path, title: str) -> None:
    """Save a difference plot against the naive reference."""
    diff = difference_image(reference, candidate)
    plt.figure(figsize=(8, 8))
    plt.imshow(diff, cmap=DIFF_CMAP, interpolation="nearest", aspect="equal", vmin=0, vmax=3)
    plt.title(title)
    plt.xlabel("Cartesian DOF")
    plt.ylabel("Cartesian DOF")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="black", label="Shared nonzero"),
        plt.Rectangle((0, 0), 1, 1, color="#d62728", label="Missing vs naive"),
        plt.Rectangle((0, 0), 1, 1, color="#1f77b4", label="Extra vs naive"),
    ]
    plt.legend(handles=handles, loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(path, dpi=250, bbox_inches="tight")
    plt.close()


def save_summary(summary: dict[str, object], path: Path) -> None:
    """Write a human-readable JSON summary."""
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=("chignolin", "abeta40"),
        default="chignolin",
        help="Benchmark preset: default xyz/output paths (abeta40 uses medium folded PED00531_model_0853_middle.xyz)",
    )
    parser.add_argument("--xyz", type=Path, default=None, help="Override structure path")
    parser.add_argument(
        "--random-tensornet-nblocks",
        type=int,
        default=2,
        help="TensorNet interaction blocks (random init)",
    )
    parser.add_argument(
        "--dgl-backend",
        action="store_true",
        help="Use DGL TensorNet via Potential instead of default PyG TensorNet",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=5.0,
        help="Graph cutoff (Å) for TensorNet and graph construction",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Seed for random TensorNet initialization",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="Torch default device")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for plots and npz data")
    parser.add_argument(
        "--float64",
        action="store_true",
        help="Run the model and naive Hessian in float64 (overrides matgl.float_th) to rule out float32 noise.",
    )
    args = parser.parse_args()

    if args.float64:
        matgl.float_th = torch.float64
        matgl.float_np = np.float64

    if args.preset == "chignolin":
        xyz_default = DEFAULT_XYZ_CHIGNOLIN
        out_default = DEFAULT_OUTPUT_CHIGNOLIN
    else:
        xyz_default = DEFAULT_XYZ_ABETA40
        out_default = DEFAULT_OUTPUT_ABETA40

    args.xyz = args.xyz or xyz_default
    args.output_dir = args.output_dir or out_default

    device = set_default_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 80)
    print("Sparsity-pattern comparison (naive Hessian vs graph receptive field)")
    print("=" * 80)
    print(f"Preset:    {args.preset}")
    print(f"Structure: {args.xyz}")
    print(f"Device:    {device}")
    print(f"Dtype:     {matgl.float_th}")
    print("Threshold: exact zero comparison")

    atoms = read(str(args.xyz))

    use_pyg = not args.dgl_backend

    if use_pyg:
        model = build_random_tensornet_pyg(
            atoms, args.random_tensornet_nblocks, args.random_seed, cutoff=args.cutoff
        )
        model = model.to(device=device, dtype=matgl.float_th)
        model.eval()
        model_tag = (
            f"TensorNet-PyG-random-{args.random_tensornet_nblocks}layers-cut{args.cutoff:g}-seed{args.random_seed}"
        )
        print(f"Model:     {model_tag}")
        graph_converter = Atoms2GraphPYG(model.element_types, cutoff=args.cutoff)
        graph, lat, _state_attr = graph_converter.get_graph(atoms)
        num_layers = infer_num_layers(model)

        print(f"Atoms:     {len(atoms)}")
        print(f"Layers:    {num_layers} (interaction blocks; graph sparsity uses +1 hop for tensor embedding)")

        print("\n[1/4] Computing naive Hessian reference (PyG TensorNet)...")
        hessian_naive = naive_hessian_pyg(model, graph, lat)
        naive_atom_pattern = hessian_to_atom_block_pattern(hessian_naive, 0.0)
        naive_cart_pattern = expand_atom_pattern(naive_atom_pattern)
        ndof = hessian_naive.shape[0]

        print("\n[2/4] Building graph-based sparsity pattern (PyG graph)...")
        sparsity_hops = num_graph_hops_for_hessian_sparsity(model)
        print(f"Using {sparsity_hops} graph hops for receptive-field pattern")
        graph_edges, graph_n = create_sparsity_pattern_from_graph(graph, sparsity_hops)
    else:
        potential = build_random_tensornet_potential(
            atoms, args.random_tensornet_nblocks, args.random_seed, cutoff=args.cutoff
        )
        model_tag = (
            f"TensorNet-DGL-random-{args.random_tensornet_nblocks}layers-cut{args.cutoff:g}-seed{args.random_seed}"
        )
        print(
            f"Model:     random DGL TensorNet ({args.random_tensornet_nblocks} blocks, "
            f"cutoff={args.cutoff}, seed={args.random_seed})"
        )
        potential.calc_hessian = True
        potential.return_seed_matrix = False
        potential.hessian_sparsity_cutoff = None

        calc = PESCalculator(potential)
        graph_converter = Atoms2Graph(potential.model.element_types, potential.model.cutoff)
        graph, _, _ = graph_converter.get_graph(atoms)
        num_layers = infer_num_layers(potential.model)

        print(f"Atoms:     {len(atoms)}")
        print(f"Layers:    {num_layers}")

        print("\n[1/4] Computing naive Hessian reference...")
        potential.hessian_algorithm = "naive"
        calc.calculate(atoms, properties=["hessian"])
        hessian_naive = np.asarray(calc.results["hessian"], dtype=float)
        naive_atom_pattern = hessian_to_atom_block_pattern(hessian_naive, 0.0)
        naive_cart_pattern = expand_atom_pattern(naive_atom_pattern)
        ndof = hessian_naive.shape[0]

        print("\n[2/4] Building graph-based sparsity pattern...")
        sparsity_hops = num_graph_hops_for_hessian_sparsity(potential.model)
        print(f"Using {sparsity_hops} graph hops for receptive-field pattern")
        graph_edges, graph_n = create_sparsity_pattern_from_graph(graph, sparsity_hops)

    if ndof != graph_n:
        raise RuntimeError(f"Pattern dimensions disagree: naive={ndof}, graph={graph_n}")

    graph_pattern = edge_list_to_dense_pattern(np.asarray(graph_edges, dtype=int), graph_n)

    print("\n[3/4] Saving sparsity plots...")
    stem = f"{args.xyz.stem}_{model_tag}_{timestamp}"
    save_pattern_plot(naive_cart_pattern, args.output_dir / f"{stem}_naive_pattern.png", "Naive Hessian sparsity")
    save_pattern_plot(graph_pattern, args.output_dir / f"{stem}_graph_pattern.png", "Graph sparsity")
    save_difference_plot(
        naive_cart_pattern,
        graph_pattern,
        args.output_dir / f"{stem}_graph_vs_naive_diff.png",
        "Graph sparsity vs naive",
    )

    print("\n[4/4] Writing summary data...")
    graph_atom_pattern = graph_pattern.reshape(len(atoms), 3, len(atoms), 3).any(axis=(1, 3))
    distance_report = graph_distance_diagnostic(graph, naive_atom_pattern, graph_atom_pattern)

    summary = {
        "preset": args.preset,
        "xyz": str(args.xyz),
        "backend": "pyg" if use_pyg else "dgl",
        "cutoff_angstrom": args.cutoff,
        "dtype": str(matgl.float_th),
        "model_tag": model_tag,
        "random_tensornet_nblocks": args.random_tensornet_nblocks,
        "random_seed": args.random_seed,
        "device": device,
        "num_atoms": len(atoms),
        "num_layers": num_layers,
        "sparsity_graph_hops": sparsity_hops,
        "ndof": ndof,
        "naive_nonzeros": int(np.count_nonzero(naive_cart_pattern)),
        "graph": compare_patterns(naive_cart_pattern, graph_pattern),
        "graph_is_symmetric": bool(np.array_equal(graph_pattern, graph_pattern.T)),
        "atom_level_distance_report": distance_report,
    }

    npz_path = args.output_dir / f"{stem}_patterns.npz"
    json_path = args.output_dir / f"{stem}_summary.json"
    np.savez_compressed(
        npz_path,
        naive_cart_pattern=naive_cart_pattern.astype(np.uint8),
        naive_atom_pattern=naive_atom_pattern.astype(np.uint8),
        graph_pattern=graph_pattern.astype(np.uint8),
        hessian_naive=hessian_naive.astype(np.float32),
        num_layers=np.asarray([num_layers], dtype=int),
        sparsity_graph_hops=np.asarray([sparsity_hops], dtype=int),
        num_atoms=np.asarray([len(atoms)], dtype=int),
    )
    save_summary(summary, json_path)

    print(f"Saved archive:  {npz_path}")
    print(f"Saved summary:  {json_path}")
    print("Comparison stats:")
    stats = summary["graph"]
    print(
        f"  {'graph':>11}: candidate={stats['candidate_nonzeros']}, "
        f"missing={stats['missing_nonzeros']}, extra={stats['extra_nonzeros']}, "
        f"jaccard={stats['jaccard']:.6f}"
    )
    print(
        f"  atom-level: missing_pairs={distance_report['missing_atom_pairs']}, "
        f"extra_pairs={distance_report['extra_atom_pairs']}"
    )
    if distance_report["missing_by_distance"]:
        print("  missing atom pairs by graph distance:")
        for k, v in sorted(distance_report["missing_by_distance"].items()):
            print(f"    {k:>14}: {v}")
    if distance_report["extra_by_distance"]:
        print("  extra   atom pairs by graph distance:")
        for k, v in sorted(distance_report["extra_by_distance"].items()):
            print(f"    {k:>14}: {v}")


if __name__ == "__main__":
    main()
