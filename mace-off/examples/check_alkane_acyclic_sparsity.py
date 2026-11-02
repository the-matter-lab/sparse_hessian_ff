"""Run acyclic Hessian on largest alkane; verify sparsity pattern vs real Hessian.

1. Exact pattern agreement on a small alkane (naive vs graph pattern vs acyclic).
2. Acyclic benchmark on alkane_C300 with pattern vs recovered-Hessian support check,
   plus sampled outside-pattern autodiff probes (should be ~0 if the pattern is correct).
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from ase.io import read
from matplotlib.colors import ListedColormap
from torch_geometric.data import Data

from colpackpy.torch_utils import compute_sparse_hessian
from matgl.graph.compute import create_sparsity_pattern_from_graph
from mace.calculators import MACECalculator

SCRIPT_DIR = Path(__file__).resolve().parent
MACE_OFF_ROOT = SCRIPT_DIR.parent
REPO_ROOT = MACE_OFF_ROOT.parent
MODEL_PATH = MACE_OFF_ROOT / "mace_off23" / "MACE-OFF23_medium.model"
ALKANE_SMALL = REPO_ROOT / "data" / "Alkanes" / "alkane_C10.xyz"
ALKANE_LARGE = REPO_ROOT / "data" / "Alkanes" / "alkane_C300.xyz"
OUT_DIR = SCRIPT_DIR / "sparse_hessian_results" / "alkanes_acyclic_sparsity_check"

DIFF_CMAP = ListedColormap(["white", "black", "#d62728", "#1f77b4"])


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def load_model(device: str):
    calc = MACECalculator(
        model_paths=str(MODEL_PATH),
        device=device,
        default_dtype="float64",
    )
    model = calc.models[0]
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    hops = max(int(model.num_interactions.item()), 1)
    return calc, model, hops


def prepare(calc, atoms):
    batch = calc._atoms_to_batch(atoms)
    pos = batch.positions.detach().clone().requires_grad_(True)
    batch_dict = batch.to_dict()
    batch_dict["positions"] = pos
    graph = Data(edge_index=batch.edge_index, num_nodes=int(pos.shape[0]))
    return batch_dict, pos, graph


def energy_on(model, batch_dict, pos):
    bd = dict(batch_dict)
    bd["positions"] = pos
    energy = model(bd, compute_force=False, compute_stress=False, training=True)["energy"]
    if energy.dim() > 0:
        energy = energy.sum()
    return energy


def pattern_dense(edges: np.ndarray, n: int) -> np.ndarray:
    pat = np.zeros((n, n), dtype=bool)
    if len(edges):
        pat[edges[:, 0], edges[:, 1]] = True
    return pat


def naive_hessian(model, batch_dict, pos) -> np.ndarray:
    energy = energy_on(model, batch_dict, pos)
    forces = torch.autograd.grad(energy, pos, create_graph=True)[0].view(-1)
    s = forces.numel()
    H = pos.new_zeros((s, s))
    for i in range(s):
        row = torch.autograd.grad(forces[i], pos, retain_graph=i < s - 1)[0]
        H[i] = row.view(-1)
    return H.detach().cpu().numpy()


def acyclic_hessian(model, batch_dict, pos, graph, hops: int, ordering: str = "SMALLEST_LAST"):
    energy = energy_on(model, batch_dict, pos)
    edges, n = create_sparsity_pattern_from_graph(graph, hops)
    grads0 = torch.autograd.grad(energy, pos, create_graph=True, retain_graph=True)[0]
    H = compute_sparse_hessian(
        energy,
        pos,
        edges,
        n,
        "acyclic",
        ordering=ordering,
        precomputed_grads=grads0,
        return_seed_matrix=True,
        return_stats=True,
        hessian_return_type="dense",
    )
    if isinstance(H, tuple) and len(H) == 3:
        hessian, seed, stats = H
    else:
        hessian, seed = H  # type: ignore[misc]
        stats = {}
    return (
        hessian.detach().cpu().numpy(),
        seed.detach().cpu().numpy(),
        edges,
        n,
        stats,
    )


def compare_patterns(name: str, H_ref: np.ndarray, pattern: np.ndarray, out_dir: Path, tag: str):
    """Compare reference Hessian sparsity (exact != 0) to graph pattern."""
    ref_nz = H_ref != 0
    missing = ref_nz & ~pattern  # true nonzero, pattern says zero (BAD)
    extra = pattern & ~ref_nz  # pattern allows, ref is exact zero (OK / conservative)
    agree_nz = ref_nz & pattern

    report = {
        "system": name,
        "n": int(H_ref.shape[0]),
        "ref_nnz": int(ref_nz.sum()),
        "pattern_nnz": int(pattern.sum()),
        "agree_nnz": int(agree_nz.sum()),
        "missing_nnz": int(missing.sum()),
        "extra_nnz": int(extra.sum()),
        "pattern_covers_all_ref_nonzeros": bool(missing.sum() == 0),
        "max_abs_missing": float(np.abs(H_ref[missing]).max()) if missing.any() else 0.0,
        "max_abs_ref": float(np.abs(H_ref).max()),
    }

    # 0 white, 1 black=agree nz, 2 red=missing, 3 blue=extra
    vis = np.zeros(pattern.shape, dtype=int)
    vis[extra] = 3
    vis[agree_nz] = 1
    vis[missing] = 2
    plt.figure(figsize=(10, 9))
    plt.imshow(vis, cmap=DIFF_CMAP, vmin=0, vmax=3, aspect="equal", interpolation="nearest")
    plt.title(f"{name}: pattern vs naive (black=agree, red=missing, blue=extra)")
    plt.xlabel("DOF")
    plt.ylabel("DOF")
    plt.tight_layout()
    fig_path = out_dir / f"{tag}_pattern_vs_naive_{name}.png"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    report["figure"] = str(fig_path)
    return report


def compare_acyclic_to_naive(H_naive: np.ndarray, H_acyclic: np.ndarray, pattern: np.ndarray):
    diff = H_acyclic - H_naive
    on = pattern
    off = ~pattern
    return {
        "max_abs_diff_on_pattern": float(np.abs(diff[on]).max()) if on.any() else 0.0,
        "rms_diff_on_pattern": float(np.sqrt(np.mean(diff[on] ** 2))) if on.any() else 0.0,
        "max_abs_acyclic_off_pattern": float(np.abs(H_acyclic[off]).max()) if off.any() else 0.0,
        "max_abs_naive_off_pattern": float(np.abs(H_naive[off]).max()) if off.any() else 0.0,
        "fro_rel_on_pattern": float(
            np.linalg.norm(diff[on]) / max(np.linalg.norm(H_naive[on]), 1e-30)
        )
        if on.any()
        else 0.0,
    }


def probe_outside_pattern(model, calc, atoms, hops: int, n_probes: int = 32, seed: int = 0):
    """Autodiff selected Hessian entries that the pattern claims are zero."""
    batch_dict, pos, graph = prepare(calc, atoms)
    edges, n = create_sparsity_pattern_from_graph(graph, hops)
    pattern = pattern_dense(edges, n)
    off = np.argwhere(~pattern)
    if len(off) == 0:
        return {"n_probes": 0, "max_abs": 0.0, "note": "pattern is dense"}

    rng = np.random.default_rng(seed)
    pick = off[rng.choice(len(off), size=min(n_probes, len(off)), replace=False)]

    energy = energy_on(model, batch_dict, pos)
    forces = torch.autograd.grad(energy, pos, create_graph=True)[0].view(-1)
    vals = []
    for i, j in pick:
        # one second derivative entry H_ij
        row = torch.autograd.grad(forces[int(i)], pos, retain_graph=True)[0].view(-1)
        vals.append(float(row[int(j)].detach().cpu()))
    vals = np.asarray(vals, dtype=float)
    return {
        "n_probes": int(len(vals)),
        "max_abs": float(np.abs(vals).max()),
        "mean_abs": float(np.abs(vals).mean()),
        "median_abs": float(np.median(np.abs(vals))),
        "pattern_nnz": int(pattern.sum()),
        "pattern_density": float(pattern.mean()),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.set_default_device("cuda")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log(f"device={device} model={MODEL_PATH}")

    calc, model, hops = load_model(device)
    log(f"num_interactions / sparsity hops = {hops}, r_max={float(calc.r_max)}")

    summary = {
        "timestamp": ts,
        "model": str(MODEL_PATH),
        "device": device,
        "sparsity_hops": hops,
        "r_max": float(calc.r_max),
    }

    # ---- 1) Exact check on C10: naive vs pattern vs acyclic ----
    log(f"Exact sparsity check on {ALKANE_SMALL.name}")
    atoms_s = read(ALKANE_SMALL)
    batch_dict, pos, graph = prepare(calc, atoms_s)
    t0 = time.perf_counter()
    H_naive = naive_hessian(model, batch_dict, pos)
    log(f"  naive done in {time.perf_counter()-t0:.2f}s, shape={H_naive.shape}")

    batch_dict, pos, graph = prepare(calc, atoms_s)
    t0 = time.perf_counter()
    H_acyc, seed, edges, n, stats = acyclic_hessian(model, batch_dict, pos, graph, hops)
    log(
        f"  acyclic done in {time.perf_counter()-t0:.2f}s, HVPs={seed.shape[-1]}, "
        f"stats_keys={list(stats)}"
    )
    pattern = pattern_dense(edges, n)
    c10_pattern = compare_patterns(ALKANE_SMALL.name, H_naive, pattern, OUT_DIR, ts)
    c10_vals = compare_acyclic_to_naive(H_naive, H_acyc, pattern)
    summary["c10_pattern_vs_naive"] = c10_pattern
    summary["c10_acyclic_vs_naive"] = c10_vals
    summary["c10_hvps"] = int(seed.shape[-1])
    log(
        f"  pattern covers all naive nonzeros? {c10_pattern['pattern_covers_all_ref_nonzeros']} "
        f"(missing={c10_pattern['missing_nnz']}, extra={c10_pattern['extra_nnz']})"
    )
    log(
        f"  acyclic vs naive on-pattern: max|Δ|={c10_vals['max_abs_diff_on_pattern']:.3e}, "
        f"relFro={c10_vals['fro_rel_on_pattern']:.3e}"
    )
    log(
        f"  off-pattern max|H_naive|={c10_vals['max_abs_naive_off_pattern']:.3e}, "
        f"max|H_acyclic|={c10_vals['max_abs_acyclic_off_pattern']:.3e}"
    )

    # ---- 2) Acyclic on C300 + pattern support / outside probes ----
    log(f"Acyclic Hessian on {ALKANE_LARGE.name}")
    atoms_l = read(ALKANE_LARGE)
    n_atoms = len(atoms_l)
    log(f"  n_atoms={n_atoms}, n_dof={3*n_atoms}")

    # warm-up
    batch_dict, pos, graph = prepare(calc, atoms_l)
    _ = acyclic_hessian(model, batch_dict, pos, graph, hops)
    if device == "cuda":
        torch.cuda.synchronize()
    log("  warm-up done")

    batch_dict, pos, graph = prepare(calc, atoms_l)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    H_c300, seed_c300, edges_c300, n_c300, stats_c300 = acyclic_hessian(
        model, batch_dict, pos, graph, hops
    )
    if device == "cuda":
        torch.cuda.synchronize()
    t_acyc = time.perf_counter() - t0
    pattern_c300 = pattern_dense(edges_c300, n_c300)
    H_nz = H_c300 != 0
    missing_rec = H_nz & ~pattern_c300
    extra_rec = pattern_c300 & ~H_nz

    c300_support = {
        "n_atoms": n_atoms,
        "n_dof": int(n_c300),
        "time_s": float(t_acyc),
        "hvps": int(seed_c300.shape[-1]),
        "pattern_nnz": int(pattern_c300.sum()),
        "pattern_density": float(pattern_c300.mean()),
        "acyclic_nnz_exact": int(H_nz.sum()),
        "acyclic_entries_outside_pattern": int(missing_rec.sum()),
        "max_abs_acyclic_outside_pattern": float(np.abs(H_c300[missing_rec]).max())
        if missing_rec.any()
        else 0.0,
        "pattern_zeros_that_are_exact_zero_in_acyclic": int(extra_rec.sum()),
        "stats": {
            k: (float(v) if isinstance(v, (float, int, np.floating, np.integer)) else v)
            for k, v in (stats_c300 or {}).items()
            if k != "timings_s"
        },
        "timings_s": (stats_c300 or {}).get("timings_s", {}),
    }
    summary["c300_acyclic"] = c300_support
    log(
        f"  acyclic: {t_acyc:.2f}s, HVPs={c300_support['hvps']}, "
        f"pattern density={c300_support['pattern_density']:.4f}"
    )
    log(
        f"  recovered nnz outside pattern: {c300_support['acyclic_entries_outside_pattern']} "
        f"(max|H|={c300_support['max_abs_acyclic_outside_pattern']:.3e})"
    )

    # Save Hessian / pattern for later
    np.savez_compressed(
        OUT_DIR / f"{ts}_alkane_C300_acyclic.npz",
        hessian=H_c300,
        pattern=pattern_c300,
        edges=edges_c300,
        hvps=np.asarray([seed_c300.shape[-1]]),
        time_s=np.asarray([t_acyc]),
    )

    # Outside-pattern autodiff probes on C300 (true Hessian entries that should vanish)
    log("  probing true Hessian entries outside pattern (should be ~0)")
    probes = probe_outside_pattern(model, calc, atoms_l, hops, n_probes=48, seed=0)
    summary["c300_outside_pattern_probes"] = probes
    log(
        f"  probes={probes['n_probes']}, max|H_ij|={probes['max_abs']:.3e}, "
        f"mean|H_ij|={probes['mean_abs']:.3e}"
    )

    # Plot C300 sparsity pattern
    plt.figure(figsize=(10, 9))
    plt.imshow(pattern_c300, cmap=ListedColormap(["white", "black"]), aspect="equal", interpolation="nearest")
    plt.title(f"alkane_C300 graph sparsity pattern (hops={hops})")
    plt.xlabel("DOF")
    plt.ylabel("DOF")
    plt.tight_layout()
    fig_pat = OUT_DIR / f"{ts}_alkane_C300_pattern.png"
    plt.savefig(fig_pat, dpi=200, bbox_inches="tight")
    plt.close()
    summary["c300_pattern_figure"] = str(fig_pat)

    out_json = OUT_DIR / f"{ts}_sparsity_check_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"Wrote {out_json}")

    ok_c10 = c10_pattern["pattern_covers_all_ref_nonzeros"] and c10_vals["max_abs_naive_off_pattern"] == 0.0
    ok_c300_support = c300_support["acyclic_entries_outside_pattern"] == 0
    # numerical probe tolerance: float64 second derivatives can have tiny noise
    ok_probes = probes["max_abs"] < 1e-8
    log("=" * 60)
    log(f"PASS C10 pattern vs naive: {ok_c10}")
    log(f"PASS C300 acyclic ⊆ pattern: {ok_c300_support}")
    log(f"PASS C300 outside-pattern probes (~0): {ok_probes} (max={probes['max_abs']:.3e})")
    if not (ok_c10 and ok_c300_support and ok_probes):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
