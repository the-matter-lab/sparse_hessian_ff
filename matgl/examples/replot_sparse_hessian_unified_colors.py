#!/usr/bin/env python3
"""
Replot figures from saved sparse_hessian_results using the same method colors as
plot_chignolin_hvp_timing_combined.py (Tableau-style blues / orange / green).

Reads disk only — no benchmark or cutoff reruns.

Coverage
--------
* sparse_hessian_benchmark.py outputs:
  - Per-dataset HVP vs n_atoms and timing vs n_atoms (from *_benchmark_results_*.json
    or *_plot_data_*.npz).
  - Chignolin conformation timing / HVP bar plots (JSON or *_conformation_plot_data_*.npz).
  - Hessian heatmaps + sparsity patterns from sparsity_pattern_data_*_{timestamp}.npz.

* Cutoff / ColPack-style Hessian compression (sparse_hessian_cutoff_real.py):
  The ColPack C++ driver ``05_Compression_and_direct_recovery...cpp`` only prints to
  stdout and does not save figures. The cutoff experiment implements the same
  star/acyclic sparsity-cutoff workflow and saves ``cutoff_plot_series_*.npz`` /
  ``cutoff_debug_data_*.npz`` — those figures are replotted here.

Outputs are written next to the source data with filename suffix ``_unified_colors.png``
(by default) so original PNGs are preserved.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METHODS = ["naive", "star", "acyclic"]
# Matches plot_chignolin_hvp_timing_combined.py
COLORS = {"naive": "#4e79a7", "star": "#f28e2b", "acyclic": "#59a14f"}
COLORS_LIGHT = {"naive": "#a0c4e8", "star": "#f6c98c", "acyclic": "#9fcd8f"}
MARKERS = {"naive": "o", "star": "s", "acyclic": "^"}
# Optional 4th algorithm in cutoff runs
COLORS["distance_two"] = "#b07aa1"
COLORS_LIGHT["distance_two"] = "#d4b3d4"
MARKERS["distance_two"] = "D"

LABEL_FS = int(round(14 * 1.3))
LEGEND_FS = int(round(12 * 1.4))
CBAR_FS = LABEL_FS

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = SCRIPT_DIR / "sparse_hessian_results"


def _ts_from_name(name: str) -> str | None:
    m = re.search(r"_(\d{8}_\d{6})(?:\.[^.]+)?$", name)
    return m.group(1) if m else None


def _hessian_npz_timestamp_for_json(json_ts: str, subdir: Path) -> str | None:
    """Hessian NPZ is written a few seconds before JSON; pick latest sparsity ts <= json_ts."""
    try:
        tj = datetime.strptime(json_ts, "%Y%m%d_%H%M%S")
    except ValueError:
        return None
    best: tuple[datetime, str] | None = None
    for p in subdir.glob("sparsity_pattern_data_*.npz"):
        ts = _ts_from_name(p.stem)
        if not ts:
            continue
        try:
            t = datetime.strptime(ts, "%Y%m%d_%H%M%S")
        except ValueError:
            continue
        if t <= tj and (best is None or t > best[0]):
            best = (t, ts)
    return best[1] if best else None


def load_benchmark_json(path: Path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return d


def plot_system_benchmark_from_results(
    results: list,
    dataset_name: str,
    out_dir: Path,
    timestamp: str,
    suffix: str,
) -> None:
    """HVP and timing vs n_atoms (alkanes, water cluster, …)."""
    plot_data = {m: {"n_atoms": [], "hvp_counts": [], "times": [], "time_stds": []} for m in METHODS}
    for fr in results:
        n_atoms = fr["n_atoms"]
        for m in METHODS:
            mr = fr["methods"].get(m)
            if mr is not None:
                plot_data[m]["n_atoms"].append(n_atoms)
                plot_data[m]["hvp_counts"].append(mr["hvp_count"])
                plot_data[m]["times"].append(mr["time"])
                plot_data[m]["time_stds"].append(mr.get("time_std", 0.0))

    base = f"{dataset_name}_hvp_comparison_{timestamp}{suffix}.png"
    plt.figure(figsize=(12, 8))
    for m in METHODS:
        if plot_data[m]["n_atoms"]:
            plt.plot(
                plot_data[m]["n_atoms"],
                plot_data[m]["hvp_counts"],
                color=COLORS[m],
                marker=MARKERS[m],
                linewidth=2,
                markersize=8,
                label=m.capitalize(),
            )
    plt.xlabel("Number of Atoms", fontsize=LABEL_FS)
    plt.ylabel("Number of HVPs", fontsize=LABEL_FS)
    plt.legend(fontsize=LEGEND_FS)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / base, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  wrote {base}")

    base_t = f"{dataset_name}_timing_comparison_{timestamp}{suffix}.png"
    plt.figure(figsize=(12, 8))
    for m in METHODS:
        if plot_data[m]["n_atoms"]:
            plt.errorbar(
                plot_data[m]["n_atoms"],
                plot_data[m]["times"],
                yerr=plot_data[m]["time_stds"],
                color=COLORS[m],
                marker=MARKERS[m],
                linewidth=2,
                markersize=8,
                label=m.capitalize(),
                capsize=5,
                capthick=2,
            )
    plt.xlabel("Number of Atoms", fontsize=LABEL_FS)
    plt.ylabel("Wall Clock Time (seconds)", fontsize=LABEL_FS)
    plt.legend(fontsize=LEGEND_FS)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / base_t, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  wrote {base_t}")


def sort_conformation_results(results: list) -> list:
    def avg_runtime(fr):
        ts = []
        for m in METHODS:
            mr = fr["methods"].get(m)
            if mr is not None:
                ts.append(mr["time"])
        return float(np.mean(ts)) if ts else float("inf")

    return sorted(results, key=avg_runtime)


def plot_chignolin_conformation_bars(
    results: list,
    dataset_name: str,
    out_dir: Path,
    timestamp: str,
    suffix: str,
) -> None:
    sorted_results = sort_conformation_results(results)
    conf_names = [f"Conformation {i + 1}" for i in range(len(sorted_results))]
    method_times = {m: [] for m in METHODS}
    method_hvps = {m: [] for m in METHODS}
    for fr in sorted_results:
        for m in METHODS:
            mr = fr["methods"].get(m)
            if mr is not None:
                method_times[m].append(mr["time"])
                method_hvps[m].append(mr["hvp_count"])
            else:
                method_times[m].append(0.0)
                method_hvps[m].append(0.0)

    x = np.arange(len(conf_names))
    w = 0.25

    plt.figure(figsize=(14, 8))
    for i, m in enumerate(METHODS):
        if any(t > 0 for t in method_times[m]):
            plt.bar(x + i * w, method_times[m], w, label=m.capitalize(), color=COLORS[m])
    plt.ylabel("Time (seconds)", fontsize=LABEL_FS)
    plt.xticks(x + w, conf_names, rotation=45, fontsize=14)
    plt.legend(fontsize=LEGEND_FS)
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    p = out_dir / f"{dataset_name}_conformation_timing_bars_{timestamp}{suffix}.png"
    plt.savefig(p, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  wrote {p.name}")

    plt.figure(figsize=(14, 8))
    for i, m in enumerate(METHODS):
        if any(h > 0 for h in method_hvps[m]):
            plt.bar(x + i * w, method_hvps[m], w, label=m.capitalize(), color=COLORS[m])
    plt.ylabel("Number of HVPs", fontsize=LABEL_FS)
    plt.xticks(x + w, conf_names, rotation=45, fontsize=14)
    plt.legend(fontsize=LEGEND_FS)
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    p2 = out_dir / f"{dataset_name}_conformation_hvp_bars_{timestamp}{suffix}.png"
    plt.savefig(p2, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  wrote {p2.name}")


def replot_hessian_sparsity_from_npz(npz_path: Path, suffix: str) -> None:
    from matplotlib.colors import BoundaryNorm, ListedColormap

    z = np.load(npz_path, allow_pickle=True)
    hessian = np.asarray(z["hessian"])
    if "sparsity_pattern" in z.files:
        sp = np.asarray(z["sparsity_pattern"], dtype=bool)
    else:
        sp = hessian != 0

    tag = npz_path.stem.replace("sparsity_pattern_data_", "")
    out_dir = npz_path.parent

    plt.figure(figsize=(12, 10))
    plt.imshow(hessian, cmap="RdBu_r", aspect="equal")
    cbar = plt.colorbar()
    cbar.set_label("Hessian Value", fontsize=CBAR_FS)
    plt.xlabel("DOF Index", fontsize=LABEL_FS)
    plt.ylabel("DOF Index", fontsize=LABEL_FS)
    out = out_dir / f"hessian_matrix_{tag}{suffix}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  wrote {out.name}")

    plt.figure(figsize=(12, 10))
    sparse_cmap = ListedColormap(["white", "black"])
    sparse_norm = BoundaryNorm([-0.5, 0.5, 1.5], sparse_cmap.N)
    img = plt.imshow(sp.astype(int), cmap=sparse_cmap, norm=sparse_norm, aspect="equal")
    plt.xlabel("DOF Index", fontsize=LABEL_FS)
    plt.ylabel("DOF Index", fontsize=LABEL_FS)
    cbar = plt.colorbar(img, ticks=[0, 1])
    cbar.ax.set_yticklabels(["0", "1"])
    cbar.set_label("Sparsity Entry", fontsize=CBAR_FS)
    out2 = out_dir / f"sparsity_pattern_{tag}{suffix}.png"
    plt.savefig(out2, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  wrote {out2.name}")


def _eigenval_first_keys(z: np.lib.npyio.NpzFile) -> dict[str, str]:
    """Map algorithm name -> npz key for first-N eigenvalue error series."""
    out = {}
    for k in z.files:
        m = re.match(r"eigenval_errors_first(\d+)_(.+)$", k)
        if m:
            out[m.group(2)] = k
    return out


def replot_cutoff_from_series(npz_path: Path, suffix: str) -> None:
    z = np.load(npz_path, allow_pickle=True)
    cutoffs = np.asarray(z["cutoffs"], dtype=float)
    algorithms = [str(a) for a in z["algorithms"]]
    full_rows = int(z["full_hessian_rows"].flat[0])
    label_fs = int(z["label_fontsize"].flat[0]) if "label_fontsize" in z.files else LABEL_FS
    leg_fs = int(z["legend_fontsize"].flat[0]) if "legend_fontsize" in z.files else LEGEND_FS
    eig_first_keys = _eigenval_first_keys(z)
    n_first = None
    for k in eig_first_keys.values():
        m = re.match(r"eigenval_errors_first(\d+)_", k)
        if m:
            n_first = int(m.group(1))
            break
    n_first = n_first or 10
    timestamp = str(z["timestamp"].flat[0]) if "timestamp" in z.files else _ts_from_name(npz_path.name) or "unknown"
    out_dir = npz_path.parent

    def col(alg: str) -> str:
        return COLORS.get(alg, "#666666")

    # 1) Error vs cutoff
    plt.figure(figsize=(10, 6))
    for alg in algorithms:
        ek = f"errors_{alg}"
        if ek not in z.files:
            continue
        y = np.asarray(z[ek], dtype=float)
        mask = np.isfinite(y) & (y > 0)
        if np.any(mask):
            plt.plot(cutoffs[mask], y[mask], "o-", color=col(alg), linewidth=2, markersize=8, label=alg.replace("_", " ").title())
    plt.xlabel("Cutoff Radius (Angstroms)", fontsize=label_fs)
    plt.ylabel("Max Absolute Error vs Naive", fontsize=label_fs)
    plt.legend(fontsize=leg_fs)
    plt.grid(True, alpha=0.3)
    plt.yscale("log", nonpositive="mask")
    plt.tight_layout()
    p1 = out_dir / f"cutoff_error_comparison_{timestamp}{suffix}.png"
    plt.savefig(p1, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  wrote {p1.name}")

    # 2) HVPs vs cutoff
    plt.figure(figsize=(10, 6))
    for alg in algorithms:
        nk = f"n_hvps_{alg}"
        if nk not in z.files:
            continue
        y = np.asarray(z[nk], dtype=float)
        mask = np.isfinite(y)
        if np.any(mask):
            plt.plot(cutoffs[mask], y[mask], "s-", color=col(alg), linewidth=2, markersize=8, label=alg.replace("_", " ").title())
    plt.axhline(full_rows, color=COLORS["naive"], linestyle="--", linewidth=1.5, label="Full Hessian rows")
    plt.xlabel("Cutoff Radius (Angstroms)", fontsize=label_fs)
    plt.ylabel("Number of Hessian-Vector Products", fontsize=label_fs)
    plt.legend(fontsize=leg_fs)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    p2 = out_dir / f"cutoff_hvp_comparison_{timestamp}{suffix}.png"
    plt.savefig(p2, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  wrote {p2.name}")

    # 3–4) Eigenvalue errors vs cutoff
    for key_suffix, ylabel, fname in (
        ("first", f"Average Relative Error (First {n_first} Eigenvalues)", "cutoff_eigenval_comparison"),
        ("all", "Average Relative Error (All Eigenvalues)", "cutoff_eigenval_all_comparison"),
    ):
        plt.figure(figsize=(10, 6))
        for alg in algorithms:
            if key_suffix == "first":
                ek = eig_first_keys.get(alg) or f"eigenval_errors_first{n_first}_{alg}"
            else:
                ek = f"eigenval_errors_all_{alg}"
            if ek not in z.files:
                continue
            y = np.asarray(z[ek], dtype=float)
            mask = np.isfinite(y) & (y > 0)
            if np.any(mask):
                plt.plot(cutoffs[mask], y[mask], "^-", color=col(alg), linewidth=2, markersize=8, label=alg.replace("_", " ").title())
        plt.xlabel("Cutoff Radius (Angstroms)", fontsize=label_fs)
        plt.ylabel(ylabel, fontsize=label_fs)
        plt.legend(fontsize=leg_fs)
        plt.grid(True, alpha=0.3)
        plt.yscale("log", nonpositive="mask")
        plt.tight_layout()
        pn = out_dir / f"{fname}_{timestamp}{suffix}.png"
        plt.savefig(pn, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  wrote {pn.name}")

    # 5–7) vs HVP count
    for ykey_tpl, ylabel, fname in (
        ("errors", "Max Absolute Error vs Naive", "error_vs_hvp_comparison"),
        ("eigenval_first", f"Average Relative Error (First {n_first} Eigenvalues)", "eigenval_vs_hvp_comparison"),
        ("eigenval_all", "Average Relative Error (All Eigenvalues)", "eigenval_all_vs_hvp_comparison"),
    ):
        plt.figure(figsize=(10, 6))
        for alg in algorithms:
            xs = np.asarray(z[f"n_hvps_{alg}"], dtype=float)
            if ykey_tpl == "errors":
                ys = np.asarray(z[f"errors_{alg}"], dtype=float)
            elif ykey_tpl == "eigenval_first":
                ek = eig_first_keys.get(alg) or f"eigenval_errors_first{n_first}_{alg}"
                ys = np.asarray(z[ek], dtype=float)
            else:
                ys = np.asarray(z[f"eigenval_errors_all_{alg}"], dtype=float)
            mask = np.isfinite(xs) & np.isfinite(ys) & (ys > 0)
            if np.any(mask):
                plt.plot(xs[mask], ys[mask], "o-", color=col(alg), linewidth=2, markersize=8, label=alg.replace("_", " ").title())
        plt.axvline(full_rows, color=COLORS["naive"], linestyle="--", linewidth=1.5, label="Full Hessian rows")
        plt.xlabel("Number of Hessian-Vector Products", fontsize=label_fs)
        plt.ylabel(ylabel, fontsize=label_fs)
        plt.legend(fontsize=leg_fs)
        plt.grid(True, alpha=0.3)
        plt.yscale("log", nonpositive="mask")
        plt.tight_layout()
        pn = out_dir / f"{fname}_{timestamp}{suffix}.png"
        plt.savefig(pn, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  wrote {pn.name}")

    # 8) Per-eigenvalue bars
    fig, axes = plt.subplots(2, len(algorithms), figsize=(20, 10))
    if len(algorithms) == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    per_keys = {alg: f"per_eig_errors_all_{alg}" for alg in algorithms}

    for col_idx, alg in enumerate(algorithms):
        pk = per_keys.get(alg)
        if pk not in z.files:
            continue
        mat = np.asarray(z[pk], dtype=float)
        rep_i = None
        for i in range(mat.shape[0] - 1, -1, -1):
            row = mat[i]
            if np.any(np.isfinite(row) & (row >= 0)):
                rep_i = i
                break
        ax_top = axes[0, col_idx]
        ax_bot = axes[1, col_idx]
        if rep_i is None:
            ax_top.text(0.5, 0.5, "No valid run", ha="center", va="center", transform=ax_top.transAxes)
            ax_bot.text(0.5, 0.5, "No valid run", ha="center", va="center", transform=ax_bot.transAxes)
            continue

        eigvals_naive = None
        dbg = npz_path.parent / f"cutoff_debug_data_{timestamp}.npz"
        if dbg.is_file():
            dz = np.load(dbg, allow_pickle=True)
            if "eigvals_naive_eckart_projected_sorted" in dz.files:
                eigvals_naive = np.asarray(dz["eigvals_naive_eckart_projected_sorted"], dtype=float)

        err = mat[rep_i]
        finite = err[np.isfinite(err)]
        top_cap = float(np.percentile(finite, 99.5)) if finite.size else 1e-12
        top_cap = max(top_cap, 1e-12)
        err_plot = np.clip(err, 0.0, top_cap)
        eig_indices = np.arange(len(err_plot))

        ax_top.bar(eig_indices, err_plot, width=0.9, color=COLORS[alg], alpha=0.92, edgecolor="white", linewidth=0.4)
        ax_top.set_ylim(0.0, top_cap * 1.05)
        ax_top.grid(True, alpha=0.3, axis="y")
        if col_idx == 0:
            ax_top.set_ylabel("Eigenvalue % Error", fontsize=label_fs)
        ax_top.set_xlabel("Eigenvalue Index", fontsize=label_fs)
        ax_top.set_title(f"{alg.replace('_', ' ').title()} (cutoff={cutoffs[rep_i]:.0f} Å)", fontsize=label_fs - 1)

        if eigvals_naive is not None and len(eigvals_naive) == len(err_plot):
            span = float(np.max(eigvals_naive) - np.min(eigvals_naive))
            bw = span / max(len(eigvals_naive) * 1.5, 1)
            if not np.isfinite(bw) or bw <= 0:
                bw = 1e-3
            ax_bot.bar(
                eigvals_naive,
                err_plot,
                width=bw,
                color=COLORS_LIGHT[alg],
                alpha=0.92,
                edgecolor="white",
                linewidth=0.4,
            )
        ax_bot.set_ylim(0.0, top_cap * 1.05)
        ax_bot.grid(True, alpha=0.3, axis="y")
        if col_idx == 0:
            ax_bot.set_ylabel("Eigenvalue % Error", fontsize=label_fs)
        ax_bot.set_xlabel("Ground-Truth Eigenvalue", fontsize=label_fs)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    p8 = out_dir / f"per_eigenvalue_error_bars_{timestamp}{suffix}.png"
    plt.savefig(p8, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  wrote {p8.name}")


def process_benchmark_folder(subdir: Path, suffix: str) -> None:
    jsons = sorted(subdir.glob("*_benchmark_results_*.json"))
    if not jsons:
        print(f"(skip) no benchmark JSON in {subdir.name}")
        return
    jp = jsons[-1]
    ts = _ts_from_name(jp.name)
    if not ts:
        print(f"(skip) bad timestamp: {jp}")
        return
    d = load_benchmark_json(jp)
    dataset_name = d.get("dataset", jp.stem.split("_benchmark_results")[0])
    results = d["results"]
    print(f"{subdir.name}: {jp.name}")

    # Same n_atoms for every structure => conformation benchmark (e.g. Chignolin).
    n_atoms_set = {r["n_atoms"] for r in results}
    if len(results) > 1 and len(n_atoms_set) == 1:
        plot_chignolin_conformation_bars(results, dataset_name, subdir, ts, suffix)
    else:
        plot_system_benchmark_from_results(results, dataset_name, subdir, ts, suffix)

    hts = _hessian_npz_timestamp_for_json(ts, subdir)
    if hts:
        for npz in sorted(subdir.glob(f"sparsity_pattern_data_*_{hts}.npz")):
            replot_hessian_sparsity_from_npz(npz, suffix)
    else:
        print(f"  (no sparsity_pattern_data_* <= {ts})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--suffix",
        default="_unified_colors",
        help="Inserted before .png (default: _unified_colors)",
    )
    ap.add_argument(
        "--results-root",
        type=Path,
        default=RESULTS_ROOT,
        help="sparse_hessian_results directory",
    )
    ap.add_argument(
        "--only",
        choices=("benchmark", "cutoff", "all"),
        default="all",
    )
    args = ap.parse_args()
    suffix = args.suffix if args.suffix.startswith("_") else f"_{args.suffix}"
    root = args.results_root

    if args.only in ("benchmark", "all"):
        for name in ("alkanes_experiments", "water_cluster_experiments", "chignolin_experiments"):
            p = root / name
            if p.is_dir():
                print(f"=== Benchmark: {name} ===")
                process_benchmark_folder(p, suffix)

    if args.only in ("cutoff", "all"):
        cut = root / "cutoff_experiments"
        series = sorted(cut.glob("cutoff_plot_series_*.npz"))
        if series:
            print("=== Cutoff (from cutoff_plot_series) ===")
            replot_cutoff_from_series(series[-1], suffix)
        else:
            print("(skip) no cutoff_plot_series_*.npz")

    print("Done.")


if __name__ == "__main__":
    main()
