#!/usr/bin/env python3
"""
Load saved Chignolin benchmark data (JSON or NPZ) and plot HVP count + timing
in one figure. For each method, two vertical bars side by side: left = HVP count,
right = wall time (twin y-axes so both use raw units).

No benchmark run — only reads disk. Default: latest Chignolin_benchmark_results_*.json
in sparse_hessian_results/chignolin_experiments/.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

METHODS = ["naive", "star", "acyclic"]
COLORS_HVP = {"naive": "#4e79a7", "star": "#f28e2b", "acyclic": "#59a14f"}
COLORS_TIME = {
    "naive": "#a0c4e8",
    "star": "#f6c98c",
    "acyclic": "#9fcd8f",
}

LABEL_FONTSIZE = int(round(14 * 1.3))
LEGEND_FONTSIZE = int(round(12 * 1.4))


def _default_json() -> Path:
    base = Path(__file__).resolve().parent / "sparse_hessian_results" / "chignolin_experiments"
    files = sorted(base.glob("Chignolin_benchmark_results_*.json"))
    if not files:
        raise FileNotFoundError(f"No Chignolin_benchmark_results_*.json under {base}")
    return files[-1]


def load_from_json(path: Path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    n = len(d["results"])
    times = {m: np.zeros(n) for m in METHODS}
    hvps = {m: np.zeros(n) for m in METHODS}
    names = []
    for i, fr in enumerate(d["results"]):
        names.append(fr.get("filename", f"Conf. {i + 1}").replace(".xyz", ""))
        for m in METHODS:
            mr = fr["methods"].get(m)
            if mr is None:
                times[m][i] = np.nan
                hvps[m][i] = np.nan
            else:
                times[m][i] = mr["time"]
                hvps[m][i] = mr["hvp_count"]
    return names, times, hvps, d.get("timestamp", "")


def load_from_npz(path: Path):
    z = np.load(path, allow_pickle=True)
    names = list(z["conformation_names"])
    methods = list(z["methods"])
    times_arr = z["times"]
    hvps_arr = z["hvps"]
    times = {methods[j]: times_arr[j] for j in range(len(methods))}
    hvps = {methods[j]: hvps_arr[j] for j in range(len(methods))}
    ts = ""
    m = re.search(r"_(\d{8}_\d{6})\.npz$", path.name)
    if m:
        ts = m.group(1)
    return names, times, hvps, ts


def sort_by_acyclic_time(names: list, times: dict, hvps: dict):
    """Fastest acyclic time first; non-finite acyclic times last."""
    t_ac = np.asarray(times["acyclic"], dtype=float)
    key = np.where(np.isfinite(t_ac), t_ac, np.inf)
    order = np.argsort(key)
    names_s = [names[i] for i in order]
    times_s = {m: times[m][order] for m in METHODS}
    hvps_s = {m: hvps[m][order] for m in METHODS}
    return names_s, times_s, hvps_s


def plot_combined_paired_bars(times: dict, hvps: dict, out_path: Path):
    """
    Per conformation: three methods. Each method = two bars: left (HVP count), right (time s).
    Left axis: HVP count; right axis: wall time. Method pairs abut (no gap between methods).
    X ticks: Conformation 1, 2, … (same style as benchmark conformation bar plots).
    """
    n_conf = len(times["acyclic"])
    x = np.arange(n_conf, dtype=float)
    bar_w = 0.072
    inner_gap = 0.006  # small gap only between HVP and time within each method
    pair_w = 2 * bar_w + inner_gap

    fig, ax_hvp = plt.subplots(figsize=(14, 8))
    ax_time = ax_hvp.twinx()

    total_w = 3 * pair_w
    for j in range(n_conf):
        left = x[j] - total_w / 2
        for mi, method in enumerate(METHODS):
            base = left + mi * pair_w
            hvp_x = base + bar_w / 2
            time_x = base + bar_w + inner_gap + bar_w / 2
            h = hvps[method][j]
            t = times[method][j]
            if np.isfinite(h) and h >= 0:
                ax_hvp.bar(
                    hvp_x,
                    h,
                    width=bar_w,
                    color=COLORS_HVP[method],
                    edgecolor="white",
                    linewidth=0.6,
                )
            if np.isfinite(t) and t >= 0:
                ax_time.bar(
                    time_x,
                    t,
                    width=bar_w,
                    color=COLORS_TIME[method],
                    edgecolor="white",
                    linewidth=0.6,
                )

    ax_hvp.set_xticks(x)
    # Match sparse_hessian_benchmark.create_conformation_bar_plot
    ax_hvp.set_xticklabels(
        [f"Conformation {j + 1}" for j in range(n_conf)],
        rotation=45,
        fontsize=14,
    )
    ax_hvp.set_ylabel("HVP count", fontsize=LABEL_FONTSIZE, color="#2c3e50")
    ax_time.set_ylabel("Wall time (s)", fontsize=LABEL_FONTSIZE, color="#5d4e37")
    ax_hvp.tick_params(axis="y", labelcolor="#2c3e50")
    ax_time.tick_params(axis="y", labelcolor="#5d4e37")

    ax_hvp.grid(True, alpha=0.3, axis="y")
    ax_hvp.set_axisbelow(True)

    leg_handles = [
        plt.Rectangle((0, 0), 1, 1, fc=COLORS_HVP[m], ec="white", label=m.capitalize())
        for m in METHODS
    ]
    ax_hvp.legend(
        handles=leg_handles,
        fontsize=LEGEND_FONTSIZE - 1,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        framealpha=0.95,
        borderaxespad=0.0,
    )

    plt.tight_layout(rect=(0, 0, 1, 0.9))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Path to Chignolin_benchmark_results_*.json (default: latest in chignolin_experiments)",
    )
    p.add_argument(
        "--npz",
        type=Path,
        default=None,
        help="Path to Chignolin_conformation_plot_data_*.npz (overrides --json)",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output PNG path",
    )
    args = p.parse_args()

    if args.npz:
        names, times, hvps, ts = load_from_npz(args.npz)
    else:
        jp = args.json or _default_json()
        names, times, hvps, ts = load_from_json(jp)

    out = args.output
    if out is None:
        out_dir = Path(__file__).resolve().parent / "sparse_hessian_results" / "chignolin_experiments"
        out = out_dir / f"Chignolin_hvp_time_paired_{ts or 'from_data'}.png"

    _, times, hvps = sort_by_acyclic_time(names, times, hvps)
    plot_combined_paired_bars(times, hvps, out)


if __name__ == "__main__":
    main()
