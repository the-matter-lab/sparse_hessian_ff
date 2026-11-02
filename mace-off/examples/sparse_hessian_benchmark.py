"""Sparse Hessian benchmark for MACE-OFF23 on the largest alkane and water cluster.

Mirrors ``matgl/examples/sparse_hessian_benchmark.py``, but evaluates only:
  - largest alkane in ``data/Alkanes`` (``alkane_C300.xyz``)
  - largest water cluster ``data/Water Cluster/964.xyz``
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from ase.io import read
from torch_geometric.data import Data

try:
    from colpackpy.torch_utils import compute_sparse_hessian  # type: ignore[import-not-found]
except ImportError:
    compute_sparse_hessian = None

from matgl.graph.compute import create_sparsity_pattern_from_graph

SCRIPT_DIR = Path(__file__).resolve().parent
MACE_OFF_ROOT = SCRIPT_DIR.parent
REPO_ROOT = MACE_OFF_ROOT.parent
DEFAULT_MODEL = MACE_OFF_ROOT / "mace_off23" / "MACE-OFF23_medium.model"
RESULTS_ROOT = SCRIPT_DIR / "sparse_hessian_results"

METHODS = ["naive", "star", "acyclic"]
DEFAULT_ORDERINGS_TO_COMPARE = ["SMALLEST_LAST", "NATURAL"]
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
COLORS = {"naive": "#4e79a7", "star": "#f28e2b", "acyclic": "#59a14f"}

LABEL_FONTSIZE = int(round(14 * 1.3))
LEGEND_FONTSIZE = int(round(12 * 1.4))
TICK_FONTSIZE = 16
BAR_FIGSIZE = (10, 6)

# Largest systems only (by atom count in each dataset).
ALKANE_XYZ = REPO_ROOT / "data" / "Alkanes" / "alkane_C300.xyz"
WATER_XYZ = REPO_ROOT / "data" / "Water Cluster" / "964.xyz"


def _log_progress(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def _style_current_ticks() -> None:
    plt.xticks(fontsize=TICK_FONTSIZE)
    plt.yticks(fontsize=TICK_FONTSIZE)


def _ordering_label(ordering: str) -> str:
    return "".join(part.capitalize() for part in ordering.lower().split("_"))


def _series_label(method: str, ordering: str | None = None) -> str:
    if method == "naive":
        return "Naive"
    pretty_method = method.capitalize()
    if ordering is None:
        return pretty_method
    return f"{pretty_method} ({_ordering_label(ordering)})"


def _resolve_device(device: str | None) -> str:
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


class _MACEHessianCalcProxy:
    """Holds Hessian results; ``calculate`` delegates to the benchmark runner."""

    def __init__(self, bench: "MACEHessianBenchmark"):
        self._bench = bench
        self.results: dict = {}
        self.return_seed_matrix = True

    def calculate(self, atoms, properties=None):
        self._bench._run_mace_hessian(atoms)


class MACEHessianBenchmark:
    """Benchmark naive / star / acyclic Hessians for a local MACE-OFF23 model."""

    def __init__(
        self,
        model_path: Path,
        device: str = "cuda",
        default_dtype: str = "float64",
        coloring_ordering: str = "SMALLEST_LAST",
        collect_breakdown: bool = False,
        methods: list[str] | None = None,
    ):
        self.model_path = Path(model_path)
        self.device = _resolve_device(device)
        self.default_dtype = default_dtype
        self.coloring_ordering = coloring_ordering.upper()
        self.collect_breakdown = collect_breakdown
        self.methods = list(methods) if methods is not None else list(METHODS)

        self.calc_ase = None
        self.model = None
        self.calc: _MACEHessianCalcProxy | None = None
        self.hessian_algorithm = "naive"
        self.sparsity_hops: int | None = None

    def ensure_calculator(self, atoms=None) -> None:
        if self.calc_ase is not None and self.calc is not None and self.model is not None:
            return

        from mace.calculators import MACECalculator

        if not self.model_path.is_file():
            raise FileNotFoundError(f"MACE-OFF23 model not found: {self.model_path}")

        _log_progress(f"Loading MACE-OFF23 model: {self.model_path} (device={self.device})")
        self.calc_ase = MACECalculator(
            model_paths=str(self.model_path),
            device=self.device,
            default_dtype=self.default_dtype,
        )
        self.model = self.calc_ase.models[0]
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.sparsity_hops = max(int(self.model.num_interactions.item()), 1)
        self.calc = _MACEHessianCalcProxy(self)
        _log_progress(
            f"Model ready: r_max={float(self.calc_ase.r_max):.3f} Å, "
            f"num_interactions={self.sparsity_hops} (Hessian sparsity hops)"
        )

    def _configure_method(self, method: str) -> None:
        self.ensure_calculator()
        self.hessian_algorithm = method
        assert self.calc is not None
        self.calc.return_seed_matrix = method != "naive"

    def _build_batch_and_positions(self, atoms):
        assert self.calc_ase is not None
        batch = self.calc_ase._atoms_to_batch(atoms)
        pos = batch.positions.detach().clone().requires_grad_(True)
        batch_dict = batch.to_dict()
        batch_dict["positions"] = pos
        graph = Data(edge_index=batch.edge_index, num_nodes=int(pos.shape[0]))
        return batch_dict, pos, graph

    def _run_mace_hessian(self, atoms) -> None:
        assert self.model is not None and self.calc is not None and self.sparsity_hops is not None
        self.calc.results.clear()
        method = self.hessian_algorithm

        batch_dict, pos, graph = self._build_batch_and_positions(atoms)
        out = self.model(
            batch_dict,
            compute_force=False,
            compute_stress=False,
            training=True,
        )
        energy = out["energy"]
        if energy.dim() > 0:
            energy = energy.sum()

        if method == "naive":
            forces = torch.autograd.grad(energy, pos, create_graph=True)[0].view(-1)
            s = forces.numel()
            hessian = pos.new_zeros((s, s))
            for i in range(s):
                row = torch.autograd.grad(forces[i], pos, retain_graph=i < s - 1)[0]
                if row is not None:
                    hessian[i] = row.view(-1)
            if self.calc.return_seed_matrix:
                hessian_payload = (
                    hessian,
                    torch.eye(s, device=pos.device, dtype=pos.dtype),
                )
            else:
                hessian_payload = hessian
        else:
            if compute_sparse_hessian is None:
                raise RuntimeError(
                    "colpackpy is required for sparse Hessian methods (star, acyclic); "
                    "install colpackpy or run with --methods naive only."
                )
            sparsity_pattern, n = create_sparsity_pattern_from_graph(graph, self.sparsity_hops)
            if n != pos.numel():
                raise RuntimeError(f"Sparsity n={n} != pos.numel()={pos.numel()}")
            grads0 = torch.autograd.grad(energy, pos, create_graph=True, retain_graph=True)[0]
            hessian_payload = compute_sparse_hessian(
                energy,
                pos,
                sparsity_pattern,
                n,
                method,
                ordering=self.coloring_ordering,
                precomputed_grads=grads0,
                return_seed_matrix=self.calc.return_seed_matrix,
                return_stats=self.collect_breakdown,
                hessian_return_type="dense",
            )

        breakdown = None
        if self.calc.return_seed_matrix:
            if (
                isinstance(hessian_payload, tuple)
                and len(hessian_payload) == 3
                and isinstance(hessian_payload[2], dict)
            ):
                hessian_tensor, seed_tensor, breakdown = hessian_payload
            else:
                hessian_tensor, seed_tensor = hessian_payload  # type: ignore[misc]
            self.calc.results["hessian"] = hessian_tensor.detach().cpu().numpy()
            self.calc.results["seed_matrix"] = seed_tensor.detach().cpu().numpy()
        else:
            if isinstance(hessian_payload, tuple) and len(hessian_payload) == 2 and isinstance(
                hessian_payload[1], dict
            ):
                hessian_tensor, breakdown = hessian_payload
            else:
                hessian_tensor = hessian_payload  # type: ignore[assignment]
            self.calc.results["hessian"] = hessian_tensor.detach().cpu().numpy()
        if breakdown is not None:
            self.calc.results["hessian_breakdown"] = breakdown

    def _release_method_memory(self) -> None:
        if self.calc is not None:
            self.calc.results.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_atoms_from_file(self, file_path):
        return read(file_path)

    def warmup_gpu(self, atoms) -> None:
        self.ensure_calculator(atoms)
        warmup_start = time.perf_counter()
        _log_progress(f"Warming up kernels for {len(atoms)} atoms...")
        for method in self.methods:
            self._configure_method(method)
            method_start = time.perf_counter()
            try:
                _log_progress(f"  Warm-up start: method={method}")
                assert self.calc is not None
                self.calc.calculate(atoms, properties=["hessian"])
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                _log_progress(
                    f"  Warm-up done: method={method}, "
                    f"elapsed={time.perf_counter() - method_start:.4f}s"
                )
            except Exception as e:
                _log_progress(f"  Warning: warm-up failed for {method}: {e}")
            finally:
                self._release_method_memory()
        _log_progress(f"Warm-up completed in {time.perf_counter() - warmup_start:.4f}s")

    def benchmark_method(self, atoms, method, n_repeats=1, keep_hessian=False):
        self.ensure_calculator(atoms)
        self._configure_method(method)
        assert self.calc is not None

        times = []
        breakdown_samples = []
        for repeat_idx in range(n_repeats):
            _log_progress(
                f"    Repeat {repeat_idx + 1}/{n_repeats} start: method={method}, "
                f"n_atoms={len(atoms)}"
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            self.calc.calculate(atoms, properties=["hessian"])
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)
            if self.collect_breakdown:
                raw_breakdown = self.calc.results.get("hessian_breakdown")
                if isinstance(raw_breakdown, dict):
                    breakdown_samples.append(
                        {
                            "coloring_s": float(raw_breakdown.get("timings_s", {}).get("coloring_s", 0.0)),
                            "hvp_loop_s": float(raw_breakdown.get("timings_s", {}).get("hvp_loop_s", 0.0)),
                            "recovery_s": float(raw_breakdown.get("timings_s", {}).get("recovery_s", 0.0)),
                        }
                    )
            _log_progress(
                f"    Repeat {repeat_idx + 1}/{n_repeats} complete: elapsed={times[-1]:.4f}s"
            )

        avg_time = float(np.mean(times))
        std_time = float(np.std(times))
        results = self.calc.results
        hessian_array = results["hessian"]
        hessian_shape = hessian_array.shape
        hessian = np.array(hessian_array, copy=True) if keep_hessian else None

        if "seed_matrix" in results and results["seed_matrix"] is not None:
            hvp_count = int(results["seed_matrix"].shape[-1])
        else:
            hvp_count = 3 * len(atoms)

        if breakdown_samples:
            avg_breakdown = {
                key: float(np.mean([sample[key] for sample in breakdown_samples]))
                for key in breakdown_samples[0]
            }
        else:
            avg_breakdown = {"coloring_s": 0.0, "hvp_loop_s": 0.0, "recovery_s": 0.0}
        avg_breakdown["overhead_s"] = max(
            0.0,
            float(
                avg_time
                - avg_breakdown["coloring_s"]
                - avg_breakdown["hvp_loop_s"]
                - avg_breakdown["recovery_s"]
            ),
        )

        self._release_method_memory()
        return {
            "method": method,
            "time": avg_time,
            "time_std": std_time,
            "times": times,
            "hvp_count": hvp_count,
            "hessian_shape": hessian_shape,
            "hessian": hessian,
            "n_atoms": len(atoms),
            "timing_breakdown": avg_breakdown,
        }

    def benchmark_files(
        self,
        xyz_files: list[Path],
        output_dir: Path,
        dataset_name: str,
        n_repeats: int = 1,
    ):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        _log_progress(f"Dataset start: {dataset_name} -> {output_path}")

        all_results = []
        slowest_hessian = None
        fastest_hessian = None
        slowest_hvp = -1
        fastest_hvp = float("inf")
        slowest_file = ""
        fastest_file = ""

        for i, file_path in enumerate(xyz_files):
            file_path = Path(file_path)
            file_start = time.perf_counter()
            _log_progress(f"Processing file {i + 1}/{len(xyz_files)}: {file_path}")
            atoms = self.get_atoms_from_file(file_path)
            n_atoms = len(atoms)
            _log_progress(f"  Loaded {file_path.name}: n_atoms={n_atoms}")
            self.warmup_gpu(atoms)

            file_results = {"filename": file_path.name, "n_atoms": n_atoms, "methods": {}}
            for method in self.methods:
                _log_progress(
                    f"  Benchmarking method={method} repeats={n_repeats} for {file_path.name}"
                )
                try:
                    result = self.benchmark_method(
                        atoms,
                        method,
                        n_repeats,
                        keep_hessian=(method == "acyclic"),
                    )
                    times_str = ", ".join([f"{t:.3f}s" for t in result["times"]])
                    _log_progress(f"    Times: [{times_str}]")
                    _log_progress(f"    Average: {result['time']:.4f}s ± {result['time_std']:.4f}s")
                    _log_progress(f"    HVPs: {result['hvp_count']}")
                    stored_result = {k: v for k, v in result.items() if k != "hessian"}
                    file_results["methods"][method] = stored_result

                    if result["hessian"] is not None and method == "acyclic":
                        hvp = result["hvp_count"]
                        if hvp > slowest_hvp:
                            slowest_hvp = hvp
                            slowest_hessian = result["hessian"]
                            slowest_file = file_path.name
                        if hvp < fastest_hvp:
                            fastest_hvp = hvp
                            fastest_hessian = result["hessian"]
                            fastest_file = file_path.name
                        del result["hessian"]
                        gc.collect()
                except Exception as e:
                    _log_progress(f"    Error with {method}: {e}")
                    file_results["methods"][method] = None

            all_results.append(file_results)
            _log_progress(
                f"Completed file {file_path.name} in {time.perf_counter() - file_start:.2f}s"
            )
            del atoms
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if slowest_hessian is not None:
            self.plot_hessian_matrix(
                slowest_hessian, output_path, f"slowest_{slowest_file}", timestamp
            )
        if fastest_hessian is not None:
            self.plot_hessian_matrix(
                fastest_hessian, output_path, f"fastest_{fastest_file}", timestamp
            )

        self.save_results_and_plots(all_results, output_path, dataset_name, timestamp)
        return all_results

    def plot_hessian_matrix(self, hessian, output_path, file_identifier, timestamp):
        from matplotlib.colors import BoundaryNorm, ListedColormap

        sparsity_pattern = hessian != 0
        plt.figure(figsize=(12, 10))
        plt.imshow(hessian, cmap="RdBu_r", aspect="equal")
        cbar = plt.colorbar()
        cbar.set_label("Hessian Value", fontsize=LABEL_FONTSIZE)
        cbar.ax.tick_params(labelsize=TICK_FONTSIZE)
        plt.xlabel("DOF Index", fontsize=LABEL_FONTSIZE)
        plt.ylabel("DOF Index", fontsize=LABEL_FONTSIZE)
        _style_current_ticks()
        hessian_plot_file = output_path / f"hessian_matrix_{file_identifier}_{timestamp}.png"
        plt.savefig(hessian_plot_file, dpi=300, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(12, 10))
        sparse_cmap = ListedColormap(["white", "black"])
        sparse_norm = BoundaryNorm([-0.5, 0.5, 1.5], sparse_cmap.N)
        sparse_img = plt.imshow(
            sparsity_pattern.astype(int), cmap=sparse_cmap, norm=sparse_norm, aspect="equal"
        )
        plt.xlabel("DOF Index", fontsize=LABEL_FONTSIZE)
        plt.ylabel("DOF Index", fontsize=LABEL_FONTSIZE)
        cbar = plt.colorbar(sparse_img, ticks=[0, 1])
        cbar.ax.set_yticklabels(["0", "1"])
        cbar.set_label("Sparsity Entry", fontsize=LABEL_FONTSIZE)
        cbar.ax.tick_params(labelsize=TICK_FONTSIZE)
        _style_current_ticks()
        sparsity_plot_file = output_path / f"sparsity_pattern_{file_identifier}_{timestamp}.png"
        plt.savefig(sparsity_plot_file, dpi=300, bbox_inches="tight")
        plt.close()
        _log_progress(f"  Hessian matrix: {hessian_plot_file}")
        _log_progress(f"  Sparsity pattern: {sparsity_plot_file}")

    def save_results_and_plots(self, results, output_path, dataset_name, timestamp):
        results_for_json = []
        for file_result in results:
            json_result = {
                "filename": file_result["filename"],
                "n_atoms": file_result["n_atoms"],
                "methods": {},
            }
            for method, method_result in file_result["methods"].items():
                if method_result is not None:
                    json_result["methods"][method] = {
                        k: v for k, v in method_result.items() if k != "hessian"
                    }
                else:
                    json_result["methods"][method] = None
            results_for_json.append(json_result)

        results_file = output_path / f"{dataset_name}_benchmark_results_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump(
                {
                    "dataset": dataset_name,
                    "model": str(self.model_path),
                    "device": self.device,
                    "coloring_ordering": self.coloring_ordering,
                    "sparsity_hops": self.sparsity_hops,
                    "timestamp": timestamp,
                    "note": "Largest alkane + largest water cluster only; GPU warm-up before each system.",
                    "results": results_for_json,
                },
                f,
                indent=2,
            )
        _log_progress(f"Results saved to: {results_file}")
        self.create_plots(results, output_path, dataset_name, timestamp)

    def create_plots(self, results, output_path, dataset_name, timestamp):
        plot_data = {
            method: {"n_atoms": [], "hvp_counts": [], "times": [], "time_stds": [], "labels": []}
            for method in self.methods
        }
        for file_result in results:
            for method in self.methods:
                method_result = file_result["methods"].get(method)
                if method_result is not None:
                    plot_data[method]["n_atoms"].append(file_result["n_atoms"])
                    plot_data[method]["hvp_counts"].append(method_result["hvp_count"])
                    plot_data[method]["times"].append(method_result["time"])
                    plot_data[method]["time_stds"].append(method_result["time_std"])
                    plot_data[method]["labels"].append(file_result["filename"])

        # Bar plots (few systems): clearer than line plots for 1–2 points.
        labels = [r["filename"] for r in results]
        x = np.arange(len(labels))
        bar_width = 0.25

        plt.figure(figsize=BAR_FIGSIZE)
        for i, method in enumerate(self.methods):
            times = []
            for file_result in results:
                method_result = file_result["methods"].get(method)
                times.append(method_result["time"] if method_result is not None else 0.0)
            plt.bar(x + i * bar_width, times, bar_width, label=method.capitalize(), color=COLORS[method])
        plt.ylabel("Wall Clock Time (seconds)", fontsize=LABEL_FONTSIZE)
        plt.xticks(x + bar_width, labels, rotation=20, ha="right", fontsize=TICK_FONTSIZE)
        plt.yticks(fontsize=TICK_FONTSIZE)
        plt.legend(fontsize=LEGEND_FONTSIZE)
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        timing_plot_file = output_path / f"{dataset_name}_timing_comparison_{timestamp}.png"
        plt.savefig(timing_plot_file, dpi=300, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=BAR_FIGSIZE)
        for i, method in enumerate(self.methods):
            hvps = []
            for file_result in results:
                method_result = file_result["methods"].get(method)
                hvps.append(method_result["hvp_count"] if method_result is not None else 0)
            plt.bar(x + i * bar_width, hvps, bar_width, label=method.capitalize(), color=COLORS[method])
        plt.ylabel("Number of HVPs", fontsize=LABEL_FONTSIZE)
        plt.xticks(x + bar_width, labels, rotation=20, ha="right", fontsize=TICK_FONTSIZE)
        plt.yticks(fontsize=TICK_FONTSIZE)
        plt.legend(fontsize=LEGEND_FONTSIZE)
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        hvp_plot_file = output_path / f"{dataset_name}_hvp_comparison_{timestamp}.png"
        plt.savefig(hvp_plot_file, dpi=300, bbox_inches="tight")
        plt.close()

        data_file = output_path / f"{dataset_name}_plot_data_{timestamp}.npz"
        payload = {
            "methods": np.asarray(self.methods, dtype=object),
            "filenames": np.asarray(labels, dtype=object),
        }
        for method in self.methods:
            payload[f"n_atoms_{method}"] = np.asarray(plot_data[method]["n_atoms"], dtype=int)
            payload[f"hvp_counts_{method}"] = np.asarray(plot_data[method]["hvp_counts"], dtype=float)
            payload[f"times_{method}"] = np.asarray(plot_data[method]["times"], dtype=float)
            payload[f"time_stds_{method}"] = np.asarray(plot_data[method]["time_stds"], dtype=float)
        np.savez_compressed(data_file, **payload)
        _log_progress(f"  Timing comparison: {timing_plot_file}")
        _log_progress(f"  HVP comparison: {hvp_plot_file}")
        self.print_summary(results, dataset_name)

    def print_summary(self, results, dataset_name):
        print(f"\n=== BENCHMARK SUMMARY FOR {dataset_name.upper()} ===")
        for file_result in results:
            print(f"\n{file_result['filename']} ({file_result['n_atoms']} atoms)")
            for method in self.methods:
                method_result = file_result["methods"].get(method)
                if method_result is None:
                    print(f"  {method}: FAILED")
                else:
                    print(
                        f"  {method}: {method_result['time']:.4f}s ± {method_result['time_std']:.4f}s, "
                        f"HVPs={method_result['hvp_count']}"
                    )


def create_ordering_comparison_plots(ordering_results, output_path, dataset_name, model_tag):
    """Compare NATURAL vs SMALLEST_LAST (and any other requested orderings)."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    orderings = list(ordering_results.keys())
    if not orderings:
        return

    # Collect filenames / atom counts from the first ordering that has results.
    ref_results = next(iter(ordering_results.values()))
    filenames = [r["filename"] for r in ref_results]
    n_atoms = [r["n_atoms"] for r in ref_results]
    x = np.arange(len(filenames))
    bar_width = 0.35 / max(len(orderings), 1)

    for method in ("star", "acyclic"):
        plt.figure(figsize=BAR_FIGSIZE)
        for i, ordering in enumerate(orderings):
            results = ordering_results[ordering]
            times = []
            for file_result in results:
                method_result = file_result["methods"].get(method)
                times.append(method_result["time"] if method_result is not None else 0.0)
            offset = (i - 0.5 * (len(orderings) - 1)) * bar_width
            plt.bar(
                x + offset,
                times,
                bar_width,
                label=_series_label(method, ordering),
                alpha=0.9,
            )
        plt.ylabel("Wall Clock Time (seconds)", fontsize=LABEL_FONTSIZE)
        plt.xticks(x, filenames, rotation=20, ha="right", fontsize=TICK_FONTSIZE)
        plt.yticks(fontsize=TICK_FONTSIZE)
        plt.legend(fontsize=LEGEND_FONTSIZE)
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        timing_file = (
            output_path / f"{dataset_name}_{model_tag}_ordering_timing_{method}_{timestamp}.png"
        )
        plt.savefig(timing_file, dpi=300, bbox_inches="tight")
        plt.close()
        _log_progress(f"  Ordering timing ({method}): {timing_file}")

        plt.figure(figsize=BAR_FIGSIZE)
        for i, ordering in enumerate(orderings):
            results = ordering_results[ordering]
            hvps = []
            for file_result in results:
                method_result = file_result["methods"].get(method)
                hvps.append(method_result["hvp_count"] if method_result is not None else 0)
            offset = (i - 0.5 * (len(orderings) - 1)) * bar_width
            plt.bar(
                x + offset,
                hvps,
                bar_width,
                label=_series_label(method, ordering),
                alpha=0.9,
            )
        plt.ylabel("Number of HVPs", fontsize=LABEL_FONTSIZE)
        plt.xticks(x, filenames, rotation=20, ha="right", fontsize=TICK_FONTSIZE)
        plt.yticks(fontsize=TICK_FONTSIZE)
        plt.legend(fontsize=LEGEND_FONTSIZE)
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        hvp_file = output_path / f"{dataset_name}_{model_tag}_ordering_hvp_{method}_{timestamp}.png"
        plt.savefig(hvp_file, dpi=300, bbox_inches="tight")
        plt.close()
        _log_progress(f"  Ordering HVP ({method}): {hvp_file}")

    data_file = output_path / f"{dataset_name}_{model_tag}_ordering_comparison_data_{timestamp}.npz"
    payload = {
        "filenames": np.asarray(filenames, dtype=object),
        "n_atoms": np.asarray(n_atoms, dtype=int),
        "orderings": np.asarray(orderings, dtype=object),
    }
    for ordering in orderings:
        for method in METHODS:
            times, hvps = [], []
            for file_result in ordering_results[ordering]:
                method_result = file_result["methods"].get(method)
                times.append(method_result["time"] if method_result is not None else np.nan)
                hvps.append(method_result["hvp_count"] if method_result is not None else np.nan)
            payload[f"times_{method}_{ordering}"] = np.asarray(times, dtype=float)
            payload[f"hvps_{method}_{ordering}"] = np.asarray(hvps, dtype=float)
    np.savez_compressed(data_file, **payload)


def _select_xyz_files(alkanes_only: bool, water_only: bool) -> list[tuple[str, list[Path]]]:
    """Return named groups of xyz files (largest alkane / water only)."""
    groups = []
    if not water_only:
        if not ALKANE_XYZ.is_file():
            raise FileNotFoundError(f"Missing largest alkane: {ALKANE_XYZ}")
        groups.append(("Alkanes_largest", [ALKANE_XYZ]))
    if not alkanes_only:
        if not WATER_XYZ.is_file():
            raise FileNotFoundError(f"Missing largest water cluster: {WATER_XYZ}")
        groups.append(("WaterCluster_largest", [WATER_XYZ]))
    return groups


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="MACE-OFF23 sparse Hessian benchmark (largest alkane + largest water only)"
    )
    dataset_group = parser.add_mutually_exclusive_group()
    dataset_group.add_argument("--alkanes", action="store_true", help="Only alkane_C300.xyz")
    dataset_group.add_argument(
        "--water-cluster",
        action="store_true",
        help="Only Water Cluster/964.xyz",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"Path to MACE-OFF23 .model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--model-size",
        choices=["small", "medium", "large"],
        default=None,
        help="Convenience selector under mace-off/mace_off23/ (overrides --model)",
    )
    parser.add_argument("--device", default=None, help="cuda|cpu (default: cuda if available)")
    parser.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    parser.add_argument("--n-repeats", type=int, default=1, metavar="N")
    parser.add_argument(
        "--coloring-ordering",
        nargs="+",
        choices=COLORING_ORDERINGS,
        default=None,
        help="ColPack orderings (default: SMALLEST_LAST NATURAL)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=None,
        help="Hessian methods to run (default: naive star acyclic)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_ROOT,
        help="Root directory for results",
    )
    parser.add_argument(
        "--collect-breakdown",
        action="store_true",
        help="Collect sparse coloring/HVP/recovery timing breakdown",
    )
    args = parser.parse_args(argv)

    if args.model_size is not None:
        model_path = MACE_OFF_ROOT / "mace_off23" / f"MACE-OFF23_{args.model_size}.model"
    else:
        model_path = Path(args.model)

    selected_orderings = args.coloring_ordering or DEFAULT_ORDERINGS_TO_COMPARE
    methods = args.methods or METHODS
    device = _resolve_device(args.device)
    model_tag = model_path.stem

    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            _log_progress("CUDA requested/default but unavailable; falling back to CPU")
            device = "cpu"
        else:
            torch.set_default_device("cuda")

    groups = _select_xyz_files(alkanes_only=args.alkanes, water_only=args.water_cluster)

    _log_progress("Starting MACE-OFF23 Sparse Hessian Benchmark")
    _log_progress("=" * 60)
    _log_progress(f"Model: {model_path}")
    _log_progress(f"Device: {device}, dtype: {args.dtype}")
    _log_progress(f"Methods: {', '.join(methods)}")
    _log_progress(f"Orderings: {', '.join(selected_orderings)}")
    for name, files in groups:
        _log_progress(f"Systems ({name}): {', '.join(p.name for p in files)}")

    for group_name, xyz_files in groups:
        dataset_output_dir = Path(args.output_dir) / group_name
        dataset_ordering_results = {}
        _log_progress("-" * 60)
        _log_progress(f"Dataset: {group_name}")

        for ordering in selected_orderings:
            ordering_suffix = _ordering_label(ordering)
            run_output_dir = dataset_output_dir / f"{model_tag}_{ordering_suffix}"
            _log_progress(f"Coloring ordering: {ordering}")
            _log_progress(f"Run output directory: {run_output_dir}")
            bench = MACEHessianBenchmark(
                model_path=model_path,
                device=device,
                default_dtype=args.dtype,
                coloring_ordering=ordering,
                collect_breakdown=args.collect_breakdown,
                methods=methods,
            )
            results = bench.benchmark_files(
                xyz_files=xyz_files,
                output_dir=run_output_dir,
                dataset_name=group_name,
                n_repeats=args.n_repeats,
            )
            dataset_ordering_results[ordering] = results

        create_ordering_comparison_plots(
            ordering_results=dataset_ordering_results,
            output_path=dataset_output_dir,
            dataset_name=group_name,
            model_tag=model_tag,
        )

    _log_progress("All requested MACE-OFF23 benchmarks finished.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        _log_progress(f"FATAL: {exc}")
        raise
