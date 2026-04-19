import json
import time
import gc
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matgl
import numpy as np
import torch
from ase.io import read
from matgl.ext.ase import Atoms2GraphPYG
from matgl.graph.compute import create_sparsity_pattern_from_graph, num_graph_hops_for_hessian_sparsity
from matgl.models._tensornet_pyg import TensorNet as TensorNetPYG

try:
    from colpackpy.torch_utils import compute_sparse_hessian  # type: ignore[import-not-found]
except ImportError:
    compute_sparse_hessian = None

torch.set_default_device("cuda")

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
COLORS_LIGHT = {"naive": "#a0c4e8", "star": "#f6c98c", "acyclic": "#9fcd8f"}
MARKERS = {"naive": "o", "star": "s", "acyclic": "^"}
ORDERING_LINESTYLES = {"SMALLEST_LAST": "--", "NATURAL": "-"}
ORDERING_MARKERS = {"SMALLEST_LAST": "s", "NATURAL": "o"}
# Timing breakdown (JSON / method results); overhead merged into HVP for stacked bars.
BREAKDOWN_JSON_KEYS = ["coloring_s", "hvp_loop_s", "recovery_s", "overhead_s"]
BREAKDOWN_STACK_KEYS = ["coloring_s", "hvp_loop_s", "recovery_s"]
BREAKDOWN_STACK_LABELS = {
    "coloring_s": "Coloring",
    "hvp_loop_s": "HVP",
    "recovery_s": "Recovery",
}
BREAKDOWN_STACK_HATCHES = {
    "coloring_s": "|||",
    "hvp_loop_s": "///",
    "recovery_s": "...",
}
BREAKDOWN_SPARSE_KEYS = ["coloring_s", "hvp_loop_s", "recovery_s"]

LABEL_FONTSIZE = int(round(14 * 1.3))
LEGEND_FONTSIZE = int(round(12 * 1.4))
CBAR_LABEL_FONTSIZE = LABEL_FONTSIZE
TICK_FONTSIZE = 16
LINEWIDTH = 4.5
MARKERSIZE = 10
LINE_FIGSIZE = (10, 6)
BAR_FIGSIZE = (10, 6)


def _style_current_ticks():
    plt.xticks(fontsize=TICK_FONTSIZE)
    plt.yticks(fontsize=TICK_FONTSIZE)


def _style_axis_ticks(ax):
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)


def _log_progress(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


class _PyGHessianCalcProxy:
    """Holds Hessian results; ``calculate`` delegates to ``HessianBenchmark._run_pyg_hessian``."""

    def __init__(self, bench: "HessianBenchmark"):
        self._bench = bench
        self.results: dict = {}
        self.return_seed_matrix = True

    def calculate(self, atoms, properties=None):
        self._bench._run_pyg_hessian(atoms)


def _ordering_label(ordering: str) -> str:
    parts = ordering.lower().split("_")
    return "".join(part.capitalize() for part in parts)


def _build_model_specs(random_seed: int) -> list[dict]:
    """Two benchmarks: 1 and 0 TensorNet *interaction* blocks (embedding is always on).

    With ``tensor_embedding`` counted as one graph hop, those are **2** and **1** effective
    message-passing depths (same reach as the old ``nblocks=2`` / ``nblocks=1`` configs).
    """
    return [
        {
            "tag": "tensornet_1interaction_5A",
            "display_name": "Random TensorNet (1 interaction + embedding ≈ 2 eff. MP hops, 5 Å)",
            "kind": "random_tensornet",
            "nblocks": 1,
            "cutoff": 5.0,
            "seed": random_seed,
        },
        {
            "tag": "tensornet_0interaction_5A",
            "display_name": "Random TensorNet (embedding only ≈ 1 eff. MP hop, 5 Å)",
            "kind": "random_tensornet",
            "nblocks": 0,
            "cutoff": 5.0,
            "seed": random_seed,
        },
    ]


def _series_label(method: str, ordering: str | None = None) -> str:
    if method == "naive":
        return "Naive"
    pretty_method = method.capitalize()
    if ordering is None:
        return pretty_method
    return f"{pretty_method} ({_ordering_label(ordering)})"


def _uses_conformation_plots(dataset_name: str) -> bool:
    """Return True only for datasets that should be treated as conformations."""
    normalized = dataset_name.lower().replace("_", "").replace(" ", "")
    return "abeta40" in normalized or "chignolin" in normalized


def _sort_system_series(x_values, *ys):
    x_array = np.asarray(x_values, dtype=float)
    order = np.argsort(x_array, kind="stable")
    sorted_arrays = [x_array[order]]
    for values in ys:
        sorted_arrays.append(np.asarray(values)[order])
    return tuple(sorted_arrays)


def _adjust_timing_breakdown_for_display(method: str, total: float, raw: dict[str, float]) -> dict[str, float]:
    """Match JSON semantics for stacked timing bars (replot_sparse_hessian_unified_colors).

    Naive runs often store the full wall time under ``overhead_s`` with no sparse
    phases; for the breakdown figure that should read as HVP-only time.
    """
    out = {c: float(raw.get(c, 0.0)) for c in BREAKDOWN_JSON_KEYS}
    if not np.isfinite(total) or total <= 0:
        return out
    sparse_sum = out["coloring_s"] + out["hvp_loop_s"] + out["recovery_s"]
    if method == "naive" and sparse_sum <= 1e-9 * total and out["overhead_s"] >= 0.5 * total:
        return {
            "coloring_s": 0.0,
            "hvp_loop_s": total,
            "recovery_s": 0.0,
            "overhead_s": 0.0,
        }
    return out


def _timing_stack_segments_for_display(method: str, method_result: dict | None) -> dict[str, float] | None:
    """Wall-time stack segments (overhead folded into HVP) for one method result, or None if missing."""
    if method_result is None:
        return None
    breakdown = method_result.get("timing_breakdown") or {}
    t_wall = float(method_result["time"])
    raw_parts = {}
    for c in BREAKDOWN_JSON_KEYS:
        default_value = t_wall if c == "overhead_s" else 0.0
        raw_parts[c] = float(breakdown.get(c, default_value))
    parts = _adjust_timing_breakdown_for_display(method, t_wall, raw_parts)
    oh = max(0.0, float(parts.get("overhead_s", 0.0)))
    return {
        "coloring_s": float(parts["coloring_s"]),
        "hvp_loop_s": float(parts["hvp_loop_s"]) + oh,
        "recovery_s": float(parts["recovery_s"]),
    }


class HessianBenchmark:
    """Comprehensive benchmarking class for Hessian calculation methods with GPU warm-up."""
    
    def __init__(
        self,
        model_spec=None,
        test_over_conformations=False,
        coloring_ordering="SMALLEST_LAST",
        collect_breakdown=False,
    ):
        """Initialize the benchmark with a model and optional test_over_conformations flag."""
        if model_spec is None:
            model_spec = {
                "tag": "tensornet_1interaction_5A",
                "display_name": "Random TensorNet (1 interaction + embedding ≈ 2 eff. MP hops, 5 Å)",
                "kind": "random_tensornet",
                "nblocks": 1,
                "cutoff": 5.0,
                "seed": 0,
            }
        self.model_spec = dict(model_spec)
        self.test_over_conformations = test_over_conformations
        self.coloring_ordering = coloring_ordering.upper()
        self.collect_breakdown = collect_breakdown
        self.model: TensorNetPYG | None = None
        self._graph_converter: Atoms2GraphPYG | None = None
        self.calc: _PyGHessianCalcProxy | None = None
        self._element_types = None
        self.hessian_algorithm = "naive"

        # Methods to benchmark
        self.methods = METHODS

    def _build_random_tensornet_pyg(self, atoms) -> TensorNetPYG:
        seed = int(self.model_spec.get("seed", 0))
        cutoff = float(self.model_spec.get("cutoff", 5.0))
        nblocks = int(self.model_spec.get("nblocks", 1))

        torch.manual_seed(seed)
        np.random.seed(seed)

        element_types = tuple(sorted(set(atoms.get_chemical_symbols())))
        model = TensorNetPYG(
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
        self._element_types = element_types
        return model

    def _build_model(self, atoms=None) -> TensorNetPYG:
        kind = self.model_spec["kind"]
        if kind == "random_tensornet":
            if atoms is None:
                raise ValueError("atoms are required to initialize a random TensorNet benchmark")
            model = self._build_random_tensornet_pyg(atoms)
        else:
            raise ValueError(f"Unknown model kind: {kind}")

        for param in model.parameters():
            param.requires_grad_(False)

        return model

    def ensure_calculator(self, atoms=None):
        current_element_types = None if atoms is None else tuple(sorted(set(atoms.get_chemical_symbols())))
        if self.model is not None and self.calc is not None:
            if atoms is None:
                return
            if self.model_spec["kind"] != "random_tensornet" or current_element_types == self._element_types:
                return

        if atoms is None:
            raise ValueError("atoms are required to initialize the PyG TensorNet benchmark")
        self.model = self._build_model(atoms)
        _dev = (
            torch.get_default_device()
            if hasattr(torch, "get_default_device")
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(device=_dev, dtype=matgl.float_th)
        self.model.eval()
        self._graph_converter = Atoms2GraphPYG(self.model.element_types, cutoff=float(self.model.cutoff))
        self.calc = _PyGHessianCalcProxy(self)

    def _configure_method(self, method):
        """Configure algorithm and whether we need to materialize a seed matrix."""
        self.ensure_calculator()
        self.hessian_algorithm = method
        need_seed_matrix = method != "naive"
        assert self.calc is not None
        self.calc.return_seed_matrix = need_seed_matrix

    def _run_pyg_hessian(self, atoms) -> None:
        """Compute Hessian via PyG TensorNet; fills ``self.calc.results`` like ``PESCalculator``."""
        assert self.model is not None and self.calc is not None and self._graph_converter is not None
        self.calc.results.clear()
        method = self.hessian_algorithm
        device = next(self.model.parameters()).device
        dtype = matgl.float_th

        graph, lat, _state = self._graph_converter.get_graph(atoms)
        graph = graph.to(device)
        graph.frac_coords = graph.frac_coords.to(device=device, dtype=dtype)
        graph.pbc_offset = graph.pbc_offset.to(device=device, dtype=dtype)
        lat = lat.to(device=device, dtype=dtype)
        graph.pbc_offshift = torch.matmul(graph.pbc_offset, lat[0])
        pos = torch.matmul(graph.frac_coords, lat[0]).detach().clone().requires_grad_(True)
        graph.pos = pos

        energy = self.model(graph, None)
        if energy.dim() > 0:
            energy = energy.sum()

        if method == "naive":
            r = torch.autograd.grad(energy, pos, create_graph=True)[0].view(-1)
            s = r.numel()
            hessian = pos.new_zeros((s, s))
            for iatom in range(s):
                tmp = torch.autograd.grad(r[iatom], pos, retain_graph=iatom < s - 1)[0]
                if tmp is not None:
                    hessian[iatom] = tmp.view(-1)
            if self.calc.return_seed_matrix:
                hessian_payload = (hessian, torch.eye(s, device=device, dtype=dtype))
            else:
                hessian_payload = hessian
        else:
            if compute_sparse_hessian is None:
                raise RuntimeError(
                    "colpackpy is required for sparse Hessian methods (star, acyclic, …); "
                    "install colpackpy or run with --methods naive only."
                )
            sparsity_hops = num_graph_hops_for_hessian_sparsity(self.model)
            sparsity_pattern, n = create_sparsity_pattern_from_graph(graph, sparsity_hops)
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
            if isinstance(hessian_payload, tuple) and len(hessian_payload) == 2 and isinstance(hessian_payload[1], dict):
                hessian_tensor, breakdown = hessian_payload
            else:
                hessian_tensor = hessian_payload  # type: ignore[assignment]
            self.calc.results["hessian"] = hessian_tensor.detach().cpu().numpy()
        if breakdown is not None:
            self.calc.results["hessian_breakdown"] = breakdown

    def _release_method_memory(self):
        """Drop retained calculator outputs and release unused cached memory."""
        self.calc.results.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    def get_atoms_from_file(self, file_path):
        """Load atoms from xyz file and return atoms object."""
        return read(file_path)
    
    def warmup_gpu(self, atoms):
        """Warm up GPU kernels by running each method once before benchmarking."""
        self.ensure_calculator(atoms)
        warmup_start = time.perf_counter()
        _log_progress(f"Warming up GPU kernels for {len(atoms)} atoms...")
        
        for method in self.methods:
            self._configure_method(method)
            method_start = time.perf_counter()
            try:
                _log_progress(f"  Warm-up start: method={method}")
                # Run once to warm up kernels
                self.calc.calculate(atoms, properties=['hessian'])
                torch.cuda.synchronize()  # Ensure GPU operations complete
                _log_progress(
                    f"  Warm-up done: method={method}, "
                    f"elapsed={time.perf_counter() - method_start:.4f}s"
                )
            except Exception as e:
                _log_progress(f"  Warning: warm-up failed for {method}: {e}")
            finally:
                self._release_method_memory()
        
        _log_progress(
            f"GPU warm-up completed in {time.perf_counter() - warmup_start:.4f}s"
        )
    
    def benchmark_method(self, atoms, method, n_repeats=1, keep_hessian=False):
        """Benchmark a single method and return timing and HVP count."""
        self.ensure_calculator(atoms)
        self._configure_method(method)
        
        # Accurate timing with GPU synchronization
        times = []
        breakdown_samples = []
        
        for repeat_idx in range(n_repeats):
            _log_progress(
                f"    Repeat {repeat_idx + 1}/{n_repeats} start: method={method}, "
                f"n_atoms={len(atoms)}"
            )
            _log_progress(f"    Repeat {repeat_idx + 1}/{n_repeats}: pre-sync start")
            torch.cuda.synchronize()  # Ensure previous operations complete
            _log_progress(f"    Repeat {repeat_idx + 1}/{n_repeats}: pre-sync done")
            t0 = time.perf_counter()
            _log_progress(f"    Repeat {repeat_idx + 1}/{n_repeats}: calculate start")
            self.calc.calculate(atoms, properties=['hessian'])
            _log_progress(f"    Repeat {repeat_idx + 1}/{n_repeats}: calculate done")
            _log_progress(f"    Repeat {repeat_idx + 1}/{n_repeats}: post-sync start")
            torch.cuda.synchronize()  # Ensure calculation completes
            _log_progress(f"    Repeat {repeat_idx + 1}/{n_repeats}: post-sync done")
            t1 = time.perf_counter()
            start_time, end_time = t0, t1  # perf_counter monotonic, better than time.time()
            times.append(end_time - start_time)
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
                f"    Repeat {repeat_idx + 1}/{n_repeats} complete: "
                f"elapsed={times[-1]:.4f}s"
            )
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        # Get the results to extract seed matrix info
        _log_progress(f"    Result extraction start: method={method}")
        results = self.calc.results
        hessian_array = results['hessian']
        hessian_shape = hessian_array.shape
        hessian = np.array(hessian_array, copy=True) if keep_hessian else None
        _log_progress(
            f"    Hessian extracted: method={method}, shape={hessian_shape}"
        )
        
        # Get HVP count from seed matrix
        hvp_count = None
        if 'seed_matrix' in results and results['seed_matrix'] is not None:
            seed_matrix_shape = results['seed_matrix'].shape
            hvp_count = seed_matrix_shape[-1]  # Last dimension gives number of HVPs
            _log_progress(
                f"    Seed matrix extracted: method={method}, shape={seed_matrix_shape}"
            )
        else:
            # For naive method, estimate based on problem size
            n_atoms = len(atoms)
            n_dof = 3 * n_atoms
            hvp_count = n_dof  # Naive method uses one HVP per degree of freedom
            _log_progress(
                f"    No seed matrix for method={method}; using naive estimate hvp_count={hvp_count}"
            )

        if breakdown_samples:
            avg_breakdown = {
                key: float(np.mean([sample[key] for sample in breakdown_samples]))
                for key in breakdown_samples[0]
            }
        else:
            avg_breakdown = {
                "coloring_s": 0.0,
                "hvp_loop_s": 0.0,
                "recovery_s": 0.0,
            }
        avg_breakdown["overhead_s"] = max(
            0.0,
            float(avg_time - avg_breakdown["coloring_s"] - avg_breakdown["hvp_loop_s"] - avg_breakdown["recovery_s"]),
        )
        if self.collect_breakdown:
            _log_progress(
                "    Breakdown: "
                f"coloring={avg_breakdown['coloring_s']:.4f}s, "
                f"hvp={avg_breakdown['hvp_loop_s']:.4f}s, "
                f"recovery={avg_breakdown['recovery_s']:.4f}s, "
                f"overhead={avg_breakdown['overhead_s']:.4f}s"
            )

        self._release_method_memory()
        
        return {
            'method': method,
            'time': avg_time,
            'time_std': std_time,
            'times': times,
            'hvp_count': hvp_count,
            'hessian_shape': hessian_shape,
            'hessian': hessian,  # Include the actual hessian
            'n_atoms': len(atoms),
            'timing_breakdown': avg_breakdown,
        }
    
    def benchmark_directory(
        self,
        data_dir,
        output_dir="sparse_hessian_results",
        n_repeats=1,
        reverse_file_order: bool = False,
    ):
        """Benchmark all xyz files in a directory, with optional test_over_conformations.

        reverse_file_order: process files in reverse order (after sorting). Use with
        test_over_conformations to check GPU thermal / ordering bias on timings.
        """
        data_path = Path(data_dir)
        output_path = Path(output_dir)
        if not output_path.is_absolute():
            output_path = Path(__file__).resolve().parent / output_path
        output_path.mkdir(parents=True, exist_ok=True)
        _log_progress(f"Dataset start: data_dir={data_path} output_dir={output_path}")
        
        # Get all xyz files
        xyz_files = list(data_path.glob("*.xyz"))
        if not xyz_files:
            raise ValueError(f"No xyz files found in {data_dir}")
        
        # Sort files by number of atoms (not by filename)
        file_atom_counts = []
        for file_path in xyz_files:
            atoms = self.get_atoms_from_file(file_path)
            file_atom_counts.append((file_path, len(atoms)))
        
        if self.test_over_conformations:
            # Same atom count: stable order by filename so forward/reverse is reproducible
            file_atom_counts.sort(key=lambda x: x[0].name)
            sorted_files = [item[0] for item in file_atom_counts]
        else:
            # Sort by atom count
            file_atom_counts.sort(key=lambda x: x[1])
            sorted_files = [item[0] for item in file_atom_counts]

        if reverse_file_order:
            sorted_files = list(reversed(sorted_files))
            _log_progress("Processing order: REVERSED (last filename first)")

        _log_progress(f"Found {len(sorted_files)} xyz files in {data_dir}")
        _log_progress("Benchmark processing order:")
        for file_path in sorted_files:
            n_atoms = next(n for p, n in file_atom_counts if p == file_path)
            _log_progress(f"  {file_path.name}: {n_atoms} atoms")
        
        # Store results
        all_results = []
        # Fastest/slowest by acyclic HVP count (same algorithm, different systems)
        slowest_hessian = None
        fastest_hessian = None
        slowest_hvp = -1
        fastest_hvp = float('inf')
        slowest_file = ""
        fastest_file = ""
        
        # Benchmark each file
        for i, file_path in enumerate(sorted_files):
            file_start = time.perf_counter()
            _log_progress(
                f"Processing file {i + 1}/{len(sorted_files)}: {file_path.name} "
                f"(path={file_path}, atoms={next(n for p, n in file_atom_counts if p == file_path)})"
            )
            
            try:
                _log_progress(f"  Reading atoms from {file_path}")
                atoms = self.get_atoms_from_file(file_path)
                n_atoms = len(atoms)
                _log_progress(f"  Loaded {file_path.name}: n_atoms={n_atoms}")
                
                # Warm up GPU kernels before benchmarking this system
                self.warmup_gpu(atoms)
                
                file_results = {
                    'filename': file_path.name,
                    'n_atoms': n_atoms,
                    'methods': {}
                }
                
                # Benchmark each method
                for method in self.methods:
                    _log_progress(
                        f"  Benchmarking method={method} repeats={n_repeats} "
                        f"for file={file_path.name}"
                    )
                    try:
                        result = self.benchmark_method(
                            atoms,
                            method,
                            n_repeats,
                            keep_hessian=(method == "acyclic"),
                        )
                        
                        # Show individual run times for transparency
                        times_str = ", ".join([f"{t:.3f}s" for t in result['times']])
                        _log_progress(f"    Times: [{times_str}]")
                        _log_progress(f"    Average: {result['time']:.4f}s ± {result['time_std']:.4f}s")
                        _log_progress(f"    HVPs: {result['hvp_count']}")
                        
                        stored_result = {k: v for k, v in result.items() if k != 'hessian'}
                        file_results['methods'][method] = stored_result

                        if result['hessian'] is not None and method == 'acyclic':
                            hvp = result['hvp_count']
                            if hvp > slowest_hvp:
                                slowest_hvp = hvp
                                slowest_hessian = result['hessian']
                                slowest_file = file_path.name
                            if hvp < fastest_hvp:
                                fastest_hvp = hvp
                                fastest_hessian = result['hessian']
                                fastest_file = file_path.name
                            del result['hessian']
                            gc.collect()

                    except Exception as e:
                        _log_progress(f"    Error with {method}: {e}")
                        file_results['methods'][method] = None
                
                all_results.append(file_results)
                _log_progress(
                    f"Completed file {file_path.name} in {time.perf_counter() - file_start:.2f}s"
                )
                del atoms
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                _log_progress(f"  Error processing {file_path.name}: {e}")
                continue
        
        # Create plots of Hessians and sparsity patterns (by acyclic HVP: slowest=most HVPs, fastest=least HVPs)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if slowest_hessian is not None:
            _log_progress(
                f"Creating slowest Hessian plots for {slowest_file} (acyclic HVPs={slowest_hvp})"
            )
            self.plot_hessian_matrix(slowest_hessian, output_path, f"slowest_{slowest_file}", timestamp)
            _log_progress(f"Slowest (acyclic HVPs={slowest_hvp}) Hessian plots saved for {slowest_file}")
        if fastest_hessian is not None:
            _log_progress(
                f"Creating fastest Hessian plots for {fastest_file} (acyclic HVPs={fastest_hvp})"
            )
            self.plot_hessian_matrix(fastest_hessian, output_path, f"fastest_{fastest_file}", timestamp)
            _log_progress(f"Fastest (acyclic HVPs={fastest_hvp}) Hessian plots saved for {fastest_file}")
        
        # Save results and create plots
        _log_progress(f"Saving aggregate results and plots for dataset={data_path.name}")
        self.save_results_and_plots(all_results, output_path, data_path.name)
        _log_progress(f"Dataset finished: {data_path.name}")
        
        return all_results
    
    def plot_hessian_matrix(self, hessian, output_path, file_identifier, timestamp):
        """Create plots of the Hessian matrix and its sparsity pattern."""
        import matplotlib.pyplot as plt
        from matplotlib.colors import BoundaryNorm, ListedColormap
        
        # Create sparsity pattern (boolean mask)
        sparsity_pattern = hessian != 0
        
        # Plot the full Hessian matrix
        plt.figure(figsize=(12, 10))
        plt.imshow(hessian, cmap="RdBu_r", aspect="equal")
        cbar = plt.colorbar()
        cbar.set_label("Hessian Value", fontsize=CBAR_LABEL_FONTSIZE)
        cbar.ax.tick_params(labelsize=TICK_FONTSIZE)
        plt.xlabel("DOF Index", fontsize=LABEL_FONTSIZE)
        plt.ylabel("DOF Index", fontsize=LABEL_FONTSIZE)
        _style_current_ticks()
        
        hessian_plot_file = output_path / f"hessian_matrix_{file_identifier}_{timestamp}.png"
        plt.savefig(hessian_plot_file, dpi=300, bbox_inches="tight")
        plt.close()
        
        # Plot the sparsity pattern
        plt.figure(figsize=(12, 10))
        sparse_cmap = ListedColormap(["white", "black"])
        sparse_norm = BoundaryNorm([-0.5, 0.5, 1.5], sparse_cmap.N)
        sparse_img = plt.imshow(sparsity_pattern.astype(int), cmap=sparse_cmap, norm=sparse_norm, aspect="equal")
        plt.xlabel("DOF Index", fontsize=LABEL_FONTSIZE)
        plt.ylabel("DOF Index", fontsize=LABEL_FONTSIZE)
        cbar = plt.colorbar(sparse_img, ticks=[0, 1])
        cbar.ax.set_yticklabels(["0", "1"])
        cbar.set_label("Sparsity Entry", fontsize=CBAR_LABEL_FONTSIZE)
        cbar.ax.tick_params(labelsize=TICK_FONTSIZE)
        _style_current_ticks()
        
        sparsity_plot_file = output_path / f"sparsity_pattern_{file_identifier}_{timestamp}.png"
        plt.savefig(sparsity_plot_file, dpi=300, bbox_inches="tight")
        plt.close()

        # Save the underlying arrays for design iteration.
        data_file = output_path / f"sparsity_pattern_data_{file_identifier}_{timestamp}.npz"
        np.savez_compressed(
            data_file,
            hessian=np.asarray(hessian),
            sparsity_pattern=np.asarray(sparsity_pattern, dtype=bool),
            label_fontsize=np.asarray([LABEL_FONTSIZE], dtype=int),
            legend_fontsize=np.asarray([LEGEND_FONTSIZE], dtype=int),
            cbar_label_fontsize=np.asarray([CBAR_LABEL_FONTSIZE], dtype=int),
        )
        
        _log_progress(f"  Hessian matrix: {hessian_plot_file}")
        _log_progress(f"  Sparsity pattern: {sparsity_plot_file}")
    
    def create_conformation_bar_plot(self, results, output_path, dataset_name, timestamp):
        """Create bar plots for conformations with timings/HVPs on the y-axis."""
        # Sort results by average runtime across all methods (leftmost bar = fastest overall).
        # Note: bar order is NOT execution order; naive times can differ slightly due to
        # GPU thermal state or variance when n_repeats is small.
        def get_avg_runtime(file_result):
            times = []
            for method in self.methods:
                method_result = file_result['methods'].get(method)
                if method_result is not None:
                    times.append(method_result['time'])
            return np.mean(times) if times else float('inf')
        
        sorted_results = sorted(results, key=get_avg_runtime)
        
        # Extract data for plotting - handle None values
        conformation_names = []
        method_times = {method: [] for method in self.methods}
        method_hvps = {method: [] for method in self.methods}
        
        for i, file_result in enumerate(sorted_results):
            conformation_names.append(f"Conformation {i+1}")
            for method in self.methods:
                method_result = file_result['methods'].get(method)
                if method_result is not None:
                    method_times[method].append(method_result['time'])
                    method_hvps[method].append(method_result['hvp_count'])
                else:
                    method_times[method].append(0)  # Use 0 for failed methods
                    method_hvps[method].append(0)
        
        # Create timing bar plot
        plt.figure(figsize=BAR_FIGSIZE)
        bar_width = 0.25
        x = np.arange(len(conformation_names))
        
        for i, method in enumerate(self.methods):
            # Only plot if we have non-zero data
            if any(t > 0 for t in method_times[method]):
                plt.bar(x + i * bar_width, method_times[method], bar_width,
                        label=method.capitalize(), color=COLORS[method])
        
        plt.ylabel("Time (seconds)", fontsize=LABEL_FONTSIZE)
        plt.xticks(x + bar_width, conformation_names, rotation=45, fontsize=TICK_FONTSIZE)
        plt.yticks(fontsize=TICK_FONTSIZE)
        plt.legend(fontsize=LEGEND_FONTSIZE)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        timing_bar_plot_file = output_path / f"{dataset_name}_conformation_timing_bars_{timestamp}.png"
        plt.savefig(timing_bar_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create HVP count bar plot
        plt.figure(figsize=BAR_FIGSIZE)
        
        for i, method in enumerate(self.methods):
            # Only plot if we have non-zero data
            if any(h > 0 for h in method_hvps[method]):
                plt.bar(x + i * bar_width, method_hvps[method], bar_width,
                        label=method.capitalize(), color=COLORS[method])
        
        plt.ylabel("Number of HVPs", fontsize=LABEL_FONTSIZE)
        plt.xticks(x + bar_width, conformation_names, rotation=45, fontsize=TICK_FONTSIZE)
        plt.yticks(fontsize=TICK_FONTSIZE)
        plt.legend(fontsize=LEGEND_FONTSIZE)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        hvp_bar_plot_file = output_path / f"{dataset_name}_conformation_hvp_bars_{timestamp}.png"
        plt.savefig(hvp_bar_plot_file, dpi=300, bbox_inches='tight')
        plt.close()

        # Save series used by both conformation plots.
        data_file = output_path / f"{dataset_name}_conformation_plot_data_{timestamp}.npz"
        np.savez_compressed(
            data_file,
            conformation_names=np.asarray(conformation_names, dtype=object),
            methods=np.asarray(self.methods, dtype=object),
            times=np.asarray([method_times[m] for m in self.methods], dtype=float),
            hvps=np.asarray([method_hvps[m] for m in self.methods], dtype=float),
            label_fontsize=np.asarray([LABEL_FONTSIZE], dtype=int),
            legend_fontsize=np.asarray([LEGEND_FONTSIZE], dtype=int),
        )
        
        _log_progress("Bar plots saved:")
        _log_progress(f"  Timing: {timing_bar_plot_file}")
        _log_progress(f"  HVPs: {hvp_bar_plot_file}")
        
        self.create_conformation_combined_plot(results, output_path, dataset_name, timestamp)
        self.create_conformation_combined_breakdown_plot(results, output_path, dataset_name, timestamp)

    def create_conformation_combined_plot(self, results, output_path, dataset_name, timestamp):
        """Create a paired-bar plot with HVP counts and timings on twin y-axes."""

        def acyclic_time_key(file_result):
            acyclic_result = file_result["methods"].get("acyclic")
            if acyclic_result is None:
                return float("inf")
            return acyclic_result["time"]

        sorted_results = sorted(results, key=acyclic_time_key)
        method_times = {method: [] for method in self.methods}
        method_hvps = {method: [] for method in self.methods}

        for file_result in sorted_results:
            for method in self.methods:
                method_result = file_result["methods"].get(method)
                if method_result is not None:
                    method_times[method].append(method_result["time"])
                    method_hvps[method].append(method_result["hvp_count"])
                else:
                    method_times[method].append(np.nan)
                    method_hvps[method].append(np.nan)

        n_conformations = len(sorted_results)
        x = np.arange(n_conformations, dtype=float)
        bar_width = 0.072
        inner_gap = 0.006
        pair_width = 2 * bar_width + inner_gap
        total_width = len(self.methods) * pair_width

        fig, ax_hvp = plt.subplots(figsize=(14, 8))
        ax_time = ax_hvp.twinx()

        for j in range(n_conformations):
            left = x[j] - total_width / 2
            for method_index, method in enumerate(self.methods):
                base = left + method_index * pair_width
                hvp_x = base + bar_width / 2
                time_x = base + bar_width + inner_gap + bar_width / 2
                hvp_value = method_hvps[method][j]
                time_value = method_times[method][j]

                if np.isfinite(hvp_value) and hvp_value >= 0:
                    ax_hvp.bar(
                        hvp_x,
                        hvp_value,
                        width=bar_width,
                        color=COLORS[method],
                        edgecolor="white",
                        linewidth=0.6,
                    )
                if np.isfinite(time_value) and time_value >= 0:
                    ax_time.bar(
                        time_x,
                        time_value,
                        width=bar_width,
                        color=COLORS_LIGHT[method],
                        edgecolor="white",
                        linewidth=0.6,
                    )

        ax_hvp.set_xticks(x)
        ax_hvp.set_xticklabels(
            [f"Conformation {j + 1}" for j in range(n_conformations)],
            rotation=45,
            fontsize=TICK_FONTSIZE,
        )
        ax_hvp.set_ylabel("HVP count", fontsize=LABEL_FONTSIZE, color="#2c3e50")
        ax_time.set_ylabel("Wall time (s)", fontsize=LABEL_FONTSIZE, color="#5d4e37")
        ax_hvp.tick_params(axis="y", labelcolor="#2c3e50", labelsize=TICK_FONTSIZE)
        ax_time.tick_params(axis="y", labelcolor="#5d4e37", labelsize=TICK_FONTSIZE)
        ax_hvp.grid(True, alpha=0.3, axis="y")
        ax_hvp.set_axisbelow(True)

        legend_handles = [
            Patch(facecolor=COLORS[method], edgecolor="white", linewidth=0.5, label=method.capitalize())
            for method in self.methods
        ]
        ax_hvp.legend(
            handles=legend_handles,
            fontsize=LEGEND_FONTSIZE - 1,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=len(self.methods),
            framealpha=0.95,
            borderaxespad=0.0,
        )

        plt.tight_layout(rect=(0, 0, 1, 0.9))

        combined_plot_file = output_path / f"{dataset_name}_conformation_hvp_timing_combined_{timestamp}.png"
        fig.savefig(combined_plot_file, dpi=300, bbox_inches="tight")
        plt.close(fig)

        _log_progress(f"  Combined timing/HVPs: {combined_plot_file}")

    def create_conformation_combined_breakdown_plot(self, results, output_path, dataset_name, timestamp):
        """HVP bars (saturated) + stacked timing (light); hatches distinguish timing phases."""

        def acyclic_time_key(file_result):
            acyclic_result = file_result["methods"].get("acyclic")
            if acyclic_result is None:
                return float("inf")
            return acyclic_result["time"]

        sorted_results = sorted(results, key=acyclic_time_key)
        method_hvps = {method: [] for method in self.methods}
        method_breakdowns = {
            method: {c: [] for c in BREAKDOWN_STACK_KEYS} for method in self.methods
        }

        has_any_breakdown = False
        for file_result in sorted_results:
            for method in self.methods:
                method_result = file_result["methods"].get(method)
                if method_result is not None:
                    method_hvps[method].append(method_result["hvp_count"])
                    breakdown = method_result.get("timing_breakdown") or {}
                    if any(float(breakdown.get(k, 0.0)) > 0 for k in BREAKDOWN_SPARSE_KEYS):
                        has_any_breakdown = True
                    stack = _timing_stack_segments_for_display(method, method_result)
                    for c in BREAKDOWN_STACK_KEYS:
                        method_breakdowns[method][c].append(stack[c])
                else:
                    method_hvps[method].append(np.nan)
                    for c in BREAKDOWN_STACK_KEYS:
                        method_breakdowns[method][c].append(np.nan)

        if not has_any_breakdown:
            _log_progress("  (skip breakdown combined: no plottable timing breakdown)")
            return

        n_conformations = len(sorted_results)
        x = np.arange(n_conformations, dtype=float)
        bar_width = 0.072
        inner_gap = 0.006
        pair_width = 2 * bar_width + inner_gap
        total_width = len(self.methods) * pair_width

        fig, ax_hvp = plt.subplots(figsize=(14, 8))
        ax_time = ax_hvp.twinx()

        for j in range(n_conformations):
            left = x[j] - total_width / 2
            for method_index, method in enumerate(self.methods):
                base = left + method_index * pair_width
                hvp_x = base + bar_width / 2
                time_x = base + bar_width + inner_gap + bar_width / 2
                hvp_value = method_hvps[method][j]

                if np.isfinite(hvp_value) and hvp_value >= 0:
                    ax_hvp.bar(
                        hvp_x,
                        hvp_value,
                        width=bar_width,
                        color=COLORS[method],
                        edgecolor="white",
                        linewidth=0.6,
                    )

                stack_bottom = 0.0
                for c in BREAKDOWN_STACK_KEYS:
                    value = method_breakdowns[method][c][j]
                    if np.isfinite(value) and value > 0:
                        ax_time.bar(
                            time_x,
                            value,
                            width=bar_width,
                            bottom=stack_bottom,
                            color=COLORS_LIGHT[method],
                            hatch=BREAKDOWN_STACK_HATCHES[c],
                            edgecolor="#2a2a2a",
                            linewidth=0.35,
                        )
                        stack_bottom += value

        ax_hvp.set_xticks(x)
        ax_hvp.set_xticklabels(
            [f"Conformation {k + 1}" for k in range(n_conformations)],
            rotation=45,
            fontsize=TICK_FONTSIZE,
        )
        ax_hvp.set_ylabel("HVP count", fontsize=LABEL_FONTSIZE, color="#2c3e50")
        ax_time.set_ylabel("Wall time (s)", fontsize=LABEL_FONTSIZE, color="#5d4e37")
        ax_hvp.tick_params(axis="y", labelcolor="#2c3e50", labelsize=TICK_FONTSIZE)
        ax_time.tick_params(axis="y", labelcolor="#5d4e37", labelsize=TICK_FONTSIZE)
        ax_hvp.grid(True, alpha=0.3, axis="y")
        ax_hvp.set_axisbelow(True)

        method_handles = [
            Patch(facecolor=COLORS[method], edgecolor="white", linewidth=0.5, label=method.capitalize())
            for method in self.methods
        ]
        neutral = "#d8d8d8"
        component_handles = [
            Patch(
                facecolor=neutral,
                edgecolor="#333333",
                linewidth=0.8,
                hatch=BREAKDOWN_STACK_HATCHES[c],
                label=BREAKDOWN_STACK_LABELS[c],
            )
            for c in BREAKDOWN_STACK_KEYS
        ]
        ax_hvp.legend(
            handles=method_handles + component_handles,
            fontsize=LEGEND_FONTSIZE - 1,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=6,
            framealpha=0.95,
            borderaxespad=0.0,
        )

        plt.tight_layout(rect=(0, 0, 1, 0.9))

        breakdown_plot_file = output_path / f"{dataset_name}_conformation_hvp_timing_combined_breakdown_{timestamp}.png"
        fig.savefig(breakdown_plot_file, dpi=300, bbox_inches="tight")
        plt.close(fig)

        breakdown_data_file = output_path / f"{dataset_name}_conformation_breakdown_plot_data_{timestamp}.npz"
        payload = {
            "methods": np.asarray(self.methods, dtype=object),
            "stack_components": np.asarray(BREAKDOWN_STACK_KEYS, dtype=object),
            "label_fontsize": np.asarray([LABEL_FONTSIZE], dtype=int),
            "legend_fontsize": np.asarray([LEGEND_FONTSIZE], dtype=int),
        }
        for method in self.methods:
            payload[f"hvp_{method}"] = np.asarray(method_hvps[method], dtype=float)
            for c in BREAKDOWN_STACK_KEYS:
                payload[f"{method}_{c}"] = np.asarray(method_breakdowns[method][c], dtype=float)
        np.savez_compressed(breakdown_data_file, **payload)

        _log_progress(f"  Combined timing/HVPs with breakdown: {breakdown_plot_file}")

    def save_results_and_plots(self, results, output_path, dataset_name):
        """Save benchmark results and create plots."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save raw results to JSON (remove hessian arrays for JSON compatibility)
        results_for_json = []
        for file_result in results:
            json_result = {
                'filename': file_result['filename'],
                'n_atoms': file_result['n_atoms'],
                'methods': {}
            }
            for method, method_result in file_result['methods'].items():
                if method_result is not None:
                    # Copy all fields except the hessian array
                    json_method_result = {k: v for k, v in method_result.items() if k != 'hessian'}
                    json_result['methods'][method] = json_method_result
                else:
                    json_result['methods'][method] = None
            results_for_json.append(json_result)
        
        results_file = output_path / f"{dataset_name}_benchmark_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'dataset': dataset_name,
                'timestamp': timestamp,
                'note': 'Results include GPU warm-up before each system. Hessian matrices saved separately as .npy files.',
                'results': results_for_json
            }, f, indent=2)
        
        _log_progress(f"Results saved to: {results_file}")
        
        # Create plots
        if self.test_over_conformations:
            self.create_conformation_bar_plot(results, output_path, dataset_name, timestamp)
        else:
            self.create_plots(results, output_path, dataset_name, timestamp)
    
    def create_plots(self, results, output_path, dataset_name, timestamp):
        """Create HVP count and timing plots."""
        # Extract data for plotting
        plot_data = {method: {'n_atoms': [], 'hvp_counts': [], 'times': [], 'time_stds': []} 
                    for method in self.methods}
        
        for file_result in results:
            n_atoms = file_result['n_atoms']
            
            for method in self.methods:
                method_result = file_result['methods'].get(method)
                if method_result is not None:
                    plot_data[method]['n_atoms'].append(n_atoms)
                    plot_data[method]['hvp_counts'].append(method_result['hvp_count'])
                    plot_data[method]['times'].append(method_result['time'])
                    plot_data[method]['time_stds'].append(method_result['time_std'])

        for method in self.methods:
            if len(plot_data[method]['n_atoms']) > 0:
                (
                    plot_data[method]['n_atoms'],
                    plot_data[method]['hvp_counts'],
                    plot_data[method]['times'],
                    plot_data[method]['time_stds'],
                ) = _sort_system_series(
                    plot_data[method]['n_atoms'],
                    plot_data[method]['hvp_counts'],
                    plot_data[method]['times'],
                    plot_data[method]['time_stds'],
                )
        
        # Create HVP count plot
        plt.figure(figsize=LINE_FIGSIZE)
        
        for method in self.methods:
            if len(plot_data[method]['n_atoms']) > 0:  # Only plot if we have data
                plt.plot(plot_data[method]['n_atoms'], 
                        plot_data[method]['hvp_counts'],
                        color=COLORS[method], marker=MARKERS[method], 
                        linewidth=LINEWIDTH, markersize=MARKERSIZE, label=method.capitalize())
        
        plt.xlabel('Number of Atoms', fontsize=LABEL_FONTSIZE)
        plt.ylabel('Number of HVPs', fontsize=LABEL_FONTSIZE)
        plt.legend(fontsize=LEGEND_FONTSIZE)
        plt.grid(True, alpha=0.3)
        _style_current_ticks()
        plt.tight_layout()
        # Use linear scale to better show differences between methods
        
        hvp_plot_file = output_path / f"{dataset_name}_hvp_comparison_{timestamp}.png"
        plt.savefig(hvp_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create timing plot with error bars
        plt.figure(figsize=LINE_FIGSIZE)
        
        for method in self.methods:
            if len(plot_data[method]['n_atoms']) > 0:  # Only plot if we have data
                plt.errorbar(plot_data[method]['n_atoms'], 
                           plot_data[method]['times'],
                           yerr=plot_data[method]['time_stds'],
                           color=COLORS[method], marker=MARKERS[method], 
                           linewidth=LINEWIDTH, markersize=MARKERSIZE, label=method.capitalize(),
                           capsize=5, capthick=2)
        
        plt.xlabel('Number of Atoms', fontsize=LABEL_FONTSIZE)
        plt.ylabel('Wall Clock Time (seconds)', fontsize=LABEL_FONTSIZE)
        plt.legend(fontsize=LEGEND_FONTSIZE)
        plt.grid(True, alpha=0.3)
        _style_current_ticks()
        plt.tight_layout()
        # Use linear scale to better show differences between methods
        
        timing_plot_file = output_path / f"{dataset_name}_timing_comparison_{timestamp}.png"
        plt.savefig(timing_plot_file, dpi=300, bbox_inches='tight')
        plt.close()

        # Save the series used by the HVP and timing plots.
        data_file = output_path / f"{dataset_name}_plot_data_{timestamp}.npz"
        payload = {
            "methods": np.asarray(self.methods, dtype=object),
            "label_fontsize": np.asarray([LABEL_FONTSIZE], dtype=int),
            "legend_fontsize": np.asarray([LEGEND_FONTSIZE], dtype=int),
        }
        for method in self.methods:
            payload[f"n_atoms_{method}"] = np.asarray(plot_data[method]["n_atoms"], dtype=int)
            payload[f"hvp_counts_{method}"] = np.asarray(plot_data[method]["hvp_counts"], dtype=float)
            payload[f"times_{method}"] = np.asarray(plot_data[method]["times"], dtype=float)
            payload[f"time_stds_{method}"] = np.asarray(plot_data[method]["time_stds"], dtype=float)
        np.savez_compressed(data_file, **payload)
        
        print(f"Plots saved:")
        print(f"  HVP comparison: {hvp_plot_file}")
        print(f"  Timing comparison: {timing_plot_file}")
        
        # Print summary statistics
        self.print_summary(results, dataset_name)
    
    def print_summary(self, results, dataset_name):
        """Print summary statistics."""
        print(f"\n=== BENCHMARK SUMMARY FOR {dataset_name.upper()} (WITH GPU WARM-UP) ===")
        
        for method in self.methods:
            method_times = []
            method_hvps = []
            method_atoms = []
            
            for file_result in results:
                method_result = file_result['methods'].get(method)
                if method_result is not None:
                    method_times.append(method_result['time'])
                    method_hvps.append(method_result['hvp_count'])
                    method_atoms.append(method_result['n_atoms'])
            
            if method_times:
                print(f"\n{method.upper()} METHOD:")
                print(f"  Systems processed: {len(method_times)}")
                print(f"  Atom range: {min(method_atoms)} - {max(method_atoms)}")
                print(f"  HVP range: {min(method_hvps)} - {max(method_hvps)}")
                print(f"  Time range: {min(method_times):.4f}s - {max(method_times):.4f}s")
                print(f"  Average time: {np.mean(method_times):.4f}s ± {np.std(method_times):.4f}s")
                print(f"  Coefficient of variation: {np.std(method_times)/np.mean(method_times)*100:.1f}%")


def create_ordering_comparison_plots(
    ordering_results: dict[str, list[dict]],
    output_path,
    dataset_name: str,
    model_tag: str,
):
    """Create dataset-level comparison plots between Natural and SmallestLast.

    Bar charts are used only for true conformation datasets (Abeta40, chignolin),
    matching ``replot_sparse_hessian_unified_colors._uses_conformation_bars``.
    System benchmarks (alkanes, water cluster, …) always use line plots vs
    ``n_atoms``, with points ordered by atom count (not directory iteration order).
    """
    required = [ordering for ordering in DEFAULT_ORDERINGS_TO_COMPARE if ordering in ordering_results]
    if len(required) < 2:
        return

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ordering_breakdown_file = None
    base_results = ordering_results["SMALLEST_LAST"]
    use_conformation_bars = _uses_conformation_plots(dataset_name)

    if use_conformation_bars:
        x_labels = [result["filename"] for result in base_results]
        x_values = np.arange(len(x_labels), dtype=float)
        x_axis_label = "Conformation"
    else:
        n_atoms_list = [int(result["n_atoms"]) for result in base_results]
        atom_order = np.argsort(np.asarray(n_atoms_list, dtype=float), kind="stable")
        x_labels = [base_results[i]["filename"] for i in atom_order]
        x_values = np.asarray(n_atoms_list, dtype=float)[atom_order]
        x_axis_label = "Number of Atoms"

    file_maps = {
        ordering: {result["filename"]: result for result in results}
        for ordering, results in ordering_results.items()
    }

    series_specs = [
        ("naive", None, COLORS["naive"], "-", MARKERS["naive"]),
        ("star", "SMALLEST_LAST", COLORS["star"], ORDERING_LINESTYLES["SMALLEST_LAST"], ORDERING_MARKERS["SMALLEST_LAST"]),
        ("star", "NATURAL", COLORS["star"], ORDERING_LINESTYLES["NATURAL"], ORDERING_MARKERS["NATURAL"]),
        ("acyclic", "SMALLEST_LAST", COLORS["acyclic"], ORDERING_LINESTYLES["SMALLEST_LAST"], ORDERING_MARKERS["SMALLEST_LAST"]),
        ("acyclic", "NATURAL", COLORS["acyclic"], ORDERING_LINESTYLES["NATURAL"], ORDERING_MARKERS["NATURAL"]),
    ]

    plot_payload = {
        "x_values": np.asarray(x_values),
        "x_labels": np.asarray(x_labels, dtype=object),
        "model_tag": np.asarray([model_tag], dtype=object),
        "dataset_name": np.asarray([dataset_name], dtype=object),
    }

    series_data = []
    for method, ordering, color, linestyle, marker in series_specs:
        source_ordering = "SMALLEST_LAST" if ordering is None else ordering
        series_label = _series_label(method, ordering)
        times = []
        time_stds = []
        hvps = []
        valid_x = []
        valid_labels = []

        for x_value, x_label in zip(x_values, x_labels):
            file_result = file_maps[source_ordering].get(x_label)
            method_result = None if file_result is None else file_result["methods"].get(method)
            if method_result is None:
                continue
            valid_x.append(x_value)
            valid_labels.append(x_label)
            times.append(method_result["time"])
            time_stds.append(method_result["time_std"])
            hvps.append(method_result["hvp_count"])

        if not valid_x:
            continue

        valid_x_array = np.asarray(valid_x, dtype=float)
        times_array = np.asarray(times, dtype=float)
        time_stds_array = np.asarray(time_stds, dtype=float)
        hvps_array = np.asarray(hvps, dtype=float)
        if not use_conformation_bars:
            order = np.argsort(valid_x_array, kind="stable")
            valid_x_array = valid_x_array[order]
            times_array = times_array[order]
            time_stds_array = time_stds_array[order]
            hvps_array = hvps_array[order]
            valid_labels = [valid_labels[i] for i in order]
        plot_payload[f"{series_label}_x"] = valid_x_array
        plot_payload[f"{series_label}_labels"] = np.asarray(valid_labels, dtype=object)
        plot_payload[f"{series_label}_times"] = times_array
        plot_payload[f"{series_label}_time_stds"] = time_stds_array
        plot_payload[f"{series_label}_hvps"] = hvps_array
        series_data.append(
            {
                "label": series_label,
                "x": valid_x_array,
                "labels": valid_labels,
                "times": times_array,
                "time_stds": time_stds_array,
                "hvps": hvps_array,
                "color": color,
                "linestyle": linestyle,
                "marker": marker,
            }
        )

    if use_conformation_bars:
        bar_width = 0.14
        base_positions = np.arange(len(x_labels), dtype=float)
        offsets = (np.arange(len(series_data)) - (len(series_data) - 1) / 2.0) * bar_width

        plt.figure(figsize=(14, 8))
        for offset, series in zip(offsets, series_data):
            value_map = dict(zip(series["labels"], series["times"]))
            error_map = dict(zip(series["labels"], series["time_stds"]))
            heights = np.asarray([value_map.get(label, np.nan) for label in x_labels], dtype=float)
            errors = np.asarray([error_map.get(label, np.nan) for label in x_labels], dtype=float)
            mask = np.isfinite(heights)
            if np.any(mask):
                plt.bar(
                    base_positions[mask] + offset,
                    heights[mask],
                    width=bar_width,
                    yerr=errors[mask],
                    color=series["color"],
                    alpha=0.9 if "Natural" in series["label"] else 0.65,
                    capsize=4,
                    label=series["label"],
                )

        plt.xlabel(x_axis_label, fontsize=LABEL_FONTSIZE)
        plt.ylabel("Wall Clock Time (seconds)", fontsize=LABEL_FONTSIZE)
        plt.xticks(base_positions, [f"C{i + 1}" for i in range(len(x_labels))], rotation=45, fontsize=TICK_FONTSIZE)
        plt.yticks(fontsize=TICK_FONTSIZE)
        plt.legend(fontsize=LEGEND_FONTSIZE - 1)
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()

        timing_plot_file = output_path / f"{dataset_name}_{model_tag}_ordering_timing_comparison_{timestamp}.png"
        plt.savefig(timing_plot_file, dpi=300, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(14, 8))
        for offset, series in zip(offsets, series_data):
            value_map = dict(zip(series["labels"], series["hvps"]))
            heights = np.asarray([value_map.get(label, np.nan) for label in x_labels], dtype=float)
            mask = np.isfinite(heights)
            if np.any(mask):
                plt.bar(
                    base_positions[mask] + offset,
                    heights[mask],
                    width=bar_width,
                    color=series["color"],
                    alpha=0.9 if "Natural" in series["label"] else 0.65,
                    label=series["label"],
                )

        plt.xlabel(x_axis_label, fontsize=LABEL_FONTSIZE)
        plt.ylabel("Number of HVPs", fontsize=LABEL_FONTSIZE)
        plt.xticks(base_positions, [f"C{i + 1}" for i in range(len(x_labels))], rotation=45, fontsize=TICK_FONTSIZE)
        plt.yticks(fontsize=TICK_FONTSIZE)
        plt.legend(fontsize=LEGEND_FONTSIZE - 1)
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()

        hvp_plot_file = output_path / f"{dataset_name}_{model_tag}_ordering_hvp_comparison_{timestamp}.png"
        plt.savefig(hvp_plot_file, dpi=300, bbox_inches="tight")
        plt.close()

        ob_available = False
        for method, ordering, *_ in series_specs:
            src = "SMALLEST_LAST" if ordering is None else ordering
            for label in x_labels:
                fr = file_maps[src].get(label)
                mr = None if fr is None else fr["methods"].get(method)
                if mr is None:
                    continue
                bd = mr.get("timing_breakdown") or {}
                if any(float(bd.get(k, 0.0)) > 0 for k in BREAKDOWN_SPARSE_KEYS):
                    ob_available = True
                    break
            if ob_available:
                break

        if ob_available:
            fig, ax_ob = plt.subplots(figsize=(14, 8))
            for offset, series, spec in zip(offsets, series_data, series_specs):
                method, ordering, *_ = spec
                source_ordering = "SMALLEST_LAST" if ordering is None else ordering
                seg_alpha = 0.9 if "Natural" in series["label"] else 0.65
                for j, label in enumerate(x_labels):
                    file_result = file_maps[source_ordering].get(label)
                    method_result = None if file_result is None else file_result["methods"].get(method)
                    stack = _timing_stack_segments_for_display(method, method_result)
                    if stack is None:
                        continue
                    xpos = base_positions[j] + offset
                    bottom = 0.0
                    for c in BREAKDOWN_STACK_KEYS:
                        v = stack[c]
                        if v > 0:
                            ax_ob.bar(
                                xpos,
                                v,
                                width=bar_width,
                                bottom=bottom,
                                color=COLORS_LIGHT[method],
                                alpha=seg_alpha,
                                hatch=BREAKDOWN_STACK_HATCHES[c],
                                edgecolor="#2a2a2a",
                                linewidth=0.35,
                            )
                            bottom += v

            ax_ob.set_xlabel(x_axis_label, fontsize=LABEL_FONTSIZE)
            ax_ob.set_ylabel("Wall Clock Time (seconds)", fontsize=LABEL_FONTSIZE)
            ax_ob.set_xticks(base_positions)
            ax_ob.set_xticklabels([f"C{i + 1}" for i in range(len(x_labels))], rotation=45, fontsize=TICK_FONTSIZE)
            ax_ob.tick_params(axis="y", labelsize=TICK_FONTSIZE)
            ax_ob.grid(True, alpha=0.3, axis="y")
            ax_ob.set_axisbelow(True)

            series_handles = [
                Patch(
                    facecolor=COLORS_LIGHT[spec[0]],
                    edgecolor="white",
                    linewidth=0.5,
                    alpha=0.9 if "Natural" in ser["label"] else 0.65,
                    label=ser["label"],
                )
                for ser, spec in zip(series_data, series_specs)
            ]
            neutral = "#d8d8d8"
            component_handles = [
                Patch(
                    facecolor=neutral,
                    edgecolor="#333333",
                    linewidth=0.8,
                    hatch=BREAKDOWN_STACK_HATCHES[c],
                    label=BREAKDOWN_STACK_LABELS[c],
                )
                for c in BREAKDOWN_STACK_KEYS
            ]
            ax_ob.legend(
                handles=series_handles + component_handles,
                fontsize=LEGEND_FONTSIZE - 2,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.14),
                ncol=4,
                framealpha=0.95,
                borderaxespad=0.0,
            )
            plt.tight_layout(rect=(0, 0, 1, 0.88))
            ordering_breakdown_file = (
                output_path / f"{dataset_name}_{model_tag}_ordering_timing_breakdown_{timestamp}.png"
            )
            fig.savefig(ordering_breakdown_file, dpi=300, bbox_inches="tight")
            plt.close(fig)
    else:
        plt.figure(figsize=(12, 7))
        for series in series_data:
            plt.errorbar(
                series["x"],
                series["times"],
                yerr=series["time_stds"],
                color=series["color"],
                linestyle=series["linestyle"],
                marker=series["marker"],
                linewidth=LINEWIDTH,
                markersize=MARKERSIZE,
                label=series["label"],
                capsize=5,
                capthick=2,
            )

        plt.xlabel(x_axis_label, fontsize=LABEL_FONTSIZE)
        plt.ylabel("Wall Clock Time (seconds)", fontsize=LABEL_FONTSIZE)
        plt.legend(fontsize=LEGEND_FONTSIZE - 1)
        plt.grid(True, alpha=0.3)
        _style_current_ticks()
        plt.tight_layout()

        timing_plot_file = output_path / f"{dataset_name}_{model_tag}_ordering_timing_comparison_{timestamp}.png"
        plt.savefig(timing_plot_file, dpi=300, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(12, 7))
        for series in series_data:
            plt.plot(
                series["x"],
                series["hvps"],
                color=series["color"],
                linestyle=series["linestyle"],
                marker=series["marker"],
                linewidth=LINEWIDTH,
                markersize=MARKERSIZE,
                label=series["label"],
            )

        plt.xlabel(x_axis_label, fontsize=LABEL_FONTSIZE)
        plt.ylabel("Number of HVPs", fontsize=LABEL_FONTSIZE)
        plt.legend(fontsize=LEGEND_FONTSIZE - 1)
        plt.grid(True, alpha=0.3)
        _style_current_ticks()
        plt.tight_layout()

        hvp_plot_file = output_path / f"{dataset_name}_{model_tag}_ordering_hvp_comparison_{timestamp}.png"
        plt.savefig(hvp_plot_file, dpi=300, bbox_inches="tight")
        plt.close()

    data_file = output_path / f"{dataset_name}_{model_tag}_ordering_comparison_data_{timestamp}.npz"
    np.savez_compressed(data_file, **plot_payload)
    _log_progress(f"Ordering comparison plots saved:")
    _log_progress(f"  Timing: {timing_plot_file}")
    _log_progress(f"  HVPs: {hvp_plot_file}")
    if ordering_breakdown_file is not None:
        _log_progress(f"  Timing (stacked breakdown): {ordering_breakdown_file}")


def main():
    """Main function to run benchmarks."""
    import argparse

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent
    results_root = script_dir / "sparse_hessian_results"

    parser = argparse.ArgumentParser(description="Sparse Hessian benchmark suite")
    dataset_group = parser.add_mutually_exclusive_group()
    dataset_group.add_argument(
        "--alkanes",
        action="store_true",
        help="Run only the Alkanes benchmark",
    )
    dataset_group.add_argument(
        "--water-cluster",
        action="store_true",
        help="Run only the Water Cluster benchmark",
    )
    dataset_group.add_argument(
        "--abeta40",
        dest="abeta40",
        action="store_true",
        help="Run only the Abeta40 conformation benchmark (data/Abeta40_PED00531)",
    )
    parser.add_argument(
        "--reverse-order",
        action="store_true",
        help="Process xyz files in reverse order (after sort); for ordering-bias checks",
    )
    parser.add_argument(
        "--stability-xyz",
        metavar="PATH",
        default=None,
        help="Run naive Hessian timing N times on one .xyz (after warmup); for outlier checks",
    )
    parser.add_argument(
        "--stability-repeats",
        type=int,
        default=3,
        help="Repeats for --stability-xyz (default: 3)",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=1,
        metavar="N",
        help="Timed runs per method per structure (default: 1)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Seed for random TensorNet benchmarks (0 and 1 interaction blocks; embedding adds 1 eff. hop) (default: 0)",
    )
    parser.add_argument(
        "--coloring-ordering",
        nargs="+",
        choices=COLORING_ORDERINGS,
        default=None,
        help="Optional restriction to one or more ColPack orderings. If omitted, runs SMALLEST_LAST and NATURAL.",
    )
    args = parser.parse_args()

    selected_orderings = args.coloring_ordering or DEFAULT_ORDERINGS_TO_COMPARE
    model_specs = _build_model_specs(args.random_seed)

    if args.stability_xyz:
        xyz = Path(args.stability_xyz)
        if not xyz.is_file():
            raise SystemExit(f"Not a file: {xyz}")
        if len(selected_orderings) > 1:
            _log_progress(
                f"Stability mode only uses one ordering; taking the first requested ordering: {selected_orderings[0]}"
            )
        print("Naive Hessian stability test (sync before/after calculate; hessian .cpu() + extra sync)")
        print(f"  {xyz}  x {args.stability_repeats} repeats\n")
        bench = HessianBenchmark(
            model_spec=model_specs[0],
            test_over_conformations=True,
            coloring_ordering=selected_orderings[0],
        )
        atoms = bench.get_atoms_from_file(xyz)
        bench.warmup_gpu(atoms)
        bench._configure_method("naive")
        times = []
        for i in range(args.stability_repeats):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            bench.calc.calculate(atoms, properties=["hessian"])
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            dt = t1 - t0
            times.append(dt)
            H = bench.calc.results["hessian"]
            print(f"  run {i + 1:2d}: {dt:.4f} s  |H|_max={np.abs(H).max():.6e}")
        arr = np.asarray(times)
        print(
            f"\n  min={arr.min():.4f}s  max={arr.max():.4f}s  "
            f"mean={arr.mean():.4f}s  std={arr.std():.4f}s  "
            f"range%={(arr.max() - arr.min()) / arr.mean() * 100:.1f}% of mean"
        )
        return

    if args.alkanes:
        runs = [
            {
                "name": "Alkanes",
                "data_dir": str(repo_root / "data" / "Alkanes"),
                "output_dir": str(results_root / "alkanes_experiments"),
                "test_over_conformations": False,
                "reverse_file_order": False,
                "collect_breakdown": False,
            },
        ]
    elif args.water_cluster:
        runs = [
            {
                "name": "Water Cluster",
                "data_dir": str(repo_root / "data" / "Water Cluster"),
                "output_dir": str(results_root / "water_cluster_experiments"),
                "test_over_conformations": False,
                "reverse_file_order": args.reverse_order,
                "collect_breakdown": False,
            },
        ]
    elif args.abeta40:
        runs = [
            {
                "name": "Abeta40",
                "data_dir": str(repo_root / "data" / "Abeta40_PED00531"),
                "output_dir": str(results_root / "abeta40_experiments"),
                "test_over_conformations": _uses_conformation_plots("Abeta40"),
                "reverse_file_order": args.reverse_order,
                "collect_breakdown": True,
            },
        ]
    else:
        runs = [
            {
                "name": "Alkanes",
                "data_dir": str(repo_root / "data" / "Alkanes"),
                "output_dir": str(results_root / "alkanes_experiments"),
                "test_over_conformations": False,
                "reverse_file_order": False,
                "collect_breakdown": False,
            },
            {
                "name": "Water Cluster",
                "data_dir": str(repo_root / "data" / "Water Cluster"),
                "output_dir": str(results_root / "water_cluster_experiments"),
                "test_over_conformations": False,
                "reverse_file_order": args.reverse_order,
                "collect_breakdown": False,
            },
            {
                "name": "Abeta40",
                "data_dir": str(repo_root / "data" / "Abeta40_PED00531"),
                "output_dir": str(results_root / "abeta40_experiments"),
                "test_over_conformations": _uses_conformation_plots("Abeta40"),
                "reverse_file_order": args.reverse_order,
                "collect_breakdown": True,
            },
        ]

    _log_progress("Starting Hessian Benchmark Suite (with GPU warm-up)")
    _log_progress("=" * 60)
    _log_progress("Note: Each system will be warmed up before benchmarking")
    _log_progress(f"Random TensorNet seed: {args.random_seed}")
    _log_progress(f"Orderings to compare: {', '.join(selected_orderings)}")

    for run in runs:
        _log_progress("-" * 60)
        _log_progress(f"Dataset: {run['name']}")
        _log_progress(f"Data directory: {run['data_dir']}")
        _log_progress(f"Output directory: {run['output_dir']}")
        if run.get("reverse_file_order"):
            _log_progress("File order: REVERSE")
        _log_progress(f"Timed repeats per method: {args.n_repeats}")
        if run.get("collect_breakdown"):
            _log_progress("Sparse timing breakdown: ENABLED")
        dataset_ordering_results = {}
        dataset_output_dir = Path(run["output_dir"])

        for model_spec in model_specs:
            model_tag = model_spec["tag"]
            dataset_ordering_results[model_tag] = {}
            _log_progress(f"Model variant: {model_spec['display_name']}")

            for ordering in selected_orderings:
                ordering_suffix = _ordering_label(ordering)
                run_output_dir = dataset_output_dir / f"{model_tag}_{ordering_suffix}"
                _log_progress(f"Coloring ordering: {ordering}")
                _log_progress(f"Run output directory: {run_output_dir}")
                benchmark = HessianBenchmark(
                    model_spec=model_spec,
                    test_over_conformations=run["test_over_conformations"],
                    coloring_ordering=ordering,
                    collect_breakdown=run.get("collect_breakdown", False),
                )

                try:
                    results = benchmark.benchmark_directory(
                        data_dir=run["data_dir"],
                        output_dir=run_output_dir,
                        n_repeats=args.n_repeats,
                        reverse_file_order=run.get("reverse_file_order", False),
                    )
                    dataset_ordering_results[model_tag][ordering] = results
                    _log_progress(
                        f"{run['name']} benchmark completed successfully for {model_tag} with {ordering}!"
                    )
                except Exception as e:
                    _log_progress(
                        f"{run['name']} benchmark failed for {model_tag} with {ordering}: {e}"
                    )
                    raise

        for model_tag, ordering_results in dataset_ordering_results.items():
            create_ordering_comparison_plots(
                ordering_results=ordering_results,
                output_path=dataset_output_dir,
                dataset_name=run["name"],
                model_tag=model_tag,
            )


if __name__ == "__main__":
    main()
