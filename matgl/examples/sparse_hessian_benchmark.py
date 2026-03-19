import numpy as np
import torch
torch.set_default_device("cuda")

import os
import json
import glob
from pathlib import Path
import numpy as np
import matgl
from ase.io import read
from matgl.ext.ase import PESCalculator
import matplotlib.pyplot as plt
import time
from datetime import datetime

# Plot styling (paper readability)
# Historically, axis labels used fontsize=14 in this script.
LABEL_FONTSIZE = int(round(14 * 1.3))  # ~30% larger
LEGEND_FONTSIZE = int(round(12 * 1.4))  # ~40% larger (was 12)
CBAR_LABEL_FONTSIZE = LABEL_FONTSIZE  # keep consistent for paper readability


class HessianBenchmark:
    """Comprehensive benchmarking class for Hessian calculation methods with GPU warm-up."""
    
    def __init__(self, model_path="../pretrained_models/TensorNet-ANI-1x-Subset-PES", test_over_conformations=False):
        """Initialize the benchmark with a model and optional test_over_conformations flag."""
        self.test_over_conformations = test_over_conformations
        self.pot = matgl.load_model(model_path)
        self.pot.calc_hessian = True
        self.pot.return_seed_matrix = True
        self.calc = PESCalculator(self.pot)
        
        # Methods to benchmark
        self.methods = ["naive", "star", "acyclic"]
        
    def get_atoms_from_file(self, file_path):
        """Load atoms from xyz file and return atoms object."""
        return read(file_path)
    
    def warmup_gpu(self, atoms):
        """Warm up GPU kernels by running each method once before benchmarking."""
        print("  Warming up GPU kernels...")
        
        for method in self.methods:
            self.pot.hessian_algorithm = method
            try:
                # Run once to warm up kernels
                self.calc.calculate(atoms, properties=['hessian'])
                torch.cuda.synchronize()  # Ensure GPU operations complete
            except Exception as e:
                print(f"    Warning: Warm-up failed for {method}: {e}")
        
        print("  GPU warm-up completed")
    
    def benchmark_method(self, atoms, method, n_repeats=1):
        """Benchmark a single method and return timing and HVP count."""
        self.pot.hessian_algorithm = method
        
        # Accurate timing with GPU synchronization
        times = []
        
        for _ in range(n_repeats):
            torch.cuda.synchronize()  # Ensure previous operations complete
            t0 = time.perf_counter()
            self.calc.calculate(atoms, properties=['hessian'])
            torch.cuda.synchronize()  # Ensure calculation completes
            t1 = time.perf_counter()
            start_time, end_time = t0, t1  # perf_counter monotonic, better than time.time()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        # Get the results to extract seed matrix info
        results = self.calc.results
        hessian = np.array(results['hessian'])
        
        # Get HVP count from seed matrix
        hvp_count = None
        if 'seed_matrix' in results and results['seed_matrix'] is not None:
            seed_matrix = np.array(results['seed_matrix'])
            hvp_count = seed_matrix.shape[-1]  # Last dimension gives number of HVPs
        else:
            # For naive method, estimate based on problem size
            n_atoms = len(atoms)
            n_dof = 3 * n_atoms
            hvp_count = n_dof  # Naive method uses one HVP per degree of freedom
        
        return {
            'method': method,
            'time': avg_time,
            'time_std': std_time,
            'times': times,
            'hvp_count': hvp_count,
            'hessian_shape': hessian.shape,
            'hessian': hessian,  # Include the actual hessian
            'n_atoms': len(atoms)
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
        output_path.mkdir(exist_ok=True)
        
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
            print("Processing order: REVERSED (last filename first)")

        print(f"Found {len(sorted_files)} xyz files in {data_dir}")
        print("Benchmark processing order:")
        for file_path in sorted_files:
            n_atoms = next(n for p, n in file_atom_counts if p == file_path)
            print(f"  {file_path.name}: {n_atoms} atoms")
        
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
            print(f"\nProcessing {file_path.name} ({i+1}/{len(sorted_files)})...")
            
            try:
                atoms = self.get_atoms_from_file(file_path)
                n_atoms = len(atoms)
                
                # Warm up GPU kernels before benchmarking this system
                self.warmup_gpu(atoms)
                
                file_results = {
                    'filename': file_path.name,
                    'n_atoms': n_atoms,
                    'methods': {}
                }
                
                # Benchmark each method
                for method in self.methods:
                    print(f"  Benchmarking {method} method ({n_repeats} runs)...")
                    try:
                        result = self.benchmark_method(atoms, method, n_repeats)
                        file_results['methods'][method] = result
                        
                        # Show individual run times for transparency
                        times_str = ", ".join([f"{t:.3f}s" for t in result['times']])
                        print(f"    Times: [{times_str}]")
                        print(f"    Average: {result['time']:.4f}s ± {result['time_std']:.4f}s")
                        print(f"    HVPs: {result['hvp_count']}")
                        
                    except Exception as e:
                        print(f"    Error with {method}: {e}")
                        file_results['methods'][method] = None
                
                # Update fastest/slowest by acyclic HVP count (use acyclic Hessian for both)
                acyclic_result = file_results['methods'].get('acyclic')
                if acyclic_result is not None:
                    hvp = acyclic_result['hvp_count']
                    if hvp > slowest_hvp:
                        slowest_hvp = hvp
                        slowest_hessian = acyclic_result['hessian']
                        slowest_file = file_path.name
                    if hvp < fastest_hvp:
                        fastest_hvp = hvp
                        fastest_hessian = acyclic_result['hessian']
                        fastest_file = file_path.name
                
                all_results.append(file_results)
                
            except Exception as e:
                print(f"  Error processing {file_path.name}: {e}")
                continue
        
        # Create plots of Hessians and sparsity patterns (by acyclic HVP: slowest=most HVPs, fastest=least HVPs)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if slowest_hessian is not None:
            self.plot_hessian_matrix(slowest_hessian, output_path, f"slowest_{slowest_file}", timestamp)
            print(f"Slowest (acyclic HVPs={slowest_hvp}) Hessian plots saved for {slowest_file}")
        if fastest_hessian is not None:
            self.plot_hessian_matrix(fastest_hessian, output_path, f"fastest_{fastest_file}", timestamp)
            print(f"Fastest (acyclic HVPs={fastest_hvp}) Hessian plots saved for {fastest_file}")
        
        # Save results and create plots
        self.save_results_and_plots(all_results, output_path, data_path.name)
        
        return all_results
    
    def plot_hessian_matrix(self, hessian, output_path, file_identifier, timestamp):
        """Create plots of the Hessian matrix and its sparsity pattern."""
        import matplotlib.pyplot as plt
        from matplotlib.colors import BoundaryNorm, ListedColormap
        
        # Create sparsity pattern (boolean mask)
        sparsity_pattern = hessian != 0
        
        # Plot the full Hessian matrix
        plt.figure(figsize=(12, 10))
        plt.imshow(hessian, cmap='RdBu_r', aspect='equal')
        cbar = plt.colorbar()
        cbar.set_label("Hessian Value", fontsize=CBAR_LABEL_FONTSIZE)
        plt.xlabel('DOF Index', fontsize=LABEL_FONTSIZE)
        plt.ylabel('DOF Index', fontsize=LABEL_FONTSIZE)
        
        hessian_plot_file = output_path / f"hessian_matrix_{file_identifier}_{timestamp}.png"
        plt.savefig(hessian_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot the sparsity pattern
        plt.figure(figsize=(12, 10))
        sparse_cmap = ListedColormap(["white", "black"])
        sparse_norm = BoundaryNorm([-0.5, 0.5, 1.5], sparse_cmap.N)
        sparse_img = plt.imshow(sparsity_pattern.astype(int), cmap=sparse_cmap, norm=sparse_norm, aspect='equal')
        plt.xlabel('DOF Index', fontsize=LABEL_FONTSIZE)
        plt.ylabel('DOF Index', fontsize=LABEL_FONTSIZE)
        cbar = plt.colorbar(sparse_img, ticks=[0, 1])
        cbar.ax.set_yticklabels(["0", "1"])
        cbar.set_label("Sparsity Entry", fontsize=CBAR_LABEL_FONTSIZE)
        
        sparsity_plot_file = output_path / f"sparsity_pattern_{file_identifier}_{timestamp}.png"
        plt.savefig(sparsity_plot_file, dpi=300, bbox_inches='tight')
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
        
        print(f"  Hessian matrix: {hessian_plot_file}")
        print(f"  Sparsity pattern: {sparsity_plot_file}")
    
    def create_conformation_bar_plot(self, results, output_path, dataset_name, timestamp):
        """Create bar plots for conformations with timings/HVPs on the y-axis."""
        colors = {'naive': 'red', 'star': 'blue', 'acyclic': 'green'}
        
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
        plt.figure(figsize=(14, 8))
        bar_width = 0.25
        x = np.arange(len(conformation_names))
        
        for i, method in enumerate(self.methods):
            # Only plot if we have non-zero data
            if any(t > 0 for t in method_times[method]):
                plt.bar(x + i * bar_width, method_times[method], bar_width,
                        label=method.capitalize(), color=colors[method])
        
        plt.ylabel('Time (seconds)', fontsize=LABEL_FONTSIZE)
        plt.xticks(x + bar_width, conformation_names, rotation=45, fontsize=14)
        plt.legend(fontsize=LEGEND_FONTSIZE)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        timing_bar_plot_file = output_path / f"{dataset_name}_conformation_timing_bars_{timestamp}.png"
        plt.savefig(timing_bar_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create HVP count bar plot
        plt.figure(figsize=(14, 8))
        
        for i, method in enumerate(self.methods):
            # Only plot if we have non-zero data
            if any(h > 0 for h in method_hvps[method]):
                plt.bar(x + i * bar_width, method_hvps[method], bar_width,
                        label=method.capitalize(), color=colors[method])
        
        plt.ylabel('Number of HVPs', fontsize=LABEL_FONTSIZE)
        plt.xticks(x + bar_width, conformation_names, rotation=45, fontsize=14)
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
        
        print(f"Bar plots saved:")
        print(f"  Timing: {timing_bar_plot_file}")
        print(f"  HVPs: {hvp_bar_plot_file}")

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
        
        print(f"\nResults saved to: {results_file}")
        
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
        
        # Create HVP count plot
        plt.figure(figsize=(12, 8))
        
        colors = {'naive': 'red', 'star': 'blue', 'acyclic': 'green'}
        markers = {'naive': 'o', 'star': 's', 'acyclic': '^'}
        
        for method in self.methods:
            if plot_data[method]['n_atoms']:  # Only plot if we have data
                plt.plot(plot_data[method]['n_atoms'], 
                        plot_data[method]['hvp_counts'],
                        color=colors[method], marker=markers[method], 
                        linewidth=2, markersize=8, label=method.capitalize())
        
        plt.xlabel('Number of Atoms', fontsize=LABEL_FONTSIZE)
        plt.ylabel('Number of HVPs', fontsize=LABEL_FONTSIZE)
        plt.legend(fontsize=LEGEND_FONTSIZE)
        plt.grid(True, alpha=0.3)
        # Use linear scale to better show differences between methods
        
        hvp_plot_file = output_path / f"{dataset_name}_hvp_comparison_{timestamp}.png"
        plt.savefig(hvp_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create timing plot with error bars
        plt.figure(figsize=(12, 8))
        
        for method in self.methods:
            if plot_data[method]['n_atoms']:  # Only plot if we have data
                plt.errorbar(plot_data[method]['n_atoms'], 
                           plot_data[method]['times'],
                           yerr=plot_data[method]['time_stds'],
                           color=colors[method], marker=markers[method], 
                           linewidth=2, markersize=8, label=method.capitalize(),
                           capsize=5, capthick=2)
        
        plt.xlabel('Number of Atoms', fontsize=LABEL_FONTSIZE)
        plt.ylabel('Wall Clock Time (seconds)', fontsize=LABEL_FONTSIZE)
        plt.legend(fontsize=LEGEND_FONTSIZE)
        plt.grid(True, alpha=0.3)
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


def main():
    """Main function to run benchmarks."""
    import argparse

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent
    results_root = script_dir / "sparse_hessian_results"

    parser = argparse.ArgumentParser(description="Sparse Hessian benchmark suite")
    parser.add_argument(
        "--chignolin",
        action="store_true",
        help="Run Chignolin conformation benchmark only (data/Chignolin)",
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
    args = parser.parse_args()

    if args.stability_xyz:
        xyz = Path(args.stability_xyz)
        if not xyz.is_file():
            raise SystemExit(f"Not a file: {xyz}")
        print("Naive Hessian stability test (sync before/after calculate; hessian .cpu() + extra sync in PESCalculator)")
        print(f"  {xyz}  x {args.stability_repeats} repeats\n")
        bench = HessianBenchmark(test_over_conformations=True)
        atoms = bench.get_atoms_from_file(xyz)
        bench.warmup_gpu(atoms)
        bench.pot.hessian_algorithm = "naive"
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

    if args.chignolin:
        runs = [
            {
                "name": "Chignolin",
                "data_dir": str(repo_root / "data" / "Chignolin"),
                "output_dir": str(results_root / "chignolin_experiments"),
                "test_over_conformations": True,
                "reverse_file_order": args.reverse_order,
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
            },
        ]

    print("Starting Hessian Benchmark Suite (with GPU warm-up)")
    print("=" * 60)
    print("Note: Each system will be warmed up before benchmarking")

    for run in runs:
        print("\n" + "-" * 60)
        print(f"Dataset: {run['name']}")
        print(f"Data directory: {run['data_dir']}")
        print(f"Output directory: {run['output_dir']}")
        if run.get("reverse_file_order"):
            print("File order: REVERSE")
        print(f"Timed repeats per method: {args.n_repeats}")
        benchmark = HessianBenchmark(test_over_conformations=run["test_over_conformations"])

        try:
            benchmark.benchmark_directory(
                data_dir=run["data_dir"],
                output_dir=run["output_dir"],
                n_repeats=args.n_repeats,
                reverse_file_order=run.get("reverse_file_order", False),
            )
            print(f"\n{run['name']} benchmark completed successfully!")
        except Exception as e:
            print(f"{run['name']} benchmark failed: {e}")
            raise


if __name__ == "__main__":
    main()
