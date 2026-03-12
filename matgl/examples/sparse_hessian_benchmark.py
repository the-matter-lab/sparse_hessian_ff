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
            start_time = time.time()
            self.calc.calculate(atoms, properties=['hessian'])
            torch.cuda.synchronize()  # Ensure calculation completes
            end_time = time.time()
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
    
    def benchmark_directory(self, data_dir, output_dir="sparse_hessian_results", n_repeats=1):
        """Benchmark all xyz files in a directory, with optional test_over_conformations."""
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
            # For conformations, just use the files as they are (all same atom count)
            sorted_files = [item[0] for item in file_atom_counts]
        else:
            # Sort by atom count
            file_atom_counts.sort(key=lambda x: x[1])
            sorted_files = [item[0] for item in file_atom_counts]
        
        print(f"Found {len(sorted_files)} xyz files in {data_dir}")
        print("Files sorted by atom count:")
        for file_path, n_atoms in file_atom_counts:
            print(f"  {file_path.name}: {n_atoms} atoms")
        
        # Store results
        all_results = []
        slowest_hessian = None
        fastest_hessian = None
        slowest_time = -1
        fastest_time = float('inf')
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
                        
                        # Check for slowest and fastest
                        if result['time'] > slowest_time:
                            slowest_time = result['time']
                            slowest_hessian = result['hessian']
                            slowest_file = file_path.name
                        if result['time'] < fastest_time:
                            fastest_time = result['time']
                            fastest_hessian = result['hessian']
                            fastest_file = file_path.name
                        
                        # Show individual run times for transparency
                        times_str = ", ".join([f"{t:.3f}s" for t in result['times']])
                        print(f"    Times: [{times_str}]")
                        print(f"    Average: {result['time']:.4f}s ± {result['time_std']:.4f}s")
                        print(f"    HVPs: {result['hvp_count']}")
                        
                    except Exception as e:
                        print(f"    Error with {method}: {e}")
                        file_results['methods'][method] = None
                
                all_results.append(file_results)
                
            except Exception as e:
                print(f"  Error processing {file_path.name}: {e}")
                continue
        
        # Create plots of Hessians and sparsity patterns
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if slowest_hessian is not None:
            self.plot_hessian_matrix(slowest_hessian, output_path, f"slowest_{slowest_file}", timestamp)
            print(f"Slowest Hessian plots saved for {slowest_file}")
        if fastest_hessian is not None:
            self.plot_hessian_matrix(fastest_hessian, output_path, f"fastest_{fastest_file}", timestamp)
            print(f"Fastest Hessian plots saved for {fastest_file}")
        
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
        plt.colorbar(label='Hessian Value')
        plt.xlabel('DOF Index')
        plt.ylabel('DOF Index')
        
        hessian_plot_file = output_path / f"hessian_matrix_{file_identifier}_{timestamp}.png"
        plt.savefig(hessian_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot the sparsity pattern
        plt.figure(figsize=(12, 10))
        sparse_cmap = ListedColormap(["white", "black"])
        sparse_norm = BoundaryNorm([-0.5, 0.5, 1.5], sparse_cmap.N)
        sparse_img = plt.imshow(sparsity_pattern.astype(int), cmap=sparse_cmap, norm=sparse_norm, aspect='equal')
        plt.xlabel('DOF Index')
        plt.ylabel('DOF Index')
        cbar = plt.colorbar(sparse_img, ticks=[0, 1])
        cbar.ax.set_yticklabels(["0", "1"])
        cbar.set_label("Sparsity Entry")
        
        sparsity_plot_file = output_path / f"sparsity_pattern_{file_identifier}_{timestamp}.png"
        plt.savefig(sparsity_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Hessian matrix: {hessian_plot_file}")
        print(f"  Sparsity pattern: {sparsity_plot_file}")
    
    def create_conformation_bar_plot(self, results, output_path, dataset_name, timestamp):
        """Create bar plots for conformations with timings/HVPs on the y-axis."""
        colors = {'naive': 'red', 'star': 'blue', 'acyclic': 'green'}
        
        # Sort results by average runtime across all methods
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
        
        plt.ylabel('Time (seconds)', fontsize=14)
        plt.xticks(x + bar_width, conformation_names, rotation=45, fontsize=14)
        plt.legend(fontsize=12)
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
        
        plt.ylabel('Number of HVPs', fontsize=14)
        plt.xticks(x + bar_width, conformation_names, rotation=45, fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        hvp_bar_plot_file = output_path / f"{dataset_name}_conformation_hvp_bars_{timestamp}.png"
        plt.savefig(hvp_bar_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
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
        
        plt.xlabel('Number of Atoms', fontsize=14)
        plt.ylabel('Number of HVPs', fontsize=14)
        plt.legend(fontsize=12)
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
        
        plt.xlabel('Number of Atoms', fontsize=14)
        plt.ylabel('Wall Clock Time (seconds)', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        # Use linear scale to better show differences between methods
        
        timing_plot_file = output_path / f"{dataset_name}_timing_comparison_{timestamp}.png"
        plt.savefig(timing_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
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
    # Configuration
    DATA_DIR = "/h/28/lucahiede/Projects/sparse_hessian_ff/data/Chignolin"
    OUTPUT_DIR = str(Path(__file__).resolve().parent / "sparse_hessian_results")
    
    print("Starting Hessian Benchmark Suite (with GPU warm-up)")
    print("=" * 60)
    print(f"Data directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("Note: Each system will be warmed up before benchmarking")
    
    # Initialize benchmark with test_over_conformations=True for Chignolin
    benchmark = HessianBenchmark(test_over_conformations=True)
    
    # Run benchmarks
    try:
        results = benchmark.benchmark_directory(
            data_dir=DATA_DIR,
            output_dir=OUTPUT_DIR,
            n_repeats=1  # Number of timing repeats for accuracy (increase for better statistics)
        )
        
        print("\nBenchmark completed successfully!")
        
    except Exception as e:
        print(f"Benchmark failed: {e}")
        raise


if __name__ == "__main__":
    main()
