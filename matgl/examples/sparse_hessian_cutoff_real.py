import os
# To supress warning from the use_deterministic_algorithms=True
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import numpy as np
import torch
torch.set_default_device("cuda")

# Enable deterministic algorithms for PyTorch
torch.use_deterministic_algorithms(True, warn_only=True)

import matgl
from ase.io import read
from matgl.ext.ase import PESCalculator
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime

# Relative-error denominator regularization used for full-spectrum comparisons.
FULL_SPECTRUM_REG_EPS = 1e-5

# Load model and structure
pot = matgl.load_model("../pretrained_models/TensorNet-ANI-1x-Subset-PES")
pot.calc_hessian = True
pot.return_seed_matrix = True
pot.hessian_delta = 5e-3
atoms = read('/h/28/lucahiede/Projects/sparse_hessian_ff/data/Water Cluster/76.xyz')

calc = PESCalculator(pot)


def get_vibrational_basis_from_eckart(atoms_obj, tol=1e-10):
    """Build a Cartesian basis orthogonal to rigid-body Eckart modes."""
    masses = np.asarray(atoms_obj.get_masses(), dtype=float)
    coords = np.asarray(atoms_obj.get_positions(), dtype=float)
    n_atoms = masses.shape[0]
    ndof = 3 * n_atoms

    center_of_mass = np.average(coords, axis=0, weights=masses)
    coords_rel = coords - center_of_mass
    sqrt_m = np.sqrt(masses)

    rigid_modes = []

    # Mass-weighted translations.
    for axis in range(3):
        mode = np.zeros(ndof, dtype=float)
        for atom_idx in range(n_atoms):
            mode[3 * atom_idx + axis] = sqrt_m[atom_idx]
        rigid_modes.append(mode)

    # Mass-weighted rotations.
    unit_axes = np.eye(3)
    for axis in range(3):
        mode = np.zeros(ndof, dtype=float)
        for atom_idx in range(n_atoms):
            rot_vec = np.cross(unit_axes[axis], coords_rel[atom_idx])
            mode[3 * atom_idx : 3 * atom_idx + 3] = sqrt_m[atom_idx] * rot_vec
        rigid_modes.append(mode)

    rigid_mat = np.column_stack(rigid_modes)  # (3N, 6)
    q_rigid, r_rigid = np.linalg.qr(rigid_mat, mode="reduced")
    rigid_rank = int(np.sum(np.abs(np.diag(r_rigid)) > tol))
    q_rigid = q_rigid[:, :rigid_rank]

    q_full, _ = np.linalg.qr(q_rigid, mode="complete")
    vib_basis = q_full[:, rigid_rank:]
    return vib_basis


def mass_weight_hessian(hessian, masses):
    """Convert Cartesian Hessian to mass-weighted form M^{-1/2} H M^{-1/2}."""
    inv_sqrt_m = 1.0 / np.sqrt(np.asarray(masses, dtype=float))
    inv_sqrt_m_xyz = np.repeat(inv_sqrt_m, 3)
    return (inv_sqrt_m_xyz[:, None] * hessian) * inv_sqrt_m_xyz[None, :]


def projected_vibrational_eigvals(hessian, masses, vib_basis):
    """Eigenvalues after projecting out rigid-body modes via Eckart basis."""
    hessian_mw = mass_weight_hessian(hessian, masses)
    hessian_vib = vib_basis.T @ hessian_mw @ vib_basis
    # Numerical guard: keep matrix Hermitian for eigvalsh.
    hessian_vib = 0.5 * (hessian_vib + hessian_vib.T)
    return np.linalg.eigvalsh(hessian_vib)

# Compute naive hessian (reference)
print("Computing naive hessian...")
pot.hessian_algorithm = "naive"
pot.hessian_sparsity_cutoff = None  # Use default graph-based
calc.calculate(atoms, properties=['hessian'])
hessian_naive = np.array(calc.results['hessian'])
full_hessian_rows = hessian_naive.shape[0]
print(f"Full Hessian dimension: {full_hessian_rows} x {full_hessian_rows}")
masses = np.asarray(atoms.get_masses(), dtype=float)
vib_basis = get_vibrational_basis_from_eckart(atoms)

# Cutoffs to test
cutoffs = np.arange(5, 13, 1)  # 5 to 12 Angstroms
algorithms = ["distance_two", "star", "acyclic"]

# Storage for results
errors = {alg: [] for alg in algorithms}
n_hvps = {alg: [] for alg in algorithms}
hessians = {alg: [] for alg in algorithms}  # Store hessians for eigenvalue comparison
first_n_eigvals = 10  # Parameter for eigenvalue comparison

# Loop through cutoffs
for cutoff in cutoffs:
    print(f"\nTesting cutoff: {cutoff} Angstroms")
    
    for alg in algorithms:
        print(f"  Computing {alg}...")
        pot.hessian_algorithm = alg
        pot.hessian_sparsity_cutoff = cutoff
        
        try:
            calc.calculate(atoms, properties=['hessian'])
            hessian_sparse = np.array(calc.results['hessian'])
            seed_matrix = np.array(calc.results['seed_matrix'])

            # Get number of HVPs (second dimension of seed matrix)
            n_hvp = seed_matrix.shape[1] if len(seed_matrix.shape) == 2 else seed_matrix.shape[0]
            
            # Filter out runs that are effectively full-Hessian calculations.
            # In these cases, eigenvalue errors can collapse to exact zero.
            if n_hvp >= full_hessian_rows:
                print(
                    f"    Skipping from plots: HVPs ({n_hvp}) >= full Hessian rows ({full_hessian_rows})"
                )
                errors[alg].append(np.nan)
                n_hvps[alg].append(np.nan)
                hessians[alg].append(None)
                continue
            
            # Store hessian for eigenvalue comparison
            hessians[alg].append(hessian_sparse)

            # Calculate error (max absolute difference)
            max_error = np.abs(hessian_naive - hessian_sparse).max()
            errors[alg].append(max_error)
            n_hvps[alg].append(n_hvp)
            
            print(f"    Max error: {max_error:.6e}, Number of HVPs: {n_hvp}")
            
        except Exception as e:
            print(f"    Error computing {alg} at cutoff {cutoff}: {e}")
            errors[alg].append(np.nan)
            n_hvps[alg].append(np.nan)
            hessians[alg].append(None)

# Create plots
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "sparse_hessian_results", "cutoff_experiments")
os.makedirs(output_dir, exist_ok=True)

# Compute eigenvalue differences
eigenval_errors = {alg: [] for alg in algorithms}
eigenval_errors_all = {alg: [] for alg in algorithms}
eigvals_naive_proj_sorted = np.sort(projected_vibrational_eigvals(hessian_naive, masses, vib_basis))
eigvals_naive_first_sorted = eigvals_naive_proj_sorted[:first_n_eigvals]

# Store per-eigenvalue percentage errors for every cutoff/algorithm.
# Shape per algorithm: (n_cutoffs, n_eigvals), NaN where run is invalid/skipped.
per_eig_errors_all = {
    alg: np.full((len(cutoffs), len(eigvals_naive_proj_sorted)), np.nan, dtype=float)
    for alg in algorithms
}

for i, cutoff in enumerate(cutoffs):
    for alg in algorithms:
        if i < len(hessians[alg]) and hessians[alg][i] is not None:
            try:
                eigvals_sparse_proj_sorted = np.sort(
                    projected_vibrational_eigvals(hessians[alg][i], masses, vib_basis)
                )
                eigvals_sparse_first_sorted = eigvals_sparse_proj_sorted[:first_n_eigvals]
                # Average percentage difference
                pct_diff = np.abs(eigvals_naive_first_sorted - eigvals_sparse_first_sorted) / (
                    np.abs(eigvals_naive_first_sorted) + 1e-5
                )
                eigenval_errors[alg].append(pct_diff.mean())
                pct_diff_all = np.abs(eigvals_naive_proj_sorted - eigvals_sparse_proj_sorted) / (
                    np.abs(eigvals_naive_proj_sorted) + FULL_SPECTRUM_REG_EPS
                )
                eigenval_errors_all[alg].append(pct_diff_all.mean())
                per_eig_errors_all[alg][i, :] = pct_diff_all
            except Exception as e:
                print(f"    Warning: Could not compute eigenvalues for {alg} at cutoff {cutoff}: {e}")
                eigenval_errors[alg].append(np.nan)
                eigenval_errors_all[alg].append(np.nan)
        else:
            eigenval_errors[alg].append(np.nan)
            eigenval_errors_all[alg].append(np.nan)

def get_valid_xy(x_vals, y_vals, require_positive_y=False):
    """Return finite plotting points and optionally keep only positive y values."""
    x_arr = np.asarray(x_vals, dtype=float)
    y_arr = np.asarray(y_vals, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if require_positive_y:
        mask &= y_arr > 0
    return x_arr[mask], y_arr[mask]

# Plot 1: Error vs Cutoff (fixed to show all algorithms)
plt.figure(figsize=(10, 6))
for alg in algorithms:
    valid_cutoffs, valid_errors = get_valid_xy(cutoffs, errors[alg], require_positive_y=True)
    if valid_errors.size > 0:
        plt.plot(valid_cutoffs, valid_errors, marker='o', label=alg.replace('_', ' ').title(), linewidth=2, markersize=6)

plt.xlabel('Cutoff Radius (Angstroms)', fontsize=12)
plt.ylabel('Max Absolute Error vs Naive', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.yscale('log', nonpositive='mask')
plt.tight_layout()
error_plot_path = f"{output_dir}/cutoff_error_comparison_{timestamp}.png"
plt.savefig(error_plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"\nSaved error plot to: {error_plot_path}")

# Plot 2: Number of HVPs vs Cutoff
plt.figure(figsize=(10, 6))
for alg in algorithms:
    valid_cutoffs, valid_hvps = get_valid_xy(cutoffs, n_hvps[alg], require_positive_y=False)
    if valid_hvps.size > 0:
        plt.plot(valid_cutoffs, valid_hvps, marker='s', label=alg.replace('_', ' ').title(), linewidth=2, markersize=6)

plt.xlabel('Cutoff Radius (Angstroms)', fontsize=12)
plt.ylabel('Number of Hessian-Vector Products', fontsize=12)
plt.axhline(
    y=full_hessian_rows,
    color='red',
    linestyle='--',
    linewidth=2,
    label='Full hessian',
)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
hvp_plot_path = f"{output_dir}/cutoff_hvp_comparison_{timestamp}.png"
plt.savefig(hvp_plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved HVP plot to: {hvp_plot_path}")

# Plot 3: Eigenvalue relative error vs Cutoff
plt.figure(figsize=(10, 6))
for alg in algorithms:
    valid_cutoffs, valid_eigenval_errors = get_valid_xy(cutoffs, eigenval_errors[alg], require_positive_y=True)
    if valid_eigenval_errors.size > 0:
        plt.plot(valid_cutoffs, valid_eigenval_errors, marker='^', label=alg.replace('_', ' ').title(), linewidth=2, markersize=6)

plt.xlabel('Cutoff Radius (Angstroms)', fontsize=12)
plt.ylabel(f'Average Relative Error (First {first_n_eigvals} Eigenvalues)', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.yscale('log', nonpositive='mask')
plt.tight_layout()
eigenval_plot_path = f"{output_dir}/cutoff_eigenval_comparison_{timestamp}.png"
plt.savefig(eigenval_plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved eigenvalue comparison plot to: {eigenval_plot_path}")

# Plot 3b: All-eigenvalue relative error vs Cutoff
plt.figure(figsize=(10, 6))
for alg in algorithms:
    valid_cutoffs, valid_eigenval_errors_all = get_valid_xy(cutoffs, eigenval_errors_all[alg], require_positive_y=True)
    if valid_eigenval_errors_all.size > 0:
        plt.plot(
            valid_cutoffs,
            valid_eigenval_errors_all,
            marker='d',
            label=alg.replace('_', ' ').title(),
            linewidth=2,
            markersize=6,
        )

plt.xlabel('Cutoff Radius (Angstroms)', fontsize=12)
plt.ylabel('Average Relative Error (All Eigenvalues)', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.yscale('log', nonpositive='mask')
plt.tight_layout()
eigenval_all_plot_path = f"{output_dir}/cutoff_eigenval_all_comparison_{timestamp}.png"
plt.savefig(eigenval_all_plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved all-eigenvalue comparison plot to: {eigenval_all_plot_path}")

# Plot 4: Error vs Number of HVPs
plt.figure(figsize=(10, 6))
for alg in algorithms:
    valid_hvps, valid_errors = get_valid_xy(n_hvps[alg], errors[alg], require_positive_y=True)
    if valid_errors.size > 0:
        plt.plot(valid_hvps, valid_errors, marker='o', label=alg.replace('_', ' ').title(), linewidth=2, markersize=6)

plt.xlabel('Number of Hessian-Vector Products', fontsize=12)
plt.ylabel('Max Absolute Error vs Naive', fontsize=12)
plt.axvline(
    x=full_hessian_rows,
    color='red',
    linestyle='--',
    linewidth=2,
    label='Full hessian',
)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.yscale('log', nonpositive='mask')
plt.tight_layout()
error_hvp_plot_path = f"{output_dir}/error_vs_hvp_comparison_{timestamp}.png"
plt.savefig(error_hvp_plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved error vs HVP plot to: {error_hvp_plot_path}")

# Plot 5: Eigenvalue relative error vs Number of HVPs
plt.figure(figsize=(10, 6))
for alg in algorithms:
    valid_hvps, valid_eigenval_errors = get_valid_xy(n_hvps[alg], eigenval_errors[alg], require_positive_y=True)
    if valid_eigenval_errors.size > 0:
        plt.plot(valid_hvps, valid_eigenval_errors, marker='^', label=alg.replace('_', ' ').title(), linewidth=2, markersize=6)

plt.xlabel('Number of Hessian-Vector Products', fontsize=12)
plt.ylabel(f'Average Relative Error (First {first_n_eigvals} Eigenvalues)', fontsize=12)
plt.axvline(
    x=full_hessian_rows,
    color='red',
    linestyle='--',
    linewidth=2,
    label='Full hessian',
)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.yscale('log', nonpositive='mask')
plt.tight_layout()
eigenval_hvp_plot_path = f"{output_dir}/eigenval_vs_hvp_comparison_{timestamp}.png"
plt.savefig(eigenval_hvp_plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved eigenvalue vs HVP plot to: {eigenval_hvp_plot_path}")

# Plot 5b: All-eigenvalue relative error vs Number of HVPs
plt.figure(figsize=(10, 6))
for alg in algorithms:
    valid_hvps, valid_eigenval_errors_all = get_valid_xy(n_hvps[alg], eigenval_errors_all[alg], require_positive_y=True)
    if valid_eigenval_errors_all.size > 0:
        plt.plot(
            valid_hvps,
            valid_eigenval_errors_all,
            marker='d',
            label=alg.replace('_', ' ').title(),
            linewidth=2,
            markersize=6,
        )

plt.xlabel('Number of Hessian-Vector Products', fontsize=12)
plt.ylabel('Average Relative Error (All Eigenvalues)', fontsize=12)
plt.axvline(
    x=full_hessian_rows,
    color='red',
    linestyle='--',
    linewidth=2,
    label='Full hessian',
)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.yscale('log', nonpositive='mask')
plt.tight_layout()
eigenval_all_hvp_plot_path = f"{output_dir}/eigenval_all_vs_hvp_comparison_{timestamp}.png"
plt.savefig(eigenval_all_hvp_plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved all-eigenvalue vs HVP plot to: {eigenval_all_hvp_plot_path}")

# Plot 6: Per-eigenvalue error bars (2 rows x 3 columns)
fig, axes = plt.subplots(2, len(algorithms), figsize=(20, 10))

for col_idx, alg in enumerate(algorithms):
    # Pick largest-cutoff valid run for each algorithm.
    representative_idx = None
    for i in range(len(cutoffs) - 1, -1, -1):
        if i < len(hessians[alg]) and hessians[alg][i] is not None:
            representative_idx = i
            break

    ax_top = axes[0, col_idx]
    ax_bottom = axes[1, col_idx]

    if representative_idx is None:
        ax_top.text(0.5, 0.5, 'No valid run', ha='center', va='center', transform=ax_top.transAxes)
        ax_bottom.text(0.5, 0.5, 'No valid run', ha='center', va='center', transform=ax_bottom.transAxes)
        continue

    cutoff_value = cutoffs[representative_idx]
    hvp_value = n_hvps[alg][representative_idx]

    eigvals_sparse_proj_sorted = np.sort(
        projected_vibrational_eigvals(hessians[alg][representative_idx], masses, vib_basis)
    )
    eigval_pct_error = np.abs(eigvals_naive_proj_sorted - eigvals_sparse_proj_sorted) / (
        np.abs(eigvals_naive_proj_sorted) + FULL_SPECTRUM_REG_EPS
    )
    finite_errors = eigval_pct_error[np.isfinite(eigval_pct_error)]
    if finite_errors.size > 0:
        top_cap = float(np.percentile(finite_errors, 99.5))
        top_cap = max(top_cap, 1e-12)
    else:
        top_cap = 1e-12
    eigval_pct_error_plot = np.clip(eigval_pct_error, 0.0, top_cap)

    eig_indices = np.arange(len(eigval_pct_error))
    ax_top.bar(eig_indices, eigval_pct_error_plot, width=0.9, color='tab:blue', alpha=0.9, edgecolor='none')
    ax_top.set_ylim(0.0, top_cap * 1.05)
    ax_top.grid(True, alpha=0.3, axis='y')
    if col_idx == 0:
        ax_top.set_ylabel('Eigenvalue % Error', fontsize=11)
    ax_top.set_xlabel('Eigenvalue Index', fontsize=10)

    eig_span = float(np.max(eigvals_naive_proj_sorted) - np.min(eigvals_naive_proj_sorted))
    bar_width = eig_span / max(len(eigvals_naive_proj_sorted) * 1.5, 1)
    if not np.isfinite(bar_width) or bar_width <= 0:
        bar_width = 1e-3

    ax_bottom.bar(
        eigvals_naive_proj_sorted,
        eigval_pct_error_plot,
        width=bar_width,
        color='tab:orange',
        alpha=0.9,
        edgecolor='none',
    )
    ax_bottom.set_ylim(0.0, top_cap * 1.05)
    ax_bottom.grid(True, alpha=0.3, axis='y')
    if col_idx == 0:
        ax_bottom.set_ylabel('Eigenvalue % Error', fontsize=11)
    ax_bottom.set_xlabel('Ground-Truth Eigenvalue', fontsize=10)

plt.tight_layout(rect=[0, 0, 1, 0.96])
per_eigen_plot_path = f"{output_dir}/per_eigenvalue_error_bars_{timestamp}.png"
plt.savefig(per_eigen_plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved per-eigenvalue error bar plot to: {per_eigen_plot_path}")

# Save raw data used in plots for debugging without rerunning.
debug_data_path = f"{output_dir}/cutoff_debug_data_{timestamp}.npz"
np.savez_compressed(
    debug_data_path,
    cutoffs=np.asarray(cutoffs, dtype=float),
    full_hessian_rows=np.asarray([full_hessian_rows], dtype=int),
    eigvals_naive_eckart_projected_sorted=np.asarray(eigvals_naive_proj_sorted, dtype=float),
    errors_distance_two=np.asarray(errors["distance_two"], dtype=float),
    errors_star=np.asarray(errors["star"], dtype=float),
    errors_acyclic=np.asarray(errors["acyclic"], dtype=float),
    n_hvps_distance_two=np.asarray(n_hvps["distance_two"], dtype=float),
    n_hvps_star=np.asarray(n_hvps["star"], dtype=float),
    n_hvps_acyclic=np.asarray(n_hvps["acyclic"], dtype=float),
    eigenval_errors_first10_distance_two=np.asarray(eigenval_errors["distance_two"], dtype=float),
    eigenval_errors_first10_star=np.asarray(eigenval_errors["star"], dtype=float),
    eigenval_errors_first10_acyclic=np.asarray(eigenval_errors["acyclic"], dtype=float),
    eigenval_errors_all_distance_two=np.asarray(eigenval_errors_all["distance_two"], dtype=float),
    eigenval_errors_all_star=np.asarray(eigenval_errors_all["star"], dtype=float),
    eigenval_errors_all_acyclic=np.asarray(eigenval_errors_all["acyclic"], dtype=float),
    per_eig_errors_all_distance_two=per_eig_errors_all["distance_two"],
    per_eig_errors_all_star=per_eig_errors_all["star"],
    per_eig_errors_all_acyclic=per_eig_errors_all["acyclic"],
)
print(f"Saved debug data to: {debug_data_path}")

# Print summary
print("\n" + "="*60)
print("Summary:")
print("="*60)
print(f"{'Cutoff':<8} {'Distance Two':<20} {'Star':<20} {'Acyclic':<20}")
print(f"{'':<8} {'Error':<10} {'HVPs':<10} {'Error':<10} {'HVPs':<10} {'Error':<10} {'HVPs':<10}")
print("-"*60)
for i, cutoff in enumerate(cutoffs):
    print(f"{cutoff:<8.1f}", end="")
    for alg in algorithms:
        if i < len(errors[alg]) and not np.isnan(errors[alg][i]):
            print(f" {errors[alg][i]:<10.2e} {n_hvps[alg][i]:<10}", end="")
        else:
            print(f" {'N/A':<10} {'N/A':<10}", end="")
    print()

print("\nDone!")

