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
from matplotlib.colors import BoundaryNorm, ListedColormap
from datetime import datetime
from pathlib import Path
import time

# Relative-error denominator regularization used for full-spectrum comparisons.
FULL_SPECTRUM_REG_EPS = 1e-5

# Plot styling — match replot_cutoff_experiments.py (unified experiment figures).
LABEL_FONTSIZE = int(round(14 * 1.3))
LEGEND_FONTSIZE = int(round(12 * 1.4))
CBAR_LABEL_FONTSIZE = LABEL_FONTSIZE
TICK_FONTSIZE = 16
LINEWIDTH = 4.5
REF_LINEWIDTH = 2.0
MARKERSIZE = 10
LINE_FIGSIZE = (10, 6)
PLOT_DATA_DUMP = True

# Gate optional algorithms/series in plots & runs.
INCLUDE_DISTANCE_TWO = False

COLORS = {
    "naive": "#4e79a7",
    "star": "#f28e2b",
    "acyclic": "#59a14f",
    "distance_two": "#b07aa1",
}
COLORS_LIGHT = {
    "naive": "#a0c4e8",
    "star": "#f6c98c",
    "acyclic": "#9fcd8f",
    "distance_two": "#d4b3d4",
}
MARKERS = {
    "naive": "o",
    "star": "s",
    "acyclic": "^",
    "distance_two": "D",
}
# Untimed Hessian runs per algorithm before the timed cutoff sweep (GPU kernel compile, caches, etc.).
KERNEL_WARMUP_ROUNDS = 1
YLABEL_POINTWISE_REL_ERR = (
    r"$\frac{|\lambda_i-\lambda_i^{\mathrm{GT}}|}{|\lambda_i^{\mathrm{GT}}|}$"
)


def _mean_rel_err_label(n_terms: int | None = None) -> str:
    """Same LaTeX as replot_cutoff_experiments.py for mean relative eigenvalue error."""
    if n_terms is None:
        return (
            r"$\frac{1}{N_{\mathrm{eig}}}\sum_i"
            r"\frac{|\lambda_i-\lambda_i^{\mathrm{GT}}|}{|\lambda_i^{\mathrm{GT}}|}$"
        )
    return (
        rf"$\frac{{1}}{{{n_terms}}}\sum_{{i=1}}^{{{n_terms}}}"
        r"\frac{|\lambda_i-\lambda_i^{\mathrm{GT}}|}{|\lambda_i^{\mathrm{GT}}|}$"
    )


def _style_plot_ticks() -> None:
    plt.xticks(fontsize=TICK_FONTSIZE)
    plt.yticks(fontsize=TICK_FONTSIZE)


def _style_axis_ticks(ax) -> None:
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)


def _plot_abs_hessian_matrix(matrix, output_path, title, cbar_label):
    """Plot absolute Hessian values on a linear scale."""
    abs_matrix = np.abs(np.asarray(matrix, dtype=float))
    vmax = float(np.nanmax(abs_matrix)) if abs_matrix.size else 1.0
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0

    plt.figure(figsize=(12, 10))
    im = plt.imshow(
        abs_matrix,
        vmin=0.0,
        vmax=vmax,
        cmap="magma",
        aspect="equal",
        interpolation="nearest",
    )
    cbar = plt.colorbar(im)
    cbar.set_label(cbar_label, fontsize=CBAR_LABEL_FONTSIZE)
    cbar.ax.tick_params(labelsize=TICK_FONTSIZE)
    plt.title(title, fontsize=LABEL_FONTSIZE)
    plt.xlabel("Cartesian DOF Index", fontsize=LABEL_FONTSIZE)
    plt.ylabel("Cartesian DOF Index", fontsize=LABEL_FONTSIZE)
    _style_plot_ticks()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def _plot_sparsity_pattern(sparsity_pattern, output_path, title):
    """Plot a boolean sparsity pattern where True means exactly nonzero."""
    plt.figure(figsize=(12, 10))
    sparse_cmap = ListedColormap(["white", "black"])
    sparse_norm = BoundaryNorm([-0.5, 0.5, 1.5], sparse_cmap.N)
    im = plt.imshow(
        np.asarray(sparsity_pattern, dtype=int),
        cmap=sparse_cmap,
        norm=sparse_norm,
        aspect="equal",
        interpolation="nearest",
    )
    cbar = plt.colorbar(im, ticks=[0, 1])
    cbar.ax.set_yticklabels(["exactly zero", "nonzero"])
    cbar.set_label("Naive Hessian entry", fontsize=CBAR_LABEL_FONTSIZE)
    cbar.ax.tick_params(labelsize=TICK_FONTSIZE)
    plt.title(title, fontsize=LABEL_FONTSIZE)
    plt.xlabel("Cartesian DOF Index", fontsize=LABEL_FONTSIZE)
    plt.ylabel("Cartesian DOF Index", fontsize=LABEL_FONTSIZE)
    _style_plot_ticks()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def _plot_marker(alg: str) -> str:
    return MARKERS.get(alg, "o")

# Experiment configuration
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
MODEL_NAME = "CHGNet-MatPES-r2SCAN-2025.2.10-2.7M-PES"
MODEL_PATH = REPO_ROOT / "matgl" / "pretrained_models" / MODEL_NAME
STRUCTURE_PATH = REPO_ROOT / "data" / "Chignolin" / "Coordinates_9999.xyz"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = SCRIPT_DIR / "sparse_hessian_results" / "cutoff_experiments"
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("Sparse Hessian cutoff experiment")
print("=" * 80)
print(f"Model: {MODEL_NAME}")
print(f"Structure: {STRUCTURE_PATH}")
print("Loading model and structure...")

# Load model and structure
pot = matgl.load_model(str(MODEL_PATH))
pot.calc_hessian = True
pot.return_seed_matrix = True
pot.hessian_delta = 5e-3
atoms = read(str(STRUCTURE_PATH))
print(f"Loaded structure with {len(atoms)} atoms")

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
print("\n[1/4] Computing naive reference Hessian...")
pot.hessian_algorithm = "naive"
pot.hessian_sparsity_cutoff = None  # Use default graph-based
if torch.cuda.is_available():
    torch.cuda.synchronize()
t_naive0 = time.perf_counter()
calc.calculate(atoms, properties=['hessian'])
if torch.cuda.is_available():
    torch.cuda.synchronize()
naive_hessian_time_s = time.perf_counter() - t_naive0
hessian_naive = np.array(calc.results['hessian'])
full_hessian_rows = hessian_naive.shape[0]
naive_hessian_npy = output_dir / f"full_naive_hessian_{timestamp}.npy"
np.save(naive_hessian_npy, hessian_naive)
naive_hessian_offdiag = hessian_naive - np.diag(np.diag(hessian_naive))
naive_sparsity_pattern = hessian_naive != 0
naive_sparsity_npy = output_dir / f"naive_hessian_exact_zero_sparsity_{timestamp}.npy"
np.save(naive_sparsity_npy, naive_sparsity_pattern)
naive_plot_data_path = output_dir / f"naive_hessian_plot_data_{timestamp}.npz"
np.savez_compressed(
    naive_plot_data_path,
    full_naive_hessian=hessian_naive,
    naive_hessian_offdiag=naive_hessian_offdiag,
    naive_sparsity_pattern=naive_sparsity_pattern,
)
naive_hessian_plot_path = output_dir / f"naive_hessian_offdiag_abs_{timestamp}.png"
_plot_abs_hessian_matrix(
    naive_hessian_offdiag,
    naive_hessian_plot_path,
    r"$|H_{\mathrm{naive}} - \mathrm{diag}(H_{\mathrm{naive}})|$",
    "|value| (linear scale)",
)
naive_sparsity_plot_path = output_dir / f"naive_hessian_exact_zero_sparsity_{timestamp}.png"
_plot_sparsity_pattern(
    naive_sparsity_pattern,
    naive_sparsity_plot_path,
    "Naive Hessian Exact-Zero Sparsity Pattern",
)
print(
    f"Full Hessian dimension: {full_hessian_rows} x {full_hessian_rows} "
    f"(naive wall time: {naive_hessian_time_s:.4f}s)"
)
print(f"Saved full naive reference Hessian to: {naive_hessian_npy}")
print(f"Saved naive Hessian plot data to: {naive_plot_data_path}")
print(f"Saved naive off-diagonal Hessian plot to: {naive_hessian_plot_path}")
print(f"Saved naive exact-zero sparsity array to: {naive_sparsity_npy}")
print(f"Saved naive exact-zero sparsity plot to: {naive_sparsity_plot_path}")
masses = np.asarray(atoms.get_masses(), dtype=float)
vib_basis = get_vibrational_basis_from_eckart(atoms)

# Cutoffs to test
cutoffs = np.arange(5, 33, 3)  # 5 to 32 Angstroms, step size 3
algorithms = ["star", "acyclic"]
if INCLUDE_DISTANCE_TWO:
    algorithms = ["distance_two", *algorithms]

if KERNEL_WARMUP_ROUNDS > 0:
    print(
        f"\nKernel warmup: {KERNEL_WARMUP_ROUNDS} untimed Hessian(s) per algorithm "
        f"at cutoff={float(cutoffs[0])} Å (excluded from sparse timings)..."
    )
    for alg in algorithms:
        pot.hessian_algorithm = alg
        pot.hessian_sparsity_cutoff = float(cutoffs[0])
        for _ in range(KERNEL_WARMUP_ROUNDS):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            calc.calculate(atoms, properties=["hessian"])
            if torch.cuda.is_available():
                torch.cuda.synchronize()

# Storage for results
errors = {alg: [] for alg in algorithms}
n_hvps = {alg: [] for alg in algorithms}
timings_s = {alg: [] for alg in algorithms}  # Wall time per Hessian solve (seconds)
hessians = {alg: [] for alg in algorithms}  # Store hessians for eigenvalue comparison
first_n_eigvals = 10  # Parameter for eigenvalue comparison
included_full_case = {alg: False for alg in algorithms}

# Loop through cutoffs
print(f"\n[2/4] Sweeping {len(cutoffs)} cutoff values: {cutoffs.tolist()}")
for cutoff in cutoffs:
    print(f"\nTesting cutoff: {cutoff} Angstroms")
    
    for alg in algorithms:
        print(f"  Computing {alg}...")
        pot.hessian_algorithm = alg
        pot.hessian_sparsity_cutoff = cutoff
        
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            calc.calculate(atoms, properties=['hessian'])
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed_s = time.perf_counter() - t0
            hessian_sparse = np.array(calc.results['hessian'])
            seed_matrix = np.array(calc.results['seed_matrix'])

            # Get number of HVPs (second dimension of seed matrix)
            n_hvp = seed_matrix.shape[1] if len(seed_matrix.shape) == 2 else seed_matrix.shape[0]
            
            # Full-Hessian (or overcomplete) cases can collapse error to exact zero.
            # Keep exactly one such datapoint per algorithm to show the limiting behavior.
            if n_hvp >= full_hessian_rows and included_full_case[alg]:
                print(
                    f"    Skipping extra full-Hessian case: HVPs ({n_hvp}) >= rows ({full_hessian_rows})"
                )
                errors[alg].append(np.nan)
                n_hvps[alg].append(np.nan)
                timings_s[alg].append(np.nan)
                hessians[alg].append(None)
                continue
            
            # Store hessian for eigenvalue comparison
            hessians[alg].append(hessian_sparse)

            # Calculate error (max absolute difference)
            max_error = np.abs(hessian_naive - hessian_sparse).max()
            errors[alg].append(max_error)
            n_hvps[alg].append(n_hvp)
            timings_s[alg].append(elapsed_s)
            if n_hvp >= full_hessian_rows:
                included_full_case[alg] = True
            
            print(f"    Max error: {max_error:.6e}, Number of HVPs: {n_hvp}, Time: {elapsed_s:.4f}s")
            
        except Exception as e:
            print(f"    Error computing {alg} at cutoff {cutoff}: {e}")
            errors[alg].append(np.nan)
            n_hvps[alg].append(np.nan)
            timings_s[alg].append(np.nan)
            hessians[alg].append(None)

# Create plots
print("\n[3/4] Generating plots and saving outputs...")

# Compute eigenvalue differences
eigenval_errors = {alg: [] for alg in algorithms}
eigenval_errors_all = {alg: [] for alg in algorithms}
EIGENVAL_ERROR_PLOT_EPS = 1e-7
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
                eigenval_errors[alg].append(pct_diff.mean() + EIGENVAL_ERROR_PLOT_EPS)
                pct_diff_all = np.abs(eigvals_naive_proj_sorted - eigvals_sparse_proj_sorted) / (
                    np.abs(eigvals_naive_proj_sorted) + FULL_SPECTRUM_REG_EPS
                )
                eigenval_errors_all[alg].append(pct_diff_all.mean() + EIGENVAL_ERROR_PLOT_EPS)
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


def get_valid_xy_sorted_by_x(x_vals, y_vals, require_positive_y=False):
    """Like get_valid_xy, but sort points by x so line plots follow the x-axis order."""
    x_arr, y_arr = get_valid_xy(x_vals, y_vals, require_positive_y=require_positive_y)
    if x_arr.size == 0:
        return x_arr, y_arr
    order = np.argsort(x_arr, kind="stable")
    return x_arr[order], y_arr[order]

# Plot 1: Error vs Cutoff (fixed to show all algorithms)
plt.figure(figsize=LINE_FIGSIZE)
for alg in algorithms:
    valid_cutoffs, valid_errors = get_valid_xy(cutoffs, errors[alg], require_positive_y=True)
    if valid_errors.size > 0:
        plt.plot(
            valid_cutoffs,
            valid_errors,
            marker=_plot_marker(alg),
            color=COLORS.get(alg),
            label=alg.replace("_", " ").title(),
            linewidth=LINEWIDTH,
            markersize=MARKERSIZE,
        )

plt.xlabel('Cutoff Radius (Angstroms)', fontsize=LABEL_FONTSIZE)
plt.ylabel('Max Absolute Error vs Naive', fontsize=LABEL_FONTSIZE)
plt.legend(fontsize=LEGEND_FONTSIZE)
plt.grid(True, alpha=0.3)
plt.yscale('log', nonpositive='mask')
_style_plot_ticks()
plt.tight_layout()
error_plot_path = output_dir / f"cutoff_error_comparison_{timestamp}.png"
plt.savefig(error_plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"\nSaved error plot to: {error_plot_path}")

# Plot 2: Number of HVPs vs Cutoff
plt.figure(figsize=LINE_FIGSIZE)
for alg in algorithms:
    valid_cutoffs, valid_hvps = get_valid_xy(cutoffs, n_hvps[alg], require_positive_y=False)
    if valid_hvps.size > 0:
        plt.plot(
            valid_cutoffs,
            valid_hvps,
            marker=_plot_marker(alg),
            color=COLORS.get(alg),
            label=alg.replace("_", " ").title(),
            linewidth=LINEWIDTH,
            markersize=MARKERSIZE,
        )

plt.xlabel('Cutoff Radius (Angstroms)', fontsize=LABEL_FONTSIZE)
plt.ylabel('Number of Hessian-Vector Products', fontsize=LABEL_FONTSIZE)
plt.axhline(
    y=full_hessian_rows,
    color=COLORS["naive"],
    linestyle="--",
    linewidth=REF_LINEWIDTH,
    label="Full Hessian rows",
)
plt.legend(fontsize=LEGEND_FONTSIZE)
plt.grid(True, alpha=0.3)
_style_plot_ticks()
plt.tight_layout()
hvp_plot_path = output_dir / f"cutoff_hvp_comparison_{timestamp}.png"
plt.savefig(hvp_plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved HVP plot to: {hvp_plot_path}")

# Plot 2b: Wall time vs cutoff (linear time axis; always meaningful when runs succeed)
plt.figure(figsize=LINE_FIGSIZE)
for alg in algorithms:
    valid_cutoffs, valid_t = get_valid_xy_sorted_by_x(cutoffs, timings_s[alg], require_positive_y=False)
    pos_t = valid_t > 0
    if np.any(pos_t):
        plt.plot(
            valid_cutoffs[pos_t],
            valid_t[pos_t],
            marker=_plot_marker(alg),
            color=COLORS.get(alg),
            label=alg.replace("_", " ").title(),
            linewidth=LINEWIDTH,
            markersize=MARKERSIZE,
        )
if np.isfinite(naive_hessian_time_s) and naive_hessian_time_s > 0:
    plt.axhline(
        y=naive_hessian_time_s,
        color=COLORS["naive"],
        linestyle="--",
        linewidth=REF_LINEWIDTH,
        label="Naive reference time",
    )
plt.xlabel("Cutoff Radius (Angstroms)", fontsize=LABEL_FONTSIZE)
plt.ylabel("Wall Time per Hessian (s)", fontsize=LABEL_FONTSIZE)
plt.legend(fontsize=LEGEND_FONTSIZE)
plt.grid(True, alpha=0.3)
_style_plot_ticks()
plt.tight_layout()
time_cutoff_plot_path = output_dir / f"cutoff_wall_time_comparison_{timestamp}.png"
plt.savefig(time_cutoff_plot_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved wall time vs cutoff plot to: {time_cutoff_plot_path}")

# Plot 3: Eigenvalue relative error vs Cutoff
plt.figure(figsize=LINE_FIGSIZE)
for alg in algorithms:
    valid_cutoffs, valid_eigenval_errors = get_valid_xy(cutoffs, eigenval_errors[alg], require_positive_y=True)
    if valid_eigenval_errors.size > 0:
        plt.plot(
            valid_cutoffs,
            valid_eigenval_errors,
            marker=_plot_marker(alg),
            color=COLORS.get(alg),
            label=alg.replace("_", " ").title(),
            linewidth=LINEWIDTH,
            markersize=MARKERSIZE,
        )

plt.xlabel('Cutoff Radius (Angstroms)', fontsize=LABEL_FONTSIZE)
plt.ylabel(_mean_rel_err_label(first_n_eigvals), fontsize=LABEL_FONTSIZE)
plt.legend(fontsize=LEGEND_FONTSIZE)
plt.grid(True, alpha=0.3)
plt.yscale('log', nonpositive='mask')
_style_plot_ticks()
plt.tight_layout()
eigenval_plot_path = output_dir / f"cutoff_eigenval_comparison_{timestamp}.png"
plt.savefig(eigenval_plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved eigenvalue comparison plot to: {eigenval_plot_path}")

# Plot 3b: All-eigenvalue relative error vs Cutoff
plt.figure(figsize=LINE_FIGSIZE)
for alg in algorithms:
    valid_cutoffs, valid_eigenval_errors_all = get_valid_xy(cutoffs, eigenval_errors_all[alg], require_positive_y=True)
    if valid_eigenval_errors_all.size > 0:
        plt.plot(
            valid_cutoffs,
            valid_eigenval_errors_all,
            marker=_plot_marker(alg),
            color=COLORS.get(alg),
            label=alg.replace('_', ' ').title(),
            linewidth=LINEWIDTH,
            markersize=MARKERSIZE,
        )

plt.xlabel('Cutoff Radius (Angstroms)', fontsize=LABEL_FONTSIZE)
plt.ylabel(_mean_rel_err_label(None), fontsize=LABEL_FONTSIZE)
plt.legend(fontsize=LEGEND_FONTSIZE)
plt.grid(True, alpha=0.3)
plt.yscale('log', nonpositive='mask')
_style_plot_ticks()
plt.tight_layout()
eigenval_all_plot_path = output_dir / f"cutoff_eigenval_all_comparison_{timestamp}.png"
plt.savefig(eigenval_all_plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved all-eigenvalue comparison plot to: {eigenval_all_plot_path}")

# Plot 4: Error vs Number of HVPs (points ordered by HVP count on x-axis)
plt.figure(figsize=LINE_FIGSIZE)
for alg in algorithms:
    valid_hvps, valid_errors = get_valid_xy_sorted_by_x(n_hvps[alg], errors[alg], require_positive_y=True)
    if valid_errors.size > 0:
        plt.plot(
            valid_hvps,
            valid_errors,
            marker=_plot_marker(alg),
            color=COLORS.get(alg),
            label=alg.replace("_", " ").title(),
            linewidth=LINEWIDTH,
            markersize=MARKERSIZE,
        )

plt.xlabel('Number of Hessian-Vector Products', fontsize=LABEL_FONTSIZE)
plt.ylabel('Max Absolute Error vs Naive', fontsize=LABEL_FONTSIZE)
plt.axvline(
    x=full_hessian_rows,
    color=COLORS["naive"],
    linestyle="--",
    linewidth=REF_LINEWIDTH,
    label="Full Hessian rows",
)
plt.legend(fontsize=LEGEND_FONTSIZE)
plt.grid(True, alpha=0.3)
plt.yscale('log', nonpositive='mask')
_style_plot_ticks()
plt.tight_layout()
error_hvp_plot_path = output_dir / f"error_vs_hvp_comparison_{timestamp}.png"
plt.savefig(error_hvp_plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved error vs HVP plot to: {error_hvp_plot_path}")

# Plot 5: Eigenvalue relative error vs Number of HVPs (points ordered by HVP count)
plt.figure(figsize=LINE_FIGSIZE)
for alg in algorithms:
    valid_hvps, valid_eigenval_errors = get_valid_xy_sorted_by_x(
        n_hvps[alg], eigenval_errors[alg], require_positive_y=True
    )
    if valid_eigenval_errors.size > 0:
        plt.plot(
            valid_hvps,
            valid_eigenval_errors,
            marker=_plot_marker(alg),
            color=COLORS.get(alg),
            label=alg.replace("_", " ").title(),
            linewidth=LINEWIDTH,
            markersize=MARKERSIZE,
        )

plt.xlabel('Number of Hessian-Vector Products', fontsize=LABEL_FONTSIZE)
plt.ylabel(_mean_rel_err_label(first_n_eigvals), fontsize=LABEL_FONTSIZE)
plt.axvline(
    x=full_hessian_rows,
    color=COLORS["naive"],
    linestyle="--",
    linewidth=REF_LINEWIDTH,
    label="Full Hessian rows",
)
plt.legend(fontsize=LEGEND_FONTSIZE)
plt.grid(True, alpha=0.3)
plt.yscale('log', nonpositive='mask')
_style_plot_ticks()
plt.tight_layout()
eigenval_hvp_plot_path = output_dir / f"eigenval_vs_hvp_comparison_{timestamp}.png"
plt.savefig(eigenval_hvp_plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved eigenvalue vs HVP plot to: {eigenval_hvp_plot_path}")

# Plot 5b: All-eigenvalue relative error vs Number of HVPs (points ordered by HVP count)
plt.figure(figsize=LINE_FIGSIZE)
for alg in algorithms:
    valid_hvps, valid_eigenval_errors_all = get_valid_xy_sorted_by_x(
        n_hvps[alg], eigenval_errors_all[alg], require_positive_y=True
    )
    if valid_eigenval_errors_all.size > 0:
        plt.plot(
            valid_hvps,
            valid_eigenval_errors_all,
            marker=_plot_marker(alg),
            color=COLORS.get(alg),
            label=alg.replace('_', ' ').title(),
            linewidth=LINEWIDTH,
            markersize=MARKERSIZE,
        )

plt.xlabel('Number of Hessian-Vector Products', fontsize=LABEL_FONTSIZE)
plt.ylabel(_mean_rel_err_label(None), fontsize=LABEL_FONTSIZE)
plt.axvline(
    x=full_hessian_rows,
    color=COLORS["naive"],
    linestyle="--",
    linewidth=REF_LINEWIDTH,
    label="Full Hessian rows",
)
plt.legend(fontsize=LEGEND_FONTSIZE)
plt.grid(True, alpha=0.3)
plt.yscale('log', nonpositive='mask')
_style_plot_ticks()
plt.tight_layout()
eigenval_all_hvp_plot_path = output_dir / f"eigenval_all_vs_hvp_comparison_{timestamp}.png"
plt.savefig(eigenval_all_hvp_plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved all-eigenvalue vs HVP plot to: {eigenval_all_hvp_plot_path}")

# Plot 5c: Max error vs wall time (x-axis: seconds; points ordered by time)
# Use finite errors (not require_positive_y): exact zeros are common at full sparsity; log-y only
# when there is at least one positive error — otherwise log scale breaks on an empty/default y-axis.
plt.figure(figsize=LINE_FIGSIZE)
ax_time_err = plt.gca()
min_positive_error = None
for alg in algorithms:
    valid_t, valid_errors = get_valid_xy_sorted_by_x(timings_s[alg], errors[alg], require_positive_y=False)
    pos = (valid_errors > 0) & np.isfinite(valid_errors) & (valid_t > 0) & np.isfinite(valid_t)
    if np.any(pos):
        plt.plot(
            valid_t[pos],
            valid_errors[pos],
            marker=_plot_marker(alg),
            color=COLORS.get(alg),
            label=alg.replace("_", " ").title(),
            linewidth=LINEWIDTH,
            markersize=MARKERSIZE,
        )
        mp = float(np.min(valid_errors[pos]))
        min_positive_error = mp if min_positive_error is None else min(min_positive_error, mp)

if np.isfinite(naive_hessian_time_s) and naive_hessian_time_s > 0:
    plt.axvline(
        x=naive_hessian_time_s,
        color=COLORS["naive"],
        linestyle="--",
        linewidth=REF_LINEWIDTH,
        label="Naive reference time",
    )

plt.xlabel("Wall Time per Hessian (s)", fontsize=LABEL_FONTSIZE)
plt.ylabel("Max Absolute Error vs Naive", fontsize=LABEL_FONTSIZE)
plt.legend(fontsize=LEGEND_FONTSIZE)
plt.grid(True, alpha=0.3)
if min_positive_error is not None:
    ax_time_err.set_ylim(bottom=max(min_positive_error * 0.5, 1e-30))
    plt.yscale("log", nonpositive="mask")
else:
    ax_time_err.text(
        0.5,
        0.5,
        "No positive max-error points\n(all exact or missing data);\nsee wall time vs cutoff plot.",
        transform=ax_time_err.transAxes,
        ha="center",
        va="center",
        fontsize=LABEL_FONTSIZE,
    )
_style_plot_ticks()
plt.tight_layout()
error_time_plot_path = output_dir / f"error_vs_time_comparison_{timestamp}.png"
plt.savefig(error_time_plot_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved error vs wall time plot to: {error_time_plot_path}")

# Plot 5d: Mean eigenvalue relative error (first N) vs wall time
plt.figure(figsize=LINE_FIGSIZE)
ax_ev_t = plt.gca()
min_pos_eig = None
for alg in algorithms:
    valid_t, valid_y = get_valid_xy_sorted_by_x(timings_s[alg], eigenval_errors[alg], require_positive_y=False)
    pos = (valid_y > 0) & np.isfinite(valid_y) & (valid_t > 0) & np.isfinite(valid_t)
    if np.any(pos):
        plt.plot(
            valid_t[pos],
            valid_y[pos],
            marker=_plot_marker(alg),
            color=COLORS.get(alg),
            label=alg.replace("_", " ").title(),
            linewidth=LINEWIDTH,
            markersize=MARKERSIZE,
        )
        mp = float(np.min(valid_y[pos]))
        min_pos_eig = mp if min_pos_eig is None else min(min_pos_eig, mp)
if np.isfinite(naive_hessian_time_s) and naive_hessian_time_s > 0:
    plt.axvline(
        x=naive_hessian_time_s,
        color=COLORS["naive"],
        linestyle="--",
        linewidth=REF_LINEWIDTH,
        label="Naive reference time",
    )
plt.xlabel("Wall Time per Hessian (s)", fontsize=LABEL_FONTSIZE)
plt.ylabel(_mean_rel_err_label(first_n_eigvals), fontsize=LABEL_FONTSIZE)
plt.legend(fontsize=LEGEND_FONTSIZE)
plt.grid(True, alpha=0.3)
if min_pos_eig is not None:
    ax_ev_t.set_ylim(bottom=max(min_pos_eig * 0.5, 1e-30))
    plt.yscale("log", nonpositive="mask")
else:
    ax_ev_t.text(
        0.5,
        0.5,
        "No positive mean eigenvalue-error points\n(all exact or missing data).",
        transform=ax_ev_t.transAxes,
        ha="center",
        va="center",
        fontsize=LABEL_FONTSIZE,
    )
_style_plot_ticks()
plt.tight_layout()
eigenval_time_plot_path = output_dir / f"eigenval_vs_time_comparison_{timestamp}.png"
plt.savefig(eigenval_time_plot_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved eigenvalue vs wall time plot to: {eigenval_time_plot_path}")

# Plot 5e: Mean eigenvalue relative error (all modes) vs wall time
plt.figure(figsize=LINE_FIGSIZE)
ax_eva_t = plt.gca()
min_pos_eig_all = None
for alg in algorithms:
    valid_t, valid_y = get_valid_xy_sorted_by_x(timings_s[alg], eigenval_errors_all[alg], require_positive_y=False)
    pos = (valid_y > 0) & np.isfinite(valid_y) & (valid_t > 0) & np.isfinite(valid_t)
    if np.any(pos):
        plt.plot(
            valid_t[pos],
            valid_y[pos],
            marker=_plot_marker(alg),
            color=COLORS.get(alg),
            label=alg.replace("_", " ").title(),
            linewidth=LINEWIDTH,
            markersize=MARKERSIZE,
        )
        mp = float(np.min(valid_y[pos]))
        min_pos_eig_all = mp if min_pos_eig_all is None else min(min_pos_eig_all, mp)
if np.isfinite(naive_hessian_time_s) and naive_hessian_time_s > 0:
    plt.axvline(
        x=naive_hessian_time_s,
        color=COLORS["naive"],
        linestyle="--",
        linewidth=REF_LINEWIDTH,
        label="Naive reference time",
    )
plt.xlabel("Wall Time per Hessian (s)", fontsize=LABEL_FONTSIZE)
plt.ylabel(_mean_rel_err_label(None), fontsize=LABEL_FONTSIZE)
plt.legend(fontsize=LEGEND_FONTSIZE)
plt.grid(True, alpha=0.3)
if min_pos_eig_all is not None:
    ax_eva_t.set_ylim(bottom=max(min_pos_eig_all * 0.5, 1e-30))
    plt.yscale("log", nonpositive="mask")
else:
    ax_eva_t.text(
        0.5,
        0.5,
        "No positive mean eigenvalue-error points\n(all exact or missing data).",
        transform=ax_eva_t.transAxes,
        ha="center",
        va="center",
        fontsize=LABEL_FONTSIZE,
    )
_style_plot_ticks()
plt.tight_layout()
eigenval_all_time_plot_path = output_dir / f"eigenval_all_vs_time_comparison_{timestamp}.png"
plt.savefig(eigenval_all_time_plot_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved all-eigenvalue vs wall time plot to: {eigenval_all_time_plot_path}")

# Plot 6: Per-eigenvalue error bars (2 rows x N columns)
fig, axes = plt.subplots(2, len(algorithms), figsize=(20, 10))
if len(algorithms) == 1:
    axes = np.array([[axes[0]], [axes[1]]])

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
    ax_top.bar(
        eig_indices,
        eigval_pct_error_plot,
        width=0.9,
        color=COLORS.get(alg, "#666666"),
        alpha=0.92,
        edgecolor="white",
        linewidth=0.4,
    )
    ax_top.set_ylim(0.0, top_cap * 1.05)
    ax_top.grid(True, alpha=0.3, axis='y')
    if col_idx == 0:
        ax_top.set_ylabel(YLABEL_POINTWISE_REL_ERR, fontsize=LABEL_FONTSIZE)
    ax_top.set_xlabel('Eigenvalue Index', fontsize=LABEL_FONTSIZE)
    ax_top.set_title(
        f"{alg.replace('_', ' ').title()} (cutoff={cutoffs[representative_idx]:.0f} Å)",
        fontsize=LABEL_FONTSIZE - 1,
    )
    _style_axis_ticks(ax_top)

    eig_span = float(np.max(eigvals_naive_proj_sorted) - np.min(eigvals_naive_proj_sorted))
    bar_width = eig_span / max(len(eigvals_naive_proj_sorted) * 1.5, 1)
    if not np.isfinite(bar_width) or bar_width <= 0:
        bar_width = 1e-3

    ax_bottom.bar(
        eigvals_naive_proj_sorted,
        eigval_pct_error_plot,
        width=bar_width,
        color=COLORS_LIGHT.get(alg, COLORS.get(alg, "#666666")),
        alpha=0.92,
        edgecolor="white",
        linewidth=0.4,
    )
    ax_bottom.set_ylim(0.0, top_cap * 1.05)
    ax_bottom.grid(True, alpha=0.3, axis='y')
    if col_idx == 0:
        ax_bottom.set_ylabel(YLABEL_POINTWISE_REL_ERR, fontsize=LABEL_FONTSIZE)
    ax_bottom.set_xlabel('Ground-Truth Eigenvalue', fontsize=LABEL_FONTSIZE)
    _style_axis_ticks(ax_bottom)

plt.tight_layout(rect=[0, 0, 1, 0.96])
per_eigen_plot_path = output_dir / f"per_eigenvalue_error_bars_{timestamp}.png"
plt.savefig(per_eigen_plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved per-eigenvalue error bar plot to: {per_eigen_plot_path}")

# Save raw data used in plots for debugging without rerunning.
debug_data_path = output_dir / f"cutoff_debug_data_{timestamp}.npz"
debug_payload = {
    "cutoffs": np.asarray(cutoffs, dtype=float),
    "naive_hessian_time_s": np.asarray([naive_hessian_time_s], dtype=float),
    "full_hessian_rows": np.asarray([full_hessian_rows], dtype=int),
    "eigvals_naive_eckart_projected_sorted": np.asarray(eigvals_naive_proj_sorted, dtype=float),
    "errors_star": np.asarray(errors["star"], dtype=float),
    "errors_acyclic": np.asarray(errors["acyclic"], dtype=float),
    "n_hvps_star": np.asarray(n_hvps["star"], dtype=float),
    "n_hvps_acyclic": np.asarray(n_hvps["acyclic"], dtype=float),
    "timings_s_star": np.asarray(timings_s["star"], dtype=float),
    "timings_s_acyclic": np.asarray(timings_s["acyclic"], dtype=float),
    "eigenval_errors_first10_star": np.asarray(eigenval_errors["star"], dtype=float),
    "eigenval_errors_first10_acyclic": np.asarray(eigenval_errors["acyclic"], dtype=float),
    "eigenval_errors_all_star": np.asarray(eigenval_errors_all["star"], dtype=float),
    "eigenval_errors_all_acyclic": np.asarray(eigenval_errors_all["acyclic"], dtype=float),
    "per_eig_errors_all_star": per_eig_errors_all["star"],
    "per_eig_errors_all_acyclic": per_eig_errors_all["acyclic"],
}
if INCLUDE_DISTANCE_TWO:
    debug_payload.update(
        {
            "errors_distance_two": np.asarray(errors["distance_two"], dtype=float),
            "n_hvps_distance_two": np.asarray(n_hvps["distance_two"], dtype=float),
            "timings_s_distance_two": np.asarray(timings_s["distance_two"], dtype=float),
            "eigenval_errors_first10_distance_two": np.asarray(eigenval_errors["distance_two"], dtype=float),
            "eigenval_errors_all_distance_two": np.asarray(eigenval_errors_all["distance_two"], dtype=float),
            "per_eig_errors_all_distance_two": per_eig_errors_all["distance_two"],
        }
    )
np.savez_compressed(debug_data_path, **debug_payload)
print(f"Saved debug data to: {debug_data_path}")

# Save a single consolidated payload containing all plot series.
if PLOT_DATA_DUMP:
    plot_series_path = output_dir / f"cutoff_plot_series_{timestamp}.npz"
    series_payload = {
        "timestamp": np.asarray([timestamp], dtype=object),
        "cutoffs": np.asarray(cutoffs, dtype=float),
        "algorithms": np.asarray(algorithms, dtype=object),
        "include_distance_two": np.asarray([INCLUDE_DISTANCE_TWO], dtype=bool),
        "full_hessian_rows": np.asarray([full_hessian_rows], dtype=int),
        "naive_hessian_time_s": np.asarray([naive_hessian_time_s], dtype=float),
        "label_fontsize": np.asarray([LABEL_FONTSIZE], dtype=int),
        "legend_fontsize": np.asarray([LEGEND_FONTSIZE], dtype=int),
        "colors_naive": np.asarray([COLORS["naive"]], dtype=object),
        "colors_star": np.asarray([COLORS["star"]], dtype=object),
        "colors_acyclic": np.asarray([COLORS["acyclic"]], dtype=object),
    }
    for alg in algorithms:
        series_payload[f"errors_{alg}"] = np.asarray(errors[alg], dtype=float)
        series_payload[f"n_hvps_{alg}"] = np.asarray(n_hvps[alg], dtype=float)
        series_payload[f"timings_s_{alg}"] = np.asarray(timings_s[alg], dtype=float)
        series_payload[f"eigenval_errors_first{first_n_eigvals}_{alg}"] = np.asarray(eigenval_errors[alg], dtype=float)
        series_payload[f"eigenval_errors_all_{alg}"] = np.asarray(eigenval_errors_all[alg], dtype=float)
        series_payload[f"per_eig_errors_all_{alg}"] = np.asarray(per_eig_errors_all[alg], dtype=float)
        series_payload[f"color_{alg}"] = np.asarray([COLORS.get(alg)], dtype=object)
    np.savez_compressed(plot_series_path, **series_payload)
    print(f"Saved plot series to: {plot_series_path}")

# Print summary
print("\n[4/4] Final summary")
print("\n" + "="*60)
print("Summary:")
print("="*60)
header = ["Cutoff"]
for alg in algorithms:
    header.append(alg.replace("_", " ").title())
print(" ".join([f"{h:<20}" if i else f"{h:<8}" for i, h in enumerate(header)]))
sub = ["", *["Error     HVPs     " for _ in algorithms]]
print(" ".join([f"{s:<20}" if i else f"{s:<8}" for i, s in enumerate(sub)]))
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

