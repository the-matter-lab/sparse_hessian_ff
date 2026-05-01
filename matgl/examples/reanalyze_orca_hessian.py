"""Reanalyze sparse Hessian recovery using a parsed ORCA .hess reference.

The dense ORCA Cartesian Hessian is parsed once, then Hessian-vector products are
emulated as ``H_true @ v`` for cutoff-based sparsity patterns. This mirrors
``reanalize_dft_hessian.py`` / ``sparse_hessian_cutoff_dft.py`` without running
any quantum chemistry during the sweep.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from ase import Atoms
from scipy.sparse import coo_matrix, kron


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
COLPACK_SRC = REPO_ROOT / "colpackpy" / "src"
if str(COLPACK_SRC) not in sys.path:
    sys.path.insert(0, str(COLPACK_SRC))

from colpackpy.torch_utils import compute_sparse_hessian_from_hvp_fn  # type: ignore[import-not-found]
from plot_orca_hessian import parse_orca_hessian


BOHR_TO_ANGSTROM = 0.529177210903
COLORING_ORDERING = "SMALLEST_LAST"
FULL_SPECTRUM_REG_EPS = 1e-5
# Keep exact-zero eigenvalue errors visible on log-scale plots.
EIGENVAL_ERROR_PLOT_EPS = 1e-7
LABEL_FONTSIZE = 18
LEGEND_FONTSIZE = 17
TICK_FONTSIZE = 16
LINEWIDTH = 4.5
REF_LINEWIDTH = 2.0
MARKERSIZE = 10
LINE_FIGSIZE = (10, 6)
DEFAULT_HESSIAN = (
    SCRIPT_DIR / "sparse_hessian_results" / "orca_chingolin" / "pbe_orca_hessian_chingolin.hess"
)

COLORS = {
    "naive": "#4e79a7",
    "star": "#f28e2b",
    "acyclic": "#59a14f",
    "distance_two": "#b07aa1",
}
MARKERS = {
    "star": "s",
    "acyclic": "^",
    "distance_two": "D",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hessian",
        type=Path,
        default=DEFAULT_HESSIAN,
        help="Input ORCA .hess file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <hessian-dir>/reanalyze.",
    )
    parser.add_argument(
        "--cutoffs",
        type=str,
        default="5,8,11,14,17,20,23,26,29,32",
        help="Comma-separated cutoff radii in Angstrom.",
    )
    parser.add_argument(
        "--algorithms",
        type=str,
        default="star,acyclic",
        help="Comma-separated ColPack algorithms.",
    )
    parser.add_argument("--first-n-eigvals", type=int, default=10)
    parser.add_argument("--plot-dpi", type=int, default=300, help="DPI for saved PNG figures.")
    parser.add_argument(
        "--save-csv",
        type=Path,
        default=None,
        help="Optional CSV path for recovery and eigenvalue-error metrics.",
    )
    parser.add_argument(
        "--no-heatmaps",
        action="store_true",
        help="Skip Hessian off-block and error heatmaps.",
    )
    return parser.parse_args()


def parse_orca_atoms(path: Path) -> Atoms:
    """Parse symbols, masses, and positions from ORCA's ``$atoms`` section."""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip() == "$atoms":
                break
        else:
            raise ValueError(f"No $atoms section found in {path}")

        try:
            n_atoms = int(next(handle).strip())
        except StopIteration as exc:
            raise ValueError(f"Missing atom count after $atoms in {path}") from exc

        symbols: list[str] = []
        masses: list[float] = []
        positions_bohr = np.zeros((n_atoms, 3), dtype=float)
        for atom_idx in range(n_atoms):
            try:
                parts = next(handle).split()
            except StopIteration as exc:
                raise ValueError(f"Unexpected end of $atoms section in {path}") from exc
            if len(parts) < 5:
                raise ValueError(f"Malformed $atoms row: {' '.join(parts)!r}")
            symbols.append(parts[0])
            masses.append(float(parts[1]))
            positions_bohr[atom_idx] = [float(parts[2]), float(parts[3]), float(parts[4])]

    atoms = Atoms(symbols=symbols, positions=positions_bohr * BOHR_TO_ANGSTROM)
    atoms.set_masses(masses)
    return atoms


def create_sparsity_pattern_from_cutoff_positions(
    positions: np.ndarray, cutoff: float
) -> tuple[np.ndarray, int]:
    pos_np = np.asarray(positions, dtype=float)
    num_atoms = pos_np.shape[0]
    diff = pos_np[:, np.newaxis, :] - pos_np[np.newaxis, :, :]
    distances = np.linalg.norm(diff, axis=2)
    adjacency = coo_matrix((distances <= cutoff).astype(float))
    expanded_sparsity = kron(adjacency, coo_matrix(np.ones((3, 3)))).tocoo()
    edges = np.vstack((expanded_sparsity.row, expanded_sparsity.col)).T
    return edges.astype(np.intp, copy=False), num_atoms * 3


def max_pair_distance(positions: np.ndarray) -> float:
    pos_np = np.asarray(positions, dtype=float)
    diff = pos_np[:, np.newaxis, :] - pos_np[np.newaxis, :, :]
    return float(np.linalg.norm(diff, axis=2).max())


def hessian_tensor_to_numpy(hessian) -> np.ndarray:
    if hasattr(hessian, "detach"):
        return np.asarray(hessian.detach().cpu().numpy(), dtype=float)
    if hasattr(hessian, "toarray"):
        return np.asarray(hessian.toarray(), dtype=float)
    return np.asarray(hessian, dtype=float)


def block_diag_3x3_per_atom(hessian: np.ndarray, n_atoms: int) -> np.ndarray:
    n_dof = hessian.shape[0]
    if n_dof != 3 * n_atoms:
        raise ValueError(f"Hessian shape {hessian.shape} inconsistent with {n_atoms} atoms")
    out = np.zeros_like(hessian, dtype=float)
    for atom_idx in range(n_atoms):
        block = slice(3 * atom_idx, 3 * (atom_idx + 1))
        out[block, block] = hessian[block, block]
    return out


def get_vibrational_basis_from_eckart(atoms_obj: Atoms, tol: float = 1e-10) -> np.ndarray:
    masses = np.asarray(atoms_obj.get_masses(), dtype=float)
    coords = np.asarray(atoms_obj.get_positions(), dtype=float)
    n_atoms = masses.shape[0]
    ndof = 3 * n_atoms
    center_of_mass = np.average(coords, axis=0, weights=masses)
    coords_rel = coords - center_of_mass
    sqrt_m = np.sqrt(masses)

    rigid_modes = []
    for axis in range(3):
        mode = np.zeros(ndof, dtype=float)
        for atom_idx in range(n_atoms):
            mode[3 * atom_idx + axis] = sqrt_m[atom_idx]
        rigid_modes.append(mode)

    unit_axes = np.eye(3)
    for axis in range(3):
        mode = np.zeros(ndof, dtype=float)
        for atom_idx in range(n_atoms):
            rot_vec = np.cross(unit_axes[axis], coords_rel[atom_idx])
            mode[3 * atom_idx : 3 * atom_idx + 3] = sqrt_m[atom_idx] * rot_vec
        rigid_modes.append(mode)

    rigid_mat = np.column_stack(rigid_modes)
    q_rigid, r_rigid = np.linalg.qr(rigid_mat, mode="reduced")
    rigid_rank = int(np.sum(np.abs(np.diag(r_rigid)) > tol))
    q_full, _ = np.linalg.qr(q_rigid[:, :rigid_rank], mode="complete")
    return q_full[:, rigid_rank:]


def mass_weight_hessian(hessian: np.ndarray, masses: np.ndarray) -> np.ndarray:
    inv_sqrt_m = 1.0 / np.sqrt(np.asarray(masses, dtype=float))
    inv_sqrt_m_xyz = np.repeat(inv_sqrt_m, 3)
    return (inv_sqrt_m_xyz[:, None] * hessian) * inv_sqrt_m_xyz[None, :]


def projected_vibrational_eigvals(
    hessian: np.ndarray, masses: np.ndarray, vib_basis: np.ndarray
) -> np.ndarray:
    hessian_mw = mass_weight_hessian(hessian, masses)
    hessian_vib = vib_basis.T @ hessian_mw @ vib_basis
    hessian_vib = 0.5 * (hessian_vib + hessian_vib.T)
    return np.linalg.eigvalsh(hessian_vib)


def max_error_location(error_matrix: np.ndarray, atoms: Atoms) -> tuple[float, int, int, int, int, str, str]:
    abs_error = np.abs(error_matrix)
    flat_idx = int(np.argmax(abs_error))
    row, col = np.unravel_index(flat_idx, abs_error.shape)
    atom_i, coord_i = divmod(int(row), 3)
    atom_j, coord_j = divmod(int(col), 3)
    symbols = atoms.get_chemical_symbols()
    return (
        float(abs_error[row, col]),
        atom_i,
        coord_i,
        atom_j,
        coord_j,
        symbols[atom_i],
        symbols[atom_j],
    )


def _mean_rel_err_label(n_terms: int | None = None) -> str:
    if n_terms is None:
        return (
            r"$\frac{1}{N_{\mathrm{eig}}}\sum_i"
            r"\frac{|\lambda_i-\lambda_i^{\mathrm{ORCA}}|}{|\lambda_i^{\mathrm{ORCA}}|}$"
        )
    return (
        rf"$\frac{{1}}{{{n_terms}}}\sum_{{i=1}}^{{{n_terms}}}"
        r"\frac{|\lambda_i-\lambda_i^{\mathrm{ORCA}}|}{|\lambda_i^{\mathrm{ORCA}}|}$"
    )


def _plot_marker(alg: str) -> str:
    return MARKERS.get(alg, "o")


def _style_plot_ticks() -> None:
    plt.xticks(fontsize=TICK_FONTSIZE)
    plt.yticks(fontsize=TICK_FONTSIZE)


def get_valid_xy(x_vals, y_vals, require_positive_y: bool = False) -> tuple[np.ndarray, np.ndarray]:
    x_arr = np.asarray(x_vals, dtype=float)
    y_arr = np.asarray(y_vals, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if require_positive_y:
        mask &= y_arr > 0
    return x_arr[mask], y_arr[mask]


def get_valid_xy_sorted_by_x(
    x_vals, y_vals, require_positive_y: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    x_arr, y_arr = get_valid_xy(x_vals, y_vals, require_positive_y=require_positive_y)
    if x_arr.size == 0:
        return x_arr, y_arr
    order = np.argsort(x_arr, kind="stable")
    return x_arr[order], y_arr[order]


def _panel_side_inches(n_dof: int, inches_per_dof: float = 0.038) -> float:
    return float(np.clip(n_dof * inches_per_dof, 8.0, 26.0))


def _imshow_abs_linear(ax, data: np.ndarray, title: str, cbar_label: str) -> None:
    abs_data = np.abs(np.asarray(data, dtype=float))
    vmax = float(np.nanmax(abs_data)) if abs_data.size else 1.0
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0
    im = ax.imshow(
        abs_data,
        vmin=0.0,
        vmax=vmax,
        cmap="magma",
        aspect="equal",
        interpolation="nearest",
        rasterized=True,
    )
    ax.set_title(title)
    ax.set_xlabel("column (Cartesian DOF)")
    ax.set_ylabel("row (Cartesian DOF)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)


def save_line_plot(
    output_path: Path,
    *,
    x_by_alg: dict[str, np.ndarray],
    y_by_alg: dict[str, np.ndarray],
    xlabel: str,
    ylabel: str,
    reference_x: float | None = None,
    reference_y: float | None = None,
    reference_label: str = "Full Hessian rows",
    log_y: bool = True,
) -> None:
    plt.figure(figsize=LINE_FIGSIZE)
    for alg, x_vals in x_by_alg.items():
        y_vals = y_by_alg[alg]
        vx, vy = get_valid_xy_sorted_by_x(x_vals, y_vals, require_positive_y=log_y)
        if vy.size > 0:
            plt.plot(
                vx,
                vy,
                marker=_plot_marker(alg),
                color=COLORS.get(alg),
                label=alg.replace("_", " ").title(),
                linewidth=LINEWIDTH,
                markersize=MARKERSIZE,
            )
    if reference_x is not None:
        plt.axvline(
            x=reference_x,
            color=COLORS["naive"],
            linestyle="--",
            linewidth=REF_LINEWIDTH,
            label=reference_label,
            zorder=10,
        )
        xmin, xmax = plt.xlim()
        if reference_x <= xmax:
            pad = max(0.05 * (max(xmax, reference_x) - min(xmin, reference_x)), 1.0)
            plt.xlim(min(xmin, reference_x - pad), max(xmax, reference_x + pad))
    if reference_y is not None:
        plt.axhline(
            y=reference_y,
            color=COLORS["naive"],
            linestyle="--",
            linewidth=REF_LINEWIDTH,
            label=reference_label,
            zorder=10,
        )
        ymin, ymax = plt.ylim()
        if reference_y <= ymax:
            pad = max(0.05 * (max(ymax, reference_y) - min(ymin, reference_y)), 1.0)
            plt.ylim(min(ymin, reference_y - pad), max(ymax, reference_y + pad))
    plt.xlabel(xlabel, fontsize=LABEL_FONTSIZE)
    plt.ylabel(ylabel, fontsize=LABEL_FONTSIZE)
    plt.legend(fontsize=LEGEND_FONTSIZE)
    plt.grid(True, alpha=0.3)
    if log_y:
        plt.yscale("log", nonpositive="mask")
    _style_plot_ticks()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()
    hessian_path = args.hessian.resolve()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = hessian_path.parent / "reanalyze"
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    cutoffs = np.array([float(x) for x in args.cutoffs.split(",") if x.strip()], dtype=float)
    algorithms = [x.strip().lower() for x in args.algorithms.split(",") if x.strip()]
    if not algorithms:
        raise ValueError("At least one ColPack algorithm is required")

    print("=" * 80)
    print("Reanalyzing ORCA Hessian with sparse recovery from emulated HVPs")
    print("=" * 80)
    print(f"Hessian: {hessian_path}")
    print(f"Output:  {output_dir}")

    hessian_ref = parse_orca_hessian(hessian_path)
    hessian_ref = 0.5 * (hessian_ref + hessian_ref.T)
    atoms = parse_orca_atoms(hessian_path)
    positions = np.asarray(atoms.get_positions(), dtype=float)
    masses = np.asarray(atoms.get_masses(), dtype=float)
    n_atoms = len(atoms)
    n_dof = 3 * n_atoms
    if hessian_ref.shape != (n_dof, n_dof):
        raise ValueError(f"Parsed Hessian shape {hessian_ref.shape} does not match {n_dof} Cartesian DOFs")

    max_dist = max_pair_distance(positions)
    full_edges = n_dof * n_dof
    full_hessian_rows = n_dof
    print(f"Atoms: {n_atoms}, DOFs: {n_dof}")
    print(f"Maximum interatomic distance: {max_dist:.6f} Angstrom")
    print(f"Complete atom-pair cutoff threshold: >= {max_dist:.6f} Angstrom")
    print(f"Cutoffs: {cutoffs.tolist()}")
    print(f"Algorithms: {algorithms}")

    x_t = torch.zeros(n_dof, dtype=torch.float64)
    hessian_ref_t = torch.tensor(hessian_ref, dtype=torch.float64)

    def hvp_from_true_hessian(x_tensor: torch.Tensor, v_tensor: torch.Tensor) -> torch.Tensor:
        del x_tensor
        return (hessian_ref_t @ v_tensor.reshape(-1)).reshape_as(v_tensor)

    errors = {alg: [] for alg in algorithms}
    n_hvps = {alg: [] for alg in algorithms}
    timings_s = {alg: [] for alg in algorithms}
    hessians = {alg: [] for alg in algorithms}
    csv_rows = [
        "cutoff,pattern_fraction,pattern_is_dense,algorithm,max_abs_hessian_element_error,"
        "mean_rel_eig_error_first,mean_rel_eig_error_all,num_hvps,elapsed_s,"
        "row,col,atom_i,coord_i,atom_j,coord_j,symbol_i,symbol_j"
    ]

    print()
    header = ["Cutoff", "Pattern", *[alg.title() for alg in algorithms]]
    print(" ".join(f"{h:<24}" for h in header))
    print("-" * (24 * len(header)))

    for cutoff in cutoffs:
        edges, n_pat = create_sparsity_pattern_from_cutoff_positions(positions, float(cutoff))
        assert n_pat == n_dof
        pattern_fraction = len(edges) / full_edges
        pattern_status = "dense" if len(edges) == full_edges else f"{pattern_fraction:.3%}"
        row_summary = [f"{cutoff:.3f}", pattern_status]
        for alg in algorithms:
            t0 = time.perf_counter()
            h_out, stats = compute_sparse_hessian_from_hvp_fn(
                hvp_from_true_hessian,
                x_t,
                edges,
                n_dof,
                coloring_algorithm=alg,
                ordering=COLORING_ORDERING,
                verify=False,
                return_stats=True,
                hessian_return_type="dense",
            )
            elapsed_s = time.perf_counter() - t0
            h_np = hessian_tensor_to_numpy(h_out)
            h_np = 0.5 * (h_np + h_np.T)
            hessians[alg].append(h_np)

            err = hessian_ref - h_np
            max_abs_err, atom_i, coord_i, atom_j, coord_j, sym_i, sym_j = max_error_location(err, atoms)
            n_hvp = int(stats["num_hvps"])
            errors[alg].append(max_abs_err)
            n_hvps[alg].append(n_hvp)
            timings_s[alg].append(elapsed_s)
            coord_labels = ("x", "y", "z")
            loc = f"{atom_i}{sym_i}{coord_labels[coord_i]}-{atom_j}{sym_j}{coord_labels[coord_j]}"
            row_summary.append(f"{max_abs_err:.3e}/{n_hvp} HVP/{loc}/{elapsed_s:.2f}s")
        print(" ".join(f"{x:<24}" for x in row_summary), flush=True)

    print("\nComputing Eckart-projected relative eigenvalue errors...")
    vib_basis = get_vibrational_basis_from_eckart(atoms)
    eigvals_ref = np.sort(projected_vibrational_eigvals(hessian_ref, masses, vib_basis))
    eigvals_ref_first = eigvals_ref[: args.first_n_eigvals]
    eigenval_errors = {alg: [] for alg in algorithms}
    eigenval_errors_all = {alg: [] for alg in algorithms}

    for i, cutoff in enumerate(cutoffs):
        edges, _ = create_sparsity_pattern_from_cutoff_positions(positions, float(cutoff))
        pattern_fraction = len(edges) / full_edges
        for alg in algorithms:
            h_np = hessians[alg][i]
            eig_sparse = np.sort(projected_vibrational_eigvals(h_np, masses, vib_basis))
            eig_sparse_first = eig_sparse[: args.first_n_eigvals]
            rel_first = np.abs(eigvals_ref_first - eig_sparse_first) / (np.abs(eigvals_ref_first) + 1e-5)
            rel_all = np.abs(eigvals_ref - eig_sparse) / (np.abs(eigvals_ref) + FULL_SPECTRUM_REG_EPS)
            mean_rel_first = float(np.mean(rel_first) + EIGENVAL_ERROR_PLOT_EPS)
            mean_rel_all = float(np.mean(rel_all) + EIGENVAL_ERROR_PLOT_EPS)
            eigenval_errors[alg].append(mean_rel_first)
            eigenval_errors_all[alg].append(mean_rel_all)

            err = hessian_ref - h_np
            max_abs_err, atom_i, coord_i, atom_j, coord_j, sym_i, sym_j = max_error_location(err, atoms)
            row_idx = atom_i * 3 + coord_i
            col_idx = atom_j * 3 + coord_j
            csv_rows.append(
                f"{cutoff:.12g},{pattern_fraction:.12g},{len(edges) == full_edges},{alg},"
                f"{max_abs_err:.16g},{mean_rel_first:.16g},{mean_rel_all:.16g},"
                f"{int(n_hvps[alg][i])},{timings_s[alg][i]:.12g},"
                f"{row_idx},{col_idx},{atom_i},{coord_i},{atom_j},{coord_j},{sym_i},{sym_j}"
            )

    x_cutoffs = {alg: cutoffs for alg in algorithms}
    hvps_by_alg = {alg: np.asarray(n_hvps[alg], dtype=float) for alg in algorithms}
    errors_by_alg = {alg: np.asarray(errors[alg], dtype=float) for alg in algorithms}
    eig_first_by_alg = {alg: np.asarray(eigenval_errors[alg], dtype=float) for alg in algorithms}
    eig_all_by_alg = {alg: np.asarray(eigenval_errors_all[alg], dtype=float) for alg in algorithms}

    save_line_plot(
        output_dir / f"cutoff_error_comparison_{timestamp}.png",
        x_by_alg=x_cutoffs,
        y_by_alg=errors_by_alg,
        xlabel="Cutoff radius (Angstrom)",
        ylabel="Max absolute error vs ORCA Hessian",
    )
    save_line_plot(
        output_dir / f"cutoff_hvp_comparison_{timestamp}.png",
        x_by_alg=x_cutoffs,
        y_by_alg=hvps_by_alg,
        xlabel="Cutoff radius (Angstrom)",
        ylabel="Number of Hessian-vector products",
        reference_y=float(full_hessian_rows),
        reference_label="Full Hessian rows",
        log_y=False,
    )
    save_line_plot(
        output_dir / f"cutoff_eigenval_comparison_{timestamp}.png",
        x_by_alg=x_cutoffs,
        y_by_alg=eig_first_by_alg,
        xlabel="Cutoff radius (Angstrom)",
        ylabel=_mean_rel_err_label(args.first_n_eigvals),
    )
    save_line_plot(
        output_dir / f"cutoff_eigenval_all_comparison_{timestamp}.png",
        x_by_alg=x_cutoffs,
        y_by_alg=eig_all_by_alg,
        xlabel="Cutoff radius (Angstrom)",
        ylabel=_mean_rel_err_label(None),
    )
    save_line_plot(
        output_dir / f"error_vs_hvp_comparison_{timestamp}.png",
        x_by_alg=hvps_by_alg,
        y_by_alg=errors_by_alg,
        xlabel="Number of Hessian-vector products",
        ylabel="Max absolute error vs ORCA Hessian",
        reference_x=float(full_hessian_rows),
        reference_label="Full Hessian rows",
    )
    save_line_plot(
        output_dir / f"eigenval_vs_hvp_comparison_{timestamp}.png",
        x_by_alg=hvps_by_alg,
        y_by_alg=eig_first_by_alg,
        xlabel="Number of Hessian-vector products",
        ylabel=_mean_rel_err_label(args.first_n_eigvals),
        reference_x=float(full_hessian_rows),
        reference_label="Full Hessian rows",
    )
    save_line_plot(
        output_dir / f"eigenval_all_vs_hvp_comparison_{timestamp}.png",
        x_by_alg=hvps_by_alg,
        y_by_alg=eig_all_by_alg,
        xlabel="Number of Hessian-vector products",
        ylabel=_mean_rel_err_label(None),
        reference_x=float(full_hessian_rows),
        reference_label="Full Hessian rows",
    )

    if not args.no_heatmaps:
        side = _panel_side_inches(n_dof)
        hessian_offdiag = hessian_ref - block_diag_3x3_per_atom(hessian_ref, n_atoms)
        fig_gt, ax_gt = plt.subplots(figsize=(side + 1.6, side + 1.3))
        _imshow_abs_linear(
            ax_gt,
            hessian_offdiag,
            r"$|H_{\mathrm{ORCA}} - \mathrm{blkdiag}_{3\times3}(H_{\mathrm{ORCA}})|$",
            "|value| (linear scale)",
        )
        fig_gt.tight_layout()
        gt_path = output_dir / "hessian_orca_offdiag_abs.png"
        fig_gt.savefig(gt_path, dpi=args.plot_dpi, bbox_inches="tight")
        plt.close(fig_gt)
        print(f"Saved: {gt_path}")

        for i, cutoff in enumerate(cutoffs):
            ncols = len(algorithms)
            fig_err, axes_err = plt.subplots(
                1,
                ncols,
                figsize=(ncols * (side + 1.15) + 0.6, side + 1.25),
                squeeze=False,
            )
            for ax, alg in zip(np.ravel(axes_err), algorithms):
                err_mat = hessian_ref - hessians[alg][i]
                _imshow_abs_linear(
                    ax,
                    err_mat,
                    rf"$|H_{{\mathrm{{ORCA}}}} - H_{{\mathrm{{{alg}}}}}|$ ({cutoff:g} A)",
                    "|error| (linear scale)",
                )
            fig_err.suptitle(f"Sparse recovery error (cutoff = {cutoff:g} Angstrom)", fontsize=12, y=1.02)
            fig_err.tight_layout()
            ctag = str(cutoff).replace(".", "p")
            err_path = output_dir / f"hessian_abs_error_cutoff_{ctag}.png"
            fig_err.savefig(err_path, dpi=args.plot_dpi, bbox_inches="tight")
            plt.close(fig_err)
            print(f"Saved: {err_path}")

    npz_path = output_dir / f"orca_sparse_reanalysis_{timestamp}.npz"
    np.savez_compressed(
        npz_path,
        hessian_ref=hessian_ref,
        cutoffs=cutoffs,
        algorithms=np.asarray(algorithms, dtype=object),
        errors={alg: np.asarray(errors[alg], dtype=float) for alg in algorithms},
        n_hvps={alg: np.asarray(n_hvps[alg], dtype=float) for alg in algorithms},
        timings_s={alg: np.asarray(timings_s[alg], dtype=float) for alg in algorithms},
        eigenval_errors={alg: np.asarray(eigenval_errors[alg], dtype=float) for alg in algorithms},
        eigenval_errors_all={alg: np.asarray(eigenval_errors_all[alg], dtype=float) for alg in algorithms},
        hessian_path=np.asarray([str(hessian_path)], dtype=object),
    )
    print(f"Saved NPZ: {npz_path}")

    csv_path = args.save_csv or output_dir / f"orca_sparse_reanalysis_{timestamp}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text("\n".join(csv_rows) + "\n", encoding="utf-8")
    print(f"Saved CSV: {csv_path}")


if __name__ == "__main__":
    main()
