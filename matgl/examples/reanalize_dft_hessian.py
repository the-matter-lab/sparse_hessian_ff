"""Reanalyze a saved reference Hessian (MLIP, DFT, or xTB) with larger cutoffs.

Does not recompute the dense reference matrix. Loads ``H`` from, in order of
preference:

- ``full_naive_hessian_*.npy`` / ``full_analytical_hessian_*.npy`` /
  ``full_xtb_reference_hessian_*.npy`` (standalone arrays; **preferred**)
- ``cutoff_debug_data_*.npz`` only if it contains ``full_naive_hessian``,
  ``full_reference_hessian``, or ``full_analytical_hessian`` (older MLIP runs
  often have no matrix in the NPZ — use ``export_mlip_naive_hessian.py`` or
  re-run ``sparse_hessian_cutoff_real.py``, which now saves ``full_naive_hessian``)

Then reruns ColPack recovery with emulated HVPs, ``H_true @ v``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from ase.io import read
from scipy.sparse import coo_matrix, kron


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
COLPACK_SRC = REPO_ROOT / "colpackpy" / "src"
if str(COLPACK_SRC) not in sys.path:
    sys.path.insert(0, str(COLPACK_SRC))

from colpackpy.torch_utils import compute_sparse_hessian_from_hvp_fn  # type: ignore[import-not-found]


COLORING_ORDERING = "SMALLEST_LAST"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--structure",
        type=Path,
        default=REPO_ROOT / "data" / "Chignolin" / "Coordinates_9999.xyz",
        help="ASE-readable geometry used for the saved Hessian.",
    )
    parser.add_argument(
        "--hessian",
        type=Path,
        default=None,
        help="Path to a full Hessian .npy or cutoff_debug_data_*.npz (otherwise latest in --output-dir).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "sparse_hessian_results" / "cutoff_dft_experiments",
        help="Directory containing previous cutoff experiment outputs.",
    )
    parser.add_argument(
        "--cutoffs",
        type=str,
        default="30,31,32,33,34,35",
        help="Comma-separated cutoff radii in Angstrom.",
    )
    parser.add_argument(
        "--algorithms",
        type=str,
        default="star,acyclic",
        help="Comma-separated ColPack algorithms to rerun.",
    )
    parser.add_argument(
        "--center-com",
        dest="center_com",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Translate the structure to its center of mass before building cutoff patterns.",
    )
    parser.add_argument(
        "--save-csv",
        type=Path,
        default=None,
        help="Optional CSV path for the max Hessian element error table.",
    )
    parser.add_argument(
        "--reanalize-subdir",
        type=str,
        default="reanalize",
        help="Subdirectory under --output-dir for plots (default: reanalize).",
    )
    parser.add_argument(
        "--plot-dpi",
        type=int,
        default=360,
        help="DPI for saved PNG figures (higher = sharper single-matrix cells).",
    )
    parser.add_argument(
        "--inches-per-dof",
        type=float,
        default=0.038,
        help="Heatmap panel side length ≈ n_dof × this (inches), clamped for very large systems.",
    )
    parser.add_argument(
        "--combined-inches-per-dof",
        type=float,
        default=0.009,
        help="Panel side for --combined-error-plot rows (smaller than single plots).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip saving Hessian heatmap figures.",
    )
    parser.add_argument(
        "--combined-error-plot",
        action="store_true",
        help="Also save one tall multi-cutoff figure hessian_abs_errors_all_cutoffs.png.",
    )
    parser.add_argument(
        "--plot-eigenvalues",
        dest="plot_eigenvalues",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save Eckart-projected eigenvalue scatter + vline plots under reanalize/ (default: on).",
    )
    return parser.parse_args()


def latest_existing(paths: list[Path]) -> Path | None:
    existing = [p for p in paths if p.exists()]
    if not existing:
        return None
    return max(existing, key=lambda p: p.stat().st_mtime)


def find_latest_hessian(output_dir: Path) -> Path:
    """Prefer standalone dense .npy files that are always loadable."""
    dense_latest = latest_existing(
        [
            *output_dir.glob("full_naive_hessian_*.npy"),
            *output_dir.glob("full_analytical_hessian_*.npy"),
            *output_dir.glob("full_xtb_reference_hessian_*.npy"),
        ]
    )
    if dense_latest is not None:
        return dense_latest

    npz_latest = latest_existing([*output_dir.glob("cutoff_debug_data_*.npz")])
    if npz_latest is None:
        raise FileNotFoundError(
            f"No dense Hessian .npy or cutoff_debug_data_*.npz found in {output_dir}"
        )
    return npz_latest


def load_hessian(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.asarray(np.load(path), dtype=float)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=True) as data:
            for key in ("full_naive_hessian", "full_reference_hessian", "full_analytical_hessian"):
                if key in data:
                    return np.asarray(data[key], dtype=float)
            keys = ", ".join(data.files)
            raise KeyError(
                f"{path} does not contain a full Hessian array "
                f"(expected one of full_naive_hessian, full_reference_hessian, full_analytical_hessian). "
                f"Available keys: {keys}"
            )
    raise ValueError(f"Unsupported Hessian file type: {path}")


def create_sparsity_pattern_from_cutoff_positions(
    positions: np.ndarray, cutoff: float
) -> tuple[np.ndarray, int]:
    pos_np = np.asarray(positions, dtype=float)
    num_atoms = pos_np.shape[0]
    diff = pos_np[:, np.newaxis, :] - pos_np[np.newaxis, :, :]
    distances = np.linalg.norm(diff, axis=2)
    adjacency = (distances <= cutoff).astype(float)
    expanded_sparsity = kron(coo_matrix(adjacency), coo_matrix(np.ones((3, 3)))).tocoo()
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


def block_diag_3x3_per_atom(H: np.ndarray, n_atoms: int) -> np.ndarray:
    """Keep only per-atom 3×3 blocks on the diagonal; zero elsewhere."""
    n = H.shape[0]
    if n != 3 * n_atoms:
        raise ValueError(f"H shape {H.shape} inconsistent with n_atoms={n_atoms}")
    out = np.zeros_like(H, dtype=float)
    for i in range(n_atoms):
        sl = slice(3 * i, 3 * (i + 1))
        out[sl, sl] = H[sl, sl]
    return out


def get_vibrational_basis_from_eckart(atoms_obj, tol: float = 1e-10) -> np.ndarray:
    """Same construction as ``sparse_hessian_cutoff_dft.py`` (Eckart frame)."""
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
    q_rigid = q_rigid[:, :rigid_rank]
    q_full, _ = np.linalg.qr(q_rigid, mode="complete")
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


def save_eigenvalue_comparison_plots(
    out_dir: Path,
    *,
    cutoff: float,
    alg: str,
    lam_gt: np.ndarray,
    lam_pred: np.ndarray,
    dpi: int,
) -> tuple[Path, Path]:
    """Scatter (GT vs pred) and rug plot: x = λ, uniform-height vertical ticks per mode."""
    ctag = str(cutoff).replace(".", "p")
    color_gt = "#1f77b4"
    color_pred = "#ff7f0e"
    stem_h = 1.0

    fig_s, ax_s = plt.subplots(figsize=(7.0, 6.8))
    ax_s.scatter(
        lam_gt,
        lam_pred,
        s=12,
        alpha=0.55,
        c=color_pred,
        edgecolors="none",
        label=r"$(\lambda_{\mathrm{GT}}, \lambda_{\mathrm{pred}})$",
    )
    lo = min(float(np.min(lam_gt)), float(np.min(lam_pred)))
    hi = max(float(np.max(lam_gt)), float(np.max(lam_pred)))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = lo - 1.0, hi + 1.0
    ax_s.plot([lo, hi], [lo, hi], color="gray", linestyle="--", linewidth=1.2, label="y = x")
    ax_s.set_aspect("equal", adjustable="box")
    ax_s.set_xlabel(r"$\lambda_{\mathrm{GT}}$ (Eckart projected)")
    ax_s.set_ylabel(r"$\lambda_{\mathrm{pred}}$ (Eckart projected)")
    ax_s.set_title(f"Eigenvalues: {alg} @ {cutoff:g} Å")
    ax_s.legend(loc="upper left", fontsize=9)
    ax_s.grid(True, alpha=0.3)
    fig_s.tight_layout()
    scatter_path = out_dir / f"eigenval_scatter_{alg}_cutoff_{ctag}.png"
    fig_s.savefig(scatter_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig_s)

    fig_v, ax_v = plt.subplots(figsize=(10.0, 3.8))
    ax_v.vlines(
        lam_gt,
        0.0,
        stem_h,
        colors=color_gt,
        linewidth=0.7,
        label=r"$\lambda_{\mathrm{GT}}$",
    )
    ax_v.vlines(
        lam_pred,
        -stem_h,
        0.0,
        colors=color_pred,
        linewidth=0.7,
        label=r"$\lambda_{\mathrm{pred}}$",
    )
    ax_v.axhline(0.0, color="black", linewidth=0.65, zorder=0)
    x_lo = min(float(np.min(lam_gt)), float(np.min(lam_pred)))
    x_hi = max(float(np.max(lam_gt)), float(np.max(lam_pred)))
    pad = 0.02 * max(abs(x_hi - x_lo), 1.0)
    ax_v.set_xlim(x_lo - pad, x_hi + pad)
    ax_v.set_ylim(-stem_h * 1.15, stem_h * 1.15)
    ax_v.set_xlabel(r"$\lambda$ (Eckart projected, sorted modes)")
    ax_v.set_ylabel("tick (uniform height)")
    ax_v.set_yticks([-stem_h, 0.0, stem_h])
    ax_v.set_yticklabels([r"pred", "0", r"GT"])
    ax_v.set_title(f"Spectrum rug plot: {alg} @ {cutoff:g} Å")
    ax_v.legend(loc="upper right", fontsize=9)
    ax_v.grid(True, alpha=0.25, axis="x")
    fig_v.tight_layout()
    vline_path = out_dir / f"eigenval_vlines_{alg}_cutoff_{ctag}.png"
    fig_v.savefig(vline_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig_v)

    return scatter_path, vline_path


def _panel_side_inches(n_dof: int, inches_per_dof: float) -> float:
    """Square panel side length so imshow has enough physical size to resolve cells."""
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


def max_error_location(error_matrix: np.ndarray, atoms) -> tuple[float, int, int, int, int, str, str]:
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


def main() -> None:
    args = parse_args()
    cutoffs = np.array([float(x) for x in args.cutoffs.split(",") if x.strip()], dtype=float)
    algorithms = [x.strip().lower() for x in args.algorithms.split(",") if x.strip()]

    hessian_path = args.hessian.resolve() if args.hessian is not None else find_latest_hessian(args.output_dir)
    hessian_ref = load_hessian(hessian_path)

    atoms = read(str(args.structure.resolve()))
    if args.center_com:
        atoms.translate(-atoms.get_center_of_mass())
    positions = atoms.get_positions()
    n_dof = 3 * len(atoms)
    if hessian_ref.shape != (n_dof, n_dof):
        raise ValueError(
            f"Loaded Hessian shape {hessian_ref.shape} does not match structure DOFs {(n_dof, n_dof)}"
        )

    max_dist = max_pair_distance(positions)
    full_edges = n_dof * n_dof
    print("=" * 80)
    print("Reanalyzing saved reference Hessian with larger cutoffs")
    print("=" * 80)
    print(f"Structure: {args.structure.resolve()}")
    print(f"Hessian:   {hessian_path}")
    print(f"Atoms: {len(atoms)}, DOFs: {n_dof}")
    print(f"Maximum interatomic distance: {max_dist:.6f} Å")
    print(f"Complete atom-pair cutoff threshold: >= {max_dist:.6f} Å")
    print(f"Cutoffs: {cutoffs.tolist()}")
    print(f"Algorithms: {algorithms}")

    reanalize_dir = (args.output_dir / args.reanalize_subdir).resolve()
    reanalize_dir.mkdir(parents=True, exist_ok=True)

    x_t = torch.zeros(n_dof, dtype=torch.float64)
    hessian_ref_t = torch.tensor(hessian_ref, dtype=torch.float64)

    def hvp_from_true_hessian(x_tensor: torch.Tensor, v_tensor: torch.Tensor) -> torch.Tensor:
        del x_tensor
        return (hessian_ref_t @ v_tensor.reshape(-1)).reshape_as(v_tensor)

    print()
    header = ["Cutoff", "Pattern", *[alg.title() for alg in algorithms]]
    print(" ".join(f"{h:<24}" for h in header))
    print("-" * (24 * len(header)))

    csv_rows = [
        "cutoff,pattern_fraction,pattern_is_dense,algorithm,max_abs_hessian_element_error,"
        "num_hvps,elapsed_s,row,col,atom_i,coord_i,atom_j,coord_j,symbol_i,symbol_j"
    ]

    recovered: dict[float, dict[str, np.ndarray]] = {}

    for cutoff in cutoffs:
        edges, n_pat = create_sparsity_pattern_from_cutoff_positions(positions, float(cutoff))
        assert n_pat == n_dof
        pattern_fraction = len(edges) / full_edges
        pattern_status = "dense" if len(edges) == full_edges else f"{pattern_fraction:.3%}"
        row = [f"{cutoff:.3f}", pattern_status]
        recovered[float(cutoff)] = {}
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
            recovered[float(cutoff)][alg] = h_np
            err = hessian_ref - h_np
            max_abs_err, atom_i, coord_i, atom_j, coord_j, sym_i, sym_j = max_error_location(err, atoms)
            coord_labels = ("x", "y", "z")
            loc = f"{atom_i}{sym_i}{coord_labels[coord_i]}-{atom_j}{sym_j}{coord_labels[coord_j]}"
            row.append(f"{max_abs_err:.3e}/{int(stats['num_hvps'])} HVP/{loc}/{elapsed_s:.2f}s")
            row_idx = atom_i * 3 + coord_i
            col_idx = atom_j * 3 + coord_j
            csv_rows.append(
                f"{cutoff:.12g},{pattern_fraction:.12g},{len(edges) == full_edges},{alg},"
                f"{max_abs_err:.16g},{int(stats['num_hvps'])},{elapsed_s:.12g},"
                f"{row_idx},{col_idx},{atom_i},{coord_i},{atom_j},{coord_j},{sym_i},{sym_j}"
            )
        print(" ".join(f"{x:<24}" for x in row), flush=True)

    if args.plot_eigenvalues:
        masses = np.asarray(atoms.get_masses(), dtype=float)
        vib = get_vibrational_basis_from_eckart(atoms)
        lam_gt = projected_vibrational_eigvals(hessian_ref, masses, vib)
        print("\nEckart-projected eigenvalue comparison plots (scatter + vertical stems)…")
        for cutoff in cutoffs:
            ckey = float(cutoff)
            for alg in algorithms:
                lam_pred = projected_vibrational_eigvals(recovered[ckey][alg], masses, vib)
                scatter_p, vline_p = save_eigenvalue_comparison_plots(
                    reanalize_dir,
                    cutoff=float(cutoff),
                    alg=alg,
                    lam_gt=lam_gt,
                    lam_pred=lam_pred,
                    dpi=args.plot_dpi,
                )
                print(f"Saved: {scatter_p}")
                print(f"Saved: {vline_p}")

    if not args.no_plots:
        dpi = args.plot_dpi
        side = _panel_side_inches(n_dof, args.inches_per_dof)
        n_at = len(atoms)
        H_gt = hessian_ref
        H_blk = block_diag_3x3_per_atom(H_gt, n_at)
        H_gt_offdiag = H_gt - H_blk

        fig_gt, ax_gt = plt.subplots(figsize=(side + 1.6, side + 1.3))
        _imshow_abs_linear(
            ax_gt,
            H_gt_offdiag,
            r"$|H_{\mathrm{GT}} - \mathrm{blkdiag}_{3\times3}(H_{\mathrm{GT}})|$",
            "|value| (linear scale)",
        )
        fig_gt.tight_layout()
        gt_path = reanalize_dir / "hessian_gt_offdiag_abs.png"
        fig_gt.savefig(gt_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig_gt)
        print(f"\nSaved: {gt_path}")

        for cutoff in cutoffs:
            ckey = float(cutoff)
            w = len(algorithms) * (side + 1.15) + 0.6
            fig_err, axes_err = plt.subplots(1, len(algorithms), figsize=(w, side + 1.25))
            if len(algorithms) == 1:
                axes_err = np.array([axes_err])
            for ax, alg in zip(np.ravel(axes_err), algorithms):
                err_mat = H_gt - recovered[ckey][alg]
                _imshow_abs_linear(
                    ax,
                    err_mat,
                    rf"$|H_{{\mathrm{{GT}}}} - H_{{\mathrm{{{alg}}}}}|$  ({cutoff:g} Å)",
                    "|error| (linear scale)",
                )
            fig_err.suptitle(f"Sparse recovery error (cutoff = {cutoff:g} Å)", fontsize=12, y=1.02)
            fig_err.tight_layout()
            ctag = str(cutoff).replace(".", "p")
            err_path = reanalize_dir / f"hessian_abs_error_cutoff_{ctag}.png"
            fig_err.savefig(err_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig_err)
            print(f"Saved: {err_path}")

        if args.combined_error_plot and len(cutoffs) > 0:
            nrows = len(cutoffs)
            ncols = len(algorithms)
            row_side = _panel_side_inches(n_dof, args.combined_inches_per_dof)
            fig_all, axes_all = plt.subplots(
                nrows,
                ncols,
                figsize=(
                    ncols * (row_side + 1.1) + 0.5,
                    nrows * (row_side + 0.85) + 0.5,
                ),
                squeeze=False,
            )
            for ri, cutoff in enumerate(cutoffs):
                ckey = float(cutoff)
                for ci, alg in enumerate(algorithms):
                    err_mat = H_gt - recovered[ckey][alg]
                    _imshow_abs_linear(
                        axes_all[ri, ci],
                        err_mat,
                        rf"{alg}  {cutoff:g} Å",
                        "|error| (linear)",
                    )
            fig_all.suptitle(r"$|H_{\mathrm{GT}} - H_{\mathrm{recovered}}|$ by cutoff", fontsize=13, y=1.002)
            fig_all.tight_layout()
            comb_path = reanalize_dir / "hessian_abs_errors_all_cutoffs.png"
            fig_all.savefig(comb_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig_all)
            print(f"Saved: {comb_path}")

    if args.save_csv is not None:
        args.save_csv.parent.mkdir(parents=True, exist_ok=True)
        args.save_csv.write_text("\n".join(csv_rows) + "\n", encoding="utf-8")
        print(f"\nSaved CSV: {args.save_csv}")


if __name__ == "__main__":
    main()
