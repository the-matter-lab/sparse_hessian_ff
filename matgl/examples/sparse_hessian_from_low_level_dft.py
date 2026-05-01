"""Recover dense reference Hessians using sparsity patterns from a low-level Hessian.

This script is the same basic experiment as ``sparse_hessian_cutoff_dft.py`` and
``reanalize_dft_hessian.py``: no electronic-structure calculations are run here.
Instead, a saved high-level/target Hessian is used to emulate HVPs as
``H_target @ v``.

The difference is how the sparsity pattern is built. Rather than using an
interatomic cutoff, this script loads a saved low-level Hessian (typically xTB),
thresholds it as ``abs(H_low_level) > epsilon`` for logarithmically spaced
epsilon values, and uses that pattern for sparse Hessian recovery of the target
Hessian.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from ase.io import read  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
COLPACK_SRC = REPO_ROOT / "colpackpy" / "src"
if str(COLPACK_SRC) not in sys.path:
    sys.path.insert(0, str(COLPACK_SRC))

from colpackpy.torch_utils import compute_sparse_hessian_from_hvp_fn  # type: ignore[import-not-found]  # noqa: E402

FULL_SPECTRUM_REG_EPS = 1e-5
LABEL_FONTSIZE = int(round(14 * 1.3))
LEGEND_FONTSIZE = int(round(12 * 1.4))
TICK_FONTSIZE = 16
LINEWIDTH = 4.5
REF_LINEWIDTH = 2.0
MARKERSIZE = 10
LINE_FIGSIZE = (10, 6)
COLORING_ORDERING = "SMALLEST_LAST"

COLORS = {
    "naive": "#4e79a7",
    "star": "#f28e2b",
    "acyclic": "#59a14f",
    "distance_two": "#b07aa1",
}
MARKERS = {
    "naive": "o",
    "star": "s",
    "acyclic": "^",
    "distance_two": "D",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--structure",
        type=Path,
        default=REPO_ROOT / "data" / "Chignolin" / "Coordinates_9999.xyz",
        help="ASE-readable geometry used for both Hessians.",
    )
    p.add_argument(
        "--target-hessian",
        type=Path,
        default=None,
        help="Saved target Hessian (.npy or cutoff_debug_data_*.npz). If omitted, use latest in --target-output-dir.",
    )
    p.add_argument(
        "--target-output-dir",
        type=Path,
        default=SCRIPT_DIR / "sparse_hessian_results" / "cutoff_dft_experiments_chingolin_pbe",
        help="Directory containing the high-level/target experiment Hessian.",
    )
    p.add_argument(
        "--pattern-hessian",
        type=Path,
        default=None,
        help="Saved low-level pattern Hessian (.npy or cutoff_debug_data_*.npz). If omitted, search xTB output dirs.",
    )
    p.add_argument(
        "--pattern-output-dir",
        type=Path,
        default=None,
        help="Directory containing the low-level/xTB Hessian. If omitted, tries common xTB result folders.",
    )
    p.add_argument(
        "--eps-min",
        type=float,
        default=1e-9,
        help="Minimum absolute low-level Hessian threshold.",
    )
    p.add_argument(
        "--eps-max",
        type=float,
        default=1.0,
        help="Maximum absolute low-level Hessian threshold.",
    )
    p.add_argument(
        "--eps-steps",
        type=int,
        default=19,
        help="Number of logarithmic epsilon values between --eps-min and --eps-max.",
    )
    p.add_argument(
        "--epsilons",
        type=str,
        default=None,
        help="Optional comma-separated epsilon values; overrides --eps-min/--eps-max/--eps-steps.",
    )
    p.add_argument(
        "--algorithms",
        type=str,
        default="star,acyclic",
        help="Comma-separated ColPack algorithms to run.",
    )
    p.add_argument("--first-n-eigvals", type=int, default=10)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <target-output-dir>/low_level_pattern_reanalize.",
    )
    p.add_argument(
        "--center-com",
        dest="center_com",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Translate the structure to center of mass before Eckart projection.",
    )
    p.add_argument(
        "--include-diagonal",
        dest="include_diagonal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Always include diagonal Hessian entries in the thresholded pattern.",
    )
    p.add_argument(
        "--plot-data-dump",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save compressed NPZ plot/debug data.",
    )
    return p.parse_args()


def _style_plot_ticks() -> None:
    plt.xticks(fontsize=TICK_FONTSIZE)
    plt.yticks(fontsize=TICK_FONTSIZE)


def _plot_marker(alg: str) -> str:
    return MARKERS.get(alg, "o")


def _mean_rel_err_label(n_terms: int | None = None) -> str:
    if n_terms is None:
        return (
            r"$\frac{1}{N_{\mathrm{eig}}}\sum_i"
            r"\frac{|\lambda_i-\lambda_i^{\mathrm{GT}}|}{|\lambda_i^{\mathrm{GT}}|}$"
        )
    return (
        rf"$\frac{{1}}{{{n_terms}}}\sum_{{i=1}}^{{{n_terms}}}"
        r"\frac{|\lambda_i-\lambda_i^{\mathrm{GT}}|}{|\lambda_i^{\mathrm{GT}}|}$"
    )


def latest_existing(paths: list[Path]) -> Path | None:
    existing = [p for p in paths if p.exists()]
    if not existing:
        return None
    return max(existing, key=lambda p: p.stat().st_mtime)


def _hessian_candidates(output_dir: Path) -> list[Path]:
    return [
        *output_dir.glob("full_naive_hessian_*.npy"),
        *output_dir.glob("full_analytical_hessian_*.npy"),
        *output_dir.glob("full_xtb_reference_hessian_*.npy"),
        *output_dir.glob("cutoff_debug_data_*.npz"),
    ]


def find_latest_hessian(output_dir: Path) -> Path:
    dense_latest = latest_existing([p for p in _hessian_candidates(output_dir) if p.suffix == ".npy"])
    if dense_latest is not None:
        return dense_latest
    npz_latest = latest_existing([p for p in _hessian_candidates(output_dir) if p.suffix == ".npz"])
    if npz_latest is None:
        raise FileNotFoundError(f"No saved Hessian candidate found in {output_dir}")
    return npz_latest


def find_default_pattern_hessian() -> Path:
    candidates = [
        SCRIPT_DIR / "sparse_hessian_results" / "cutoff_experiments_chingolin_xtb",
        SCRIPT_DIR / "sparse_hessian_results" / "cutoff_xtb_experiments",
    ]
    latest = latest_existing(
        [
            find_latest_hessian(d)
            for d in candidates
            if d.exists() and _hessian_candidates(d)
        ]
    )
    if latest is None:
        raise FileNotFoundError(
            "Could not find a default xTB/pattern Hessian. Pass --pattern-hessian or --pattern-output-dir."
        )
    return latest


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
                f"(expected full_naive_hessian, full_reference_hessian, or full_analytical_hessian). "
                f"Available keys: {keys}"
            )
    raise ValueError(f"Unsupported Hessian file type: {path}")


def parse_epsilons(args: argparse.Namespace) -> np.ndarray:
    if args.epsilons is not None:
        eps = np.array([float(x) for x in args.epsilons.split(",") if x.strip()], dtype=float)
    else:
        if args.eps_min <= 0 or args.eps_max <= 0:
            raise ValueError("--eps-min and --eps-max must be positive for log spacing")
        eps = np.logspace(np.log10(args.eps_min), np.log10(args.eps_max), int(args.eps_steps))
    if eps.size == 0 or np.any(~np.isfinite(eps)) or np.any(eps <= 0):
        raise ValueError(f"Invalid epsilon values: {eps}")
    return eps


def threshold_pattern_from_hessian(
    hessian_pattern: np.ndarray, epsilon: float, include_diagonal: bool
) -> tuple[np.ndarray, int]:
    H = np.asarray(hessian_pattern, dtype=float)
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError(f"Pattern Hessian must be square, got shape {H.shape}")
    mask = np.abs(H) > float(epsilon)
    mask |= mask.T
    if include_diagonal:
        np.fill_diagonal(mask, True)
    row, col = np.nonzero(mask)
    edges = np.column_stack((row, col)).astype(np.intp, copy=False)
    return edges, H.shape[0]


def hessian_tensor_to_numpy(hessian) -> np.ndarray:
    if hasattr(hessian, "detach"):
        return np.asarray(hessian.detach().cpu().numpy(), dtype=float)
    if hasattr(hessian, "toarray"):
        return np.asarray(hessian.toarray(), dtype=float)
    return np.asarray(hessian, dtype=float)


def get_vibrational_basis_from_eckart(atoms_obj, tol: float = 1e-10) -> np.ndarray:
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


def get_valid_xy(x_vals, y_vals, require_positive_y: bool = False):
    x_arr = np.asarray(x_vals, dtype=float)
    y_arr = np.asarray(y_vals, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if require_positive_y:
        mask &= y_arr > 0
    return x_arr[mask], y_arr[mask]


def get_valid_xy_sorted_by_x(x_vals, y_vals, require_positive_y: bool = False):
    x_arr, y_arr = get_valid_xy(x_vals, y_vals, require_positive_y=require_positive_y)
    if x_arr.size == 0:
        return x_arr, y_arr
    order = np.argsort(x_arr, kind="stable")
    return x_arr[order], y_arr[order]


def plot_eigen_vs_epsilon(
    output_dir: Path,
    epsilons: np.ndarray,
    values: dict[str, list[float]],
    algorithms: list[str],
    *,
    ylabel: str,
    filename: str,
) -> None:
    plt.figure(figsize=LINE_FIGSIZE)
    for alg in algorithms:
        vx, vy = get_valid_xy(epsilons, values[alg], require_positive_y=True)
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
    plt.xlabel(r"Low-level threshold $\epsilon$ in $|H_{\mathrm{low}}| > \epsilon$", fontsize=LABEL_FONTSIZE)
    plt.ylabel(ylabel, fontsize=LABEL_FONTSIZE)
    plt.xscale("log")
    plt.yscale("log", nonpositive="mask")
    plt.gca().invert_xaxis()
    plt.legend(fontsize=LEGEND_FONTSIZE)
    plt.grid(True, alpha=0.3)
    _style_plot_ticks()
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
    plt.close()


def plot_eigen_vs_hvp(
    output_dir: Path,
    n_hvps: dict[str, list[float]],
    values: dict[str, list[float]],
    algorithms: list[str],
    full_hessian_rows: int,
    *,
    ylabel: str,
    filename: str,
) -> None:
    plt.figure(figsize=LINE_FIGSIZE)
    for alg in algorithms:
        vx, vy = get_valid_xy_sorted_by_x(n_hvps[alg], values[alg], require_positive_y=True)
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
    plt.xlabel("Number of Hessian-vector products", fontsize=LABEL_FONTSIZE)
    plt.ylabel(ylabel, fontsize=LABEL_FONTSIZE)
    plt.axvline(
        x=full_hessian_rows,
        color=COLORS["naive"],
        linestyle="--",
        linewidth=REF_LINEWIDTH,
        label="Full Hessian rows",
    )
    plt.yscale("log", nonpositive="mask")
    plt.legend(fontsize=LEGEND_FONTSIZE)
    plt.grid(True, alpha=0.3)
    _style_plot_ticks()
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()
    epsilons = parse_epsilons(args)
    algorithms = [x.strip().lower() for x in args.algorithms.split(",") if x.strip()]

    target_path = (
        args.target_hessian.resolve()
        if args.target_hessian is not None
        else find_latest_hessian(args.target_output_dir)
    )
    if args.pattern_hessian is not None:
        pattern_path = args.pattern_hessian.resolve()
    elif args.pattern_output_dir is not None:
        pattern_path = find_latest_hessian(args.pattern_output_dir)
    else:
        pattern_path = find_default_pattern_hessian()

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = args.target_output_dir / "low_level_pattern_reanalize"
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    H_target = load_hessian(target_path)
    H_low = load_hessian(pattern_path)
    if H_target.shape != H_low.shape:
        raise ValueError(f"Shape mismatch: target {H_target.shape}, low-level {H_low.shape}")

    atoms = read(str(args.structure.resolve()))
    if args.center_com:
        atoms.translate(-atoms.get_center_of_mass())
    n_dof = 3 * len(atoms)
    if H_target.shape != (n_dof, n_dof):
        raise ValueError(
            f"Loaded Hessian shape {H_target.shape} does not match structure DOFs {(n_dof, n_dof)}"
        )

    masses = np.asarray(atoms.get_masses(), dtype=float)
    vib_basis = get_vibrational_basis_from_eckart(atoms)
    eigvals_ref = np.sort(projected_vibrational_eigvals(H_target, masses, vib_basis))
    eigvals_ref_first = eigvals_ref[: args.first_n_eigvals]

    device = torch.device("cpu")
    x_t = torch.zeros(n_dof, dtype=torch.float64, device=device)
    H_target_t = torch.tensor(H_target, dtype=torch.float64, device=device)

    def hvp_from_true_hessian(x_tensor: torch.Tensor, v_tensor: torch.Tensor) -> torch.Tensor:
        del x_tensor
        return (H_target_t @ v_tensor.reshape(-1)).reshape_as(v_tensor)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_edges = n_dof * n_dof
    full_hessian_rows = n_dof
    errors = {alg: [] for alg in algorithms}
    n_hvps = {alg: [] for alg in algorithms}
    timings_s = {alg: [] for alg in algorithms}
    statuses = {alg: [] for alg in algorithms}
    pattern_edges = []
    pattern_fractions = []
    eigenval_errors = {alg: [] for alg in algorithms}
    eigenval_errors_all = {alg: [] for alg in algorithms}
    hessians: dict[str, list[np.ndarray | None]] = {alg: [] for alg in algorithms}

    print("=" * 80)
    print("Sparse Hessian recovery from low-level thresholded Hessian pattern")
    print("=" * 80)
    print(f"Structure:      {args.structure.resolve()}")
    print(f"Target Hessian: {target_path}")
    print(f"Pattern Hessian:{pattern_path}")
    print(f"Output dir:     {output_dir}")
    print(f"Atoms: {len(atoms)}, DOFs: {n_dof}")
    print(f"Epsilons: {epsilons.tolist()}")
    print(f"Algorithms: {algorithms}")
    print()

    header = ["Epsilon", "Pattern", *[alg.title() for alg in algorithms]]
    print(" ".join(f"{h:<26}" for h in header))
    print("-" * (26 * len(header)))

    for eps in epsilons:
        edges, n_pat = threshold_pattern_from_hessian(H_low, float(eps), args.include_diagonal)
        assert n_pat == n_dof
        pattern_edges.append(int(len(edges)))
        pattern_fraction = len(edges) / full_edges
        pattern_fractions.append(float(pattern_fraction))
        pattern_status = "dense" if len(edges) == full_edges else f"{pattern_fraction:.3%}"
        row = [f"{eps:.3e}", pattern_status]

        for alg in algorithms:
            try:
                t0 = time.perf_counter()
                H_out, stats = compute_sparse_hessian_from_hvp_fn(
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
                H_np = hessian_tensor_to_numpy(H_out)
                H_np = 0.5 * (H_np + H_np.T)
                hessians[alg].append(H_np)

                max_error = float(np.abs(H_target - H_np).max())
                eig_sparse = np.sort(projected_vibrational_eigvals(H_np, masses, vib_basis))
                eig_sparse_first = eig_sparse[: args.first_n_eigvals]
                pct_diff_first = np.abs(eigvals_ref_first - eig_sparse_first) / (
                    np.abs(eigvals_ref_first) + 1e-5
                )
                pct_diff_all = np.abs(eigvals_ref - eig_sparse) / (
                    np.abs(eigvals_ref) + FULL_SPECTRUM_REG_EPS
                )

                errors[alg].append(max_error)
                n_hvps[alg].append(float(stats["num_hvps"]))
                timings_s[alg].append(float(elapsed_s))
                statuses[alg].append("ok")
                eigenval_errors[alg].append(float(pct_diff_first.mean() + 1e-7))
                eigenval_errors_all[alg].append(float(pct_diff_all.mean() + 1e-7))
                row.append(f"{eigenval_errors_all[alg][-1]:.3e}/{int(stats['num_hvps'])} HVP/{elapsed_s:.2f}s")
            except Exception as exc:
                hessians[alg].append(None)
                errors[alg].append(np.nan)
                n_hvps[alg].append(np.nan)
                timings_s[alg].append(np.nan)
                statuses[alg].append("failed")
                eigenval_errors[alg].append(np.nan)
                eigenval_errors_all[alg].append(np.nan)
                row.append(f"failed: {exc}")

        print(" ".join(f"{x:<26}" for x in row), flush=True)

    print("\nSaving plots and data...")
    plot_eigen_vs_epsilon(
        output_dir,
        epsilons,
        eigenval_errors,
        algorithms,
        ylabel=_mean_rel_err_label(args.first_n_eigvals),
        filename=f"epsilon_eigenval_comparison_{timestamp}.png",
    )
    plot_eigen_vs_epsilon(
        output_dir,
        epsilons,
        eigenval_errors_all,
        algorithms,
        ylabel=_mean_rel_err_label(None),
        filename=f"epsilon_eigenval_all_comparison_{timestamp}.png",
    )
    plot_eigen_vs_hvp(
        output_dir,
        n_hvps,
        eigenval_errors,
        algorithms,
        full_hessian_rows,
        ylabel=_mean_rel_err_label(args.first_n_eigvals),
        filename=f"eigenval_vs_hvp_low_level_pattern_{timestamp}.png",
    )
    plot_eigen_vs_hvp(
        output_dir,
        n_hvps,
        eigenval_errors_all,
        algorithms,
        full_hessian_rows,
        ylabel=_mean_rel_err_label(None),
        filename=f"eigenval_all_vs_hvp_low_level_pattern_{timestamp}.png",
    )

    plt.figure(figsize=LINE_FIGSIZE)
    plt.plot(
        epsilons,
        pattern_edges,
        marker="o",
        color=COLORS["naive"],
        linewidth=LINEWIDTH,
        markersize=MARKERSIZE,
        label="Thresholded low-level pattern",
    )
    plt.axhline(
        y=full_edges,
        color=COLORS["star"],
        linestyle="--",
        linewidth=REF_LINEWIDTH,
        label="Dense matrix entries",
    )
    plt.xlabel(r"Low-level threshold $\epsilon$ in $|H_{\mathrm{low}}| > \epsilon$", fontsize=LABEL_FONTSIZE)
    plt.ylabel("Number of pattern entries", fontsize=LABEL_FONTSIZE)
    plt.xscale("log")
    plt.gca().invert_xaxis()
    plt.legend(fontsize=LEGEND_FONTSIZE)
    plt.grid(True, alpha=0.3)
    _style_plot_ticks()
    plt.tight_layout()
    plt.savefig(output_dir / f"pattern_entries_vs_epsilon_{timestamp}.png", dpi=300, bbox_inches="tight")
    plt.close()

    if args.plot_data_dump:
        payload = {
            "timestamp": np.asarray([timestamp], dtype=object),
            "epsilons": np.asarray(epsilons, dtype=float),
            "algorithms": np.asarray(algorithms, dtype=object),
            "target_hessian_path": np.asarray([str(target_path)], dtype=object),
            "pattern_hessian_path": np.asarray([str(pattern_path)], dtype=object),
            "full_hessian_rows": np.asarray([full_hessian_rows], dtype=int),
            "pattern_edges": np.asarray(pattern_edges, dtype=int),
            "pattern_fractions": np.asarray(pattern_fractions, dtype=float),
            "include_diagonal": np.asarray([args.include_diagonal], dtype=bool),
            "eigvals_ref_eckart_projected_sorted": np.asarray(eigvals_ref, dtype=float),
        }
        for alg in algorithms:
            payload[f"errors_{alg}"] = np.asarray(errors[alg], dtype=float)
            payload[f"n_hvps_{alg}"] = np.asarray(n_hvps[alg], dtype=float)
            payload[f"timings_s_{alg}"] = np.asarray(timings_s[alg], dtype=float)
            payload[f"statuses_{alg}"] = np.asarray(statuses[alg], dtype=object)
            payload[f"eigenval_errors_first{args.first_n_eigvals}_{alg}"] = np.asarray(
                eigenval_errors[alg], dtype=float
            )
            payload[f"eigenval_errors_all_{alg}"] = np.asarray(eigenval_errors_all[alg], dtype=float)
        data_path = output_dir / f"low_level_pattern_debug_data_{timestamp}.npz"
        np.savez_compressed(data_path, **payload)
        print(f"Saved debug NPZ: {data_path}")

    print("\nSummary")
    print("=" * 60)
    for i, eps in enumerate(epsilons):
        parts = [f"eps={eps:.3e}", f"pattern={pattern_fractions[i]:.3%}"]
        for alg in algorithms:
            if np.isfinite(eigenval_errors_all[alg][i]):
                parts.append(
                    f"{alg}: all-eig={eigenval_errors_all[alg][i]:.3e}, "
                    f"HVP={int(n_hvps[alg][i])}"
                )
            else:
                parts.append(f"{alg}: failed")
        print(" | ".join(parts))
    print(f"\nOutputs in: {output_dir}")


if __name__ == "__main__":
    main()
