"""
Sparse Hessian cutoff sweep using tblite (xTB) reference Hessians.

Ground truth is one nuclear Hessian from central finite differences on the
tblite gradient (same energy-derivative convention as PySCF/Hartree/Bohr²).
Sparse HVPs are emulated as ``H_true @ v`` so the sweep measures
coloring/recovery behavior without redoing xTB work for every seed vector.

Requires: tblite, ase, scipy, matplotlib, torch, colpackpy.
"""
from __future__ import annotations

import argparse
import os
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
from ase.units import Bohr, Hartree, kB  # noqa: E402
from scipy.sparse import coo_matrix, kron  # noqa: E402

# -----------------------------------------------------------------------------
# Optional: editable install of colpackpy from repo checkout
# -----------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
# Same default Chignolin frame as sparse_hessian_cutoff_dft.py and sparse_hessian_cutoff_real.py
# (REPO_ROOT / "data" / "Chignolin" / "Coordinates_9999.xyz").
_DEFAULT_CHIGNOLIN_STRUCTURE = _REPO_ROOT / "data" / "Chignolin" / "Coordinates_9999.xyz"
_colpack_src = _REPO_ROOT / "colpackpy" / "src"
if _colpack_src.is_dir():
    sys.path.insert(0, str(_colpack_src))

from tblite.interface import Calculator  # noqa: E402

from colpackpy.coloring import get_seeds_from_pattern  # noqa: E402
from colpackpy.torch_utils import compute_sparse_hessian_from_hvp_fn  # noqa: E402

FULL_SPECTRUM_REG_EPS = 1e-5
LABEL_FONTSIZE = int(round(14 * 1.3))
LEGEND_FONTSIZE = int(round(12 * 1.4))
TICK_FONTSIZE = 16
LINEWIDTH = 4.5
REF_LINEWIDTH = 2.0
MARKERSIZE = 10
LINE_FIGSIZE = (10, 6)
PLOT_DATA_DUMP = True
INCLUDE_DISTANCE_TWO = False
KERNEL_WARMUP_ROUNDS = 0  # Untimed dry runs; keep 0 by default.
COLORING_ORDERING = "SMALLEST_LAST"
YLABEL_POINTWISE_REL_ERR = (
    r"$\frac{|\lambda_i-\lambda_i^{\mathrm{GT}}|}{|\lambda_i^{\mathrm{GT}}|}$"
)

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--structure",
        type=Path,
        default=_DEFAULT_CHIGNOLIN_STRUCTURE,
        help=(
            "ASE-readable geometry (default: data/Chignolin/Coordinates_9999.xyz — same as "
            "sparse_hessian_cutoff_dft.py and sparse_hessian_cutoff_real.py)."
        ),
    )
    p.add_argument(
        "--method",
        type=str,
        default="GFN2-xTB",
        choices=(
            "GFN2-xTB",
            "GFN1-xTB",
            "IPEA1-xTB",
        ),
        help="tblite xTB parametrization (default: GFN2-xTB).",
    )
    p.add_argument("--charge", type=int, default=0)
    p.add_argument(
        "--spin",
        type=int,
        default=0,
        help="Unpaired electrons (tblite ``uhf``); 0 closed shell.",
    )
    p.add_argument(
        "--accuracy",
        type=float,
        default=1.0,
        help="tblite SCC numerical thresholds (same as tblite Calculator accuracy).",
    )
    p.add_argument(
        "--max-iter",
        type=int,
        default=250,
        help="Max SCC iterations for tblite single-points.",
    )
    p.add_argument(
        "--hessian-delta",
        type=float,
        default=5e-3,
        help="Central finite-difference step for reference Hessian, in Å (converted to Bohr internally).",
    )
    p.add_argument(
        "--tblite-guess",
        type=str,
        default="sad",
        choices=("sad", "eeq", "eeqbc"),
        help="Initial guess label for tblite (passed as integer code).",
    )
    p.add_argument(
        "--mixer-damping",
        type=float,
        default=0.4,
        help="SCC mixer damping (tblite default 0.4).",
    )
    p.add_argument(
        "--tblite-temperature-K",
        type=float,
        default=300.0,
        help="Electronic temperature in Kelvin (converted for tblite Calculator).",
    )
    p.add_argument(
        "--tblite-verbosity",
        type=int,
        default=0,
        help="tblite print level during the many Hessian displaced single-points.",
    )
    p.add_argument(
        "--load-full-hessian",
        type=Path,
        default=None,
        help="Load a previously saved reference Hessian .npy and skip tblite Hessian computation.",
    )
    p.add_argument(
        "--cutoffs",
        type=str,
        default="5,8,11,14,17,20,23,26,29,32",
        help="Comma-separated cutoff radii (Å) for the sparsity pattern (default: same as sparse_hessian_cutoff_real.py).",
    )
    p.add_argument("--first-n-eigvals", type=int, default=10)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to matgl/examples/sparse_hessian_results/cutoff_xtb_experiments/",
    )
    p.add_argument(
        "--center-com",
        dest="center_com",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Translate to center of mass before the xTB reference (default: on).",
    )
    return p.parse_args()


ARGS = parse_args()
STRUCTURE_PATH = ARGS.structure.resolve()
OUTPUT_DIR = ARGS.output_dir
if OUTPUT_DIR is None:
    OUTPUT_DIR = _SCRIPT_DIR / "sparse_hessian_results" / "cutoff_xtb_experiments"
OUTPUT_DIR = Path(OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

cutoffs = np.array([float(x) for x in ARGS.cutoffs.split(",") if x.strip()], dtype=float)
first_n_eigvals = ARGS.first_n_eigvals
algorithms = ["star", "acyclic"]
if INCLUDE_DISTANCE_TWO:
    algorithms = ["distance_two", *algorithms]


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


def _style_plot_ticks() -> None:
    plt.xticks(fontsize=TICK_FONTSIZE)
    plt.yticks(fontsize=TICK_FONTSIZE)


def _style_axis_ticks(ax) -> None:
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)


def _plot_marker(alg: str) -> str:
    return MARKERS.get(alg, "o")


def create_sparsity_pattern_from_cutoff_positions(
    positions: np.ndarray, cutoff: float
) -> tuple[np.ndarray, int]:
    """Match ``matgl.graph.compute.create_sparsity_pattern_from_cutoff`` (Å, 3×3 blocks)."""
    pos_np = np.asarray(positions, dtype=float)
    num_atoms = pos_np.shape[0]
    diff = pos_np[:, np.newaxis, :] - pos_np[np.newaxis, :, :]
    distances = np.linalg.norm(diff, axis=2)
    adjacency = (distances <= cutoff).astype(float)
    adjacency = coo_matrix(adjacency)
    n = num_atoms * 3
    ones_3x3 = coo_matrix(np.ones((3, 3)))
    expanded_sparsity = kron(adjacency, ones_3x3).tocoo()
    edges = np.vstack((expanded_sparsity.row, expanded_sparsity.col)).T
    return edges.astype(np.intp, copy=False), n


def get_vibrational_basis_from_eckart(atoms_obj, tol=1e-10):
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
    vib_basis = q_full[:, rigid_rank:]
    return vib_basis


def mass_weight_hessian(hessian, masses):
    inv_sqrt_m = 1.0 / np.sqrt(np.asarray(masses, dtype=float))
    inv_sqrt_m_xyz = np.repeat(inv_sqrt_m, 3)
    return (inv_sqrt_m_xyz[:, None] * hessian) * inv_sqrt_m_xyz[None, :]


def projected_vibrational_eigvals(hessian, masses, vib_basis):
    hessian_mw = mass_weight_hessian(hessian, masses)
    hessian_vib = vib_basis.T @ hessian_mw @ vib_basis
    hessian_vib = 0.5 * (hessian_vib + hessian_vib.T)
    return np.linalg.eigvalsh(hessian_vib)


_GUESS_TBLITE = {"sad": 0, "eeq": 1, "eeqbc": 2}


def numerical_hessian_tblite_centdiff(
    atoms,
    *,
    delta_ang: float,
    method: str,
    charge: int,
    spin: int,
    accuracy: float,
    max_iter: int,
    mixer_damping: float,
    electronic_temperature_k: float,
    tblite_verbosity: int,
    guess_key: str,
) -> tuple[np.ndarray, float]:
    """
    Cartesian nuclear Hessian ∂²E/∂x_i∂x_j from central differences on the
    gradient (Hartree/Bohr²), matching the PySCF convention used in cutoff_dft.

    One column at a time: ``H[:,j] = (g(x+h e_j) - g(x-h e_j)) / (2h)`` where
    ``g`` is the energy gradient in Hartree/Bohr.
    """
    numbers = np.asarray(atoms.get_atomic_numbers(), dtype=np.int32)
    x0_ang = atoms.get_positions().astype(float).reshape(-1)
    n_dof = x0_ang.size
    nat = len(numbers)
    if 3 * nat != n_dof:
        raise ValueError("internal: positions size mismatch")
    delta_bohr = float(delta_ang) / Bohr

    calc = Calculator(method, numbers, x0_ang.reshape(-1, 3) / Bohr, charge=float(charge), uhf=int(spin))
    calc.set("accuracy", float(accuracy))
    calc.set("max-iter", int(max_iter))
    calc.set("mixer-damping", float(mixer_damping))
    calc.set("verbosity", int(tblite_verbosity))
    calc.set("temperature", float(electronic_temperature_k) * kB / Hartree)
    calc.set("guess", _GUESS_TBLITE[guess_key])

    print(
        f"  Reference Hessian via central differences: {n_dof} Cartesian DOF, "
        f"step={delta_ang:g} Å (= {delta_bohr:.6f} Bohr), method={method!r} — "
        f"{2 * n_dof} displaced single-points…",
        flush=True,
    )
    t0 = time.perf_counter()
    x0_bohr = x0_ang / Bohr

    res0 = calc.singlepoint(None)
    _ = res0.get("gradient")

    hess = np.zeros((n_dof, n_dof), dtype=float)
    for j in range(n_dof):
        if j % max(1, n_dof // 10) == 0 or j == n_dof - 1:
            elapsed = time.perf_counter() - t0
            print(f"    Hessian FD progress: {j}/{n_dof} DOF (elapsed {elapsed:.1f}s)", flush=True)

        x_p = x0_bohr.copy()
        x_p[j] += delta_bohr
        calc.update(positions=x_p.reshape(-1, 3))
        res_p = calc.singlepoint(res0)
        g_p = np.asarray(res_p.get("gradient"), dtype=float).reshape(-1)

        x_m = x0_bohr.copy()
        x_m[j] -= delta_bohr
        calc.update(positions=x_m.reshape(-1, 3))
        res_m = calc.singlepoint(res0)
        g_m = np.asarray(res_m.get("gradient"), dtype=float).reshape(-1)

        hess[:, j] = (g_p - g_m) / (2.0 * delta_bohr)

        calc.update(positions=x0_bohr.reshape(-1, 3))

    hess = 0.5 * (hess + hess.T)
    elapsed = time.perf_counter() - t0
    print(f"  Finite-difference Hessian finished in {elapsed:.1f}s.", flush=True)
    return hess, elapsed


def hessian_tensor_to_numpy(H) -> np.ndarray:
    if hasattr(H, "detach"):
        return np.asarray(H.detach().cpu().numpy(), dtype=float)
    if hasattr(H, "toarray"):
        return np.asarray(H.toarray(), dtype=float)
    return np.asarray(H, dtype=float)


def planned_hvps_per_algorithm(
    positions: np.ndarray, cutoff: float, algorithm_names: list[str]
) -> dict[str, int]:
    """ColPack coloring only: number of seed-column HVPs each algorithm will request."""
    edges, n = create_sparsity_pattern_from_cutoff_positions(positions, float(cutoff))
    out: dict[str, int] = {}
    for name in algorithm_names:
        S_np, _, _ = get_seeds_from_pattern(
            edges, n, COLORING_ORDERING, name.upper()
        )
        S_np = np.asarray(S_np)
        out[name] = int(S_np.shape[1])
    return out


print("=" * 80)
print("Sparse Hessian cutoff experiment (tblite xTB reference Hessian + emulated HVPs)")
print("=" * 80)
print(f"Structure: {STRUCTURE_PATH}")
print(
    f"xTB method: {ARGS.method!r}, SCC accuracy={ARGS.accuracy:g}, "
    f"max_iter={ARGS.max_iter}, Hessian Δ={ARGS.hessian_delta:g} Å"
)
atoms = read(str(STRUCTURE_PATH))
if ARGS.center_com:
    atoms.center()
    print("Centered geometry at center of mass (use --no-center-com to keep original coords).", flush=True)
n_atom = len(atoms)
print(f"Loaded {n_atom} atoms")

x0 = atoms.get_positions().reshape(-1).astype(float)
n_dof = x0.size
full_hessian_rows = n_dof

if n_dof >= 120:
    print("\n" + "!" * 72, flush=True)
    print(
        f"WARNING: {n_atom} atoms → {n_dof} Cartesian DOF. The tblite Hessian uses "
        f"{2 * n_dof} finite-difference single-points plus one equilibrium gradient — costly.",
        flush=True,
    )
    print(
        "Sparse HVPs are still emulated from that matrix once computed.",
        flush=True,
    )
    print("Use a smaller --structure or --load-full-hessian for quick repeats.", flush=True)
    print("!" * 72 + "\n", flush=True)

if ARGS.load_full_hessian is None:
    print("\nComputing tblite finite-difference reference Hessian...", flush=True)
    print(f"[1/4] xTB reference Hessian ({n_dof}×{n_dof})", flush=True)
    hessian_ref, ref_hessian_time_s = numerical_hessian_tblite_centdiff(
        atoms,
        delta_ang=ARGS.hessian_delta,
        method=ARGS.method,
        charge=int(ARGS.charge),
        spin=int(ARGS.spin),
        accuracy=float(ARGS.accuracy),
        max_iter=int(ARGS.max_iter),
        mixer_damping=float(ARGS.mixer_damping),
        electronic_temperature_k=float(ARGS.tblite_temperature_K),
        tblite_verbosity=int(ARGS.tblite_verbosity),
        guess_key=ARGS.tblite_guess,
    )
    full_hessian_npy_path = OUTPUT_DIR / f"full_xtb_reference_hessian_{timestamp}.npy"
    np.save(full_hessian_npy_path, hessian_ref)
    print(f"Saved full reference Hessian to: {full_hessian_npy_path}", flush=True)
else:
    full_hessian_npy_path = ARGS.load_full_hessian.resolve()
    print(f"\n[1/4] Loading full reference Hessian from {full_hessian_npy_path}", flush=True)
    hessian_ref = np.asarray(np.load(full_hessian_npy_path), dtype=float)
    if hessian_ref.shape != (n_dof, n_dof):
        raise ValueError(
            f"Loaded Hessian shape {hessian_ref.shape} does not match expected {(n_dof, n_dof)}"
        )
    ref_hessian_time_s = np.nan

masses = np.asarray(atoms.get_masses(), dtype=float)
vib_basis = get_vibrational_basis_from_eckart(atoms)

device = torch.device("cpu")
dtype = torch.float64
x_t = torch.tensor(x0, dtype=dtype, device=device)
hessian_ref_t = torch.tensor(hessian_ref, dtype=dtype, device=device)


def hvp_from_true_hessian(x_tensor: torch.Tensor, v_tensor: torch.Tensor) -> torch.Tensor:
    """Emulate an HVP with the already-computed reference Hessian."""
    del x_tensor
    return (hessian_ref_t @ v_tensor.reshape(-1)).reshape_as(v_tensor)


if KERNEL_WARMUP_ROUNDS > 0:
    print(
        f"\nKernel warmup: {KERNEL_WARMUP_ROUNDS} untimed sparse Hessian(s) per algorithm "
        f"at cutoff={float(cutoffs[0]):.3f} Å..."
    )
    edges_w, n_w = create_sparsity_pattern_from_cutoff_positions(atoms.get_positions(), float(cutoffs[0]))
    assert n_w == n_dof
    for alg in algorithms:
        for _ in range(KERNEL_WARMUP_ROUNDS):
            compute_sparse_hessian_from_hvp_fn(
                hvp_from_true_hessian,
                x_t,
                edges_w,
                n_dof,
                coloring_algorithm=alg,
                verify=False,
                return_stats=False,
                hessian_return_type="dense",
            )

errors = {alg: [] for alg in algorithms}
n_hvps = {alg: [] for alg in algorithms}
timings_s = {alg: [] for alg in algorithms}
hessians = {alg: [] for alg in algorithms}
statuses = {alg: [] for alg in algorithms}
included_full_case = {alg: False for alg in algorithms}

print(f"\n[2/4] Sweeping {len(cutoffs)} cutoffs: {cutoffs.tolist()}")
pos = atoms.get_positions()
n_cut = len(cutoffs)
prev_cutoff_summary: dict[str, tuple[float, int]] = {}
prev_cutoff_val: float | None = None

for ic, cutoff in enumerate(cutoffs):
    print("\n" + "=" * 72)
    print(
        f"Cutoff {ic + 1}/{n_cut}: {cutoff:.3f} Å   "
        f"(reference Hessian DOFs = {full_hessian_rows}; HVPs are emulated as H_true @ v)"
    )
    print("=" * 72)

    if (
        ic > 0
        and prev_cutoff_summary
        and prev_cutoff_val is not None
    ):
        parts = [
            f"{a}: {prev_cutoff_summary[a][0]:.2f}s, {prev_cutoff_summary[a][1]} HVPs"
            for a in algorithms
            if a in prev_cutoff_summary
        ]
        if parts:
            print(
                f"Previous cutoff {prev_cutoff_val:.3f} Å completed — "
                + " | ".join(parts)
            )

    planned = planned_hvps_per_algorithm(pos, float(cutoff), algorithms)
    plan_parts = [f"{a}: {planned[a]} HVPs" for a in algorithms]
    print(f"This cutoff — ColPack plan: {' | '.join(plan_parts)}")

    if ic + 1 < n_cut:
        next_c = float(cutoffs[ic + 1])
        next_plan = planned_hvps_per_algorithm(pos, next_c, algorithms)
        nxt_parts = [f"{a}: {next_plan[a]} HVPs" for a in algorithms]
        print(
            f"Next cutoff {next_c:.3f} Å (after this one) — ColPack plan: {' | '.join(nxt_parts)}"
        )
    else:
        print("Next cutoff: (none — this is the last)")

    edges, n_pat = create_sparsity_pattern_from_cutoff_positions(pos, float(cutoff))
    assert n_pat == n_dof

    cutoff_wall_start = time.perf_counter()
    cutoff_results: dict[str, tuple[float, int]] = {}
    for alg in algorithms:
        print(f"\n  >>> {alg} @ {cutoff:.3f} Å ...")
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
            n_hvp = int(stats["num_hvps"])

            if n_hvp >= full_hessian_rows and included_full_case[alg]:
                print(f"    Skip duplicate full case (HVPs {n_hvp} >= {full_hessian_rows}).")
                errors[alg].append(np.nan)
                n_hvps[alg].append(np.nan)
                timings_s[alg].append(np.nan)
                hessians[alg].append(None)
                statuses[alg].append("skipped_duplicate_full")
                continue

            hessians[alg].append(H_np)
            max_error = np.abs(hessian_ref - H_np).max()
            errors[alg].append(max_error)
            n_hvps[alg].append(n_hvp)
            timings_s[alg].append(elapsed_s)
            statuses[alg].append("ok")
            if n_hvp >= full_hessian_rows:
                included_full_case[alg] = True
            cutoff_results[alg] = (elapsed_s, n_hvp)
            print(
                f"    done: wall {elapsed_s:.2f}s, {n_hvp} HVPs, "
                f"max|H−H_ref| = {max_error:.6e}"
            )
        except Exception as exc:
            print(f"    failed: {exc}")
            errors[alg].append(np.nan)
            n_hvps[alg].append(np.nan)
            timings_s[alg].append(np.nan)
            hessians[alg].append(None)
            statuses[alg].append("failed")

    cutoff_wall_total = time.perf_counter() - cutoff_wall_start
    print(
        f"\n  --- Cutoff {cutoff:.3f} Å all algorithms wall time: {cutoff_wall_total:.2f}s ---"
    )
    prev_cutoff_summary = cutoff_results
    prev_cutoff_val = float(cutoff)

# --------------------------------------------------------------------------- #
# Plots and dumps (same layout as sparse_hessian_cutoff_real.py)
# --------------------------------------------------------------------------- #
print("\n[3/4] Plots and NPZ dumps...")

eigenval_errors = {alg: [] for alg in algorithms}
eigenval_errors_all = {alg: [] for alg in algorithms}
EIGENVAL_ERROR_PLOT_EPS = 1e-7
eigvals_ref_proj_sorted = np.sort(projected_vibrational_eigvals(hessian_ref, masses, vib_basis))
eigvals_ref_first_sorted = eigvals_ref_proj_sorted[:first_n_eigvals]

per_eig_errors_all = {
    alg: np.full((len(cutoffs), len(eigvals_ref_proj_sorted)), np.nan, dtype=float)
    for alg in algorithms
}

for i, _cutoff in enumerate(cutoffs):
    for alg in algorithms:
        if i < len(hessians[alg]) and hessians[alg][i] is not None:
            try:
                eig_sparse = np.sort(projected_vibrational_eigvals(hessians[alg][i], masses, vib_basis))
                eig_sparse_first = eig_sparse[:first_n_eigvals]
                pct_diff = np.abs(eigvals_ref_first_sorted - eig_sparse_first) / (
                    np.abs(eigvals_ref_first_sorted) + 1e-5
                )
                eigenval_errors[alg].append(pct_diff.mean() + EIGENVAL_ERROR_PLOT_EPS)
                pct_diff_all = np.abs(eigvals_ref_proj_sorted - eig_sparse) / (
                    np.abs(eigvals_ref_proj_sorted) + FULL_SPECTRUM_REG_EPS
                )
                eigenval_errors_all[alg].append(pct_diff_all.mean() + EIGENVAL_ERROR_PLOT_EPS)
                per_eig_errors_all[alg][i, :] = pct_diff_all
            except Exception as exc:
                print(f"    eigen warning {alg} idx {i}: {exc}")
                eigenval_errors[alg].append(np.nan)
                eigenval_errors_all[alg].append(np.nan)
        else:
            eigenval_errors[alg].append(np.nan)
            eigenval_errors_all[alg].append(np.nan)


def get_valid_xy(x_vals, y_vals, require_positive_y=False):
    x_arr = np.asarray(x_vals, dtype=float)
    y_arr = np.asarray(y_vals, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if require_positive_y:
        mask &= y_arr > 0
    return x_arr[mask], y_arr[mask]


def get_valid_xy_sorted_by_x(x_vals, y_vals, require_positive_y=False):
    x_arr, y_arr = get_valid_xy(x_vals, y_vals, require_positive_y=require_positive_y)
    if x_arr.size == 0:
        return x_arr, y_arr
    order = np.argsort(x_arr, kind="stable")
    return x_arr[order], y_arr[order]


ylabel_maxabs = "Max absolute error vs xTB reference Hessian"

plt.figure(figsize=LINE_FIGSIZE)
for alg in algorithms:
    vx, vy = get_valid_xy(cutoffs, errors[alg], require_positive_y=True)
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
plt.xlabel("Cutoff radius (Å)", fontsize=LABEL_FONTSIZE)
plt.ylabel(ylabel_maxabs, fontsize=LABEL_FONTSIZE)
plt.legend(fontsize=LEGEND_FONTSIZE)
plt.grid(True, alpha=0.3)
plt.yscale("log", nonpositive="mask")
_style_plot_ticks()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / f"cutoff_error_comparison_{timestamp}.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=LINE_FIGSIZE)
for alg in algorithms:
    vx, vy = get_valid_xy(cutoffs, n_hvps[alg], require_positive_y=False)
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
plt.xlabel("Cutoff radius (Å)", fontsize=LABEL_FONTSIZE)
plt.ylabel("Number of Hessian–vector products", fontsize=LABEL_FONTSIZE)
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
plt.savefig(OUTPUT_DIR / f"cutoff_hvp_comparison_{timestamp}.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=LINE_FIGSIZE)
for alg in algorithms:
    vx, vy = get_valid_xy_sorted_by_x(cutoffs, timings_s[alg], require_positive_y=False)
    pos = vy > 0
    if np.any(pos):
        plt.plot(
            vx[pos],
            vy[pos],
            marker=_plot_marker(alg),
            color=COLORS.get(alg),
            label=alg.replace("_", " ").title(),
            linewidth=LINEWIDTH,
            markersize=MARKERSIZE,
        )
if np.isfinite(ref_hessian_time_s) and ref_hessian_time_s > 0:
    plt.axhline(
        y=ref_hessian_time_s,
        color=COLORS["naive"],
        linestyle="--",
        linewidth=REF_LINEWIDTH,
        label="xTB Hessian construction time",
    )
plt.xlabel("Cutoff radius (Å)", fontsize=LABEL_FONTSIZE)
plt.ylabel("Wall time per sparse Hessian (s)", fontsize=LABEL_FONTSIZE)
plt.legend(fontsize=LEGEND_FONTSIZE)
plt.grid(True, alpha=0.3)
_style_plot_ticks()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / f"cutoff_wall_time_comparison_{timestamp}.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=LINE_FIGSIZE)
for alg in algorithms:
    vx, vy = get_valid_xy(cutoffs, eigenval_errors[alg], require_positive_y=True)
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
plt.xlabel("Cutoff radius (Å)", fontsize=LABEL_FONTSIZE)
plt.ylabel(_mean_rel_err_label(first_n_eigvals), fontsize=LABEL_FONTSIZE)
plt.legend(fontsize=LEGEND_FONTSIZE)
plt.grid(True, alpha=0.3)
plt.yscale("log", nonpositive="mask")
_style_plot_ticks()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / f"cutoff_eigenval_comparison_{timestamp}.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=LINE_FIGSIZE)
for alg in algorithms:
    vx, vy = get_valid_xy(cutoffs, eigenval_errors_all[alg], require_positive_y=True)
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
plt.xlabel("Cutoff radius (Å)", fontsize=LABEL_FONTSIZE)
plt.ylabel(_mean_rel_err_label(None), fontsize=LABEL_FONTSIZE)
plt.legend(fontsize=LEGEND_FONTSIZE)
plt.grid(True, alpha=0.3)
plt.yscale("log", nonpositive="mask")
_style_plot_ticks()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / f"cutoff_eigenval_all_comparison_{timestamp}.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=LINE_FIGSIZE)
for alg in algorithms:
    vx, vy = get_valid_xy_sorted_by_x(n_hvps[alg], errors[alg], require_positive_y=True)
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
plt.xlabel("Number of Hessian–vector products", fontsize=LABEL_FONTSIZE)
plt.ylabel(ylabel_maxabs, fontsize=LABEL_FONTSIZE)
plt.axvline(
    x=full_hessian_rows,
    color=COLORS["naive"],
    linestyle="--",
    linewidth=REF_LINEWIDTH,
    label="Full Hessian rows",
)
plt.legend(fontsize=LEGEND_FONTSIZE)
plt.grid(True, alpha=0.3)
plt.yscale("log", nonpositive="mask")
_style_plot_ticks()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / f"error_vs_hvp_comparison_{timestamp}.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=LINE_FIGSIZE)
for alg in algorithms:
    vx, vy = get_valid_xy_sorted_by_x(n_hvps[alg], eigenval_errors[alg], require_positive_y=True)
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
plt.xlabel("Number of Hessian–vector products", fontsize=LABEL_FONTSIZE)
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
plt.yscale("log", nonpositive="mask")
_style_plot_ticks()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / f"eigenval_vs_hvp_comparison_{timestamp}.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=LINE_FIGSIZE)
for alg in algorithms:
    vx, vy = get_valid_xy_sorted_by_x(n_hvps[alg], eigenval_errors_all[alg], require_positive_y=True)
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
plt.xlabel("Number of Hessian–vector products", fontsize=LABEL_FONTSIZE)
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
plt.yscale("log", nonpositive="mask")
_style_plot_ticks()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / f"eigenval_all_vs_hvp_comparison_{timestamp}.png", dpi=300, bbox_inches="tight")
plt.close()

# Error vs time + eigenvalue vs time (condensed from sparse_hessian_cutoff_real.py)
fig, ax = plt.subplots(figsize=LINE_FIGSIZE)
min_positive_error = None
for alg in algorithms:
    vt, ve = get_valid_xy_sorted_by_x(timings_s[alg], errors[alg], require_positive_y=False)
    pos = (ve > 0) & np.isfinite(ve) & (vt > 0) & np.isfinite(vt)
    if np.any(pos):
        ax.plot(
            vt[pos],
            ve[pos],
            marker=_plot_marker(alg),
            color=COLORS.get(alg),
            label=alg.replace("_", " ").title(),
            linewidth=LINEWIDTH,
            markersize=MARKERSIZE,
        )
        mp = float(np.min(ve[pos]))
        min_positive_error = mp if min_positive_error is None else min(min_positive_error, mp)
if np.isfinite(ref_hessian_time_s) and ref_hessian_time_s > 0:
    ax.axvline(
        x=ref_hessian_time_s,
        color=COLORS["naive"],
        linestyle="--",
        linewidth=REF_LINEWIDTH,
        label="xTB Hessian construction time",
    )
ax.set_xlabel("Wall time per sparse Hessian (s)", fontsize=LABEL_FONTSIZE)
ax.set_ylabel(ylabel_maxabs, fontsize=LABEL_FONTSIZE)
ax.legend(fontsize=LEGEND_FONTSIZE)
ax.grid(True, alpha=0.3)
if min_positive_error is not None:
    ax.set_ylim(bottom=max(min_positive_error * 0.5, 1e-30))
    ax.set_yscale("log", nonpositive="mask")
_style_plot_ticks()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / f"error_vs_time_comparison_{timestamp}.png", dpi=300, bbox_inches="tight")
plt.close()

fig, ax = plt.subplots(figsize=LINE_FIGSIZE)
min_pos_eig = None
for alg in algorithms:
    vt, vy = get_valid_xy_sorted_by_x(timings_s[alg], eigenval_errors[alg], require_positive_y=False)
    pos = (vy > 0) & np.isfinite(vy) & (vt > 0) & np.isfinite(vt)
    if np.any(pos):
        ax.plot(
            vt[pos],
            vy[pos],
            marker=_plot_marker(alg),
            color=COLORS.get(alg),
            label=alg.replace("_", " ").title(),
            linewidth=LINEWIDTH,
            markersize=MARKERSIZE,
        )
        mp = float(np.min(vy[pos]))
        min_pos_eig = mp if min_pos_eig is None else min(min_pos_eig, mp)
if np.isfinite(ref_hessian_time_s) and ref_hessian_time_s > 0:
    ax.axvline(
        x=ref_hessian_time_s,
        color=COLORS["naive"],
        linestyle="--",
        linewidth=REF_LINEWIDTH,
        label="xTB Hessian construction time",
    )
ax.set_xlabel("Wall time per sparse Hessian (s)", fontsize=LABEL_FONTSIZE)
ax.set_ylabel(_mean_rel_err_label(first_n_eigvals), fontsize=LABEL_FONTSIZE)
ax.legend(fontsize=LEGEND_FONTSIZE)
ax.grid(True, alpha=0.3)
if min_pos_eig is not None:
    ax.set_ylim(bottom=max(min_pos_eig * 0.5, 1e-30))
    ax.set_yscale("log", nonpositive="mask")
_style_plot_ticks()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / f"eigenval_vs_time_comparison_{timestamp}.png", dpi=300, bbox_inches="tight")
plt.close()

fig, ax = plt.subplots(figsize=LINE_FIGSIZE)
min_pos_eig_all = None
for alg in algorithms:
    vt, vy = get_valid_xy_sorted_by_x(timings_s[alg], eigenval_errors_all[alg], require_positive_y=False)
    pos = (vy > 0) & np.isfinite(vy) & (vt > 0) & np.isfinite(vt)
    if np.any(pos):
        ax.plot(
            vt[pos],
            vy[pos],
            marker=_plot_marker(alg),
            color=COLORS.get(alg),
            label=alg.replace("_", " ").title(),
            linewidth=LINEWIDTH,
            markersize=MARKERSIZE,
        )
        mp = float(np.min(vy[pos]))
        min_pos_eig_all = mp if min_pos_eig_all is None else min(min_pos_eig_all, mp)
if np.isfinite(ref_hessian_time_s) and ref_hessian_time_s > 0:
    ax.axvline(
        x=ref_hessian_time_s,
        color=COLORS["naive"],
        linestyle="--",
        linewidth=REF_LINEWIDTH,
        label="xTB Hessian construction time",
    )
ax.set_xlabel("Wall time per sparse Hessian (s)", fontsize=LABEL_FONTSIZE)
ax.set_ylabel(_mean_rel_err_label(None), fontsize=LABEL_FONTSIZE)
ax.legend(fontsize=LEGEND_FONTSIZE)
ax.grid(True, alpha=0.3)
if min_pos_eig_all is not None:
    ax.set_ylim(bottom=max(min_pos_eig_all * 0.5, 1e-30))
    ax.set_yscale("log", nonpositive="mask")
_style_plot_ticks()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / f"eigenval_all_vs_time_comparison_{timestamp}.png", dpi=300, bbox_inches="tight")
plt.close()

# Per-eigenvalue bars
fig, axes = plt.subplots(2, len(algorithms), figsize=(20, 10))
if len(algorithms) == 1:
    axes = np.array([[axes[0]], [axes[1]]])

for col_idx, alg in enumerate(algorithms):
    representative_idx = None
    for i in range(len(cutoffs) - 1, -1, -1):
        if i < len(hessians[alg]) and hessians[alg][i] is not None:
            representative_idx = i
            break
    ax_top = axes[0, col_idx]
    ax_bottom = axes[1, col_idx]
    if representative_idx is None:
        ax_top.text(0.5, 0.5, "No valid run", ha="center", va="center", transform=ax_top.transAxes)
        ax_bottom.text(0.5, 0.5, "No valid run", ha="center", va="center", transform=ax_bottom.transAxes)
        continue
    eig_sparse = np.sort(projected_vibrational_eigvals(hessians[alg][representative_idx], masses, vib_basis))
    eigval_pct_error = np.abs(eigvals_ref_proj_sorted - eig_sparse) / (
        np.abs(eigvals_ref_proj_sorted) + FULL_SPECTRUM_REG_EPS
    )
    finite_errors = eigval_pct_error[np.isfinite(eigval_pct_error)]
    top_cap = float(np.percentile(finite_errors, 99.5)) if finite_errors.size > 0 else 1e-12
    top_cap = max(top_cap, 1e-12)
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
    ax_top.grid(True, alpha=0.3, axis="y")
    if col_idx == 0:
        ax_top.set_ylabel(YLABEL_POINTWISE_REL_ERR, fontsize=LABEL_FONTSIZE)
    ax_top.set_xlabel("Eigenvalue index", fontsize=LABEL_FONTSIZE)
    ax_top.set_title(
        f"{alg.replace('_', ' ').title()} (cutoff={cutoffs[representative_idx]:.0f} Å)",
        fontsize=LABEL_FONTSIZE - 1,
    )
    _style_axis_ticks(ax_top)
    eig_span = float(np.max(eigvals_ref_proj_sorted) - np.min(eigvals_ref_proj_sorted))
    bar_width = eig_span / max(len(eigvals_ref_proj_sorted) * 1.5, 1)
    if not np.isfinite(bar_width) or bar_width <= 0:
        bar_width = 1e-3
    ax_bottom.bar(
        eigvals_ref_proj_sorted,
        eigval_pct_error_plot,
        width=bar_width,
        color=COLORS_LIGHT.get(alg, COLORS.get(alg, "#666666")),
        alpha=0.92,
        edgecolor="white",
        linewidth=0.4,
    )
    ax_bottom.set_ylim(0.0, top_cap * 1.05)
    ax_bottom.grid(True, alpha=0.3, axis="y")
    if col_idx == 0:
        ax_bottom.set_ylabel(YLABEL_POINTWISE_REL_ERR, fontsize=LABEL_FONTSIZE)
    ax_bottom.set_xlabel("Reference eigenvalue", fontsize=LABEL_FONTSIZE)
    _style_axis_ticks(ax_bottom)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUTPUT_DIR / f"per_eigenvalue_error_bars_{timestamp}.png", dpi=300, bbox_inches="tight")
plt.close()

debug_data_path = OUTPUT_DIR / f"cutoff_debug_data_{timestamp}.npz"
debug_payload = {
    "cutoffs": np.asarray(cutoffs, dtype=float),
    "ref_full_hessian_time_s": np.asarray([ref_hessian_time_s], dtype=float),
    "full_hessian_rows": np.asarray([full_hessian_rows], dtype=int),
    "full_reference_hessian": np.asarray(hessian_ref, dtype=float),
    "full_reference_hessian_path": np.asarray([str(full_hessian_npy_path)], dtype=object),
    "xtb_method": np.asarray([ARGS.method], dtype=object),
    "hessian_fd_delta_ang": np.asarray([float(ARGS.hessian_delta)], dtype=float),
    "eigvals_ref_eckart_projected_sorted": np.asarray(eigvals_ref_proj_sorted, dtype=float),
    "errors_star": np.asarray(errors["star"], dtype=float),
    "errors_acyclic": np.asarray(errors["acyclic"], dtype=float),
    "n_hvps_star": np.asarray(n_hvps["star"], dtype=float),
    "n_hvps_acyclic": np.asarray(n_hvps["acyclic"], dtype=float),
    "timings_s_star": np.asarray(timings_s["star"], dtype=float),
    "timings_s_acyclic": np.asarray(timings_s["acyclic"], dtype=float),
    "statuses_star": np.asarray(statuses["star"], dtype=object),
    "statuses_acyclic": np.asarray(statuses["acyclic"], dtype=object),
    "eigenval_errors_first10_star": np.asarray(eigenval_errors["star"], dtype=float),
    "eigenval_errors_first10_acyclic": np.asarray(eigenval_errors["acyclic"], dtype=float),
    "eigenval_errors_all_star": np.asarray(eigenval_errors_all["star"], dtype=float),
    "eigenval_errors_all_acyclic": np.asarray(eigenval_errors_all["acyclic"], dtype=float),
    "per_eig_errors_all_star": per_eig_errors_all["star"],
    "per_eig_errors_all_acyclic": per_eig_errors_all["acyclic"],
}
np.savez_compressed(debug_data_path, **debug_payload)
print(f"Saved debug NPZ: {debug_data_path}")

if PLOT_DATA_DUMP:
    plot_series_path = OUTPUT_DIR / f"cutoff_plot_series_{timestamp}.npz"
    series_payload = {
        "timestamp": np.asarray([timestamp], dtype=object),
        "cutoffs": np.asarray(cutoffs, dtype=float),
        "algorithms": np.asarray(algorithms, dtype=object),
        "method": np.asarray(
            [f"tblite_fd_hess_emulated_hvp_{ARGS.method.replace('-', '_').replace('.', 'p')}"],
            dtype=object,
        ),
        "xtb_method": np.asarray([ARGS.method], dtype=object),
        "full_hessian_rows": np.asarray([full_hessian_rows], dtype=int),
        "ref_full_hessian_time_s": np.asarray([ref_hessian_time_s], dtype=float),
        "full_reference_hessian_path": np.asarray([str(full_hessian_npy_path)], dtype=object),
    }
    for alg in algorithms:
        series_payload[f"errors_{alg}"] = np.asarray(errors[alg], dtype=float)
        series_payload[f"n_hvps_{alg}"] = np.asarray(n_hvps[alg], dtype=float)
        series_payload[f"timings_s_{alg}"] = np.asarray(timings_s[alg], dtype=float)
        series_payload[f"statuses_{alg}"] = np.asarray(statuses[alg], dtype=object)
        series_payload[f"eigenval_errors_first{first_n_eigvals}_{alg}"] = np.asarray(eigenval_errors[alg], dtype=float)
        series_payload[f"eigenval_errors_all_{alg}"] = np.asarray(eigenval_errors_all[alg], dtype=float)
        series_payload[f"per_eig_errors_all_{alg}"] = np.asarray(per_eig_errors_all[alg], dtype=float)
    np.savez_compressed(plot_series_path, **series_payload)
    print(f"Saved plot series NPZ: {plot_series_path}")

print("\n[4/4] Summary")
print("=" * 60)
header = ["Cutoff", *[a.replace("_", " ").title() for a in algorithms]]
print(" ".join(f"{h:<20}" for h in header))
print("-" * 60)
for i, c in enumerate(cutoffs):
    row = [f"{c:<8.3f}"]
    for alg in algorithms:
        if i < len(errors[alg]) and not np.isnan(errors[alg][i]):
            row.append(f"{errors[alg][i]:.2e}/{int(n_hvps[alg][i])}")
        elif i < len(statuses[alg]) and statuses[alg][i] == "skipped_duplicate_full":
            row.append("skipped full")
        elif i < len(statuses[alg]) and statuses[alg][i] == "failed":
            row.append("failed")
        else:
            row.append("N/A")
    print(" ".join(f"{x:<20}" for x in row))
print(f"\nOutputs in: {OUTPUT_DIR}")
print("Done.")
