"""
Sparse Hessian cutoff sweep using GPU4PySCF/PBE analytical Hessians.

Ground truth is one analytical DFT Hessian. Sparse HVPs are emulated as
``H_true @ v`` so the sweep measures coloring/recovery behavior without running
finite-difference force calculations for every seed vector.

Requires: pyscf, gpu4pyscf, ase, scipy, matplotlib, torch, colpackpy.
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
from scipy.sparse import coo_matrix, kron  # noqa: E402

# -----------------------------------------------------------------------------
# Optional: editable install of colpackpy from repo checkout
# -----------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
_colpack_src = _REPO_ROOT / "colpackpy" / "src"
if _colpack_src.is_dir():
    sys.path.insert(0, str(_colpack_src))

import pyscf  # noqa: E402
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
KERNEL_WARMUP_ROUNDS = 0  # Each round runs a full DFT + gradient; keep 0 by default.
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
        default=_REPO_ROOT / "data" / "Chignolin" / "Coordinates_9999.xyz",
        help="ASE-readable geometry (default: same Chignolin structure as sparse_hessian_cutoff_real.py).",
    )
    p.add_argument(
        "--basis",
        type=str,
        default="def2-svp",
        help="Gaussian basis (default: def2-svp; use sto-3g for cheaper debugging).",
    )
    p.add_argument(
        "--xc",
        type=str,
        default="pbe",
        help="Libxc XC string (default pbe; e.g. lda,vwn for faster LDA).",
    )
    p.add_argument(
        "--backend",
        type=str,
        default="gpu4pyscf",
        choices=("pyscf", "gpu4pyscf"),
        help="DFT driver: GPU4PySCF (default) or CPU PySCF.",
    )
    p.add_argument("--charge", type=int, default=0)
    p.add_argument("--spin", type=int, default=0)
    p.add_argument(
        "--conv-tol",
        type=float,
        default=1e-6,
        help="SCF energy convergence (looser = faster; try 1e-8 for tighter).",
    )
    p.add_argument(
        "--max-cycle",
        type=int,
        default=50,
        help="Max SCF iterations (lower can fail to converge on hard systems).",
    )
    p.add_argument(
        "--damping",
        type=float,
        default=0.0,
        help="SCF damping factor. Use e.g. 0.5 for unstable GPU4PySCF starts.",
    )
    p.add_argument(
        "--level-shift",
        type=float,
        default=0.0,
        help="SCF virtual-orbital level shift. Use e.g. 0.5 for oscillatory SCF.",
    )
    p.add_argument(
        "--second-order-scf",
        action="store_true",
        help="Use PySCF/GPU4PySCF Newton/SOSCF for the reference SCF.",
    )
    p.add_argument(
        "--atom-grid",
        type=str,
        default="43,194",
        help="DFT radial,Lebedev grid sizes (default 43,194 = cheap; e.g. 75,302 more accurate).",
    )
    p.add_argument(
        "--direct-scf-tol",
        type=float,
        default=1e-13,
        help="DF/JK screening tolerance (default 1e-13; kept conservative to avoid convergence issues).",
    )
    p.add_argument(
        "--small-rho-cutoff",
        type=float,
        default=0.0,
        help="Prune DFT grids where ρ is small; >0 can break the first SCF (NaN) — default 0 (off). Try 1e-7 once stable.",
    )
    p.add_argument(
        "--no-density-fitting",
        action="store_true",
        help="Disable density fitting (usually slower for pure DFT on molecules; for debugging).",
    )
    p.add_argument(
        "--conv-tol-cpscf",
        type=float,
        default=1e-3,
        help="CPHF/CPKS convergence tolerance used by analytical Hessian if supported.",
    )
    p.add_argument(
        "--load-full-hessian",
        type=Path,
        default=None,
        help="Load a previously saved full analytical Hessian .npy and skip SCF/Hessian.",
    )
    p.add_argument(
        "--cutoffs",
        type=str,
        default="5,8,11,14,17,20,23,26,29,32,35,38,41,44",
        help="Comma-separated cutoff radii (Å) for the sparsity pattern (default: same as sparse_hessian_cutoff_real.py).",
    )
    p.add_argument("--first-n-eigvals", type=int, default=10)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to matgl/examples/sparse_hessian_results/cutoff_dft_experiments/",
    )
    p.add_argument(
        "--center-com",
        dest="center_com",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Translate to center of mass before DFT (default: on; avoids huge coords / NaN on big clusters).",
    )
    p.add_argument(
        "--init-guess",
        type=str,
        default="atom",
        choices=("atom", "minao", "huckel", "mod_huckel", "1e", "vsap"),
        help="SCF initial guess (default atom; minao is PySCF default but can misbehave with coarse grids / pruning).",
    )
    p.add_argument(
        "--pyscf-verbose",
        type=int,
        default=0,
        help="PySCF print level (0=quiet). Try 4 or 6 to stream SCF iterations.",
    )
    return p.parse_args()


ARGS = parse_args()
STRUCTURE_PATH = ARGS.structure.resolve()
OUTPUT_DIR = ARGS.output_dir
if OUTPUT_DIR is None:
    OUTPUT_DIR = _SCRIPT_DIR / "sparse_hessian_results" / "cutoff_dft_experiments"
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


def ase_to_pyscf_atom_list(atoms) -> list:
    syms = atoms.get_chemical_symbols()
    pos = atoms.get_positions()
    return [(s, tuple(float(x) for x in p)) for s, p in zip(syms, pos)]


def _parse_atom_grid(s: str) -> tuple[int, int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"Expected 'RADIAL,LEBEDEV' for --atom-grid, got {s!r}")
    return int(parts[0]), int(parts[1])


def _make_rks_mf(mol, xc: str, *, density_fitting: bool, backend: str):
    """Return a restricted Kohn–Sham DFT mean-field object (PySCF or GPU4PySCF)."""
    b = backend.lower().strip()
    if b == "pyscf":
        from pyscf.dft import rks as rks_mod
    elif b == "gpu4pyscf":
        from gpu4pyscf.dft import rks as rks_mod
    else:
        raise ValueError(f"Unknown --backend {backend!r} (use pyscf or gpu4pyscf)")
    mf = rks_mod.RKS(mol, xc=xc)
    if density_fitting:
        mf = mf.density_fit()
    return mf


def build_scf_mf(
    atoms,
    basis: str,
    xc: str,
    charge: int,
    spin: int,
    conv_tol: float,
    *,
    verbose: int = 0,
    max_cycle: int = 50,
    damping: float = 0.0,
    level_shift: float = 0.0,
    second_order_scf: bool = False,
    atom_grid: tuple[int, int] = (43, 194),
    direct_scf_tol: float = 1e-13,
    small_rho_cutoff: float = 0.0,
    density_fitting: bool = True,
    init_guess: str = "atom",
    backend: str = "pyscf",
):
    mol = pyscf.M(
        atom=ase_to_pyscf_atom_list(atoms),
        basis=basis,
        unit="Ang",
        charge=charge,
        spin=spin,
        verbose=verbose,
    )
    mol.build()
    mf = _make_rks_mf(mol, xc, density_fitting=density_fitting, backend=backend)
    mf.verbose = verbose
    mf.conv_tol = conv_tol
    mf.max_cycle = max_cycle
    mf.damp = damping
    mf.level_shift = level_shift
    mf.direct_scf_tol = direct_scf_tol
    mf.grids.atom_grid = atom_grid
    mf.small_rho_cutoff = small_rho_cutoff
    mf.init_guess = init_guess
    if second_order_scf:
        if not hasattr(mf, "newton"):
            raise RuntimeError(f"{backend} mean-field object does not support second-order SCF")
        mf = mf.newton()
        mf.verbose = verbose
        mf.conv_tol = conv_tol
        mf.max_cycle = max_cycle
        mf.damp = damping
        mf.level_shift = level_shift
        mf.direct_scf_tol = direct_scf_tol
        if hasattr(mf, "grids"):
            mf.grids.atom_grid = atom_grid
        if hasattr(mf, "small_rho_cutoff"):
            mf.small_rho_cutoff = small_rho_cutoff
        if hasattr(mf, "init_guess"):
            mf.init_guess = init_guess
    natm = mol.natm
    df_note = "density fitting" if density_fitting else "no density fitting"
    scf_note = "second-order SCF" if second_order_scf else "DIIS SCF"
    print(
        f"  Starting first SCF (backend={backend!r}, {natm} atoms, {df_note}, {scf_note}, grid={atom_grid}, "
        f"init_guess={init_guess!r}, direct_scf_tol={direct_scf_tol:g}, "
        f"small_rho_cutoff={small_rho_cutoff:g}, damping={damping:g}, "
        f"level_shift={level_shift:g}, conv_tol={conv_tol:g}) …",
        flush=True,
    )
    print(
        "  (Large systems can sit here for many minutes with no output if --pyscf-verbose 0.)",
        flush=True,
    )
    t0 = time.perf_counter()
    e_tot = mf.kernel()
    print(f"  First SCF finished in {time.perf_counter() - t0:.1f}s.", flush=True)
    if not getattr(mf, "converged", False):
        raise RuntimeError(
            f"Reference SCF did not converge (E={e_tot}); not attempting analytical Hessian."
        )
    return mol, mf


def hessian_tensor_to_numpy(H) -> np.ndarray:
    if hasattr(H, "detach"):
        return np.asarray(H.detach().cpu().numpy(), dtype=float)
    if hasattr(H, "toarray"):
        return np.asarray(H.toarray(), dtype=float)
    return np.asarray(H, dtype=float)


def analytical_hessian_matrix(mol, mf, atom_grid: tuple[int, int], conv_tol_cpscf: float) -> np.ndarray:
    """Compute and flatten an analytical PySCF/GPU4PySCF nuclear Hessian."""
    hess_obj = mf.Hessian()
    if hasattr(hess_obj, "auxbasis_response"):
        hess_obj.auxbasis_response = 2
    if hasattr(mf, "conv_tol_cpscf"):
        mf.conv_tol_cpscf = conv_tol_cpscf
    if hasattr(mf, "cphf_grids") and hasattr(mf.cphf_grids, "atom_grid"):
        mf.cphf_grids.atom_grid = atom_grid

    t0 = time.perf_counter()
    h4 = hess_obj.kernel()
    elapsed = time.perf_counter() - t0
    print(f"Analytical Hessian finished in {elapsed:.2f}s.", flush=True)

    h4 = np.asarray(h4, dtype=float)
    natm = mol.natm
    h2 = h4.transpose(0, 2, 1, 3).reshape(3 * natm, 3 * natm)
    return 0.5 * (h2 + h2.T), elapsed


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
print("Sparse Hessian cutoff experiment (DFT analytical Hessian + emulated HVPs)")
print("=" * 80)
print(f"Structure: {STRUCTURE_PATH}")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
atom_grid = _parse_atom_grid(ARGS.atom_grid)
print(
    f"Backend: {ARGS.backend}, XC: {ARGS.xc}, basis: {ARGS.basis}, "
    f"grid: {atom_grid}, conv_tol: {ARGS.conv_tol:g}, "
    f"direct_scf_tol: {ARGS.direct_scf_tol:g}, conv_tol_cpscf: {ARGS.conv_tol_cpscf:g}"
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
        f"WARNING: {n_atom} atoms → {n_dof} Cartesian DOF. Analytical DFT Hessian is still "
        "the expensive step, but sparse HVPs are emulated from that matrix.",
        flush=True,
    )
    print(
        "Use a smaller --structure for quick tests; Chignolin can still be a long Hessian job.",
        flush=True,
    )
    print("!" * 72 + "\n", flush=True)

if ARGS.load_full_hessian is None:
    print(
        "\nBuilding DFT driver and equilibrating SCF at the reference geometry…",
        flush=True,
    )
    mol, mf = build_scf_mf(
        atoms,
        basis=ARGS.basis,
        xc=ARGS.xc,
        charge=ARGS.charge,
        spin=ARGS.spin,
        conv_tol=ARGS.conv_tol,
        verbose=ARGS.pyscf_verbose,
        max_cycle=ARGS.max_cycle,
        damping=ARGS.damping,
        level_shift=ARGS.level_shift,
        atom_grid=atom_grid,
        direct_scf_tol=ARGS.direct_scf_tol,
        small_rho_cutoff=ARGS.small_rho_cutoff,
        second_order_scf=ARGS.second_order_scf,
        density_fitting=not ARGS.no_density_fitting,
        init_guess=ARGS.init_guess,
        backend=ARGS.backend,
    )

    print(f"\n[1/4] Analytical DFT reference Hessian ({n_dof}×{n_dof})...", flush=True)
    hessian_ref, ref_hessian_time_s = analytical_hessian_matrix(
        mol, mf, atom_grid, ARGS.conv_tol_cpscf
    )
    full_hessian_npy_path = OUTPUT_DIR / f"full_analytical_hessian_{timestamp}.npy"
    np.save(full_hessian_npy_path, hessian_ref)
    print(f"Saved full analytical Hessian to: {full_hessian_npy_path}", flush=True)
else:
    full_hessian_npy_path = ARGS.load_full_hessian.resolve()
    print(f"\n[1/4] Loading full analytical Hessian from {full_hessian_npy_path}", flush=True)
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
    """Emulate an HVP with the already-computed analytical Hessian."""
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
        f"(full Hessian DOFs = {full_hessian_rows}; HVPs are emulated as H_true @ v)"
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


ylabel_maxabs = "Max absolute error vs analytical Hessian"

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
        label="Analytical Hessian reference time",
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
        label="Analytical Hessian reference time",
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
        label="Analytical Hessian reference time",
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
        label="Analytical Hessian reference time",
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
    "ref_analytical_hessian_time_s": np.asarray([ref_hessian_time_s], dtype=float),
    "full_hessian_rows": np.asarray([full_hessian_rows], dtype=int),
    "full_analytical_hessian": np.asarray(hessian_ref, dtype=float),
    "full_analytical_hessian_path": np.asarray([str(full_hessian_npy_path)], dtype=object),
    "xc": np.asarray([ARGS.xc], dtype=object),
    "dft_backend": np.asarray([ARGS.backend], dtype=object),
    "basis": np.asarray([ARGS.basis], dtype=object),
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
        "method": np.asarray([f"{ARGS.backend}_analytical_hessian_emulated_hvp_{ARGS.xc}"], dtype=object),
        "dft_backend": np.asarray([ARGS.backend], dtype=object),
        "full_hessian_rows": np.asarray([full_hessian_rows], dtype=int),
        "ref_analytical_hessian_time_s": np.asarray([ref_hessian_time_s], dtype=float),
        "full_analytical_hessian_path": np.asarray([str(full_hessian_npy_path)], dtype=object),
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
