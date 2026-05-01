"""Parse an ORCA .hess file and plot the Cartesian Hessian matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import SymLogNorm


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_HESSIAN = (
    SCRIPT_DIR / "sparse_hessian_results" / "orca_chingolin" / "pbe_orca_hessian_chingolin.hess"
)


def parse_orca_hessian(path: Path) -> np.ndarray:
    """Read the dense Hessian from ORCA's block-column ``$hessian`` section."""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip() == "$hessian":
                break
        else:
            raise ValueError(f"No $hessian section found in {path}")

        try:
            n_dof = int(next(handle).strip())
        except StopIteration as exc:
            raise ValueError(f"Missing Hessian dimension after $hessian in {path}") from exc

        hessian = np.zeros((n_dof, n_dof), dtype=float)
        filled = np.zeros((n_dof, n_dof), dtype=bool)

        while True:
            header = next_nonempty_line(handle)
            if header is None:
                raise ValueError(f"Unexpected end of file while reading $hessian in {path}")
            if header.lstrip().startswith("$"):
                break

            columns = [int(token) for token in header.split()]
            for _ in range(n_dof):
                row_line = next(handle)
                parts = row_line.split()
                if len(parts) != len(columns) + 1:
                    raise ValueError(f"Malformed Hessian row: {row_line.rstrip()!r}")
                row = int(parts[0])
                values = [float(token) for token in parts[1:]]
                hessian[row, columns] = values
                filled[row, columns] = True

        if not filled.all():
            missing = int(filled.size - filled.sum())
            raise ValueError(f"Parsed Hessian is incomplete: {missing} entries missing")
    return hessian


def next_nonempty_line(handle) -> str | None:
    for line in handle:
        stripped = line.strip()
        if stripped:
            return stripped
    return None


def parse_orca_atoms_count(path: Path) -> int | None:
    """Return the atom count from ``$atoms`` when present."""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip() == "$atoms":
                try:
                    return int(next(handle).strip())
                except StopIteration as exc:
                    raise ValueError(f"Missing atom count after $atoms in {path}") from exc
    return None


def subtract_atom_block_diagonal(hessian: np.ndarray, atoms_count: int) -> np.ndarray:
    """Zero the per-atom 3x3 diagonal blocks, leaving inter-atom couplings."""
    n_dof = hessian.shape[0]
    if hessian.shape != (3 * atoms_count, 3 * atoms_count):
        raise ValueError(
            f"Hessian shape {hessian.shape} is inconsistent with {atoms_count} atoms "
            f"({3 * atoms_count} Cartesian DOFs)"
        )

    offdiag = hessian.copy()
    for atom_idx in range(atoms_count):
        block = slice(3 * atom_idx, 3 * (atom_idx + 1))
        offdiag[block, block] = 0.0
    return offdiag


def plot_hessian(hessian: np.ndarray, output_path: Path, *, dpi: int) -> None:
    abs_max = float(np.nanmax(np.abs(hessian))) if hessian.size else 1.0
    if not np.isfinite(abs_max) or abs_max <= 0.0:
        abs_max = 1.0

    nonzero_abs = np.abs(hessian[np.nonzero(hessian)])
    if nonzero_abs.size:
        linthresh = max(float(np.percentile(nonzero_abs, 5)), abs_max * 1e-6)
    else:
        linthresh = abs_max * 1e-6

    n_dof = hessian.shape[0]
    side = float(np.clip(n_dof * 0.025, 8.0, 18.0))
    fig, ax = plt.subplots(figsize=(side + 1.4, side + 1.0))
    im = ax.imshow(
        hessian,
        cmap="coolwarm",
        norm=SymLogNorm(linthresh=linthresh, vmin=-abs_max, vmax=abs_max, base=10),
        aspect="equal",
        interpolation="nearest",
        rasterized=True,
    )
    ax.set_title(f"ORCA Cartesian Hessian ({n_dof} x {n_dof})")
    ax.set_xlabel("column (Cartesian DOF)")
    ax.set_ylabel("row (Cartesian DOF)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Hessian value (a.u., symlog scale)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_abs_hessian(hessian: np.ndarray, output_path: Path, *, dpi: int, title: str | None = None) -> None:
    abs_hessian = np.abs(hessian)
    vmax = float(np.nanmax(abs_hessian)) if abs_hessian.size else 1.0
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0

    n_dof = hessian.shape[0]
    side = float(np.clip(n_dof * 0.025, 8.0, 18.0))
    fig, ax = plt.subplots(figsize=(side + 1.4, side + 1.0))
    im = ax.imshow(
        abs_hessian,
        vmin=0.0,
        vmax=vmax,
        cmap="magma",
        aspect="equal",
        interpolation="nearest",
        rasterized=True,
    )
    ax.set_title(title or f"|ORCA Cartesian Hessian| ({n_dof} x {n_dof})")
    ax.set_xlabel("column (Cartesian DOF)")
    ax.set_ylabel("row (Cartesian DOF)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="|Hessian value| (a.u.)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hessian",
        type=Path,
        default=DEFAULT_HESSIAN,
        help="Input ORCA .hess file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Signed Hessian PNG path. Defaults to <hessian-dir>/<hessian-stem>_hessian.png.",
    )
    parser.add_argument(
        "--abs-output",
        type=Path,
        default=None,
        help="Absolute-value Hessian PNG path. Defaults to <hessian-dir>/<hessian-stem>_hessian_abs.png.",
    )
    parser.add_argument(
        "--offdiag-abs-output",
        type=Path,
        default=None,
        help=(
            "Absolute-value Hessian after subtracting per-atom 3x3 diagonal blocks. "
            "Defaults to <hessian-dir>/<hessian-stem>_hessian_offdiag_abs.png."
        ),
    )
    parser.add_argument("--dpi", type=int, default=360, help="DPI for saved PNG figures.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hessian_path = args.hessian.resolve()
    output_path = args.output or hessian_path.with_name(f"{hessian_path.stem}_hessian.png")
    abs_output_path = args.abs_output or hessian_path.with_name(f"{hessian_path.stem}_hessian_abs.png")
    offdiag_abs_output_path = args.offdiag_abs_output or hessian_path.with_name(
        f"{hessian_path.stem}_hessian_offdiag_abs.png"
    )

    print(f"Reading ORCA Hessian: {hessian_path}")
    hessian = parse_orca_hessian(hessian_path)
    atoms_count = parse_orca_atoms_count(hessian_path)
    if atoms_count is not None and hessian.shape != (3 * atoms_count, 3 * atoms_count):
        raise ValueError(
            f"Hessian shape {hessian.shape} is inconsistent with {atoms_count} atoms "
            f"({3 * atoms_count} Cartesian DOFs)"
        )

    symmetry_error = float(np.max(np.abs(hessian - hessian.T)))
    print(f"Parsed Hessian shape: {hessian.shape}")
    if atoms_count is not None:
        print(f"Parsed atoms: {atoms_count}")
    print(f"Max |H - H.T|: {symmetry_error:.3e}")

    plot_hessian(hessian, output_path.resolve(), dpi=args.dpi)
    plot_abs_hessian(hessian, abs_output_path.resolve(), dpi=args.dpi)
    if atoms_count is not None:
        offdiag_hessian = subtract_atom_block_diagonal(hessian, atoms_count)
        plot_abs_hessian(
            offdiag_hessian,
            offdiag_abs_output_path.resolve(),
            dpi=args.dpi,
            title=r"$|H_{\mathrm{ORCA}} - \mathrm{blkdiag}_{3\times3}(H_{\mathrm{ORCA}})|$",
        )
    print(f"Saved signed Hessian plot: {output_path.resolve()}")
    print(f"Saved absolute Hessian plot: {abs_output_path.resolve()}")
    if atoms_count is not None:
        print(f"Saved off-diagonal absolute Hessian plot: {offdiag_abs_output_path.resolve()}")


if __name__ == "__main__":
    main()
