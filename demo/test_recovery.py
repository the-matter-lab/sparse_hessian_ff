import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

from colpackpy.coloring import get_seeds_from_pattern   # returns (S, colors, comps)
from colpackpy.recovery import recover_matrix            # your function we modified

ALGORITHM = "ACYCLIC"
ORDERING  = "SMALLEST_LAST"

# ---- helper: build a random sparse SPD matrix (Hessian-like) ----
def make_random_spd_matrix(n=60, avg_deg=6, seed=0):
    rng = np.random.default_rng(seed)
    p = max(1e-6, avg_deg / (n - 1))  # Erdos–Renyi probability

    # Random upper-triangular sparsity (no diagonal)
    mask = rng.random((n, n)) < p
    iu, ju = np.triu_indices(n, k=1)
    keep = mask[iu, ju]
    i = iu[keep]; j = ju[keep]

    # Negative off-diagonal weights (common for stiffness/Hessians)
    w = -np.abs(rng.normal(loc=0.0, scale=1.0, size=i.size))

    # Build symmetric off-diagonals
    rows_off = np.concatenate([i, j])
    cols_off = np.concatenate([j, i])
    data_off = np.concatenate([w, w])

    # Strict diagonal dominance → SPD
    row_abs = np.bincount(rows_off, weights=np.abs(data_off), minlength=n)
    diag = row_abs + rng.uniform(0.1, 1.0, size=n)

    rows = np.concatenate([rows_off, np.arange(n)])
    cols = np.concatenate([cols_off, np.arange(n)])
    data = np.concatenate([data_off, diag])

    A = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    # Build the sparsity pattern (0/1), excluding diagonal
    edges = np.column_stack([i, j]).astype(int)
    return A, edges

# ---- build a realistic test matrix + pattern ----
n = 80
avg_deg = 12
A, edges = make_random_spd_matrix(n=n, avg_deg=avg_deg, seed=42)

print("n =", n, "nnz =", A.nnz, "avg degree ≈", 2 * len(edges) / n)

# ---- ColPack seeds (acyclic for indirect recovery) ----
S, colors, comps = get_seeds_from_pattern(edges, n, ORDERING, ALGORITHM)
S = np.asarray(S, dtype=float)
colors = np.asarray(colors, dtype=int)

print("Seed matrix shape:", S.shape, "(n_colors =", S.shape[1], ")")

# ---- Compressed evaluation Y = A @ S ----
B = A @ S

# ---- Reconstruct via indirect (acyclic) recovery using cached components ----
A_rec = recover_matrix(
    recover_from="ACYCLIC_FOR_INDIRECT_RECOVERY",  # accepted alias
    edges=edges,
    n=n,
    B=B,
    vertex_colors=colors,
    acyclic_components=comps,   # use ColPack’s cached two-colored forests
    reconstruct_sparsely=False,
    dtype=np.float64,
    verify=True
)

plt.matshow(A.toarray())
plt.savefig("A.png")

plt.matshow(A_rec)
plt.savefig("A_rec.png")

# ---- Check ----
diff = (A.toarray() - A_rec)
print("Max |diff|:", np.abs(diff).max())
print("RMS error  :", np.sqrt((diff**2).mean()))
assert np.allclose(A.toarray(), A_rec, rtol=0, atol=1e-10)