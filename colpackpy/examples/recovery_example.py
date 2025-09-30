import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

from colpackpy.coloring import get_seeds_from_pattern   # returns (S, colors, comps)
from colpackpy.recovery import recover_matrix            # your function we modified

ALGORITHM = "ACYCLIC"
ORDERING  = "SMALLEST_LAST"

# ---- helper: build a random sparse SPD matrix (Hessian-like) ----
def make_random_spd_matrix(n=60, avg_deg=6, seed=0, return_dense=False):
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
    if return_dense:
        return A.toarray(), edges
    else:
        return A, edges




# ---- build a realistic test matrix + pattern ----
n = 8
avg_deg = 12
A, edges = make_random_spd_matrix(n=n, avg_deg=avg_deg, seed=42, return_dense=False)
sparsity_pattern = edges

# ---- ColPack seeds (acyclic for indirect recovery) ----
S, colors, comps = get_seeds_from_pattern(sparsity_pattern, n, ORDERING, ALGORITHM)

print("Seed matrix shape:", np.asarray(S).shape)

# ---- Compressed evaluation Y = A @ S ----
B = A @ S


# ---- Reconstruct via indirect (acyclic) recovery using cached components ----
A_rec = recover_matrix(
    recover_from=ALGORITHM,  # accepted alias
    edges=edges,
    n=n,
    Y=B,
    vertex_colors=colors,
    acyclic_components=comps,   # use ColPack's cached two-colored forests
    reconstruct_sparsely=True,
    dtype=np.float64,
    verify=True
)



# ---- Check ----
# Convert to dense if sparse for comparison
A_dense = A.toarray() if sp.issparse(A) else A
A_rec_dense = A_rec.toarray() if sp.issparse(A_rec) else A_rec

diff = (A_dense - A_rec_dense)
print("Max |diff|:", np.abs(diff).max())
print("RMS error  :", np.sqrt((diff**2).mean()))
assert np.allclose(A_dense, A_rec_dense, rtol=0, atol=1e-10)

