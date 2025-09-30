import numpy as np
import scipy.sparse as sp
import torch

from colpackpy.coloring import get_seeds_from_pattern
from colpackpy.recovery import recover_matrix

ALGORITHM = "ACYCLIC"
ORDERING = "SMALLEST_LAST"

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
A_np, edges = make_random_spd_matrix(n=n, avg_deg=avg_deg, seed=42, return_dense=True)
sparsity_pattern = edges

# ---- ColPack seeds (acyclic for indirect recovery) ----
S, colors, comps = get_seeds_from_pattern(sparsity_pattern, n, ORDERING, ALGORITHM)
S_np = np.asarray(S)

print("Seed matrix shape:", S_np.shape)
print("Number of colors:", S_np.shape[1])

# ---- Convert to PyTorch tensors ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

A_torch = torch.tensor(A_np, dtype=torch.float64, device=device)
S_torch = torch.tensor(S_np, dtype=torch.float64, device=device)

# ---- Create a trainable scaling parameter ----
scale_param = torch.tensor([2.0], dtype=torch.float64, device=device, requires_grad=True)
print(f"\nInitial scaling parameter: {scale_param.item():.4f}")

# ---- Scale the matrix (attaches computational graph) ----
# Square the parameter to make gradient more interesting
A_scaled = (scale_param ** 2) * A_torch

# ---- Compressed evaluation: Y = (scale^2 * A) @ S ----
Y = A_scaled @ S_torch

print(f"Compressed matrix Y shape: {Y.shape}")
print(f"Y requires_grad: {Y.requires_grad}")

# ---- Reconstruct via indirect (acyclic) recovery ----
A_rec = recover_matrix(
    recover_from=ALGORITHM,
    edges=edges,
    n=n,
    Y=Y,
    vertex_colors=colors,
    acyclic_components=comps,
    reconstruct_sparsely=False,  # PyTorch backend always returns dense
    verify=True
)

print(f"\nRecovered matrix A_rec shape: {A_rec.shape}")
print(f"A_rec requires_grad: {A_rec.requires_grad}")

# ---- Compute a scalar loss (sum of all matrix elements) ----
loss = A_rec.sum()
print(f"\nLoss (sum of recovered matrix): {loss.item():.4f}")

# ---- Backpropagate to compute gradient w.r.t. scale parameter ----
loss.backward()

print(f"\nGradient of loss w.r.t. scale_param: {scale_param.grad.item():.4f}")

# ---- Verify the gradient analytically ----
# Since A_rec = scale^2 * A, and loss = sum(A_rec) = sum(scale^2 * A) = scale^2 * sum(A)
# Therefore, d(loss)/d(scale) = 2 * scale * sum(A)
expected_gradient = 2 * scale_param.item() * A_torch.sum().item()
print(f"Expected gradient (2 * scale * sum(A)): {expected_gradient:.4f}")
print(f"Gradient error: {abs(scale_param.grad.item() - expected_gradient):.2e}")

# ---- Verify the recovered matrix is correct ----
A_rec_expected = A_scaled.detach()
max_diff = torch.abs(A_rec - A_rec_expected).max().item()
print(f"\nMax difference between recovered and expected matrix: {max_diff:.2e}")

print("\n" + "="*60)
print("Gradient example completed successfully!")
print("="*60)

