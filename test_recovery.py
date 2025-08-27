import numpy as np
# from colpack_binding import get_colpack_seeds
from colpackpy import get_colpack_seeds_simplified
import scipy.sparse as sp
from recovery import recover_matrix

ALGORITHM = "ACYCLIC_FOR_INDIRECT_RECOVERY"
ORDERING = "SMALLEST_LAST"

use_simplified_binding = True  # Toggle this to switch between bindings

edges = np.array([
    [0, 1], [0, 3], [0,9], [1, 4], [2, 3], [3, 4], [4, 5],
    [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11],
    [0, 2], [1, 3], [3, 7], [2, 9] 
])
# edges =  np.array([[0,1],[0,2], [1,2],[2,3]])
n = max(max(u, v) for u, v in edges) + 1
A = sp.csr_matrix((np.ones(len(edges)), (edges[:,0], edges[:,1])), shape=(n,n))
A = (A + A.T)
print(A.toarray())

# Use the ColPack binding to get seeds_d2, K_d2, and d2_raw
seeds_d2, d2_raw = get_colpack_seeds_simplified(edges, n, ORDERING, ALGORITHM)
seeds_d2 = np.array(seeds_d2)
d2_raw = np.array(d2_raw)
B_d2 = A @ seeds_d2
    

print('Seed matrix shape:', seeds_d2.shape)





reconstructed_A = recover_matrix(ALGORITHM, edges, n, B_d2, d2_raw, reconstruct_sparsely=False)


# Verify reconstruction
print('Max reconstruction error:', np.abs(A - reconstructed_A).max())
assert np.array_equal(A.toarray(), reconstructed_A), "Reconstruction failed!"
print("Matrix A successfully reconstructed.")

# Dont forget to call before running: export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/h/28/lucahiede/Projects/sparse_hessian_ff/ColPack/build/cmake/mywork/lib/shared_library