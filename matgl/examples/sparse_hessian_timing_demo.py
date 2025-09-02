import numpy as np
import torch
# torch.set_default_device("cuda")

import numpy as np
import matgl
from ase.build import molecule
from ase.io import read
from matgl.ext.ase import Atoms2Graph, PESCalculator
import matplotlib
# matplotlib.use('TkAgg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
import time

from colpackpy.coloring import get_seeds_from_pattern   # returns (S, colors, comps)
from colpackpy.recovery import recover_matrix            # your function we modified

def create_sparse_adjacency_matrix(atoms, cutoff, element_types):
    graph, lattice, state_attr_default = Atoms2Graph(element_types, cutoff).get_graph(atoms)
    sources, destinations = graph.edges()
    num_nodes = int(max(sources.max(), destinations.max()) + 1)
    adjacency_matrix_sparse = coo_matrix((np.ones(len(sources)), (sources.cpu().numpy(), destinations.cpu().numpy())), shape=(num_nodes, num_nodes))
    return adjacency_matrix_sparse, num_nodes

pot = matgl.load_model("../pretrained_models/TensorNet-ANI-1x-Subset-PES")
n_layers = len(pot.model.layers)
pot.calc_hessian = True
# atoms = read('/h/28/lucahiede/Projects/sparse_hessian_ff/data/AT-AT-CG-CG_random_point_double.xyz')
atoms = read('/h/28/lucahiede/Projects/sparse_hessian_ff/data/water_cluster/5.xyz')

element_types = pot.model.element_types
cutoff = pot.model.cutoff 
adjacency_matrix_sparse, n = create_sparse_adjacency_matrix(atoms, cutoff, element_types)
adjacency_matrix = adjacency_matrix_sparse.toarray()


sparsity_pattern = adjacency_matrix_sparse#((adjacency_matrix_sparse @ adjacency_matrix_sparse) != 0).tocoo()
# sparsity_pattern = coo_matrix(np.kron(sparsity_pattern.toarray(), np.ones((3,3))))
# n = n * 3
edges = np.vstack((sparsity_pattern.row, sparsity_pattern.col)).T
A = sparsity_pattern


ALGORITHM = "DISTANCE_TWO"
ORDERING  = "SMALLEST_LAST"
time1 = time.time()
S, colors, comps = get_seeds_from_pattern(edges, n, ORDERING, ALGORITHM)
S = np.asarray(S, dtype=float)
colors = np.asarray(colors, dtype=int)

print("Seed matrix shape:", S.shape, "(n_colors =", S.shape[1], ")")

# ---- Compressed evaluation Y = A @ S ----
time2 = time.time()
B = A @ S
time3 = time.time()

# ---- Reconstruct via indirect (acyclic) recovery using cached components ----
A_rec = recover_matrix(
    recover_from=ALGORITHM,  # accepted alias
    edges=edges,
    n=n,
    B=B,
    vertex_colors=colors,
    acyclic_components=comps,   # use ColPack’s cached two-colored forests
    reconstruct_sparsely=False,
    dtype=np.float64,
    verify=True
)

diff = (A - A_rec)
print("Max |diff|:", np.abs(diff).max())
print("Time taken:", time.time() - time1 - (time3 - time2))