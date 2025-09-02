import numpy as np
import torch
torch.set_default_device("cuda")

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


pot = matgl.load_model("../pretrained_models/TensorNet-ANI-1x-Subset-PES")
n_layers = len(pot.model.layers)
pot.calc_hessian = True
pot.return_seed_matrix = True
atoms = read('/h/28/lucahiede/Projects/sparse_hessian_ff/data/water_cluster/32.xyz')

calc = PESCalculator(pot)

pot.hessian_algorithm = "naive"
calc.calculate(atoms, properties=['hessian'])
hessian_naive = np.array(calc.results['hessian'])

pot.hessian_algorithm = "star"
calc.calculate(atoms, properties=['hessian'])
hessian_star = np.array(calc.results['hessian'])
seed_matrix_star = np.array(calc.results['seed_matrix'])
print('Star seed matrix shape:', seed_matrix_star.shape)

pot.hessian_algorithm = "acyclic"
calc.calculate(atoms, properties=['hessian'])
hessian_acyclic = np.array(calc.results['hessian'])
seed_matrix_acyclic = np.array(calc.results['seed_matrix'])
print('Acyclic seed matrix shape:', seed_matrix_acyclic.shape)

print(np.abs(hessian_naive - hessian_star).max())
print(np.abs(hessian_naive - hessian_acyclic).max())
print(np.abs(hessian_star - hessian_acyclic).max())


