import os
# To supress warning from the use_deterministic_algorithms=True
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import numpy as np
import torch
torch.set_default_device("cuda")

# Enable deterministic algorithms for PyTorch
torch.use_deterministic_algorithms(True, warn_only=True)

import matgl
from ase.io import read
from matgl.ext.ase import PESCalculator



pot = matgl.load_model("../pretrained_models/TensorNet-ANI-1x-Subset-PES")
n_layers = len(pot.model.layers)
pot.calc_hessian = True
pot.return_seed_matrix = True
# pot.hessian_from_finite_differences = True
pot.hessian_delta = 5e-3
atoms = read('/h/28/lucahiede/Projects/sparse_hessian_ff/data/Water Cluster/76.xyz')

calc = PESCalculator(pot)

pot.hessian_algorithm = "naive"
calc.calculate(atoms, properties=['hessian'])
hessian_naive = np.array(calc.results['hessian'])


pot.hessian_algorithm = "distance_two"
calc.calculate(atoms, properties=['hessian'])
hessian_distance_two = np.array(calc.results['hessian'])
seed_matrix_distance_two = np.array(calc.results['seed_matrix'])
print('Distance two seed matrix shape:', seed_matrix_distance_two.shape)

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

print('Max percentage difference between naive and distance two:', (np.abs(hessian_naive - hessian_distance_two)/(np.abs(hessian_naive)+1e-4)).max())
print('Max percentage difference between naive and star:', (np.abs(hessian_naive - hessian_star)/(np.abs(hessian_naive)+1e-4)).max())
print('Max percentage difference between naive and acyclic:', (np.abs(hessian_naive - hessian_acyclic)/(np.abs(hessian_naive)+1e-4)).max())

first_n_eigvals = 100
print('Average percentage difference between naive and distance two of first {} eigvals:'.format(first_n_eigvals), (np.abs(np.linalg.eigvals(hessian_naive)[:first_n_eigvals] - np.linalg.eigvals(hessian_distance_two)[:first_n_eigvals])/(np.abs(np.linalg.eigvals(hessian_naive)[:first_n_eigvals])+1e-5)).mean())
print('Average percentage difference between naive and star of first {} eigvals:'.format(first_n_eigvals), (np.abs(np.linalg.eigvals(hessian_naive)[:first_n_eigvals] - np.linalg.eigvals(hessian_star)[:first_n_eigvals])/(np.abs(np.linalg.eigvals(hessian_naive)[:first_n_eigvals])+1e-5)).mean())
print('Average percentage difference between naive and acyclic of first {} eigvals:'.format(first_n_eigvals), (np.abs(np.linalg.eigvals(hessian_naive)[:first_n_eigvals] - np.linalg.eigvals(hessian_acyclic)[:first_n_eigvals])/(np.abs(np.linalg.eigvals(hessian_naive)[:first_n_eigvals])+1e-5)).mean())
