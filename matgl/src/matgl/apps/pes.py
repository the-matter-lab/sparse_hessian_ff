"""Implementation of Interatomic Potentials."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn
from torch.autograd import grad
from torch.autograd.functional import hvp as torch_hvp
from torch.func import jvp

import matgl
from matgl.layers import AtomRef, NuclearRepulsion
from matgl.utils.io import IOMixIn
from matgl.graph.compute import create_sparsity_pattern_from_graph, create_sparsity_pattern_from_cutoff
try:
    from colpackpy.coloring import get_seeds_from_pattern   # returns (S, colors, comps)
    from colpackpy.recovery import recover_matrix            # your function we modified
    from colpackpy.torch_utils import compute_sparse_hessian, compute_sparse_hessian_fn
    COLPACK_AVAILABLE = True
except ImportError:
    COLPACK_AVAILABLE = False
    print("colpackpy not available, using naive implementation for hessian calculation")
from scipy.sparse import coo_matrix


if TYPE_CHECKING:
    import dgl


def hvp(f, primals, tangents):
    """Compute Hessian-Vector Product using PyTorch's built-in hvp function.
    
    Args:
        f: Function to compute HVP for
        primals: Input variables (tuple)
        tangents: Tangent vectors (tuple)
    
    Returns:
        HVP result
    """
    # Use PyTorch's built-in hvp function
    func_output, hvp_result = torch_hvp(f, primals, tangents)
    return func_output, hvp_result


class Potential(nn.Module, IOMixIn):
    """A class representing an interatomic potential."""

    __version__ = 3

    def __init__(
        self,
        model: nn.Module,
        data_mean: torch.Tensor | float = 0.0,
        data_std: torch.Tensor | float = 1.0,
        element_refs: torch.Tensor | np.ndarray | None = None,
        calc_forces: bool = True,
        calc_stresses: bool = True,
        calc_hessian: bool = False,
        calc_magmom: bool = False,
        calc_repuls: bool = False,
        zbl_trainable: bool = False,
        debug_mode: bool = False,
        hessian_algorithm: str = "naive", # "naive" | "distance_two" | "star" | "acyclic"
        hessian_from_finite_differences: bool = False, # Whether to use finite differences or autodiff to compute Hessian-vector products
        hessian_delta: float = 1e-2, # Finite difference step size
        hessian_return_type: str = "dense", # "sparse" | "dense"
        return_seed_matrix: bool = False, # Return seed matrix along with hessian
        hessian_sparsity_cutoff: float | None = None, # Cutoff radius for sparsity pattern (Angstroms). If None, uses graph-based sparsity from num_layers
    ):
        """Initialize Potential from a model and elemental references.

        Args:
            model: Model for predicting energies.
            data_mean: Mean of target.
            data_std: Std dev of target.
            element_refs: Element reference values for each element.
            calc_forces: Enable force calculations.
            calc_stresses: Enable stress calculations.
            calc_hessian: Enable hessian calculations.
            calc_magmom: Enable site-wise property calculation.
            calc_repuls: Whether the ZBL repulsion is included
            zbl_trainable: Whether zbl repulsion is trainable
            debug_mode: Return gradient of total energy with respect to atomic positions and lattices for checking
            hessian_algorithm: Algorithm to use for hessian calculation. Either "naive" (direct), "star" (sparse, stable but medium reduction in HVPs), or "acyclic" (needs to solve linear system, optimal reduction of HVPs).
            hessian_from_finite_differences: Whether to use finite differences or autodiff to compute Hessian-vector products
            hessian_delta: Finite difference step size
            hessian_return_type: Return type for hessian calculation, sparse or dense arrays
            return_seed_matrix: Whether to return the seed matrix along with the hessian for sparse algorithms
            hessian_sparsity_cutoff: Cutoff radius for sparsity pattern (Angstroms). If None, uses graph-based sparsity from num_layers. If provided, creates sparsity pattern based on geometric distance rather than graph connectivity.
        """
        super().__init__()
        self.save_args(locals())
        self.model = model
        self.calc_forces = calc_forces or calc_hessian # if hessian is calculated, forces are also calculated
        self.calc_stresses = calc_stresses
        self.calc_hessian = calc_hessian
        self.calc_magmom = calc_magmom
        self.element_refs: AtomRef | None
        self.debug_mode = debug_mode
        self.calc_repuls = calc_repuls
        self.hessian_algorithm = hessian_algorithm
        self.hessian_return_type = hessian_return_type
        self.return_seed_matrix = return_seed_matrix
        self.hessian_from_finite_differences = hessian_from_finite_differences
        self.hessian_delta = hessian_delta
        self.hessian_sparsity_cutoff = hessian_sparsity_cutoff
        if calc_repuls:
            self.repuls = NuclearRepulsion(self.model.cutoff, trainable=zbl_trainable)

        if element_refs is not None:
            if not isinstance(element_refs, torch.Tensor):
                element_refs = torch.tensor(element_refs, dtype=matgl.float_th)
            self.element_refs = AtomRef(property_offset=element_refs)
        else:
            self.element_refs = None
        # for backward compatibility
        if data_mean is None:
            data_mean = 0.0
        if not isinstance(data_mean, torch.Tensor):
            data_mean = torch.tensor(data_mean, dtype=matgl.float_th)
        if not isinstance(data_std, torch.Tensor):
            data_std = torch.tensor(data_std, dtype=matgl.float_th)

        self.register_buffer("data_mean", data_mean)
        self.register_buffer("data_std", data_std)

    def forward(
        self,
        g: dgl.DGLGraph,
        lat: torch.Tensor,
        state_attr: torch.Tensor | None = None,
        l_g: dgl.DGLGraph | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Args:
            g: DGL graph
            lat: lattice
            state_attr: State attrs
            l_g: Line graph.

        Returns:
            (energies, forces, stresses, hessian) or (energies, forces, stresses, hessian, site-wise properties)
            When return_seed_matrix=True and using sparse algorithms, hessian becomes (hessian_matrix, seed_matrix)
        """
        # st (strain) for stress calculations
        st = lat.new_zeros([g.batch_size, 3, 3])
        if self.calc_stresses:
            st.requires_grad_(True)
        lattice = lat @ (torch.eye(3, device=lat.device) + st)
        g.edata["lattice"] = torch.repeat_interleave(lattice, g.batch_num_edges(), dim=0)
        g.edata["pbc_offshift"] = (g.edata["pbc_offset"].unsqueeze(dim=-1) * g.edata["lattice"]).sum(dim=1)
        g.ndata["pos"] = (
            g.ndata["frac_coords"].unsqueeze(dim=-1) * torch.repeat_interleave(lattice, g.batch_num_nodes(), dim=0)
        ).sum(dim=1)
        if self.calc_forces:
            g.ndata["pos"].requires_grad_(True)

        total_energies = self.model(g=g, state_attr=state_attr, l_g=l_g)

        total_energies = self.data_std * total_energies + self.data_mean

        if self.calc_repuls:
            total_energies += self.repuls(self.model.element_types, g)

        if self.element_refs is not None:
            property_offset = torch.squeeze(self.element_refs(g))
            total_energies += property_offset

        forces = torch.zeros(1)
        stresses = torch.zeros(1)
        hessian = torch.zeros(1)

        grad_vars = [g.ndata["pos"], st] if self.calc_stresses else [g.ndata["pos"]]

        if self.calc_forces:
            grads = grad(
                total_energies,
                grad_vars,
                grad_outputs=torch.ones_like(total_energies),
                create_graph=True,
                retain_graph=True,
            )
            forces = -grads[0]

            if self.calc_hessian:
                if self.hessian_algorithm == "naive":
                    r = grads[0].view(-1)
                    s = r.size(0)
                    hessian = total_energies.new_zeros((s, s))
                    for iatom in range(s):
                        tmp = grad([r[iatom]], g.ndata["pos"], retain_graph=iatom < s)[0]
                        if tmp is not None:
                            hessian[iatom] = tmp.view(-1)
                    if self.return_seed_matrix:
                        hessian = (hessian, torch.eye(hessian.shape[0], device=hessian.device))
                elif self.hessian_algorithm in ["distance_two", "star", "acyclic"]:
                    # Pass the already computed gradients to avoid recomputation
                    if self.hessian_sparsity_cutoff is not None:
                        # Use cutoff-based sparsity pattern
                        sparsity_pattern, n = create_sparsity_pattern_from_cutoff(g, self.hessian_sparsity_cutoff)
                    else:
                        # Use graph-based sparsity pattern (default)
                        num_layers = len(self.model.layers) if hasattr(self.model, 'layers') else 1
                        sparsity_pattern, n = create_sparsity_pattern_from_graph(g, num_layers)
                    if self.hessian_from_finite_differences:
                        def total_energies_fn(x):
                            g.ndata["pos"] = x
                            total_energies = self.model(g=g, state_attr=state_attr, l_g=l_g)
                            total_energies = self.data_std * total_energies + self.data_mean
                            return total_energies
                        # If we want to use finite differences, we need to use the functional API to provide the function that 
                        # then generates the finite difference Hessian-vector product.
                        hessian_out = compute_sparse_hessian_fn(
                            total_energies_fn,
                            g.ndata["pos"],
                            sparsity_pattern,
                            n,
                            coloring_algorithm=self.hessian_algorithm,
                            return_seed_matrix=self.return_seed_matrix,
                            use_finite_differences=self.hessian_from_finite_differences,
                            delta=self.hessian_delta,
                        )
                    else:
                        # If we want to use autodiff, we need to use the non-functional API because pytorch's functional API
                        # does not support hessian vector products for sparse operations like scatter
                        hessian_out = compute_sparse_hessian(
                            total_energies,
                            g.ndata["pos"],
                            sparsity_pattern,
                            n,
                            self.hessian_algorithm,
                            precomputed_grads=grads[0],
                            return_seed_matrix=self.return_seed_matrix,
                            hessian_return_type=self.hessian_return_type,
                        )
                    hessian = hessian_out

        if self.calc_stresses:
            volume = (
                torch.abs(torch.det(lattice.float())).half()
                if matgl.float_th == torch.float16
                else torch.abs(torch.det(lattice))
            )
            sts = -grads[1]
            scale = 1.0 / volume * -160.21766208
            sts = [i * j for i, j in zip(sts, scale, strict=False)] if sts.dim() == 3 else [sts * scale]  # type:ignore[assignment]
            stresses = torch.cat(sts)  # type:ignore[call-overload]

        if self.debug_mode:
            return total_energies, grads[0], grads[1]

        if self.calc_magmom:
            return total_energies, forces, stresses, hessian, g.ndata["magmom"]

        return total_energies, forces, stresses, hessian
    