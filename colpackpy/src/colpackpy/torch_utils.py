"""PyTorch utilities for efficient sparse Hessian computation using ColPack."""

from __future__ import annotations

from typing import Callable
import time
import torch
from torch import Tensor
from torch.func import jacrev, jvp, vmap
import numpy as np
from torch.autograd import grad


from colpackpy.coloring import get_seeds_from_pattern  # type: ignore[reportMissingImports]
from colpackpy.recovery import recover_matrix


def get_grad_fn(
    f: Callable[[Tensor], Tensor],
) -> Callable[[Tensor], Tensor]:
    """Generate function to compute the gradient of f at x.

    Args:
        f: The function whose gradient we want to compute.
            It should take the input tensor as argument and return a scalar output tensor.

    Returns:
        A function that computes the gradient of f at the input tensor x.
        The gradient has the same shape as x.
    """

    def grad_fn(x: Tensor) -> Tensor:
        """Compute the gradient of f evaluated at x.

        Args:
            x: The input to the function at which the gradient is computed.

        Returns:
            The gradient of f at x. Has the same shape as x.
        """
        x_grad = x.clone().detach().requires_grad_(True)
        y = f(x_grad)
        g = grad(y, x_grad, create_graph=False)[0]
        return g

    return grad_fn


def get_hessian_vector_product_fn(
    f: Callable[[Tensor], Tensor],
    *,
    use_finite_differences: bool = False,
    delta: float = 1e-2,
) -> Callable[[Tensor, Tensor], Tensor]:
    """Generate function to compute the Hessian-vector product of f at x with v.

    Args:
        f: The function whose Hessian-vector product we want to compute.
            It should take the input tensor as argument and return the output tensor.
        use_finite_differences: If True, use finite differences instead of automatic differentiation.
        delta: Finite difference step size (only used if use_finite_differences is True).

    Returns:
        A function that computes the Hessian-vector product of f at the input
        tensor x given x and the vector. Has shape [*f(x).shape, *x.shape].
    """

    if use_finite_differences:
        grad_fn = get_grad_fn(f)
        
        def hvp(x: Tensor, v: Tensor) -> Tensor:
            """Compute the Hessian-vector product of f with v evaluated at x using finite differences.

            Args:
                x: The input to the function at which the vector-Hessian-vector product
                    is computed.
                v: The vector to compute the vector-Hessian-vector product with.
                    Has same shape as `x`.

            Returns:
                The vector-Hessian-vector product. Has shape [*f(x).shape, *x.shape].
            """
            return hessian_vector_product_from_finite_differences(grad_fn, x, v, delta)
        
        return hvp
    else:
        def hvp(x: Tensor, v: Tensor) -> Tensor:
            """Compute the Hessian-vector product of f with v evaluated at x.

            Args:
                x: The input to the function at which the vector-Hessian-vector product
                    is computed.
                v: The vector to compute the vector-Hessian-vector product with.
                    Has same shape as `x`.

            Returns:
                The vector-Hessian-vector product. Has shape [*f(x).shape, *x.shape].
            """
            grad_func = jacrev(f)
            _, hv = jvp(grad_func, (x,), (v,))
            return hv

        return hvp


def hessian_vector_product_from_finite_differences(
    g: Callable,
    x: Tensor,
    v: Tensor,
    delta: float = 1e-2,
) -> Tensor:
    """
    Compute the Hessian-vector product of f with v evaluated at x using central finite differences of the gradient g of f.
    
    Args:
        g: Function that takes input tensor x and returns the gradient of the function f we want the Hessian of with respect to x.
        x: Input tensor at which to evaluate the Hessian. Shape: (n,)
        v: Vector to compute the Hessian-vector product with. Shape: (n,)
        delta: Finite difference step size
    Returns:
        Hessian-vector product. Shape: (n, n)
    """
    return (g(x + v * delta) - g(x - v * delta)) / (2 * delta)


def compute_sparse_hessian_from_hvp_fn(
    hvp_fn: Callable[[Tensor, Tensor], Tensor],
    x: Tensor,
    sparsity_pattern: np.ndarray,
    n: int,
    *,
    coloring_algorithm: str = "ACYCLIC",
    ordering: str = "SMALLEST_LAST",
    verify: bool = False,
    return_stats: bool = False,
    return_seed_matrix: bool = False,
    hessian_return_type: str = "sparse",
) -> Tensor | tuple:
    """ColPack-compressed sparse Hessian recovery using a supplied HVP routine.

    Same coloring, Y = H @ S assembly, and ``recover_matrix`` path as
    ``compute_sparse_hessian_fn``, but ``hvp_fn(x, v)`` must implement the
    Hessian-vector product :math:`H(x)\\,v` (same tensor layout as ``x``).

    Args:
        hvp_fn: ``(x, v) -> H @ v`` with ``v`` the same shape as ``x`` (after reshape).
        x: Point at which the Hessian is evaluated.
        sparsity_pattern: Sparsity pattern as (m, 2) array of edges.
        n: Number of scalar components of ``x`` (graph coloring dimension).
        coloring_algorithm: ColPack coloring algorithm to use.
        ordering: Node ordering strategy.
        verify: If True, verify that the recovered Hessian is consistent.
        return_stats: If True, append coloring / compression statistics.
        return_seed_matrix: If True, include the seed matrix ``S`` in the return.
        hessian_return_type: ``"sparse"`` or dense recovery, as in ``recover_matrix``.

    Returns:
        Same tuple conventions as ``compute_sparse_hessian_fn``.
    """
    device = x.device
    dtype = x.dtype
    coloring_algorithm = coloring_algorithm.upper()
    ordering = ordering.upper()

    S_np, colors_np, comps = get_seeds_from_pattern(
        sparsity_pattern, n, ordering, coloring_algorithm
    )
    S = torch.tensor(S_np, dtype=dtype, device=device)
    num_colors = S.shape[1]

    for i in range(0, num_colors):
        tangent = S[:, i].reshape_as(x)
        Y_block = hvp_fn(x, tangent).reshape(S.shape[0], 1)
        if i == 0:
            Y = Y_block
        else:
            Y = torch.cat([Y, Y_block], dim=1)

    H = recover_matrix(
        recover_from=coloring_algorithm,
        edges=sparsity_pattern,
        n=n,
        Y=Y,
        vertex_colors=colors_np,
        acyclic_components=comps,
        reconstruct_sparsely=(hessian_return_type == "sparse"),
        verify=verify,
    )

    stats = {
        "num_colors": num_colors,
        "num_hvps": num_colors,
        "dimension": n,
        "num_edges": len(sparsity_pattern),
        "compression_ratio": n / num_colors,
    }
    result: Tensor | tuple = H
    if return_seed_matrix:
        result = H, S
    if return_stats:
        result = result, stats

    return result


def compute_sparse_hessian_fn(
    f: Callable[[Tensor], Tensor],
    x: Tensor,
    sparsity_pattern: np.ndarray,
    n: int,
    *,
    coloring_algorithm: str = "ACYCLIC",
    ordering: str = "SMALLEST_LAST",
    verify: bool = False,
    return_stats: bool = False,
    return_seed_matrix: bool = False,
    hessian_return_type: str = "sparse",
    use_finite_differences: bool = False,
    delta: float = 1e-2,
) -> Tensor | tuple[Tensor, dict]:
    """Compute sparse Hessian with additional statistics about the computation.

    Same as compute_sparse_hessian, but also returns statistics about
    the coloring and compression achieved.

    Args:
        f: Function that takes input tensor x and returns a scalar output.
        x: Input tensor at which to evaluate the Hessian. Shape: (n,)
        sparsity_pattern: Sparsity pattern as (m, 2) array of edges.
        n: Number of dimensions of x
        coloring_algorithm: ColPack coloring algorithm to use.
        ordering: Node ordering strategy.
        verify: If True, verify that the recovered Hessian is correct.
        return_stats: If True, return statistics about the coloring and compression achieved.
        return_seed_matrix: If True, return the seed matrix used to compress the Hessian.
        hessian_return_type: Return type for hessian calculation, sparse or dense arrays
        use_finite_differences: If True, use finite differences instead of automatic differentiation.
        delta: Finite difference step size (only used if use_finite_differences is True).
    Returns:
        Tensor or tuple of (Hessian, stats) or (Hessian, seed_matrix) where stats is a dict containing:
            - 'num_colors': Number of colors used
            - 'num_hvps': Number of Hessian-vector products computed
            - 'dimension': Input dimension (n)
            - 'num_edges': Number of edges in sparsity pattern (m)
            - 'compression_ratio': n / num_colors
        If return_seed_matrix is True, returns tuple of (Hessian, seed_matrix)
        If return_stats is True, returns tuple of (Hessian, stats)
        If both return_seed_matrix and return_stats are True, returns tuple of (Hessian, seed_matrix, stats)
        If neither return_seed_matrix nor return_stats are True, returns Hessian
    """
    hvp_fn = get_hessian_vector_product_fn(
        f, use_finite_differences=use_finite_differences, delta=delta
    )
    return compute_sparse_hessian_from_hvp_fn(
        hvp_fn,
        x,
        sparsity_pattern,
        n,
        coloring_algorithm=coloring_algorithm,
        ordering=ordering,
        verify=verify,
        return_stats=return_stats,
        return_seed_matrix=return_seed_matrix,
        hessian_return_type=hessian_return_type,
    )



def compute_sparse_hessian(y, 
                        x, 
                        sparsity_pattern, 
                        n, 
                        coloring_algorithm, 
                        ordering="SMALLEST_LAST",
                        precomputed_grads=None, 
                        return_seed_matrix=False,
                        return_stats=False,
                        hessian_return_type="sparse",
                        create_graph=False,
                        ) -> Tensor | tuple[Tensor, Tensor]:
        """Compute sparse hessian using ColPack coloring algorithms.
        
        Args:
            y: scalar to get the Hessian of
            x: input tensor with repect to which we want to get the Hessian of y
            sparsity_pattern: Sparsity pattern
            n: Number of dimensions of x
            coloring_algorithm: Coloring algorithm ("star" | "acyclic" | "distance_two")
            ordering: Node ordering strategy for ColPack coloring
            precomputed_grads: Already computed gradients of y with respect to x to avoid recomputation
            return_seed_matrix: If True, return the seed matrix used to compress the Hessian
            return_stats: If True, return sparse timing/statistics metadata
            hessian_return_type: Return type for hessian calculation, sparse or dense arrays
            create_graph: If True, HVPs keep an autograd graph so losses on the
                recovered Hessian can differentiate into parameters that enter ``y``.
                Default False (cheaper; use True only for Hessian-supervised training).
        Returns:
            Hessian tensor (sparse or dense based on hessian_return_type)
        """


        # Get coloring using specified algorithm
        ALGORITHM = coloring_algorithm.upper()
        ordering = ordering.upper()
        def sync_if_needed():
            if x.is_cuda:
                torch.cuda.synchronize(x.device)

        if return_stats:
            sync_if_needed()
            t_coloring = time.perf_counter()
        S, colors, comps = get_seeds_from_pattern(sparsity_pattern, n, ordering, ALGORITHM)
        if return_stats:
            coloring_s = time.perf_counter() - t_coloring
        
        # Convert to tensor
        S = torch.tensor(S, dtype=x.dtype, device=x.device)
        
        # Compute compressed HVPs: Y = H @ S
        Y = torch.zeros(n, S.shape[1], dtype=x.dtype, device=x.device)
        
        # Use precomputed gradients if available, otherwise compute them
        if precomputed_grads is not None:
            r = precomputed_grads.view(-1)
        else:
            # Fallback: compute gradients if not provided
            r = grad(
                y,
                x,
                grad_outputs=torch.ones_like(y),
                create_graph=True,
                retain_graph=True,
            )[0].view(-1)
        
        # Compute dot products of seed matrix with gradient
        grad_seed_products = torch.mv(S.T, r)
        
        # Compute HVPs in efficient loop
        if return_stats:
            sync_if_needed()
            t_hvp = time.perf_counter()
        num_colors = S.shape[1]
        for i in range(num_colors):
            # Use the pre-computed dot product
            grad_seed_product = grad_seed_products[i]
            
            # Second derivative (HVP). Keep the shared first-order graph for later
            # colors, and whenever create_graph=True so higher-order grads stay valid.
            hvp_result = grad(
                grad_seed_product,
                x,
                create_graph=create_graph,
                retain_graph=(i < num_colors - 1) or create_graph,
            )[0]
            
            if hvp_result is not None:
                Y[:, i] = hvp_result.view(-1)
        if return_stats:
            sync_if_needed()
            hvp_loop_s = time.perf_counter() - t_hvp
        
        # Recover sparse matrix using colpackpy
        if return_stats:
            sync_if_needed()
            t_recovery = time.perf_counter()
        H_recovered = recover_matrix(
            recover_from=ALGORITHM,
            edges=sparsity_pattern,
            n=n,
            Y=Y,
            vertex_colors=colors,
            acyclic_components=comps,
            reconstruct_sparsely=(hessian_return_type == "sparse"),
            verify=False,
        )
        if return_stats:
            sync_if_needed()
            recovery_s = time.perf_counter() - t_recovery
        
        # Prepare return value
        if hessian_return_type == "sparse":
            # Return as sparse COO matrix
            hessian_result = H_recovered
        else:
            # PyTorch recovery already returns a dense tensor; keep its graph.
            if isinstance(H_recovered, torch.Tensor):
                hessian_result = H_recovered.to(dtype=x.dtype, device=x.device)
            else:
                if hasattr(H_recovered, 'toarray'):
                    H_recovered = H_recovered.toarray()
                hessian_result = torch.as_tensor(H_recovered, dtype=x.dtype, device=x.device)
        
        # Return seed matrix if requested
        if return_seed_matrix:
            out = (hessian_result, S)
        else:
            out = hessian_result
        if return_stats:
            stats = {
                "num_colors": int(S.shape[1]),
                "num_hvps": int(S.shape[1]),
                "dimension": int(n),
                "num_edges": int(len(sparsity_pattern)),
                "compression_ratio": float(n / S.shape[1]),
                "timings_s": {
                    "coloring_s": float(coloring_s),
                    "hvp_loop_s": float(hvp_loop_s),
                    "recovery_s": float(recovery_s),
                },
            }
            if return_seed_matrix:
                out = (hessian_result, S, stats)
            else:
                out = (hessian_result, stats)
        return out

