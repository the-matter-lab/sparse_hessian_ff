"""PyTorch utilities for efficient sparse Hessian computation using ColPack."""

from typing import Callable
import torch
from torch import Tensor
from torch.func import jacrev, jvp, vmap
import numpy as np

from colpackpy.coloring import get_seeds_from_pattern
from colpackpy.recovery import recover_matrix


def hessian_vector_product(
    f: Callable[[Tensor], Tensor],
) -> Callable[[Tensor, Tensor], Tensor]:
    """Generate function to compute the Hessian-vector product of f at x with v.

    Args:
        f: The function whose Hessian-vector product we want to compute.
            It should take the input tensor as argument and return the output tensor.

    Returns:
        A function that computes the Hessian-vector product of f at the input
        tensor x given x and the vector. Has shape [*f(x).shape, *x.shape].
    """

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


def compute_sparse_hessian(
    f: Callable[[Tensor], Tensor],
    x: Tensor,
    edges: np.ndarray,
    *,
    coloring_algorithm: str = "ACYCLIC",
    ordering: str = "SMALLEST_LAST",
    verify: bool = False,
) -> Tensor:
    """Compute sparse Hessian of f at x using ColPack coloring and recovery.

    This function efficiently computes the Hessian matrix by:
    1. Using ColPack to find an optimal coloring of the sparsity pattern
    2. Computing compressed Hessian-vector products (Y = H @ S)
    3. Recovering the full Hessian from the compressed representation

    Args:
        f: Function that takes input tensor x and returns a scalar output.
        x: Input tensor at which to evaluate the Hessian. Shape: (n,)
        edges: Sparsity pattern as (m, 2) array of edges (excluding diagonal).
               Each row [i, j] represents a non-zero off-diagonal element.
        coloring_algorithm: ColPack coloring algorithm to use.
            Options: "DISTANCE_TWO", "STAR", "ACYCLIC" (default).
        ordering: Node ordering strategy. Default: "SMALLEST_LAST".
        verify: If True, verify that the recovered Hessian is correct.

    Returns:
        Hessian matrix as a PyTorch tensor. Shape: (n, n).

    Example:
        >>> def f(x):
        ...     return (x ** 2).sum()
        >>> x = torch.randn(10, requires_grad=True)
        >>> edges = np.array([[0, 1], [1, 2]])  # tridiagonal structure
        >>> H = compute_sparse_hessian(f, x, edges)
    """
    n = x.shape[0]
    device = x.device
    dtype = x.dtype

    # Get ColPack seeds and coloring
    S_np, colors_np, comps = get_seeds_from_pattern(
        edges, n, ordering, coloring_algorithm
    )
    S = torch.tensor(S_np, dtype=dtype, device=device)
    num_colors = S.shape[1]

    # Create HVP function and vectorize it
    hvp = hessian_vector_product(f)
    multiple_hvps = vmap(hvp, in_dims=(None, 1), out_dims=1)

    # Compute compressed Hessian: Y = H @ S
    # Multiple HVPs evaluates [H @ s_1, H @ s_2, ..., H @ s_c] where c is num_colors
    Y = multiple_hvps(x, S)

    # Recover the full Hessian from compressed representation
    H = recover_matrix(
        recover_from=coloring_algorithm,
        edges=edges,
        n=n,
        Y=Y,
        vertex_colors=colors_np,
        acyclic_components=comps,
        reconstruct_sparsely=False,
        verify=verify,
    )

    return H


def compute_sparse_hessian_with_stats(
    f: Callable[[Tensor], Tensor],
    x: Tensor,
    edges: np.ndarray,
    *,
    coloring_algorithm: str = "ACYCLIC",
    ordering: str = "SMALLEST_LAST",
    verify: bool = False,
) -> tuple[Tensor, dict]:
    """Compute sparse Hessian with additional statistics about the computation.

    Same as compute_sparse_hessian, but also returns statistics about
    the coloring and compression achieved.

    Args:
        f: Function that takes input tensor x and returns a scalar output.
        x: Input tensor at which to evaluate the Hessian. Shape: (n,)
        edges: Sparsity pattern as (m, 2) array of edges.
        coloring_algorithm: ColPack coloring algorithm to use.
        ordering: Node ordering strategy.
        verify: If True, verify that the recovered Hessian is correct.

    Returns:
        Tuple of (Hessian, stats) where stats is a dict containing:
            - 'num_colors': Number of colors used
            - 'num_hvps': Number of Hessian-vector products computed
            - 'dimension': Input dimension
            - 'num_edges': Number of edges in sparsity pattern
            - 'compression_ratio': n / num_colors
    """
    n = x.shape[0]
    device = x.device
    dtype = x.dtype

    # Get ColPack seeds and coloring
    S_np, colors_np, comps = get_seeds_from_pattern(
        edges, n, ordering, coloring_algorithm
    )
    S = torch.tensor(S_np, dtype=dtype, device=device)
    num_colors = S.shape[1]

    # Create HVP function and vectorize it
    hvp = hessian_vector_product(f)
    multiple_hvps = vmap(hvp, in_dims=(None, 1), out_dims=1)

    # Compute compressed Hessian: Y = H @ S
    Y = multiple_hvps(x, S)

    # Recover the full Hessian from compressed representation
    H = recover_matrix(
        recover_from=coloring_algorithm,
        edges=edges,
        n=n,
        Y=Y,
        vertex_colors=colors_np,
        acyclic_components=comps,
        reconstruct_sparsely=False,
        verify=verify,
    )

    # Compute statistics
    stats = {
        'num_colors': num_colors,
        'num_hvps': num_colors,
        'dimension': n,
        'num_edges': len(edges),
        'compression_ratio': n / num_colors,
    }

    return H, stats

