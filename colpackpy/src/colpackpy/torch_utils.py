"""PyTorch utilities for efficient sparse Hessian computation using ColPack."""

from typing import Callable
import torch
from torch import Tensor
from torch.func import jacrev, jvp, vmap
import numpy as np
from torch.autograd import grad


from colpackpy.coloring import get_seeds_from_pattern
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
    device = x.device
    dtype = x.dtype
    coloring_algorithm = coloring_algorithm.upper()
    ordering = ordering.upper()

    # Get ColPack seeds and coloring
    S_np, colors_np, comps = get_seeds_from_pattern(
        sparsity_pattern, n, ordering, coloring_algorithm
    )
    S = torch.tensor(S_np, dtype=dtype, device=device)
    num_colors = S.shape[1]

    # Create HVP function and vectorize it
    hvp_fn = get_hessian_vector_product_fn(f, use_finite_differences=use_finite_differences, delta=delta)

    # Compute compressed Hessian: Y = H @ S
    for i in range(0, num_colors):
        # Match tangent shape to x so this works for both vector and tensor inputs.
        tangent = S[:, i].reshape_as(x)
        Y_block = hvp_fn(x, tangent).reshape(S.shape[0], 1)
        if i == 0:
            Y = Y_block
        else:
            Y = torch.cat([Y, Y_block], dim=1)


    # Recover the full Hessian from compressed representation
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

    # Compute statistics
    stats = {
        'num_colors': num_colors,
        'num_hvps': num_colors,
        'dimension': n,
        'num_edges': len(sparsity_pattern),
        'compression_ratio': n / num_colors,
    }
    result = H
    if return_seed_matrix:
        result = H, S
    if return_stats:
        result = result, stats

    return result



def compute_sparse_hessian(y, 
                        x, 
                        sparsity_pattern, 
                        n, 
                        algorithm, 
                        precomputed_grads=None, 
                        return_seed_matrix=False,
                        hessian_return_type="sparse"
                        ) -> Tensor | tuple[Tensor, Tensor]:
        """Compute sparse hessian using ColPack coloring algorithms.
        
        Args:
            y: scalar to get the Hessian of
            x: input tensor with repect to which we want to get the Hessian of y
            sparsity_pattern: Sparsity pattern
            n: Number of dimensions of x
            algorithm: Coloring algorithm ("star" | "acyclic" | "distance_two")
            precomputed_grads: Already computed gradients of y with respect to x to avoid recomputation
            return_seed_matrix: If True, return the seed matrix used to compress the Hessian
            hessian_return_type: Return type for hessian calculation, sparse or dense arrays
        Returns:
            Hessian tensor (sparse or dense based on hessian_return_type)
        """


        # Get coloring using specified algorithm
        ALGORITHM = algorithm.upper()
        S, colors, comps = get_seeds_from_pattern(sparsity_pattern, n, "SMALLEST_LAST", ALGORITHM)
        
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
        for i in range(S.shape[1]):
            # Use the pre-computed dot product
            grad_seed_product = grad_seed_products[i]
            
            # Compute second derivative (HVP) using existing computation graph
            hvp_result = grad(
                grad_seed_product,
                x,
                retain_graph=(i < S.shape[1] - 1)  # Only retain graph if not last iteration
            )[0]
            
            if hvp_result is not None:
                Y[:, i] = hvp_result.view(-1)
        
        # Recover sparse matrix using colpackpy
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
        
        # Prepare return value
        if hessian_return_type == "sparse":
            # Return as sparse COO matrix
            hessian_result = H_recovered
        else:
            # Convert to dense tensor
            if hasattr(H_recovered, 'toarray'):
                H_recovered = H_recovered.toarray()
            hessian_result = torch.tensor(H_recovered, dtype=x.dtype, device=x.device)
        
        # Return seed matrix if requested
        if return_seed_matrix:
            out = (hessian_result, S)
        else:
            out = hessian_result
        return out

