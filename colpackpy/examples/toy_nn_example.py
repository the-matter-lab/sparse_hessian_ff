"""Example: Efficient sparse Hessian computation using ColPackPy + PyTorch.

This example demonstrates computing a sparse Hessian efficiently using:
1. Automatic differentiation (PyTorch)
2. Sparse coloring (ColPackPy)
3. Hessian-vector products (HVPs)
4. Matrix recovery

We use a neural network-based test function with three separate networks
that don't interact, creating a known block-diagonal Hessian structure.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from colpackpy.torch_utils import compute_sparse_hessian_with_stats


class ThreeNetworkFunction(nn.Module):
    """Function composed of three independent neural networks.
    
    Takes an 8-dimensional input and processes it through three separate networks:
    - Network 1: Processes dimensions 0-3 (4 inputs)
    - Network 2: Processes dimensions 4-6 (3 inputs)
    - Network 3: Processes dimension 7 (1 input)
    
    The networks are fully connected within themselves but don't interact,
    creating a block-diagonal Hessian structure (4x4, 3x3, 1x1 blocks).
    """
    
    def __init__(self):
        super().__init__()
        
        # Network 1: 4 inputs -> hidden -> 1 output
        self.net1 = nn.Sequential(
            nn.Linear(4, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1)
        )
        
        # Network 2: 3 inputs -> hidden -> 1 output
        self.net2 = nn.Sequential(
            nn.Linear(3, 6),
            nn.Tanh(),
            nn.Linear(6, 3),
            nn.Tanh(),
            nn.Linear(3, 1)
        )
        
        # Network 3: 1 input -> hidden -> 1 output
        self.net3 = nn.Sequential(
            nn.Linear(1, 4),
            nn.Tanh(),
            nn.Linear(4, 1)
        )
    
    def forward(self, x):
        """Forward pass through all three networks.
        
        Args:
            x: Input tensor of shape (8,)
            
        Returns:
            Scalar output (sum of all three network outputs)
        """
        # Split input into three parts
        x1 = x[0:4]  # First 4 dimensions
        x2 = x[4:7]  # Next 3 dimensions
        x3 = x[7:8]  # Last dimension
        
        # Pass through respective networks
        out1 = self.net1(x1)
        out2 = self.net2(x2)
        out3 = self.net3(x3)
        
        # Combine outputs (scalar)
        return out1.squeeze() + out2.squeeze() + out3.squeeze()


def generate_block_diagonal_edges(block_sizes: list) -> np.ndarray:
    """Generate edges for a block-diagonal sparsity pattern with variable block sizes.
    
    Args:
        block_sizes: List of block sizes (e.g., [4, 3, 1] for three blocks).
        
    Returns:
        Array of edges (m, 2) representing the sparsity pattern.
    """
    edges = []
    offset = 0
    
    for block_size in block_sizes:
        # Add all edges within this block (upper triangular)
        for i in range(offset, offset + block_size):
            for j in range(i + 1, offset + block_size):
                edges.append([i, j])
        offset += block_size
    
    return np.array(edges, dtype=int) if edges else np.array([], dtype=int).reshape(0, 2)


def compute_hessian_autograd(f, x):
    """Compute Hessian using standard PyTorch autograd (for comparison)."""
    n = x.shape[0]
    H = torch.zeros(n, n, dtype=x.dtype, device=x.device)
    
    for i in range(n):
        grad_outputs = torch.zeros_like(x)
        grad_outputs[i] = 1.0
        
        grad = torch.autograd.grad(
            f(x), x, create_graph=True, retain_graph=True
        )[0]
        
        for j in range(n):
            if grad[j].requires_grad:
                second_grad = torch.autograd.grad(
                    grad[j], x, retain_graph=True
                )[0]
                H[i, j] = second_grad[i]
            else:
                H[i, j] = 0.0
    
    return H


def main():
    print("="*70)
    print("Sparse Hessian Computation Example")
    print("Neural Network-Based Function")
    print("="*70)
    
    # Setup
    n = 8  # Dimension
    block_sizes = [4, 3, 1]  # Three networks with different input dimensions
    torch.manual_seed(42)
    
    # Create the neural network function
    model = ThreeNetworkFunction()
    model.eval()  # Set to evaluation mode
    
    # Create input
    x = torch.randn(n, dtype=torch.float64, requires_grad=True)
    
    # Convert model to float64
    model = model.double()
    
    # Define function
    def f(x):
        return model(x)
    
    # Generate sparsity pattern (block diagonal)
    edges = generate_block_diagonal_edges(block_sizes)
    print(f"Number of edges (off-diagonal): {len(edges)}")
    
    # Compute sparse Hessian using ColPackPy
    print("\n" + "-"*70)
    print("Computing Hessian with ColPackPy + HVPs...")
    print("-"*70)
    
    H_sparse, stats = compute_sparse_hessian_with_stats(
        f, x, edges,
        coloring_algorithm="ACYCLIC",
        verify=True
    )
    
    print(f"\nColoring Statistics:")
    print(f"  - Number of colors: {stats['num_colors']}")
    print(f"  - Number of HVPs: {stats['num_hvps']}")
    print(f"  - Compression ratio: {stats['compression_ratio']:.2f}x")
    print(f"  - Theoretical max (naive): {n} HVPs")
    print(f"  - Speedup: {n / stats['num_hvps']:.2f}x")
    
    # Compute Hessian using PyTorch autograd (for verification)
    print("\n" + "-"*70)
    print("Computing Hessian with standard autograd (for verification)...")
    print("-"*70)
    
    x_copy = x.detach().clone().requires_grad_(True)
    H_full = compute_hessian_autograd(f, x_copy)
    
    # Compare
    print("\n" + "-"*70)
    print("Verification")
    print("-"*70)
    
    diff = torch.abs(H_sparse - H_full)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"Max difference: {max_diff:.2e}")
    print(f"Mean difference: {mean_diff:.2e}")
    print(f"Matrices match: {torch.allclose(H_sparse, H_full, atol=1e-10)}")



    # Visualize and save plots
    print("\n" + "-"*70)
    print("Generating visualizations...")
    print("-"*70)
    
    H_sparse_np = H_sparse.detach().cpu().numpy()
    H_full_np = H_full.detach().cpu().numpy()
    
    # Create figure with two subplots for comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Determine common color scale
    vmin = min(H_sparse_np.min(), H_full_np.min())
    vmax = max(H_sparse_np.max(), H_full_np.max())
    
    # Plot 1: Hessian from ColPackPy (sparse method)
    im1 = axes[0].imshow(H_sparse_np, cmap='RdBu_r', aspect='auto', vmin=vmin, vmax=vmax)
    axes[0].set_title('Hessian via ColPackPy\n(Sparse HVP Method)', fontsize=12)
    axes[0].set_xlabel('Column index')
    axes[0].set_ylabel('Row index')
    
    # Add grid to show block structure
    offset = 0
    for block_size in block_sizes[:-1]:  # Don't add line after last block
        offset += block_size
        pos = offset - 0.5
        axes[0].axhline(pos, color='white', linewidth=1.5, alpha=0.7)
        axes[0].axvline(pos, color='white', linewidth=1.5, alpha=0.7)
    
    # Plot 2: Hessian from naive autograd
    im2 = axes[1].imshow(H_full_np, cmap='RdBu_r', aspect='auto', vmin=vmin, vmax=vmax)
    axes[1].set_title('Hessian via Naive Autograd\n(Standard Method)', fontsize=12)
    axes[1].set_xlabel('Column index')
    axes[1].set_ylabel('Row index')
    
    # Add grid to show block structure
    offset = 0
    for block_size in block_sizes[:-1]:  # Don't add line after last block
        offset += block_size
        pos = offset - 0.5
        axes[1].axhline(pos, color='white', linewidth=1.5, alpha=0.7)
        axes[1].axvline(pos, color='white', linewidth=1.5, alpha=0.7)
    
    # Add shared colorbar
    plt.tight_layout()
    fig.colorbar(im2, ax=axes, label='Hessian value', fraction=0.046, pad=0.04)
    
    # Save figure in examples directory
    examples_dir = Path(__file__).parent
    output_file = examples_dir / 'sparse_hessian_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved comparison visualization to: {output_file}")
    
    plt.close('all')
    
    print("\n" + "="*70)
    print("Sparse Hessian computation completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()

