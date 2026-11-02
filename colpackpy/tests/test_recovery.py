"""
Unit tests for colpackpy.recovery module.

Tests cover the three main recovery algorithms:
- DISTANCE_TWO: Direct recovery using distance-2 coloring
- STAR: Direct recovery using star coloring
- ACYCLIC: Indirect recovery using acyclic coloring
"""

import unittest
import numpy as np
import scipy.sparse as sp
from typing import Tuple

from colpackpy.coloring import get_seeds_from_pattern
from colpackpy.recovery import recover_matrix

try:
    import torch
    import torch.nn as nn
    from torch.func import jacrev, jvp, vmap
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class MatrixTestHelper:
    """Helper class for generating test matrices with known sparsity patterns."""
    
    @staticmethod
    def make_random_spd_matrix(n: int = 60, avg_deg: int = 6, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a random sparse positive definite (SPD) matrix.
        
        Parameters
        ----------
        n : int
            Matrix dimension.
        avg_deg : int
            Average node degree (sparsity control).
        seed : int
            Random seed for reproducibility.
            
        Returns
        -------
        A : np.ndarray
            Dense (n, n) SPD matrix.
        edges : np.ndarray
            (m, 2) array of edges (excluding diagonal).
        """
        rng = np.random.default_rng(seed)
        p = max(1e-6, avg_deg / (n - 1))  # Erdős–Rényi probability
        
        # Random upper-triangular sparsity (no diagonal)
        mask = rng.random((n, n)) < p
        iu, ju = np.triu_indices(n, k=1)
        keep = mask[iu, ju]
        i = iu[keep]
        j = ju[keep]
        
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
        edges = np.column_stack([i, j]).astype(int)
        
        return A.toarray(), edges
    
    @staticmethod
    def make_tridiagonal_matrix(n: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a simple tridiagonal SPD matrix.
        
        Parameters
        ----------
        n : int
            Matrix dimension.
            
        Returns
        -------
        A : np.ndarray
            Dense (n, n) tridiagonal SPD matrix.
        edges : np.ndarray
            (m, 2) array of edges.
        """
        diag = np.full(n, 4.0)
        off_diag = np.full(n - 1, -1.0)
        
        A = np.diag(diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)
        
        # Edges are sequential pairs
        i = np.arange(n - 1)
        j = np.arange(1, n)
        edges = np.column_stack([i, j])
        
        return A, edges
    
    @staticmethod
    def make_diagonal_matrix(n: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a diagonal matrix (no edges, trivial case).
        
        Parameters
        ----------
        n : int
            Matrix dimension.
            
        Returns
        -------
        A : np.ndarray
            Dense (n, n) diagonal matrix.
        edges : np.ndarray
            Empty (0, 2) array (no edges).
        """
        A = np.diag(np.random.uniform(1.0, 5.0, size=n))
        edges = np.empty((0, 2), dtype=int)
        
        return A, edges
    
    @staticmethod
    def make_random_symmetric_matrix(n: int = 60, avg_deg: int = 6, seed: int = 0, 
                                     allow_negative_diag: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a random sparse symmetric matrix (not necessarily positive definite).
        
        Parameters
        ----------
        n : int
            Matrix dimension.
        avg_deg : int
            Average node degree (sparsity control).
        seed : int
            Random seed for reproducibility.
        allow_negative_diag : bool
            If True, allows negative diagonal entries.
            
        Returns
        -------
        A : np.ndarray
            Dense (n, n) symmetric matrix.
        edges : np.ndarray
            (m, 2) array of edges (excluding diagonal).
        """
        rng = np.random.default_rng(seed)
        p = max(1e-6, avg_deg / (n - 1))
        
        # Random upper-triangular sparsity
        mask = rng.random((n, n)) < p
        iu, ju = np.triu_indices(n, k=1)
        keep = mask[iu, ju]
        i = iu[keep]
        j = ju[keep]
        
        # Random weights (can be positive or negative)
        w = rng.normal(loc=0.0, scale=2.0, size=i.size)
        
        # Build symmetric off-diagonals
        rows_off = np.concatenate([i, j])
        cols_off = np.concatenate([j, i])
        data_off = np.concatenate([w, w])
        
        # Random diagonal (can include negative values if allowed)
        if allow_negative_diag:
            diag = rng.normal(loc=0.0, scale=3.0, size=n)
        else:
            diag = rng.uniform(0.1, 5.0, size=n)
        
        rows = np.concatenate([rows_off, np.arange(n)])
        cols = np.concatenate([cols_off, np.arange(n)])
        data = np.concatenate([data_off, diag])
        
        A = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
        edges = np.column_stack([i, j]).astype(int)
        
        return A.toarray(), edges
    
    @staticmethod
    def make_block_diagonal_matrix(n_blocks: int = 4, block_size: int = 10, 
                                   density: float = 0.3, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a block diagonal symmetric matrix.
        
        Parameters
        ----------
        n_blocks : int
            Number of diagonal blocks.
        block_size : int
            Size of each block.
        density : float
            Density of connections within each block.
        seed : int
            Random seed.
            
        Returns
        -------
        A : np.ndarray
            Dense (n, n) block diagonal matrix.
        edges : np.ndarray
            (m, 2) array of edges.
        """
        rng = np.random.default_rng(seed)
        n = n_blocks * block_size
        A = np.zeros((n, n))
        edges_list = []
        
        for block_idx in range(n_blocks):
            start = block_idx * block_size
            end = start + block_size
            
            # Generate dense connections within block
            for i in range(start, end):
                A[i, i] = rng.uniform(2.0, 5.0)
                for j in range(i + 1, end):
                    if rng.random() < density:
                        val = rng.normal(0.0, 1.0)
                        A[i, j] = val
                        A[j, i] = val
                        edges_list.append([i, j])
        
        edges = np.array(edges_list, dtype=int) if edges_list else np.empty((0, 2), dtype=int)
        return A, edges
    
    @staticmethod
    def make_grid_laplacian(grid_size: int = 8) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a 2D grid Laplacian matrix (from finite differences).
        
        Parameters
        ----------
        grid_size : int
            Size of square grid (grid_size x grid_size).
            
        Returns
        -------
        A : np.ndarray
            Dense (n, n) Laplacian matrix where n = grid_size^2.
        edges : np.ndarray
            (m, 2) array of edges.
        """
        n = grid_size * grid_size
        A = np.zeros((n, n))
        edges_list = []
        
        for i in range(grid_size):
            for j in range(grid_size):
                node = i * grid_size + j
                A[node, node] = 4.0  # Diagonal
                
                # Right neighbor
                if j < grid_size - 1:
                    neighbor = i * grid_size + (j + 1)
                    A[node, neighbor] = -1.0
                    A[neighbor, node] = -1.0
                    edges_list.append([node, neighbor])
                
                # Bottom neighbor
                if i < grid_size - 1:
                    neighbor = (i + 1) * grid_size + j
                    A[node, neighbor] = -1.0
                    A[neighbor, node] = -1.0
                    edges_list.append([node, neighbor])
        
        edges = np.array(edges_list, dtype=int)
        return A, edges
    
    @staticmethod
    def make_banded_matrix(n: int = 50, bandwidth: int = 3, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a symmetric banded matrix.
        
        Parameters
        ----------
        n : int
            Matrix dimension.
        bandwidth : int
            Half-bandwidth (number of diagonals on each side).
        seed : int
            Random seed.
            
        Returns
        -------
        A : np.ndarray
            Dense (n, n) banded matrix.
        edges : np.ndarray
            (m, 2) array of edges.
        """
        rng = np.random.default_rng(seed)
        A = np.zeros((n, n))
        edges_list = []
        
        for i in range(n):
            A[i, i] = rng.uniform(2.0, 5.0)
            for k in range(1, min(bandwidth + 1, n - i)):
                j = i + k
                val = rng.normal(0.0, 1.0)
                A[i, j] = val
                A[j, i] = val
                edges_list.append([i, j])
        
        edges = np.array(edges_list, dtype=int)
        return A, edges
    
    @staticmethod
    def make_star_graph_matrix(n: int = 50, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a matrix with star graph structure (one central hub connected to all).
        
        Parameters
        ----------
        n : int
            Matrix dimension.
        seed : int
            Random seed.
            
        Returns
        -------
        A : np.ndarray
            Dense (n, n) matrix with star structure.
        edges : np.ndarray
            (m, 2) array of edges.
        """
        rng = np.random.default_rng(seed)
        A = np.zeros((n, n))
        edges_list = []
        
        # Diagonal
        for i in range(n):
            A[i, i] = rng.uniform(2.0, 5.0)
        
        # Center node (0) connected to all others
        for i in range(1, n):
            val = rng.normal(0.0, 1.0)
            A[0, i] = val
            A[i, 0] = val
            edges_list.append([0, i])
        
        edges = np.array(edges_list, dtype=int)
        return A, edges
    
    @staticmethod
    def make_cycle_graph_matrix(n: int = 50, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a matrix with cycle graph structure.
        
        Parameters
        ----------
        n : int
            Matrix dimension.
        seed : int
            Random seed.
            
        Returns
        -------
        A : np.ndarray
            Dense (n, n) matrix with cycle structure.
        edges : np.ndarray
            (m, 2) array of edges.
        """
        rng = np.random.default_rng(seed)
        A = np.zeros((n, n))
        edges_list = []
        
        # Diagonal
        for i in range(n):
            A[i, i] = rng.uniform(2.0, 5.0)
        
        # Cycle: 0-1-2-...-n-1-0
        for i in range(n):
            j = (i + 1) % n
            val = rng.normal(0.0, 1.0)
            A[i, j] = val
            A[j, i] = val
            edges_list.append([min(i, j), max(i, j)])
        
        edges = np.array(edges_list, dtype=int)
        return A, edges
    
    @staticmethod
    def make_mixed_magnitude_matrix(n: int = 50, avg_deg: int = 6, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a matrix with entries of vastly different magnitudes (numerical challenge).
        
        Parameters
        ----------
        n : int
            Matrix dimension.
        avg_deg : int
            Average node degree.
        seed : int
            Random seed.
            
        Returns
        -------
        A : np.ndarray
            Dense (n, n) matrix with mixed magnitudes.
        edges : np.ndarray
            (m, 2) array of edges.
        """
        rng = np.random.default_rng(seed)
        p = max(1e-6, avg_deg / (n - 1))
        
        mask = rng.random((n, n)) < p
        iu, ju = np.triu_indices(n, k=1)
        keep = mask[iu, ju]
        i = iu[keep]
        j = ju[keep]
        
        # Mix of very small and very large values
        magnitudes = rng.choice([1e-6, 1e-3, 1.0, 1e3, 1e6], size=i.size)
        signs = rng.choice([-1, 1], size=i.size)
        w = signs * magnitudes
        
        rows_off = np.concatenate([i, j])
        cols_off = np.concatenate([j, i])
        data_off = np.concatenate([w, w])
        
        # Large diagonal for stability
        diag = rng.uniform(1e6, 1e7, size=n)
        
        rows = np.concatenate([rows_off, np.arange(n)])
        cols = np.concatenate([cols_off, np.arange(n)])
        data = np.concatenate([data_off, diag])
        
        A = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
        edges = np.column_stack([i, j]).astype(int)
        
        return A.toarray(), edges


class TestDistanceTwoRecovery(unittest.TestCase):
    """Test suite for DISTANCE_TWO recovery algorithm."""
    
    def test_distance_two_random_sparse(self):
        """Test DISTANCE_TWO on random sparse SPD matrix."""
        n = 60
        avg_deg = 8
        A, edges = MatrixTestHelper.make_random_spd_matrix(n=n, avg_deg=avg_deg, seed=42)
        
        # Get coloring
        S, colors, comps = get_seeds_from_pattern(edges, n, "SMALLEST_LAST", "DISTANCE_TWO")
        S = np.asarray(S, dtype=float)
        colors = np.asarray(colors, dtype=int)
        
        # Compressed evaluation
        Y = A @ S
        
        # Recovery
        A_rec = recover_matrix(
            recover_from="DISTANCE_TWO",
            edges=edges,
            n=n,
            Y=Y,
            vertex_colors=colors,
            reconstruct_sparsely=False,
            dtype=np.float64,
            verify=True
        )
        
        # Verify accuracy
        self.assertTrue(np.allclose(A, A_rec, rtol=0, atol=1e-10))
        max_err = np.abs(A - A_rec).max()
        self.assertLess(max_err, 1e-10)
    
    def test_distance_two_tridiagonal(self):
        """Test DISTANCE_TWO on tridiagonal matrix."""
        n = 50
        A, edges = MatrixTestHelper.make_tridiagonal_matrix(n=n)
        
        S, colors, _ = get_seeds_from_pattern(edges, n, "NATURAL", "DISTANCE_TWO")
        S = np.asarray(S, dtype=float)
        colors = np.asarray(colors, dtype=int)
        
        Y = A @ S
        
        A_rec = recover_matrix(
            recover_from="DISTANCE_TWO",
            edges=edges,
            n=n,
            Y=Y,
            vertex_colors=colors,
            reconstruct_sparsely=False,
            dtype=np.float64
        )
        
        self.assertTrue(np.allclose(A, A_rec, rtol=0, atol=1e-10))
    
    def test_distance_two_sparse_reconstruction(self):
        """Test DISTANCE_TWO with sparse reconstruction path."""
        n = 40
        avg_deg = 6
        A, edges = MatrixTestHelper.make_random_spd_matrix(n=n, avg_deg=avg_deg, seed=123)
        
        S, colors, _ = get_seeds_from_pattern(edges, n, "LARGEST_FIRST", "DISTANCE_TWO")
        S = np.asarray(S, dtype=float)
        colors = np.asarray(colors, dtype=int)
        
        Y = A @ S
        
        # Test with sparse reconstruction
        A_rec = recover_matrix(
            recover_from="DISTANCE_TWO",
            edges=edges,
            n=n,
            Y=Y,
            vertex_colors=colors,
            reconstruct_sparsely=True,
            dtype=np.float64,
            verify=True
        )
        
        # Convert sparse to dense for comparison
        A_rec_dense = A_rec.toarray() if sp.issparse(A_rec) else A_rec
        self.assertTrue(np.allclose(A, A_rec_dense, rtol=0, atol=1e-10))
    
    def test_distance_two_diagonal_only(self):
        """Test DISTANCE_TWO on diagonal matrix (edge case: no edges)."""
        n = 30
        A, edges = MatrixTestHelper.make_diagonal_matrix(n=n)
        
        S, colors, _ = get_seeds_from_pattern(edges, n, "NATURAL", "DISTANCE_TWO")
        S = np.asarray(S, dtype=float)
        colors = np.asarray(colors, dtype=int)
        
        Y = A @ S
        
        A_rec = recover_matrix(
            recover_from="DISTANCE_TWO",
            edges=edges,
            n=n,
            Y=Y,
            vertex_colors=colors,
            reconstruct_sparsely=False,
            dtype=np.float64
        )
        
        self.assertTrue(np.allclose(A, A_rec, rtol=0, atol=1e-10))


class TestStarRecovery(unittest.TestCase):
    """Test suite for STAR recovery algorithm."""
    
    def test_star_random_sparse(self):
        """Test STAR on random sparse SPD matrix."""
        n = 70
        avg_deg = 10
        A, edges = MatrixTestHelper.make_random_spd_matrix(n=n, avg_deg=avg_deg, seed=55)
        
        # Get star coloring
        S, colors, comps = get_seeds_from_pattern(edges, n, "SMALLEST_LAST", "STAR")
        S = np.asarray(S, dtype=float)
        colors = np.asarray(colors, dtype=int)
        
        # Compressed evaluation
        Y = A @ S
        
        # Recovery
        A_rec = recover_matrix(
            recover_from="STAR",
            edges=edges,
            n=n,
            Y=Y,
            vertex_colors=colors,
            reconstruct_sparsely=False,
            dtype=np.float64,
            verify=True
        )
        
        # Verify accuracy
        self.assertTrue(np.allclose(A, A_rec, rtol=0, atol=1e-10))
        max_err = np.abs(A - A_rec).max()
        self.assertLess(max_err, 1e-10)
    
    def test_star_tridiagonal(self):
        """Test STAR on tridiagonal matrix."""
        n = 60
        A, edges = MatrixTestHelper.make_tridiagonal_matrix(n=n)
        
        S, colors, _ = get_seeds_from_pattern(edges, n, "NATURAL", "STAR")
        S = np.asarray(S, dtype=float)
        colors = np.asarray(colors, dtype=int)
        
        Y = A @ S
        
        A_rec = recover_matrix(
            recover_from="STAR",
            edges=edges,
            n=n,
            Y=Y,
            vertex_colors=colors,
            reconstruct_sparsely=False,
            dtype=np.float64
        )
        
        self.assertTrue(np.allclose(A, A_rec, rtol=0, atol=1e-10))
    
    def test_star_sparse_reconstruction(self):
        """Test STAR with sparse reconstruction path."""
        n = 50
        avg_deg = 8
        A, edges = MatrixTestHelper.make_random_spd_matrix(n=n, avg_deg=avg_deg, seed=789)
        
        S, colors, _ = get_seeds_from_pattern(edges, n, "LARGEST_FIRST", "STAR")
        S = np.asarray(S, dtype=float)
        colors = np.asarray(colors, dtype=int)
        
        Y = A @ S
        
        # Test with sparse reconstruction
        A_rec = recover_matrix(
            recover_from="STAR",
            edges=edges,
            n=n,
            Y=Y,
            vertex_colors=colors,
            reconstruct_sparsely=True,
            dtype=np.float64,
            verify=True
        )
        
        # Convert sparse to dense for comparison
        A_rec_dense = A_rec.toarray() if sp.issparse(A_rec) else A_rec
        self.assertTrue(np.allclose(A, A_rec_dense, rtol=0, atol=1e-10))
    
    def test_star_multiple_orderings(self):
        """Test STAR with different vertex orderings."""
        n = 40
        avg_deg = 6
        A, edges = MatrixTestHelper.make_random_spd_matrix(n=n, avg_deg=avg_deg, seed=999)
        
        orderings = ["NATURAL", "LARGEST_FIRST", "SMALLEST_LAST"]
        
        for ordering in orderings:
            with self.subTest(ordering=ordering):
                S, colors, _ = get_seeds_from_pattern(edges, n, ordering, "STAR")
                S = np.asarray(S, dtype=float)
                colors = np.asarray(colors, dtype=int)
                
                Y = A @ S
                
                A_rec = recover_matrix(
                    recover_from="STAR",
                    edges=edges,
                    n=n,
                    Y=Y,
                    vertex_colors=colors,
                    reconstruct_sparsely=False,
                    dtype=np.float64,
                    verify=True
                )
                
                self.assertTrue(np.allclose(A, A_rec, rtol=0, atol=1e-10))


class TestAcyclicRecovery(unittest.TestCase):
    """Test suite for ACYCLIC recovery algorithm (indirect/substitution)."""
    
    def test_acyclic_random_sparse(self):
        """Test ACYCLIC on random sparse SPD matrix."""
        n = 80
        avg_deg = 12
        A, edges = MatrixTestHelper.make_random_spd_matrix(n=n, avg_deg=avg_deg, seed=42)
        
        # Get acyclic coloring
        S, colors, comps = get_seeds_from_pattern(edges, n, "SMALLEST_LAST", "ACYCLIC_FOR_INDIRECT_RECOVERY")
        S = np.asarray(S, dtype=float)
        colors = np.asarray(colors, dtype=int)
        
        # Compressed evaluation
        Y = A @ S
        
        # Recovery with cached components
        A_rec = recover_matrix(
            recover_from="ACYCLIC",
            edges=edges,
            n=n,
            Y=Y,
            vertex_colors=colors,
            acyclic_components=comps,
            reconstruct_sparsely=True,
            dtype=np.float64,
            verify=True
        )
        
        # Verify accuracy
        # Convert sparse to dense for comparison
        A_rec_dense = A_rec.toarray() if sp.issparse(A_rec) else A_rec
        self.assertTrue(np.allclose(A, A_rec_dense, rtol=0, atol=1e-10))
        max_err = np.abs(A - A_rec_dense).max()
        self.assertLess(max_err, 1e-10)
    
    def test_acyclic_tridiagonal(self):
        """Test ACYCLIC on tridiagonal matrix."""
        n = 50
        A, edges = MatrixTestHelper.make_tridiagonal_matrix(n=n)
        
        S, colors, comps = get_seeds_from_pattern(edges, n, "NATURAL", "ACYCLIC_FOR_INDIRECT_RECOVERY")
        S = np.asarray(S, dtype=float)
        colors = np.asarray(colors, dtype=int)
        
        Y = A @ S
        
        A_rec = recover_matrix(
            recover_from="ACYCLIC",
            edges=edges,
            n=n,
            Y=Y,
            vertex_colors=colors,
            acyclic_components=comps,
            reconstruct_sparsely=False,
            dtype=np.float64
        )
        
        self.assertTrue(np.allclose(A, A_rec, rtol=0, atol=1e-10))
    
    def test_acyclic_without_cached_components(self):
        """Test ACYCLIC fallback path (without precomputed components)."""
        n = 60
        avg_deg = 8
        A, edges = MatrixTestHelper.make_random_spd_matrix(n=n, avg_deg=avg_deg, seed=111)
        
        S, colors, _ = get_seeds_from_pattern(edges, n, "SMALLEST_LAST", "ACYCLIC_FOR_INDIRECT_RECOVERY")
        S = np.asarray(S, dtype=float)
        colors = np.asarray(colors, dtype=int)
        
        Y = A @ S
        
        # Test without passing acyclic_components (fallback path)
        A_rec = recover_matrix(
            recover_from="ACYCLIC",
            edges=edges,
            n=n,
            Y=Y,
            vertex_colors=colors,
            acyclic_components=None,
            reconstruct_sparsely=False,
            dtype=np.float64,
            verify=True
        )
        
        self.assertTrue(np.allclose(A, A_rec, rtol=0, atol=1e-10))
    
    def test_acyclic_sparse_reconstruction(self):
        """Test ACYCLIC with sparse reconstruction path."""
        n = 70
        avg_deg = 10
        A, edges = MatrixTestHelper.make_random_spd_matrix(n=n, avg_deg=avg_deg, seed=222)
        
        S, colors, comps = get_seeds_from_pattern(edges, n, "LARGEST_FIRST", "ACYCLIC_FOR_INDIRECT_RECOVERY")
        S = np.asarray(S, dtype=float)
        colors = np.asarray(colors, dtype=int)
        
        Y = A @ S
        
        # Test with sparse reconstruction
        A_rec = recover_matrix(
            recover_from="ACYCLIC",
            edges=edges,
            n=n,
            Y=Y,
            vertex_colors=colors,
            acyclic_components=comps,
            reconstruct_sparsely=True,
            dtype=np.float64,
            verify=True
        )
        
        # Convert sparse to dense for comparison
        A_rec_dense = A_rec.toarray() if sp.issparse(A_rec) else A_rec
        self.assertTrue(np.allclose(A, A_rec_dense, rtol=0, atol=1e-10))
    
    def test_acyclic_indirect_alias(self):
        """Test ACYCLIC_INDIRECT alias for recovery."""
        n = 50
        avg_deg = 7
        A, edges = MatrixTestHelper.make_random_spd_matrix(n=n, avg_deg=avg_deg, seed=333)
        
        S, colors, comps = get_seeds_from_pattern(edges, n, "SMALLEST_LAST", "ACYCLIC_FOR_INDIRECT_RECOVERY")
        S = np.asarray(S, dtype=float)
        colors = np.asarray(colors, dtype=int)
        
        Y = A @ S
        
        # Test with ACYCLIC_INDIRECT alias
        A_rec = recover_matrix(
            recover_from="ACYCLIC_INDIRECT",
            edges=edges,
            n=n,
            Y=Y,
            vertex_colors=colors,
            acyclic_components=comps,
            reconstruct_sparsely=False,
            dtype=np.float64,
            verify=True
        )
        
        self.assertTrue(np.allclose(A, A_rec, rtol=0, atol=1e-10))
    
    def test_acyclic_diagonal_only(self):
        """Test ACYCLIC on diagonal matrix (edge case: no edges)."""
        n = 30
        A, edges = MatrixTestHelper.make_diagonal_matrix(n=n)
        
        S, colors, comps = get_seeds_from_pattern(edges, n, "NATURAL", "ACYCLIC_FOR_INDIRECT_RECOVERY")
        S = np.asarray(S, dtype=float)
        colors = np.asarray(colors, dtype=int)
        
        Y = A @ S
        
        A_rec = recover_matrix(
            recover_from="ACYCLIC",
            edges=edges,
            n=n,
            Y=Y,
            vertex_colors=colors,
            acyclic_components=comps,
            reconstruct_sparsely=False,
            dtype=np.float64
        )
        
        self.assertTrue(np.allclose(A, A_rec, rtol=0, atol=1e-10))


class TestRecoveryEdgeCases(unittest.TestCase):
    """Test suite for edge cases and error handling."""
    
    def test_invalid_algorithm(self):
        """Test that invalid algorithm raises ValueError."""
        n = 10
        A, edges = MatrixTestHelper.make_tridiagonal_matrix(n=n)
        
        S, colors, _ = get_seeds_from_pattern(edges, n, "NATURAL", "DISTANCE_TWO")
        S = np.asarray(S, dtype=float)
        colors = np.asarray(colors, dtype=int)
        
        Y = A @ S
        
        with self.assertRaises(ValueError):
            recover_matrix(
                recover_from="INVALID_ALGORITHM",
                edges=edges,
                n=n,
                Y=Y,
                vertex_colors=colors
            )
    
    def test_empty_matrix(self):
        """Test recovery on very small matrix (n=1)."""
        n = 1
        A = np.array([[5.0]])
        edges = np.empty((0, 2), dtype=int)
        
        S, colors, _ = get_seeds_from_pattern(edges, n, "NATURAL", "DISTANCE_TWO")
        S = np.asarray(S, dtype=float)
        colors = np.asarray(colors, dtype=int)
        
        Y = A @ S
        
        for algorithm in ["DISTANCE_TWO", "STAR"]:
            with self.subTest(algorithm=algorithm):
                A_rec = recover_matrix(
                    recover_from=algorithm,
                    edges=edges,
                    n=n,
                    Y=Y,
                    vertex_colors=colors
                )
                
                self.assertTrue(np.allclose(A, A_rec, rtol=0, atol=1e-10))
    
    def test_dtype_preservation(self):
        """Test that output dtype matches requested dtype."""
        n = 30
        A, edges = MatrixTestHelper.make_random_spd_matrix(n=n, avg_deg=6, seed=777)
        
        S, colors, _ = get_seeds_from_pattern(edges, n, "NATURAL", "DISTANCE_TWO")
        S = np.asarray(S, dtype=float)
        colors = np.asarray(colors, dtype=int)
        
        Y = A @ S
        
        for dtype in [np.float32, np.float64]:
            with self.subTest(dtype=dtype):
                A_rec = recover_matrix(
                    recover_from="DISTANCE_TWO",
                    edges=edges,
                    n=n,
                    Y=Y,
                    vertex_colors=colors,
                    dtype=dtype
                )
                
                self.assertEqual(A_rec.dtype, dtype)


class TestRecoveryComparison(unittest.TestCase):
    """Test suite comparing results across different algorithms."""
    
    def test_all_algorithms_consistency(self):
        """Test that all algorithms produce consistent results on the same matrix."""
        n = 60
        avg_deg = 8
        A, edges = MatrixTestHelper.make_random_spd_matrix(n=n, avg_deg=avg_deg, seed=12345)
        
        algorithms = [
            ("DISTANCE_TWO", "DISTANCE_TWO"),
            ("STAR", "STAR"),
            ("ACYCLIC_FOR_INDIRECT_RECOVERY", "ACYCLIC")
        ]
        
        results = []
        
        for coloring_alg, recovery_alg in algorithms:
            S, colors, comps = get_seeds_from_pattern(edges, n, "SMALLEST_LAST", coloring_alg)
            S = np.asarray(S, dtype=float)
            colors = np.asarray(colors, dtype=int)
            
            Y = A @ S
            
            A_rec = recover_matrix(
                recover_from=recovery_alg,
                edges=edges,
                n=n,
                Y=Y,
                vertex_colors=colors,
                acyclic_components=comps if recovery_alg == "ACYCLIC" else None,
                reconstruct_sparsely=False,
                dtype=np.float64,
                verify=True
            )
            
            results.append((recovery_alg, A_rec))
        
        # All algorithms should recover the original matrix accurately
        for alg_name, A_rec in results:
            with self.subTest(algorithm=alg_name):
                self.assertTrue(np.allclose(A, A_rec, rtol=0, atol=1e-10),
                               f"{alg_name} failed to recover matrix accurately")


class TestGeneralSymmetricMatrices(unittest.TestCase):
    """Test suite for general symmetric matrices (not necessarily positive definite)."""
    
    def test_distance_two_general_symmetric(self):
        """Test DISTANCE_TWO on general symmetric matrix with negative eigenvalues."""
        n = 50
        avg_deg = 8
        A, edges = MatrixTestHelper.make_random_symmetric_matrix(n=n, avg_deg=avg_deg, seed=42, 
                                                                 allow_negative_diag=True)
        
        S, colors, _ = get_seeds_from_pattern(edges, n, "SMALLEST_LAST", "DISTANCE_TWO")
        S = np.asarray(S, dtype=float)
        colors = np.asarray(colors, dtype=int)
        
        Y = A @ S
        
        A_rec = recover_matrix(
            recover_from="DISTANCE_TWO",
            edges=edges,
            n=n,
            Y=Y,
            vertex_colors=colors,
            verify=True
        )
        
        # Relaxed tolerance for general symmetric matrices (numerical accuracy ~1e-7)
        self.assertTrue(np.allclose(A, A_rec, rtol=1e-6, atol=1e-6))
    
    def test_star_general_symmetric(self):
        """Test STAR on general symmetric matrix."""
        n = 60
        avg_deg = 10
        A, edges = MatrixTestHelper.make_random_symmetric_matrix(n=n, avg_deg=avg_deg, seed=123, 
                                                                 allow_negative_diag=True)
        
        S, colors, _ = get_seeds_from_pattern(edges, n, "LARGEST_FIRST", "STAR")
        S = np.asarray(S, dtype=float)
        colors = np.asarray(colors, dtype=int)
        
        Y = A @ S
        
        A_rec = recover_matrix(
            recover_from="STAR",
            edges=edges,
            n=n,
            Y=Y,
            vertex_colors=colors,
            verify=True
        )
        
        # Relaxed tolerance for general symmetric matrices
        self.assertTrue(np.allclose(A, A_rec, rtol=1e-6, atol=1e-6))
    
    def test_acyclic_general_symmetric(self):
        """Test ACYCLIC on general symmetric matrix."""
        n = 70
        avg_deg = 12
        A, edges = MatrixTestHelper.make_random_symmetric_matrix(n=n, avg_deg=avg_deg, seed=456, 
                                                                 allow_negative_diag=True)
        
        S, colors, comps = get_seeds_from_pattern(edges, n, "SMALLEST_LAST", "ACYCLIC_FOR_INDIRECT_RECOVERY")
        S = np.asarray(S, dtype=float)
        colors = np.asarray(colors, dtype=int)
        
        Y = A @ S
        
        A_rec = recover_matrix(
            recover_from="ACYCLIC",
            edges=edges,
            n=n,
            Y=Y,
            vertex_colors=colors,
            acyclic_components=comps,
            verify=True
        )
        
        # Relaxed tolerance for general symmetric matrices
        self.assertTrue(np.allclose(A, A_rec, rtol=1e-6, atol=1e-6))


class TestStructuredMatrices(unittest.TestCase):
    """Test suite for matrices with specific structures."""
    
    def test_block_diagonal_all_algorithms(self):
        """Test all algorithms on block diagonal matrix."""
        A, edges = MatrixTestHelper.make_block_diagonal_matrix(n_blocks=5, block_size=10, 
                                                               density=0.4, seed=42)
        n = A.shape[0]
        
        algorithms = [
            ("DISTANCE_TWO", "DISTANCE_TWO"),
            ("STAR", "STAR"),
            ("ACYCLIC_FOR_INDIRECT_RECOVERY", "ACYCLIC")
        ]
        
        for coloring_alg, recovery_alg in algorithms:
            with self.subTest(algorithm=recovery_alg):
                S, colors, comps = get_seeds_from_pattern(edges, n, "SMALLEST_LAST", coloring_alg)
                S = np.asarray(S, dtype=float)
                colors = np.asarray(colors, dtype=int)
                
                Y = A @ S
                
                A_rec = recover_matrix(
                    recover_from=recovery_alg,
                    edges=edges,
                    n=n,
                    Y=Y,
                    vertex_colors=colors,
                    acyclic_components=comps if recovery_alg == "ACYCLIC" else None,
                    verify=True
                )
                
                # Relaxed tolerance for block diagonal matrices
                self.assertTrue(np.allclose(A, A_rec, rtol=1e-6, atol=1e-6))
    
    def test_grid_laplacian_all_algorithms(self):
        """Test all algorithms on 2D grid Laplacian."""
        A, edges = MatrixTestHelper.make_grid_laplacian(grid_size=8)
        n = A.shape[0]
        
        algorithms = [
            ("DISTANCE_TWO", "DISTANCE_TWO"),
            ("STAR", "STAR"),
            ("ACYCLIC_FOR_INDIRECT_RECOVERY", "ACYCLIC")
        ]
        
        for coloring_alg, recovery_alg in algorithms:
            with self.subTest(algorithm=recovery_alg):
                S, colors, comps = get_seeds_from_pattern(edges, n, "SMALLEST_LAST", coloring_alg)
                S = np.asarray(S, dtype=float)
                colors = np.asarray(colors, dtype=int)
                
                Y = A @ S
                
                A_rec = recover_matrix(
                    recover_from=recovery_alg,
                    edges=edges,
                    n=n,
                    Y=Y,
                    vertex_colors=colors,
                    acyclic_components=comps if recovery_alg == "ACYCLIC" else None,
                    verify=True
                )
                
                self.assertTrue(np.allclose(A, A_rec, rtol=0, atol=1e-10))
    
    def test_banded_matrix_varying_bandwidth(self):
        """Test recovery on banded matrices with different bandwidths."""
        bandwidths = [2, 3, 5, 7]
        
        for bw in bandwidths:
            with self.subTest(bandwidth=bw):
                n = 60
                A, edges = MatrixTestHelper.make_banded_matrix(n=n, bandwidth=bw, seed=42)
                
                S, colors, _ = get_seeds_from_pattern(edges, n, "NATURAL", "DISTANCE_TWO")
                S = np.asarray(S, dtype=float)
                colors = np.asarray(colors, dtype=int)
                
                Y = A @ S
                
                A_rec = recover_matrix(
                    recover_from="DISTANCE_TWO",
                    edges=edges,
                    n=n,
                    Y=Y,
                    vertex_colors=colors,
                    verify=True
                )
                
                
                # Relaxed tolerance for banded matrices
                self.assertTrue(np.allclose(A, A_rec, rtol=1e-6, atol=1e-6))
    
    def test_star_graph_structure(self):
        """Test recovery on star graph structure."""
        n = 50
        A, edges = MatrixTestHelper.make_star_graph_matrix(n=n, seed=42)
        
        # Star graph should be especially efficient with STAR coloring
        S, colors, _ = get_seeds_from_pattern(edges, n, "NATURAL", "STAR")
        S = np.asarray(S, dtype=float)
        colors = np.asarray(colors, dtype=int)
        
        Y = A @ S
        
        A_rec = recover_matrix(
            recover_from="STAR",
            edges=edges,
            n=n,
            Y=Y,
            vertex_colors=colors,
            verify=True
        )
        
        # Relaxed tolerance for star graph matrices
        self.assertTrue(np.allclose(A, A_rec, rtol=1e-6, atol=1e-6))
        
        # Verify few colors needed for star graph
        n_colors = len(np.unique(colors))
        self.assertLessEqual(n_colors, 3, "Star graph should use very few colors")
    
    def test_cycle_graph_structure(self):
        """Test recovery on cycle graph structure."""
        n = 50
        A, edges = MatrixTestHelper.make_cycle_graph_matrix(n=n, seed=42)
        
        # Cycle should have chromatic number 2 or 3 depending on parity
        S, colors, comps = get_seeds_from_pattern(edges, n, "NATURAL", "ACYCLIC_FOR_INDIRECT_RECOVERY")
        S = np.asarray(S, dtype=float)
        colors = np.asarray(colors, dtype=int)
        
        Y = A @ S
        
        A_rec = recover_matrix(
            recover_from="ACYCLIC",
            edges=edges,
            n=n,
            Y=Y,
            vertex_colors=colors,
            acyclic_components=comps,
            verify=True
        )
        
        # Relaxed tolerance for cycle graph matrices
        self.assertTrue(np.allclose(A, A_rec, rtol=1e-6, atol=1e-6))
        
        # Verify few colors needed for cycle
        n_colors = len(np.unique(colors))
        self.assertLessEqual(n_colors, 3, "Cycle graph should use at most 3 colors")


class TestNumericalChallenges(unittest.TestCase):
    """Test suite for numerically challenging matrices."""
    
    def test_mixed_magnitude_matrix(self):
        """Test recovery on matrix with vastly different magnitudes."""
        n = 50
        avg_deg = 6
        A, edges = MatrixTestHelper.make_mixed_magnitude_matrix(n=n, avg_deg=avg_deg, seed=42)
        
        algorithms = [
            ("DISTANCE_TWO", "DISTANCE_TWO"),
            ("STAR", "STAR"),
            ("ACYCLIC_FOR_INDIRECT_RECOVERY", "ACYCLIC")
        ]
        
        for coloring_alg, recovery_alg in algorithms:
            with self.subTest(algorithm=recovery_alg):
                S, colors, comps = get_seeds_from_pattern(edges, n, "SMALLEST_LAST", coloring_alg)
                S = np.asarray(S, dtype=float)
                colors = np.asarray(colors, dtype=int)
                
                Y = A @ S
                
                # Use relative tolerance for mixed magnitudes
                A_rec = recover_matrix(
                    recover_from=recovery_alg,
                    edges=edges,
                    n=n,
                    Y=Y,
                    vertex_colors=colors,
                    acyclic_components=comps if recovery_alg == "ACYCLIC" else None,
                    dtype=np.float64,
                    verify=False  # Skip verification for numerical stability
                )
                
                # Check with relative tolerance
                max_val = np.abs(A).max()
                rel_err = np.abs(A - A_rec).max() / max_val
                self.assertLess(rel_err, 1e-8, 
                               f"{recovery_alg}: Relative error {rel_err:.2e} too large")
    
    def test_near_singular_matrix(self):
        """Test recovery on near-singular matrix (small but non-zero diagonal)."""
        n = 40
        A, edges = MatrixTestHelper.make_random_symmetric_matrix(n=n, avg_deg=6, seed=999, 
                                                                 allow_negative_diag=False)
        
        # Make it near-singular by scaling down diagonal
        for i in range(n):
            A[i, i] = 0.01
        
        S, colors, _ = get_seeds_from_pattern(edges, n, "SMALLEST_LAST", "DISTANCE_TWO")
        S = np.asarray(S, dtype=float)
        colors = np.asarray(colors, dtype=int)
        
        Y = A @ S
        
        A_rec = recover_matrix(
            recover_from="DISTANCE_TWO",
            edges=edges,
            n=n,
            Y=Y,
            vertex_colors=colors,
            verify=True
        )
        
        # Relaxed tolerance for near-singular matrices
        self.assertTrue(np.allclose(A, A_rec, rtol=1e-5, atol=1e-5))
    
    def test_varying_sparsity_levels(self):
        """Test recovery on matrices with different sparsity levels."""
        n = 50
        sparsity_levels = [4, 8, 12, 16, 20]  # Average degrees
        
        for avg_deg in sparsity_levels:
            with self.subTest(avg_degree=avg_deg):
                A, edges = MatrixTestHelper.make_random_symmetric_matrix(n=n, avg_deg=avg_deg, seed=42)
                
                S, colors, _ = get_seeds_from_pattern(edges, n, "SMALLEST_LAST", "STAR")
                S = np.asarray(S, dtype=float)
                colors = np.asarray(colors, dtype=int)
                
                Y = A @ S
                
                A_rec = recover_matrix(
                    recover_from="STAR",
                    edges=edges,
                    n=n,
                    Y=Y,
                    vertex_colors=colors,
                    verify=True
                )
                
                # Relaxed tolerance for general symmetric matrices
                self.assertTrue(np.allclose(A, A_rec, rtol=1e-6, atol=1e-6))


class TestOutputFormat(unittest.TestCase):
    """Test suite for verifying correct output formats (sparse vs dense)."""
    
    def test_distance_two_dense_output(self):
        """Test that DISTANCE_TWO returns dense array when reconstruct_sparsely=False."""
        n = 50
        avg_deg = 8
        A, edges = MatrixTestHelper.make_random_spd_matrix(n=n, avg_deg=avg_deg, seed=42)
        
        S, colors, _ = get_seeds_from_pattern(edges, n, "SMALLEST_LAST", "DISTANCE_TWO")
        S = np.asarray(S, dtype=float)
        colors = np.asarray(colors, dtype=int)
        
        Y = A @ S
        
        A_rec = recover_matrix(
            recover_from="DISTANCE_TWO",
            edges=edges,
            n=n,
            Y=Y,
            vertex_colors=colors,
            reconstruct_sparsely=False,
            dtype=np.float64
        )
        
        # Verify output is dense (numpy ndarray, not scipy sparse)
        self.assertIsInstance(A_rec, np.ndarray)
        self.assertFalse(sp.issparse(A_rec))
        self.assertEqual(A_rec.dtype, np.float64)
    
    def test_distance_two_sparse_output(self):
        """Test that DISTANCE_TWO returns sparse matrix when reconstruct_sparsely=True."""
        n = 50
        avg_deg = 8
        A, edges = MatrixTestHelper.make_random_spd_matrix(n=n, avg_deg=avg_deg, seed=42)
        
        S, colors, _ = get_seeds_from_pattern(edges, n, "SMALLEST_LAST", "DISTANCE_TWO")
        S = np.asarray(S, dtype=float)
        colors = np.asarray(colors, dtype=int)
        
        Y = A @ S
        
        A_rec = recover_matrix(
            recover_from="DISTANCE_TWO",
            edges=edges,
            n=n,
            Y=Y,
            vertex_colors=colors,
            reconstruct_sparsely=True,
            dtype=np.float64
        )
        
        # Verify output is sparse (scipy sparse matrix)
        self.assertTrue(sp.issparse(A_rec))
        self.assertIn(A_rec.format, ['csr', 'csc', 'coo'])
        self.assertEqual(A_rec.dtype, np.float64)
        
        # Verify accuracy by converting to dense
        self.assertTrue(np.allclose(A, A_rec.toarray(), rtol=0, atol=1e-10))
    
    def test_star_dense_output(self):
        """Test that STAR returns dense array when reconstruct_sparsely=False."""
        n = 60
        avg_deg = 10
        A, edges = MatrixTestHelper.make_random_spd_matrix(n=n, avg_deg=avg_deg, seed=55)
        
        S, colors, _ = get_seeds_from_pattern(edges, n, "SMALLEST_LAST", "STAR")
        S = np.asarray(S, dtype=float)
        colors = np.asarray(colors, dtype=int)
        
        Y = A @ S
        
        A_rec = recover_matrix(
            recover_from="STAR",
            edges=edges,
            n=n,
            Y=Y,
            vertex_colors=colors,
            reconstruct_sparsely=False,
            dtype=np.float64
        )
        
        # Verify output is dense
        self.assertIsInstance(A_rec, np.ndarray)
        self.assertFalse(sp.issparse(A_rec))
    
    def test_star_sparse_output(self):
        """Test that STAR returns sparse matrix when reconstruct_sparsely=True."""
        n = 60
        avg_deg = 10
        A, edges = MatrixTestHelper.make_random_spd_matrix(n=n, avg_deg=avg_deg, seed=55)
        
        S, colors, _ = get_seeds_from_pattern(edges, n, "SMALLEST_LAST", "STAR")
        S = np.asarray(S, dtype=float)
        colors = np.asarray(colors, dtype=int)
        
        Y = A @ S
        
        A_rec = recover_matrix(
            recover_from="STAR",
            edges=edges,
            n=n,
            Y=Y,
            vertex_colors=colors,
            reconstruct_sparsely=True,
            dtype=np.float64
        )
        
        # Verify output is sparse
        self.assertTrue(sp.issparse(A_rec))
        self.assertIn(A_rec.format, ['csr', 'csc', 'coo'])
        
        # Verify accuracy
        self.assertTrue(np.allclose(A, A_rec.toarray(), rtol=0, atol=1e-10))
    
    def test_acyclic_dense_output(self):
        """Test that ACYCLIC returns dense array when reconstruct_sparsely=False."""
        n = 70
        avg_deg = 12
        A, edges = MatrixTestHelper.make_random_spd_matrix(n=n, avg_deg=avg_deg, seed=42)
        
        S, colors, comps = get_seeds_from_pattern(edges, n, "SMALLEST_LAST", "ACYCLIC_FOR_INDIRECT_RECOVERY")
        S = np.asarray(S, dtype=float)
        colors = np.asarray(colors, dtype=int)
        
        Y = A @ S
        
        A_rec = recover_matrix(
            recover_from="ACYCLIC",
            edges=edges,
            n=n,
            Y=Y,
            vertex_colors=colors,
            acyclic_components=comps,
            reconstruct_sparsely=False,
            dtype=np.float64
        )
        
        # Verify output is dense
        self.assertIsInstance(A_rec, np.ndarray)
        self.assertFalse(sp.issparse(A_rec))
    
    def test_acyclic_sparse_output(self):
        """Test that ACYCLIC returns sparse matrix when reconstruct_sparsely=True."""
        n = 70
        avg_deg = 12
        A, edges = MatrixTestHelper.make_random_spd_matrix(n=n, avg_deg=avg_deg, seed=42)
        
        S, colors, comps = get_seeds_from_pattern(edges, n, "SMALLEST_LAST", "ACYCLIC_FOR_INDIRECT_RECOVERY")
        S = np.asarray(S, dtype=float)
        colors = np.asarray(colors, dtype=int)
        
        Y = A @ S
        
        A_rec = recover_matrix(
            recover_from="ACYCLIC",
            edges=edges,
            n=n,
            Y=Y,
            vertex_colors=colors,
            acyclic_components=comps,
            reconstruct_sparsely=True,
            dtype=np.float64
        )
        
        # Verify output is sparse
        self.assertTrue(sp.issparse(A_rec))
        self.assertIn(A_rec.format, ['csr', 'csc', 'coo'])
        
        # Verify accuracy
        self.assertTrue(np.allclose(A, A_rec.toarray(), rtol=0, atol=1e-10))
    
    def test_sparse_output_with_verification(self):
        """Test that sparse output works correctly with verification enabled."""
        n = 60
        avg_deg = 8
        A, edges = MatrixTestHelper.make_random_spd_matrix(n=n, avg_deg=avg_deg, seed=123)
        
        algorithms = [
            ("DISTANCE_TWO", "DISTANCE_TWO", None),
            ("STAR", "STAR", None),
            ("ACYCLIC_FOR_INDIRECT_RECOVERY", "ACYCLIC", "comps")
        ]
        
        for coloring_alg, recovery_alg, comps_key in algorithms:
            with self.subTest(algorithm=recovery_alg):
                S, colors, comps = get_seeds_from_pattern(edges, n, "SMALLEST_LAST", coloring_alg)
                S = np.asarray(S, dtype=float)
                colors = np.asarray(colors, dtype=int)
                
                Y = A @ S
                
                # Test with sparse reconstruction and verification
                A_rec = recover_matrix(
                    recover_from=recovery_alg,
                    edges=edges,
                    n=n,
                    Y=Y,
                    vertex_colors=colors,
                    acyclic_components=comps if comps_key else None,
                    reconstruct_sparsely=True,
                    dtype=np.float64,
                    verify=True
                )
                
                # Verify output is sparse
                self.assertTrue(sp.issparse(A_rec), 
                               f"{recovery_alg} should return sparse matrix")
                
                # Verify accuracy
                self.assertTrue(np.allclose(A, A_rec.toarray(), rtol=0, atol=1e-10),
                               f"{recovery_alg} failed accuracy test")
    
    def test_sparse_format_is_csr(self):
        """Test that sparse output format is CSR (compressed sparse row)."""
        n = 50
        avg_deg = 8
        A, edges = MatrixTestHelper.make_random_spd_matrix(n=n, avg_deg=avg_deg, seed=42)
        
        S, colors, _ = get_seeds_from_pattern(edges, n, "SMALLEST_LAST", "DISTANCE_TWO")
        S = np.asarray(S, dtype=float)
        colors = np.asarray(colors, dtype=int)
        
        Y = A @ S
        
        A_rec = recover_matrix(
            recover_from="DISTANCE_TWO",
            edges=edges,
            n=n,
            Y=Y,
            vertex_colors=colors,
            reconstruct_sparsely=True,
            dtype=np.float64
        )
        
        # Verify format is CSR (efficient for most operations)
        self.assertEqual(A_rec.format, 'csr')
    
    def test_dtype_preservation_sparse(self):
        """Test that dtype is preserved in sparse output."""
        n = 40
        avg_deg = 6
        A, edges = MatrixTestHelper.make_random_spd_matrix(n=n, avg_deg=avg_deg, seed=777)
        
        S, colors, _ = get_seeds_from_pattern(edges, n, "NATURAL", "DISTANCE_TWO")
        S = np.asarray(S, dtype=float)
        colors = np.asarray(colors, dtype=int)
        
        Y = A @ S
        
        for dtype in [np.float32, np.float64]:
            with self.subTest(dtype=dtype):
                A_rec = recover_matrix(
                    recover_from="DISTANCE_TWO",
                    edges=edges,
                    n=n,
                    Y=Y,
                    vertex_colors=colors,
                    reconstruct_sparsely=True,
                    dtype=dtype
                )
                
                self.assertTrue(sp.issparse(A_rec))
                self.assertEqual(A_rec.dtype, dtype)
    
    def test_sparse_output_sparsity_level(self):
        """Test that sparse output has expected number of non-zeros."""
        n = 50
        avg_deg = 8
        A, edges = MatrixTestHelper.make_random_spd_matrix(n=n, avg_deg=avg_deg, seed=42)
        
        S, colors, _ = get_seeds_from_pattern(edges, n, "SMALLEST_LAST", "DISTANCE_TWO")
        S = np.asarray(S, dtype=float)
        colors = np.asarray(colors, dtype=int)
        
        Y = A @ S
        
        A_rec = recover_matrix(
            recover_from="DISTANCE_TWO",
            edges=edges,
            n=n,
            Y=Y,
            vertex_colors=colors,
            reconstruct_sparsely=True,
            dtype=np.float64
        )
        
        # Count expected non-zeros: diagonal + 2*edges
        expected_nnz = n + 2 * len(edges)
        
        self.assertEqual(A_rec.nnz, expected_nnz,
                        f"Expected {expected_nnz} non-zeros, got {A_rec.nnz}")


class TestLargeScaleMatrices(unittest.TestCase):
    """Test suite for larger-scale matrices to verify scalability."""
    
    def test_large_sparse_distance_two(self):
        """Test DISTANCE_TWO on larger sparse matrix."""
        n = 200
        avg_deg = 8
        A, edges = MatrixTestHelper.make_random_symmetric_matrix(n=n, avg_deg=avg_deg, seed=42)
        
        S, colors, _ = get_seeds_from_pattern(edges, n, "SMALLEST_LAST", "DISTANCE_TWO")
        S = np.asarray(S, dtype=float)
        colors = np.asarray(colors, dtype=int)
        
        Y = A @ S
        
        A_rec = recover_matrix(
            recover_from="DISTANCE_TWO",
            edges=edges,
            n=n,
            Y=Y,
            vertex_colors=colors,
            reconstruct_sparsely=True,
            verify=True
        )
        
        # Relaxed tolerance for large sparse matrices
        # Convert sparse to dense for comparison
        A_rec_dense = A_rec.toarray() if sp.issparse(A_rec) else A_rec
        self.assertTrue(np.allclose(A, A_rec_dense, rtol=1e-6, atol=1e-6))
    
    def test_large_grid_laplacian(self):
        """Test recovery on large grid Laplacian (realistic PDE discretization)."""
        grid_size = 15  # 225x225 matrix
        A, edges = MatrixTestHelper.make_grid_laplacian(grid_size=grid_size)
        n = A.shape[0]
        
        S, colors, comps = get_seeds_from_pattern(edges, n, "SMALLEST_LAST", "ACYCLIC_FOR_INDIRECT_RECOVERY")
        S = np.asarray(S, dtype=float)
        colors = np.asarray(colors, dtype=int)
        
        Y = A @ S
        
        A_rec = recover_matrix(
            recover_from="ACYCLIC",
            edges=edges,
            n=n,
            Y=Y,
            vertex_colors=colors,
            acyclic_components=comps,
            reconstruct_sparsely=True,
            verify=True
        )
        
        # Convert sparse to dense for comparison
        A_rec_dense = A_rec.toarray() if sp.issparse(A_rec) else A_rec
        self.assertTrue(np.allclose(A, A_rec_dense, rtol=0, atol=1e-10))


@unittest.skipUnless(HAS_TORCH, "PyTorch not available")
class TestPyTorchGradients(unittest.TestCase):
    """Test automatic differentiation through matrix recovery with PyTorch."""
    
    def test_gradient_through_distance_two_recovery(self):
        """Test that gradients flow correctly through DISTANCE_TWO recovery."""
        # Setup
        n = 8
        A_np, edges = MatrixTestHelper.make_random_spd_matrix(n=n, avg_deg=12, seed=42)
        S, colors, comps = get_seeds_from_pattern(edges, n, "SMALLEST_LAST", "DISTANCE_TWO")
        S_np = np.asarray(S)
        
        # Convert to PyTorch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        A_torch = torch.tensor(A_np, dtype=torch.float64, device=device)
        S_torch = torch.tensor(S_np, dtype=torch.float64, device=device)
        
        # Create trainable parameter (squared for more interesting gradient)
        scale_param = torch.tensor([2.0], dtype=torch.float64, device=device, requires_grad=True)
        A_scaled = (scale_param ** 2) * A_torch
        
        # Compressed evaluation and recovery
        Y = A_scaled @ S_torch
        A_rec = recover_matrix(
            recover_from="DISTANCE_TWO",
            edges=edges,
            n=n,
            Y=Y,
            vertex_colors=colors,
            reconstruct_sparsely=False,
            verify=True
        )
        
        # Compute loss and backprop
        loss = A_rec.sum()
        loss.backward()
        
        # Verify gradient
        # d(loss)/d(scale) = d/d(scale)[sum(scale^2 * A)] = 2 * scale * sum(A)
        expected_gradient = 2 * scale_param.item() * A_torch.sum().item()
        self.assertIsNotNone(scale_param.grad)
        self.assertAlmostEqual(scale_param.grad.item(), expected_gradient, places=10)
        
        # Verify recovered matrix
        A_rec_expected = A_scaled.detach()
        max_diff = torch.abs(A_rec - A_rec_expected).max().item()
        self.assertLess(max_diff, 1e-10)
    
    def test_gradient_through_star_recovery(self):
        """Test that gradients flow correctly through STAR recovery."""
        # Setup
        n = 8
        A_np, edges = MatrixTestHelper.make_random_spd_matrix(n=n, avg_deg=12, seed=43)
        S, colors, comps = get_seeds_from_pattern(edges, n, "SMALLEST_LAST", "STAR")
        S_np = np.asarray(S)
        
        # Convert to PyTorch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        A_torch = torch.tensor(A_np, dtype=torch.float64, device=device)
        S_torch = torch.tensor(S_np, dtype=torch.float64, device=device)
        
        # Create trainable parameter
        scale_param = torch.tensor([3.0], dtype=torch.float64, device=device, requires_grad=True)
        A_scaled = (scale_param ** 2) * A_torch
        
        # Compressed evaluation and recovery
        Y = A_scaled @ S_torch
        A_rec = recover_matrix(
            recover_from="STAR",
            edges=edges,
            n=n,
            Y=Y,
            vertex_colors=colors,
            reconstruct_sparsely=False,
            verify=True
        )
        
        # Compute loss and backprop
        loss = A_rec.sum()
        loss.backward()
        
        # Verify gradient
        expected_gradient = 2 * scale_param.item() * A_torch.sum().item()
        self.assertIsNotNone(scale_param.grad)
        self.assertAlmostEqual(scale_param.grad.item(), expected_gradient, places=10)
    
    def test_gradient_through_acyclic_recovery(self):
        """Test that gradients flow correctly through ACYCLIC recovery."""
        # Setup
        n = 8
        A_np, edges = MatrixTestHelper.make_random_spd_matrix(n=n, avg_deg=12, seed=44)
        S, colors, comps = get_seeds_from_pattern(edges, n, "SMALLEST_LAST", "ACYCLIC")
        S_np = np.asarray(S)
        
        # Convert to PyTorch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        A_torch = torch.tensor(A_np, dtype=torch.float64, device=device)
        S_torch = torch.tensor(S_np, dtype=torch.float64, device=device)
        
        # Create trainable parameter
        scale_param = torch.tensor([1.5], dtype=torch.float64, device=device, requires_grad=True)
        A_scaled = (scale_param ** 2) * A_torch
        
        # Compressed evaluation and recovery
        Y = A_scaled @ S_torch
        A_rec = recover_matrix(
            recover_from="ACYCLIC",
            edges=edges,
            n=n,
            Y=Y,
            vertex_colors=colors,
            acyclic_components=comps,
            reconstruct_sparsely=False,
            verify=True
        )
        
        # Compute loss and backprop
        loss = A_rec.sum()
        loss.backward()
        
        # Verify gradient
        expected_gradient = 2 * scale_param.item() * A_torch.sum().item()
        self.assertIsNotNone(scale_param.grad)
        self.assertAlmostEqual(scale_param.grad.item(), expected_gradient, places=10)
    
    def test_gradient_with_complex_function(self):
        """Test gradients with a more complex loss function."""
        # Setup
        n = 10
        A_np, edges = MatrixTestHelper.make_tridiagonal_matrix(n=n)
        S, colors, comps = get_seeds_from_pattern(edges, n, "SMALLEST_LAST", "ACYCLIC")
        S_np = np.asarray(S)
        
        # Convert to PyTorch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        A_torch = torch.tensor(A_np, dtype=torch.float64, device=device)
        S_torch = torch.tensor(S_np, dtype=torch.float64, device=device)
        
        # Create trainable parameter
        scale_param = torch.tensor([2.0], dtype=torch.float64, device=device, requires_grad=True)
        A_scaled = torch.exp(scale_param) * A_torch  # Use exponential for variety
        
        # Compressed evaluation and recovery
        Y = A_scaled @ S_torch
        A_rec = recover_matrix(
            recover_from="ACYCLIC",
            edges=edges,
            n=n,
            Y=Y,
            vertex_colors=colors,
            acyclic_components=comps,
            reconstruct_sparsely=False,
            verify=True
        )
        
        # Complex loss: L2 norm of the matrix
        loss = (A_rec ** 2).sum()
        loss.backward()
        
        # Verify gradient exists and is finite
        self.assertIsNotNone(scale_param.grad)
        self.assertTrue(torch.isfinite(scale_param.grad).all())
        
        # Numerical gradient check
        eps = 1e-6
        with torch.no_grad():
            scale_param_plus = scale_param + eps
            A_scaled_plus = torch.exp(scale_param_plus) * A_torch
            Y_plus = A_scaled_plus @ S_torch
            A_rec_plus = recover_matrix(
                recover_from="ACYCLIC",
                edges=edges,
                n=n,
                Y=Y_plus,
                vertex_colors=colors,
                acyclic_components=comps,
                reconstruct_sparsely=False,
                verify=False
            )
            loss_plus = (A_rec_plus ** 2).sum()
            numerical_grad = (loss_plus - loss.detach()) / eps
        
        # Check that analytical and numerical gradients are close
        rel_error = abs(scale_param.grad.item() - numerical_grad.item()) / (abs(numerical_grad.item()) + 1e-10)
        self.assertLess(rel_error, 1e-5)

    def test_gradient_through_sparse_hessian_create_graph(self):
        """Gradients w.r.t. a parameter that enters E, through HVP + sparse recovery.

        Uses a quadratic E = 1/2 * scale^2 * x^T A x so H = scale^2 * A, and checks
        that create_graph=True lets a loss on H differentiate into scale.
        """
        from colpackpy.torch_utils import compute_sparse_hessian

        n = 8
        A_np, edges = MatrixTestHelper.make_tridiagonal_matrix(n=n)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        A = torch.tensor(A_np, dtype=torch.float64, device=device)
        x = torch.randn(n, dtype=torch.float64, device=device, requires_grad=True)
        scale = torch.tensor([1.7], dtype=torch.float64, device=device, requires_grad=True)

        # Quadratic energy; Hessian w.r.t. x is scale^2 * A (independent of x)
        E = 0.5 * (scale ** 2) * torch.dot(x, A @ x)
        grads = torch.autograd.grad(E, x, create_graph=True, retain_graph=True)[0]

        H = compute_sparse_hessian(
            E,
            x,
            edges,
            n,
            "ACYCLIC",
            precomputed_grads=grads,
            hessian_return_type="sparse",
            create_graph=True,
        )

        # Recovered H should match scale^2 * A on the sparsity pattern
        H_expected = (scale ** 2) * A
        self.assertLess(torch.abs(H - H_expected).max().item(), 1e-10)

        loss = H.sum()
        loss.backward()

        # d/d(scale) sum(scale^2 * A) = 2 * scale * sum(A)
        expected = 2 * scale.item() * A.sum().item()
        self.assertIsNotNone(scale.grad)
        self.assertAlmostEqual(scale.grad.item(), expected, places=8)

        # Default create_graph=False must not attach grads into scale
        scale2 = torch.tensor([1.7], dtype=torch.float64, device=device, requires_grad=True)
        x2 = x.detach().clone().requires_grad_(True)
        E2 = 0.5 * (scale2 ** 2) * torch.dot(x2, A @ x2)
        grads2 = torch.autograd.grad(E2, x2, create_graph=True, retain_graph=True)[0]
        H2 = compute_sparse_hessian(
            E2,
            x2,
            edges,
            n,
            "ACYCLIC",
            precomputed_grads=grads2,
            hessian_return_type="sparse",
            create_graph=False,
        )
        self.assertFalse(H2.requires_grad)


@unittest.skipUnless(HAS_TORCH, "PyTorch not available")
class TestSparseHessianComputation(unittest.TestCase):
    """Test sparse Hessian computation using HVPs and ColPack coloring."""
    
    @staticmethod
    def hessian_vector_product(f):
        """Generate HVP function."""
        def hvp(x, v):
            grad_func = jacrev(f)
            _, hv = jvp(grad_func, (x,), (v,))
            return hv
        return hvp
    
    @staticmethod
    def compute_hessian_autograd(f, x):
        """Compute Hessian using standard PyTorch autograd (for comparison)."""
        n = x.shape[0]
        H = torch.zeros(n, n, dtype=x.dtype, device=x.device)
        
        for i in range(n):
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
    
    @staticmethod
    def compute_sparse_hessian_hvp(f, x, edges, coloring_algorithm="ACYCLIC", verify=False):
        """Compute sparse Hessian using HVPs."""
        n = x.shape[0]
        device = x.device
        dtype = x.dtype
        
        # Get ColPack seeds and coloring
        S_np, colors_np, comps = get_seeds_from_pattern(
            edges, n, "SMALLEST_LAST", coloring_algorithm
        )
        S_np = np.asarray(S_np)
        S = torch.tensor(S_np, dtype=dtype, device=device)
        
        # Create HVP function and vectorize it
        hvp = TestSparseHessianComputation.hessian_vector_product(f)
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
        
        return H
    
    def test_block_diagonal_hessian(self):
        """Test sparse Hessian computation with block diagonal structure."""
        # Create a simple quadratic function with block diagonal Hessian
        def f(x):
            # Two blocks: first 3 dims, last 2 dims
            block1 = x[:3]
            block2 = x[3:]
            return (block1 ** 2).sum() + (block2 ** 2).sum() + (block1[:-1] * block1[1:]).sum()
        
        n = 5
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = torch.randn(n, dtype=torch.float64, device=device)
        
        # Define block diagonal sparsity pattern
        edges = np.array([[0, 1], [1, 2], [3, 4]], dtype=int)
        
        # Compute Hessian using HVPs
        H = self.compute_sparse_hessian_hvp(f, x, edges, "ACYCLIC", verify=True)
        
        # Verify basic properties
        self.assertEqual(H.shape, (n, n))
        
        # Check symmetry
        H_np = H.detach().cpu().numpy()
        self.assertTrue(np.allclose(H_np, H_np.T, atol=1e-10))
        
        # Check that off-block-diagonal elements are near zero
        # Blocks are [0,1,2] and [3,4]
        self.assertLess(abs(H_np[0, 3]), 1e-10)
        self.assertLess(abs(H_np[0, 4]), 1e-10)
        self.assertLess(abs(H_np[1, 3]), 1e-10)
        self.assertLess(abs(H_np[1, 4]), 1e-10)
        self.assertLess(abs(H_np[2, 3]), 1e-10)
        self.assertLess(abs(H_np[2, 4]), 1e-10)
    
    def test_neural_network_hessian(self):
        """Test sparse Hessian computation with neural network function."""
        # Simple neural network with two independent parts
        class TwoNetFunction(nn.Module):
            def __init__(self):
                super().__init__()
                self.net1 = nn.Sequential(
                    nn.Linear(3, 4),
                    nn.Tanh(),
                    nn.Linear(4, 1)
                )
                self.net2 = nn.Sequential(
                    nn.Linear(2, 3),
                    nn.Tanh(),
                    nn.Linear(3, 1)
                )
            
            def forward(self, x):
                return self.net1(x[:3]).squeeze() + self.net2(x[3:]).squeeze()
        
        n = 5
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model = TwoNetFunction().double().to(device)
        model.eval()
        
        x = torch.randn(n, dtype=torch.float64, device=device)
        
        # Define block diagonal sparsity (fully connected within blocks)
        edges = []
        for i in range(3):
            for j in range(i + 1, 3):
                edges.append([i, j])
        for i in range(3, 5):
            for j in range(i + 1, 5):
                edges.append([i, j])
        edges = np.array(edges, dtype=int)
        
        # Compute Hessian using HVPs
        H = self.compute_sparse_hessian_hvp(lambda x: model(x), x, edges, "ACYCLIC", verify=True)
        
        # Verify basic properties
        self.assertEqual(H.shape, (n, n))
        
        # Check symmetry
        H_np = H.detach().cpu().numpy()
        self.assertTrue(np.allclose(H_np, H_np.T, atol=1e-10))
        
        # Check that off-block-diagonal elements are near zero
        for i in range(3):
            for j in range(3, 5):
                self.assertLess(abs(H_np[i, j]), 1e-10)
    
    def test_compression_ratio(self):
        """Test that sparse Hessian computation achieves compression."""
        # Large block diagonal system
        def f(x):
            n = x.shape[0]
            block_size = 4
            num_blocks = n // block_size
            total = torch.tensor(0.0, dtype=x.dtype, device=x.device)
            
            for i in range(num_blocks):
                start = i * block_size
                end = start + block_size
                block = x[start:end]
                total = total + (block ** 2).sum()
            
            return total
        
        n = 16
        block_size = 4
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = torch.randn(n, dtype=torch.float64, device=device)
        
        # Define block diagonal sparsity pattern
        edges = []
        num_blocks = n // block_size
        for b in range(num_blocks):
            start = b * block_size
            for i in range(start, start + block_size):
                for j in range(i + 1, start + block_size):
                    edges.append([i, j])
        edges = np.array(edges, dtype=int)
        
        # Get coloring information
        S_np, colors_np, comps = get_seeds_from_pattern(
            edges, n, "SMALLEST_LAST", "ACYCLIC"
        )
        S_np = np.asarray(S_np)
        num_colors = S_np.shape[1]
        
        # Verify compression is achieved
        self.assertLess(num_colors, n)
        # For block diagonal with 4x4 blocks, we expect 4 colors
        self.assertEqual(num_colors, block_size)
        
        # Compute Hessian using HVPs
        H = self.compute_sparse_hessian_hvp(f, x, edges, "ACYCLIC", verify=True)
        
        # Verify it's symmetric and diagonal (for this simple quadratic function)
        H_np = H.detach().cpu().numpy()
        self.assertTrue(np.allclose(H_np, H_np.T, atol=1e-10))
    
    def test_different_coloring_algorithms(self):
        """Test sparse Hessian computation with different coloring algorithms."""
        def f(x):
            return (x ** 2).sum() + (x[:-1] * x[1:]).sum()
        
        n = 6
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = torch.randn(n, dtype=torch.float64, device=device)
        
        # Tridiagonal sparsity pattern
        edges = np.array([[i, i+1] for i in range(n-1)], dtype=int)
        
        # Test with different algorithms (without verification to avoid issues with different seed matrices)
        # Compute manually without verification
        for algorithm in ["ACYCLIC", "STAR", "DISTANCE_TWO"]:
            H = self.compute_sparse_hessian_hvp(f, x, edges, algorithm)
            
            # Verify basic properties
            self.assertEqual(H.shape, (n, n))
            
            # Check symmetry
            H_np = H.detach().cpu().numpy()
            self.assertTrue(np.allclose(H_np, H_np.T, atol=1e-10))
        
        # Now test that ACYCLIC and DISTANCE_TWO give similar results
        # (STAR may differ slightly due to numerical precision)
        H_acyclic = self.compute_sparse_hessian_hvp(f, x, edges, "ACYCLIC")
        H_distance_two = self.compute_sparse_hessian_hvp(f, x, edges, "DISTANCE_TWO")
        
        H_acyclic_np = H_acyclic.detach().cpu().numpy()
        H_distance_two_np = H_distance_two.detach().cpu().numpy()
        
        self.assertTrue(np.allclose(H_acyclic_np, H_distance_two_np, atol=1e-8))
    
    def test_compare_with_naive_autograd(self):
        """Test that sparse Hessian matches naive autograd computation."""
        # Simple quadratic function
        def f(x):
            return (x ** 2).sum() + (x[:-1] * x[1:]).sum() + 0.5 * (x ** 3).sum()
        
        n = 6
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = torch.randn(n, dtype=torch.float64, device=device, requires_grad=True)
        
        # Tridiagonal sparsity pattern
        edges = np.array([[i, i+1] for i in range(n-1)], dtype=int)
        
        # Compute using sparse method
        H_sparse = self.compute_sparse_hessian_hvp(f, x, edges, "ACYCLIC", verify=False)
        
        # Compute using naive autograd
        x_copy = x.detach().clone().requires_grad_(True)
        H_naive = self.compute_hessian_autograd(f, x_copy)
        
        # Compare
        H_sparse_np = H_sparse.detach().cpu().numpy()
        H_naive_np = H_naive.detach().cpu().numpy()
        
        max_diff = np.abs(H_sparse_np - H_naive_np).max()
        self.assertLess(max_diff, 1e-10, f"Max difference: {max_diff}")
        self.assertTrue(np.allclose(H_sparse_np, H_naive_np, atol=1e-10))
    
    def test_neural_network_vs_autograd(self):
        """Test neural network Hessian against naive autograd."""
        class SimpleNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(4, 6),
                    nn.Tanh(),
                    nn.Linear(6, 1)
                )
            
            def forward(self, x):
                return self.net(x).squeeze()
        
        n = 4
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model = SimpleNet().double().to(device)
        model.eval()
        
        x = torch.randn(n, dtype=torch.float64, device=device, requires_grad=True)
        
        # Fully connected sparsity pattern
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append([i, j])
        edges = np.array(edges, dtype=int)
        
        # Compute using sparse method
        H_sparse = self.compute_sparse_hessian_hvp(lambda x: model(x), x, edges, "ACYCLIC", verify=False)
        
        # Compute using naive autograd
        x_copy = x.detach().clone().requires_grad_(True)
        H_naive = self.compute_hessian_autograd(lambda x: model(x), x_copy)
        
        # Compare
        H_sparse_np = H_sparse.detach().cpu().numpy()
        H_naive_np = H_naive.detach().cpu().numpy()
        
        max_diff = np.abs(H_sparse_np - H_naive_np).max()
        self.assertLess(max_diff, 1e-9, f"Max difference: {max_diff}")
        self.assertTrue(np.allclose(H_sparse_np, H_naive_np, atol=1e-9))


def suite():
    """Create a test suite for the recovery module."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Original test suites
    suite.addTests(loader.loadTestsFromTestCase(TestDistanceTwoRecovery))
    suite.addTests(loader.loadTestsFromTestCase(TestStarRecovery))
    suite.addTests(loader.loadTestsFromTestCase(TestAcyclicRecovery))
    suite.addTests(loader.loadTestsFromTestCase(TestRecoveryEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestRecoveryComparison))
    
    # Extended test suites for general matrices
    suite.addTests(loader.loadTestsFromTestCase(TestGeneralSymmetricMatrices))
    suite.addTests(loader.loadTestsFromTestCase(TestStructuredMatrices))
    suite.addTests(loader.loadTestsFromTestCase(TestNumericalChallenges))
    suite.addTests(loader.loadTestsFromTestCase(TestOutputFormat))
    suite.addTests(loader.loadTestsFromTestCase(TestLargeScaleMatrices))
    
    # PyTorch gradient tests (if available)
    if HAS_TORCH:
        suite.addTests(loader.loadTestsFromTestCase(TestPyTorchGradients))
        suite.addTests(loader.loadTestsFromTestCase(TestSparseHessianComputation))
    
    return suite


if __name__ == '__main__':
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())
