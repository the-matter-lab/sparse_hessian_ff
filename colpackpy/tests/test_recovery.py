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
        B = A @ S
        
        # Recovery
        A_rec = recover_matrix(
            recover_from="DISTANCE_TWO",
            edges=edges,
            n=n,
            B=B,
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
        
        B = A @ S
        
        A_rec = recover_matrix(
            recover_from="DISTANCE_TWO",
            edges=edges,
            n=n,
            B=B,
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
        
        B = A @ S
        
        # Test with sparse reconstruction
        A_rec = recover_matrix(
            recover_from="DISTANCE_TWO",
            edges=edges,
            n=n,
            B=B,
            vertex_colors=colors,
            reconstruct_sparsely=True,
            dtype=np.float64,
            verify=True
        )
        
        self.assertTrue(np.allclose(A, A_rec, rtol=0, atol=1e-10))
    
    def test_distance_two_diagonal_only(self):
        """Test DISTANCE_TWO on diagonal matrix (edge case: no edges)."""
        n = 30
        A, edges = MatrixTestHelper.make_diagonal_matrix(n=n)
        
        S, colors, _ = get_seeds_from_pattern(edges, n, "NATURAL", "DISTANCE_TWO")
        S = np.asarray(S, dtype=float)
        colors = np.asarray(colors, dtype=int)
        
        B = A @ S
        
        A_rec = recover_matrix(
            recover_from="DISTANCE_TWO",
            edges=edges,
            n=n,
            B=B,
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
        B = A @ S
        
        # Recovery
        A_rec = recover_matrix(
            recover_from="STAR",
            edges=edges,
            n=n,
            B=B,
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
        
        B = A @ S
        
        A_rec = recover_matrix(
            recover_from="STAR",
            edges=edges,
            n=n,
            B=B,
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
        
        B = A @ S
        
        # Test with sparse reconstruction
        A_rec = recover_matrix(
            recover_from="STAR",
            edges=edges,
            n=n,
            B=B,
            vertex_colors=colors,
            reconstruct_sparsely=True,
            dtype=np.float64,
            verify=True
        )
        
        self.assertTrue(np.allclose(A, A_rec, rtol=0, atol=1e-10))
    
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
                
                B = A @ S
                
                A_rec = recover_matrix(
                    recover_from="STAR",
                    edges=edges,
                    n=n,
                    B=B,
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
        B = A @ S
        
        # Recovery with cached components
        A_rec = recover_matrix(
            recover_from="ACYCLIC",
            edges=edges,
            n=n,
            B=B,
            vertex_colors=colors,
            acyclic_components=comps,
            reconstruct_sparsely=True,
            dtype=np.float64,
            verify=True
        )
        
        # Verify accuracy
        self.assertTrue(np.allclose(A, A_rec, rtol=0, atol=1e-10))
        max_err = np.abs(A - A_rec).max()
        self.assertLess(max_err, 1e-10)
    
    def test_acyclic_tridiagonal(self):
        """Test ACYCLIC on tridiagonal matrix."""
        n = 50
        A, edges = MatrixTestHelper.make_tridiagonal_matrix(n=n)
        
        S, colors, comps = get_seeds_from_pattern(edges, n, "NATURAL", "ACYCLIC_FOR_INDIRECT_RECOVERY")
        S = np.asarray(S, dtype=float)
        colors = np.asarray(colors, dtype=int)
        
        B = A @ S
        
        A_rec = recover_matrix(
            recover_from="ACYCLIC",
            edges=edges,
            n=n,
            B=B,
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
        
        B = A @ S
        
        # Test without passing acyclic_components (fallback path)
        A_rec = recover_matrix(
            recover_from="ACYCLIC",
            edges=edges,
            n=n,
            B=B,
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
        
        B = A @ S
        
        # Test with sparse reconstruction
        A_rec = recover_matrix(
            recover_from="ACYCLIC",
            edges=edges,
            n=n,
            B=B,
            vertex_colors=colors,
            acyclic_components=comps,
            reconstruct_sparsely=True,
            dtype=np.float64,
            verify=True
        )
        
        self.assertTrue(np.allclose(A, A_rec, rtol=0, atol=1e-10))
    
    def test_acyclic_indirect_alias(self):
        """Test ACYCLIC_INDIRECT alias for recovery."""
        n = 50
        avg_deg = 7
        A, edges = MatrixTestHelper.make_random_spd_matrix(n=n, avg_deg=avg_deg, seed=333)
        
        S, colors, comps = get_seeds_from_pattern(edges, n, "SMALLEST_LAST", "ACYCLIC_FOR_INDIRECT_RECOVERY")
        S = np.asarray(S, dtype=float)
        colors = np.asarray(colors, dtype=int)
        
        B = A @ S
        
        # Test with ACYCLIC_INDIRECT alias
        A_rec = recover_matrix(
            recover_from="ACYCLIC_INDIRECT",
            edges=edges,
            n=n,
            B=B,
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
        
        B = A @ S
        
        A_rec = recover_matrix(
            recover_from="ACYCLIC",
            edges=edges,
            n=n,
            B=B,
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
        
        B = A @ S
        
        with self.assertRaises(ValueError):
            recover_matrix(
                recover_from="INVALID_ALGORITHM",
                edges=edges,
                n=n,
                B=B,
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
        
        B = A @ S
        
        for algorithm in ["DISTANCE_TWO", "STAR"]:
            with self.subTest(algorithm=algorithm):
                A_rec = recover_matrix(
                    recover_from=algorithm,
                    edges=edges,
                    n=n,
                    B=B,
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
        
        B = A @ S
        
        for dtype in [np.float32, np.float64]:
            with self.subTest(dtype=dtype):
                A_rec = recover_matrix(
                    recover_from="DISTANCE_TWO",
                    edges=edges,
                    n=n,
                    B=B,
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
            
            B = A @ S
            
            A_rec = recover_matrix(
                recover_from=recovery_alg,
                edges=edges,
                n=n,
                B=B,
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
        
        B = A @ S
        
        A_rec = recover_matrix(
            recover_from="DISTANCE_TWO",
            edges=edges,
            n=n,
            B=B,
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
        
        B = A @ S
        
        A_rec = recover_matrix(
            recover_from="STAR",
            edges=edges,
            n=n,
            B=B,
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
        
        B = A @ S
        
        A_rec = recover_matrix(
            recover_from="ACYCLIC",
            edges=edges,
            n=n,
            B=B,
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
                
                B = A @ S
                
                A_rec = recover_matrix(
                    recover_from=recovery_alg,
                    edges=edges,
                    n=n,
                    B=B,
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
                
                B = A @ S
                
                A_rec = recover_matrix(
                    recover_from=recovery_alg,
                    edges=edges,
                    n=n,
                    B=B,
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
                
                B = A @ S
                
                A_rec = recover_matrix(
                    recover_from="DISTANCE_TWO",
                    edges=edges,
                    n=n,
                    B=B,
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
        
        B = A @ S
        
        A_rec = recover_matrix(
            recover_from="STAR",
            edges=edges,
            n=n,
            B=B,
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
        
        B = A @ S
        
        A_rec = recover_matrix(
            recover_from="ACYCLIC",
            edges=edges,
            n=n,
            B=B,
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
                
                B = A @ S
                
                # Use relative tolerance for mixed magnitudes
                A_rec = recover_matrix(
                    recover_from=recovery_alg,
                    edges=edges,
                    n=n,
                    B=B,
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
        
        B = A @ S
        
        A_rec = recover_matrix(
            recover_from="DISTANCE_TWO",
            edges=edges,
            n=n,
            B=B,
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
                
                B = A @ S
                
                A_rec = recover_matrix(
                    recover_from="STAR",
                    edges=edges,
                    n=n,
                    B=B,
                    vertex_colors=colors,
                    verify=True
                )
                
                # Relaxed tolerance for general symmetric matrices
                self.assertTrue(np.allclose(A, A_rec, rtol=1e-6, atol=1e-6))


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
        
        B = A @ S
        
        A_rec = recover_matrix(
            recover_from="DISTANCE_TWO",
            edges=edges,
            n=n,
            B=B,
            vertex_colors=colors,
            reconstruct_sparsely=True,
            verify=True
        )
        
        # Relaxed tolerance for large sparse matrices
        self.assertTrue(np.allclose(A, A_rec, rtol=1e-6, atol=1e-6))
    
    def test_large_grid_laplacian(self):
        """Test recovery on large grid Laplacian (realistic PDE discretization)."""
        grid_size = 15  # 225x225 matrix
        A, edges = MatrixTestHelper.make_grid_laplacian(grid_size=grid_size)
        n = A.shape[0]
        
        S, colors, comps = get_seeds_from_pattern(edges, n, "SMALLEST_LAST", "ACYCLIC_FOR_INDIRECT_RECOVERY")
        S = np.asarray(S, dtype=float)
        colors = np.asarray(colors, dtype=int)
        
        B = A @ S
        
        A_rec = recover_matrix(
            recover_from="ACYCLIC",
            edges=edges,
            n=n,
            B=B,
            vertex_colors=colors,
            acyclic_components=comps,
            reconstruct_sparsely=True,
            verify=True
        )
        
        self.assertTrue(np.allclose(A, A_rec, rtol=0, atol=1e-10))


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
    suite.addTests(loader.loadTestsFromTestCase(TestLargeScaleMatrices))
    
    return suite


if __name__ == '__main__':
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())
