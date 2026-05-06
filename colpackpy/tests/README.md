# ColPackPy Unit Tests

Comprehensive unit tests for the `colpackpy` library, covering all recovery algorithms with diverse matrix types.

## Overview

This test suite provides **extensive coverage** of matrix recovery algorithms with:
- ✅ **All 3 algorithms** tested: DISTANCE_TWO, STAR, ACYCLIC
- ✅ **50+ test cases** covering various matrix types and edge cases
- ✅ **General matrices** beyond just positive definite (SPD)
- ✅ **Structured patterns**: grids, blocks, bands, graphs
- ✅ **Numerical challenges**: mixed magnitudes, near-singular cases
- ✅ **Scalability tests**: matrices up to 225×225

## Test Coverage

### Algorithms Tested
1. **DISTANCE_TWO** - Direct recovery using distance-2 coloring
2. **STAR** - Direct recovery using star coloring
3. **ACYCLIC** - Indirect recovery using acyclic coloring (with substitution)

### Matrix Types Tested
- **Positive definite (SPD)**: Traditional Hessian-like matrices
- **General symmetric**: Non-positive-definite, negative eigenvalues
- **Structured**: Tridiagonal, banded, block diagonal, grid Laplacians
- **Graph-based**: Star graphs, cycles
- **Numerical stress tests**: Mixed magnitudes, near-singular
- **Large-scale**: Up to 225×225 matrices

### Test Suites

#### Core Algorithm Tests

##### `TestDistanceTwoRecovery`
Tests for the DISTANCE_TWO algorithm:
- Random sparse SPD matrices
- Tridiagonal matrices
- Sparse reconstruction path
- Diagonal-only matrices (edge case)

##### `TestStarRecovery`
Tests for the STAR algorithm:
- Random sparse SPD matrices
- Tridiagonal matrices
- Sparse reconstruction path
- Multiple vertex orderings (NATURAL, LARGEST_FIRST, SMALLEST_LAST)

##### `TestAcyclicRecovery`
Tests for the ACYCLIC algorithm:
- Random sparse SPD matrices
- Tridiagonal matrices
- Fallback path (without cached components)
- Sparse reconstruction path
- Algorithm name aliases (ACYCLIC_INDIRECT)
- Diagonal-only matrices (edge case)

#### General Matrix Tests

##### `TestGeneralSymmetricMatrices`
Tests on general symmetric matrices (not necessarily positive definite):
- Matrices with negative eigenvalues
- Matrices with mixed positive/negative diagonal entries
- All three algorithms tested on non-SPD matrices

##### `TestStructuredMatrices`
Tests on matrices with specific sparsity structures:
- **Block diagonal matrices**: Disconnected dense blocks
- **Grid Laplacian**: 2D finite difference discretization (realistic PDE problems)
- **Banded matrices**: Various bandwidth sizes (2, 3, 5, 7)
- **Star graph**: Hub-and-spoke structure (verifies coloring efficiency)
- **Cycle graph**: Ring structure (tests chromatic number properties)

##### `TestNumericalChallenges`
Tests for numerically challenging cases:
- **Mixed magnitudes**: Entries ranging from 1e-6 to 1e6
- **Near-singular matrices**: Small diagonal values
- **Varying sparsity**: Different connectivity levels (avg degree 4-20)

##### `TestLargeScaleMatrices`
Scalability tests on larger problems:
- Sparse matrices (200×200)
- Large grid Laplacians (225×225 from 15×15 grid)
- Tests sparse reconstruction paths for efficiency

#### Cross-Cutting Tests

##### `TestRecoveryEdgeCases`
Edge cases and error handling:
- Invalid algorithm names
- Very small matrices (n=1)
- Data type preservation (float32 vs float64)

##### `TestRecoveryComparison`
Cross-algorithm consistency checks:
- All algorithms produce consistent results on identical matrices
- Validates that different coloring approaches recover the same matrix

## Running the Tests

### Using unittest (built-in)

```bash
# From the project root
export PYTHONPATH=$PYTHONPATH:$(pwd)/colpackpy/src
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/colpackpy/src

# Run all tests
python -m unittest discover colpackpy/tests

# Run specific test suite
python -m unittest colpackpy.tests.test_recovery.TestDistanceTwoRecovery

# Run specific test
python -m unittest colpackpy.tests.test_recovery.TestDistanceTwoRecovery.test_distance_two_random_sparse

# Run with verbose output
python -m unittest discover -v colpackpy/tests
```

### Using pytest (if installed)

```bash
# Install pytest (optional)
pip install pytest pytest-cov

# Run all tests
pytest colpackpy/tests/

# Run with coverage
pytest --cov=colpackpy --cov-report=html colpackpy/tests/

# Run specific test file
pytest colpackpy/tests/test_recovery.py -v

# Run specific test
pytest colpackpy/tests/test_recovery.py::TestStarRecovery::test_star_random_sparse -v
```

### Direct execution

```bash
# Run the test file directly
python colpackpy/tests/test_recovery.py
```

## Test Design Principles

### Modularity
- Each algorithm has its own test class
- Helper class (`MatrixTestHelper`) provides reusable test matrix generators
- Tests are independent and can run in any order

### Professional Standards
- Clear docstrings for all test methods
- Type hints in helper methods
- Comprehensive edge case coverage
- Parametric testing where appropriate (subtests)

### Verification
- All tests use `np.allclose()` with strict tolerances (atol=1e-10)
- Built-in verification flag tests internal consistency (A @ S == B)
- Cross-algorithm comparison ensures consistency

## Adding New Tests

To add new tests:

1. Choose the appropriate test class based on the algorithm
2. Follow the naming convention: `test_<algorithm>_<scenario>`
3. Use `MatrixTestHelper` methods or add new generators if needed
4. Always verify accuracy with `np.allclose()` and appropriate tolerances
5. Add descriptive docstrings

Example:
```python
def test_distance_two_custom_case(self):
    """Test DISTANCE_TWO on custom matrix structure."""
    # Setup
    A, edges = MatrixTestHelper.make_custom_matrix(params)
    S, colors, _ = get_seeds_from_pattern(edges, n, ordering, "DISTANCE_TWO")
    
    # Execute
    B = A @ S
    A_rec = recover_matrix(recover_from="DISTANCE_TWO", edges=edges, n=n, 
                          B=B, vertex_colors=colors, verify=True)
    
    # Verify
    self.assertTrue(np.allclose(A, A_rec, rtol=0, atol=1e-10))
```

## Dependencies

- `numpy` - Array operations
- `scipy` - Sparse matrix support
- `colpackpy.coloring` - Graph coloring bindings
- `colpackpy.recovery` - Matrix recovery functions

## Notes

- Tests use random seeds for reproducibility
- Default tolerance is 1e-10 for numerical accuracy
- Both sparse and dense reconstruction paths are tested
- Tests verify the mathematical property: A @ S == B where B is the compressed form
