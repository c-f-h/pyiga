from pyiga.utils import *
import numpy as np
from numpy.random import rand

def _random_banded(n, bw):
    return scipy.sparse.spdiags(rand(2*bw+1, n), np.arange(-bw,bw+1), n, n).tocsr()

def test_lazy():
    def f(x, y, z):
        return np.sin(x) * np.cos(y + np.exp(z))
    grid = 3 * (np.linspace(0, 1, 8),)
    F = grid_eval(f, grid)
    LF1 = LazyArray(f, grid)
    LF2 = LazyCachingArray(f, (), grid, 2)
    assert np.allclose(F[2:4, 2:6, 6:8], LF1[2:4, 2:6, 6:8])
    assert np.allclose(F[2:4, 2:6, 6:8], LF2[2:4, 2:6, 6:8])
    # try again to test the caching behavior
    assert np.allclose(F[2:4, 2:6, 6:8], LF2[2:4, 2:6, 6:8])

    ## test a vector-valued function
    def f(x, y, z):
        return np.stack([x*y*np.ones_like(z), x*np.ones_like(y)*z], axis=-1)
    F = grid_eval(f, grid)
    LF1 = LazyArray(f, grid)
    LF2 = LazyCachingArray(f, (2,), grid, 2)
    assert np.allclose(F[2:4, 2:6, 6:8], LF1[2:4, 2:6, 6:8])
    assert np.allclose(F[2:4, 2:6, 6:8], LF2[2:4, 2:6, 6:8])
    # try again to test the caching behavior
    assert np.allclose(F[2:4, 2:6, 6:8], LF2[2:4, 2:6, 6:8])

def test_BijectiveIndex():
    I = BijectiveIndex([ (1,2), (3,4), (2,7) ])
    assert len(I) == 3
    assert I[1] == (3,4)
    assert I.index((2,7)) == 2

def test_kron_partial():
    As = (_random_banded(5, 1), _random_banded(4, 2), _random_banded(6, 3))
    X = multi_kron_sparse(As)
    X_partial = kron_partial(As, rows=list(range(17, 25)))
    assert np.allclose(X[17:25].toarray(), X_partial[17:25].toarray())
    assert X_partial[:17].nnz == 0
    assert X_partial[25:(5*4*6)].nnz == 0
    #
    X_partial = kron_partial(As, rows=list(range(17, 25)), restrict=True)
    assert np.allclose(X[17:25].toarray(), X_partial.toarray())
    #
    X_partial = kron_partial(As, rows=[])
    assert X_partial.shape == X.shape
    assert X_partial.nnz == 0
    #
    X_partial = kron_partial(As, rows=[], restrict=True)
    assert X_partial.shape == (0, X.shape[1])
    assert X_partial.nnz == 0

def test_CSRRowSlice():
    A = scipy.sparse.rand(100, 100, density=0.05, format='csr')
    x = rand(100)
    assert np.allclose((A @ x)[12:23], CSRRowSlice(A, (12, 23)).dot(x))
    x = rand(100, 7)
    assert np.allclose((A @ x)[12:23], CSRRowSlice(A, (12, 23)).dot(x))
    #
    rows = np.array([1, 3, 10, 11, 12, 65])
    x = rand(100)
    assert np.allclose((A @ x)[rows], CSRRowSubset(A, rows).dot(x))
