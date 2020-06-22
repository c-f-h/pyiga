from pyiga.utils import *
import numpy as np

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
