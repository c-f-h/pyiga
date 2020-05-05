from pyiga.hierarchical import *
from pyiga import bspline, geometry, utils

def _make_hs(p=3, n=3):
    kv = bspline.make_knots(p, 0.0, 1.0, n)
    return HSpace((kv, kv))

def test_hspace():
    hs = _make_hs()
    assert hs.numlevels == 1
    assert tuple(len(a) for a in hs.actfun) == (36,)
    assert tuple(len(a) for a in hs.deactfun) == (0,)

    hs.refine({ 0: [(0,0),(0,1),(1,0),(1,1),(0,2)] })
    hs.refine({ 1: [(0,0),(0,1),(2,0),(1,0),(1,1)] })

    assert hs.numlevels == 3
    assert tuple(len(a) for a in hs.actfun) == (28, 21, 20)
    assert tuple(len(a) for a in hs.deactfun) == (8, 5, 0)
    assert hs.numactive == (28, 21, 20)
    assert hs.numdofs == 28 + 21 + 20

    # representation of THB-splines on the fine level
    R = hs.represent_fine(truncate=True)
    assert R.shape == (225, 28+21+20)
    # test partition of unity property
    one_func = geometry.BSplineFunc(hs.mesh(-1).kvs, R.dot(np.ones(R.shape[1])))
    vals = utils.grid_eval(one_func, 2 * (np.linspace(0.0, 1.0, 10),))
    assert np.allclose(vals, np.ones((10, 10)))

def test_cellextents():
    hs = _make_hs(p=2, n=2)
    hs.refine_region(0, lambda *X: True)    # refine globally
    assert hs.numlevels == 2
    assert np.array_equal(
            hs.cell_extents(0, (1,0)),
            ((0.5,1.0), (0.0, 0.5)))
    assert np.array_equal(
            hs.cell_extents(1, (2,1)),
            ((0.5,0.75), (0.25, 0.5)))
    assert np.array_equal(
            hs.function_support(0, (0,0)),
            ((0.0, 0.5), (0.0, 0.5)))
    assert np.array_equal(
            hs.function_support(1, (3,1)),
            ((0.25, 1.0), (0.0, 0.5)))
