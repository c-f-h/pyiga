from pyiga.hierarchical import *
from pyiga import bspline, geometry, utils, vform, geometry

from numpy.random import rand

def _make_hs(p=3, n=3):
    kv = bspline.make_knots(p, 0.0, 1.0, n)
    return HSpace((kv, kv))

def create_example_hspace(p, dim, n0, disparity=np.inf, num_levels=3):
    hs = HSpace(dim * (bspline.make_knots(p, 0.0, 1.0, n0),))
    hs.disparity = disparity
    hs.bdspecs = [(0,0), (0,1), (1,0), (1,1)] if dim==2 else [(0,0),(0,1)]
    # perform local refinement
    delta = 0.5
    for lv in range(num_levels):
        hs.refine_region(lv, lambda *X: min(X) > 1 - delta**(lv+1))
    return hs

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

def test_thb_to_hb():
    hs = create_example_hspace(p=3, dim=2, n0=4, disparity=np.inf, num_levels=3)

    T = hs.thb_to_hb()
    I_hb = hs.represent_fine()
    I_thb = hs.represent_fine(truncate=True)
    assert np.allclose((I_hb @ T).A, I_thb.A)

def test_hb_to_thb():
    hs = create_example_hspace(p=4, dim=2, n0=4, disparity=np.inf, num_levels=3)
    T = hs.thb_to_hb()
    T_inv = hs.hb_to_thb()
    assert np.allclose((T_inv @ T).A, np.eye(hs.numdofs))

def test_truncate():
    hs = create_example_hspace(p=4, dim=2, n0=4, disparity=np.inf, num_levels=3)
    for k in range(hs.numlevels - 1):
        # check that truncation and inverse truncation are inverse to each other
        Tk = hs.truncate_one_level(k)
        Tk_inv = hs.truncate_one_level(k, inverse=True)
        X = Tk_inv @ Tk
        assert np.allclose(X.A, np.eye(X.shape[0]))

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

def test_incidence():
    kv = bspline.make_knots(2, 0.0, 1.0, 4)
    hs = HSpace((kv,))
    hs.refine_region(0, lambda x: 1./4 < x < 3./4)
    hs.refine_region(1, lambda x: 3./8 < x < 5./8)

    Z = hs.incidence_matrix().A

    naf = tuple(len(A) for A in hs.active_indices())    # (6, 2, 2)
    nac = tuple(len(A) for A in hs.active_cells())      # (2, 2, 4)
    assert Z.shape == (sum(naf), sum(nac))

    # rows: functions, columns: cells
    assert np.array_equal(Z,
             ###################### level 0
            [[1,0,  0,0,  0,0,0,0],
             [1,0,  1,0,  1,1,0,0],
             [1,0,  1,1,  1,1,1,1],
             [0,1,  1,1,  1,1,1,1],
             [0,1,  0,1,  0,0,1,1],
             [0,1,  0,0,  0,0,0,0],
             ###################### level 1
             [0,0,  1,0,  1,1,1,1],
             [0,0,  0,1,  1,1,1,1],
             ###################### level 2
             [0,0,  0,0,  1,1,1,0],
             [0,0,  0,0,  0,1,1,1]])


def test_hierarchical_assemble():
    hs = create_example_hspace(p=4, dim=2, n0=4, disparity=1, num_levels=3)
    hdiscr = HDiscretization(hs, vform.stiffness_vf(dim=2), {'geo': geometry.unit_square()})
    A = hdiscr.assemble_matrix()
    # compute matrix on the finest level for comparison
    A_fine = assemble.stiffness(hs.knotvectors(-1))
    I_hb = hs.represent_fine()
    A_hb = (I_hb.T @ A_fine @ I_hb)
    error = abs(A - A_hb).max()
    assert error < 1e-14

def test_grid_eval():
    hs = create_example_hspace(p=3, dim=2, n0=6, num_levels=3)
    u = rand(hs.numdofs)
    grid = 2 * (np.linspace(0, 1, 50),)
    f_fine = bspline.BSplineFunc(hs.knotvectors(-1), hs.represent_fine() @ u)
    hsf = HSplineFunc(hs, u)
    assert hsf.dim == 1 and hsf.sdim == 2
    #
    assert np.allclose(f_fine.grid_eval(grid), hsf.grid_eval(grid))
    assert np.allclose(f_fine.grid_jacobian(grid), hsf.grid_jacobian(grid))
    assert np.allclose(f_fine.grid_hessian(grid), hsf.grid_hessian(grid))
    ## test THBs
    f_fine = bspline.BSplineFunc(hs.knotvectors(-1), hs.represent_fine(truncate=True) @ u)
    hsf = HSplineFunc(hs, u, truncate=True)
    assert np.allclose(f_fine.grid_eval(grid), hsf.grid_eval(grid))
    assert np.allclose(f_fine.grid_jacobian(grid), hsf.grid_jacobian(grid))
    assert np.allclose(f_fine.grid_hessian(grid), hsf.grid_hessian(grid))
