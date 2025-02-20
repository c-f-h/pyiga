from pyiga.hierarchical import *
from pyiga import bspline, geometry, utils, vform

from numpy.random import rand

def _make_hs(p=3, n=3):
    kv = bspline.make_knots(p, 0.0, 1.0, n)
    return HSpace((kv, kv))

def create_example_hspace(p, dim, n0, disparity=np.inf, truncate=False, num_levels=3):
    bdspecs = [(0,0), (0,1), (1,0), (1,1)] if dim==2 else [(0,0),(0,1)]
    hs = HSpace(dim * (bspline.make_knots(p, 0.0, 1.0, n0),),
            truncate=truncate, disparity=disparity, bdspecs=bdspecs)
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

    assert hs.active_functions(flat=True) == [
            (0, (1, 2)), (0, (1, 3)), (0, (1, 4)), (0, (1, 5)), (0, (2, 0)),
            (0, (2, 1)), (0, (2, 2)), (0, (2, 3)), (0, (2, 4)), (0, (2, 5)),
            (0, (3, 0)), (0, (3, 1)), (0, (3, 2)), (0, (3, 3)), (0, (3, 4)),
            (0, (3, 5)), (0, (4, 0)), (0, (4, 1)), (0, (4, 2)), (0, (4, 3)),
            (0, (4, 4)), (0, (4, 5)), (0, (5, 0)), (0, (5, 1)), (0, (5, 2)),
            (0, (5, 3)), (0, (5, 4)), (0, (5, 5)), (1, (0, 2)), (1, (0, 3)),
            (1, (0, 4)), (1, (0, 5)), (1, (0, 6)), (1, (0, 7)), (1, (0, 8)),
            (1, (1, 2)), (1, (1, 3)), (1, (1, 4)), (1, (1, 5)), (1, (1, 6)),
            (1, (1, 7)), (1, (1, 8)), (1, (2, 1)), (1, (2, 2)), (1, (2, 3)),
            (1, (3, 0)), (1, (3, 1)), (1, (3, 2)), (1, (3, 3)), (2, (0, 0)),
            (2, (0, 1)), (2, (0, 2)), (2, (0, 3)), (2, (1, 0)), (2, (1, 1)),
            (2, (1, 2)), (2, (1, 3)), (2, (2, 0)), (2, (2, 1)), (2, (2, 2)),
            (2, (2, 3)), (2, (3, 0)), (2, (3, 1)), (2, (3, 2)), (2, (3, 3)),
            (2, (4, 0)), (2, (4, 1)), (2, (5, 0)), (2, (5, 1))]

    assert hs.active_cells(flat=True) == [
            (0, (1, 2)), (0, (2, 0)), (0, (2, 1)), (0, (2, 2)), (1, (0, 2)),
            (1, (0, 3)), (1, (0, 4)), (1, (0, 5)), (1, (1, 2)), (1, (1, 3)),
            (1, (1, 4)), (1, (1, 5)), (1, (2, 1)), (1, (2, 2)), (1, (2, 3)),
            (1, (3, 0)), (1, (3, 1)), (1, (3, 2)), (1, (3, 3)), (2, (0, 0)),
            (2, (0, 1)), (2, (0, 2)), (2, (0, 3)), (2, (1, 0)), (2, (1, 1)),
            (2, (1, 2)), (2, (1, 3)), (2, (2, 0)), (2, (2, 1)), (2, (2, 2)),
            (2, (2, 3)), (2, (3, 0)), (2, (3, 1)), (2, (3, 2)), (2, (3, 3)),
            (2, (4, 0)), (2, (4, 1)), (2, (5, 0)), (2, (5, 1))]
    assert hs.total_active_cells == 39

    # representation of THB-splines on the fine level
    R = hs.represent_fine(truncate=True)
    assert R.shape == (225, 28+21+20)
    # test partition of unity property
    one_func = geometry.BSplineFunc(hs.mesh(-1).kvs, R.dot(np.ones(R.shape[1])))
    vals = utils.grid_eval(one_func, 2 * (np.linspace(0.0, 1.0, 10),))
    assert np.allclose(vals, np.ones((10, 10)))

def test_cells():
    hs = create_example_hspace(p=3, dim=2, n0=4, disparity=np.inf, num_levels=3)

    # utility functions for axis-aligned boxes
    def contains(A, B):
        return all(a[0] <= b[0] <= b[1] <= a[1] for (a,b) in zip(A,B))
    def area(A):
        return np.prod([b-a for (a,b) in A])

    L = hs.numlevels
    for f_lv in range(L):
        # last active function on level f_lv
        f = sorted(hs.active_functions(lv=f_lv))[-1]

        funcs = [[] for _ in range(L)]
        funcs[f_lv] = [f]

        f_supp = hs.function_support(f_lv, f)   # support as a box
        act_cells = hs.compute_supports(funcs)  # support as a list of active cells

        ar = 0.0
        for lv, cells in act_cells.items():
            for c in cells:
                ext = hs.cell_extents(lv, c)
                # support of f must contain all active cells in its support
                assert contains(f_supp, ext)
                # the area of the active cells must sum up to the area of f_supp
                ar += area(ext)
        assert abs(area(f_supp) - ar) < 1e-10

def test_hmesh_cells():
    hs = create_example_hspace(p=3, dim=2, n0=4, disparity=np.inf, num_levels=2)

    # coarse deactivated cell to fine active cells
    assert hs.hmesh.hmesh_cells({0: {(2,2)}}) == {1: {(4,4), (4,5), (5,4), (5,5)}}
    assert hs.hmesh.hmesh_cells({0: {(3,3)}}) == {2: set(hs.hmesh.cell_grandchildren(0, [(3,3)], 2))}

    # fine inactive cell to coarse active cell
    assert hs.hmesh.hmesh_cells({2: {(6,5)}}) == {0: {(1,1)}}
    assert hs.hmesh.cell_grandparent(2, [(6,5)], 0) == {(1,1)}

def test_thb_to_hb():
    hs = create_example_hspace(p=3, dim=2, n0=4, disparity=np.inf, num_levels=3)

    T = hs.thb_to_hb()
    I_hb = hs.represent_fine()
    I_thb = hs.represent_fine(truncate=True)
    assert np.allclose((I_hb @ T).toarray(), I_thb.toarray())

def test_hb_to_thb():
    hs = create_example_hspace(p=4, dim=2, n0=4, disparity=np.inf, num_levels=3)
    T = hs.thb_to_hb()
    T_inv = hs.hb_to_thb()
    assert np.allclose((T_inv @ T).toarray(), np.eye(hs.numdofs))

def test_truncate():
    hs = create_example_hspace(p=4, dim=2, n0=4, disparity=np.inf, num_levels=3)
    for k in range(hs.numlevels - 1):
        # check that truncation and inverse truncation are inverse to each other
        Tk = hs.truncate_one_level(k)
        Tk_inv = hs.truncate_one_level(k, inverse=True)
        X = Tk_inv @ Tk
        assert np.allclose(X.toarray(), np.eye(X.shape[0]))

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

    Z = hs.incidence_matrix().toarray()

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
    geo = geometry.bspline_quarter_annulus()
    hdiscr = HDiscretization(hs, vform.stiffness_vf(dim=2), {'geo': geo})
    A = hdiscr.assemble_matrix(symmetric=True)
    # compute matrix on the finest level for comparison
    A_fine = assemble.stiffness(hs.knotvectors(-1), geo=geo)
    I_hb = hs.represent_fine()
    A_hb = (I_hb.T @ A_fine @ I_hb)
    assert np.allclose(A.toarray(), A_hb.toarray())
    #
    A3 = assemble.assemble(vform.stiffness_vf(dim=2), hs, geo=geo)
    assert np.allclose(A.toarray(), A3.toarray())
    #
    def f(x, y):
        return np.cos(x) * np.exp(y)
    f_hb = assemble.inner_products(hs.knotvectors(-1), f, f_physical=True, geo=geo).ravel() @ I_hb
    f2 = assemble.assemble('f * v * dx', hs, f=f, geo=geo)
    assert np.allclose(f_hb, f2)

def _convdiff_vf(dim, conv_vector):
    from pyiga.vform import VForm, inner, grad, dx
    vf = VForm(dim=dim)
    u, v = vf.basisfuns()
    vf.add((inner(grad(u), grad(v)) + inner(conv_vector, grad(u)) * v) * dx)
    return vf

def test_hierarchical_assemble_nonsym():
    hs = create_example_hspace(p=6, dim=2, n0=4, disparity=1, num_levels=2)
    geo = geometry.bspline_quarter_annulus()
    A = assemble.assemble(_convdiff_vf(2, (1.0, 1.0)), hs, geo=geo)
    # compute matrix on the finest level for comparison
    A_fine = assemble.assemble(_convdiff_vf(2, (1.0, 1.0)), hs.knotvectors(-1), geo=geo)
    I_hb = hs.represent_fine()
    A_hb = (I_hb.T @ A_fine @ I_hb)
    assert np.allclose(A.toarray(), A_hb.toarray())

def test_grid_eval():
    hs = create_example_hspace(p=3, dim=2, n0=6, num_levels=3)
    u = rand(hs.numdofs)
    grid = 2 * (np.linspace(0, 1, 50),)
    f_fine = bspline.BSplineFunc(hs.knotvectors(-1), hs.represent_fine() @ u)
    hsf = HSplineFunc(hs, u)
    assert hsf.dim == 1 and hsf.sdim == 2
    assert hsf.support == ((0.0, 1.0), (0.0, 1.0))
    #
    assert np.allclose(f_fine.grid_eval(grid), hsf.grid_eval(grid))
    assert np.allclose(f_fine.grid_jacobian(grid), hsf.grid_jacobian(grid))
    assert np.allclose(f_fine.grid_hessian(grid), hsf.grid_hessian(grid))
    #
    assert np.allclose(hsf(grid[1][7], grid[0][19]), hsf.grid_eval(grid)[19, 7])
    ## test THBs
    f_fine = bspline.BSplineFunc(hs.knotvectors(-1), hs.represent_fine(truncate=True) @ u)
    hsf = HSplineFunc(hs, u, truncate=True)
    assert np.allclose(f_fine.grid_eval(grid), hsf.grid_eval(grid))
    assert np.allclose(f_fine.grid_jacobian(grid), hsf.grid_jacobian(grid))
    assert np.allclose(f_fine.grid_hessian(grid), hsf.grid_hessian(grid))
    #
    assert np.allclose(hsf(grid[1][7], grid[0][19]), hsf.grid_eval(grid)[19, 7])

def test_prolongators():
    hs = create_example_hspace(p=3, dim=2, n0=4, disparity=1, num_levels=1)
    n0 = hs.mesh(0).numbf

    # create a coarse B-spline function
    u_tp = rand(n0)
    f0 = bspline.BSplineFunc(hs.knotvectors(0), u_tp)
    # bring its coefficients into canonical order (active, then deactivated)
    u_lv0 = np.concatenate((
        u_tp[hs.active_indices()[0]], u_tp[hs.deactivated_indices()[0]],))
    X = 2 * (np.linspace(0, 1, 20),)

    #### prolongators for HB-splines ####
    # prolongate f to the finest space (hs itself)
    P_hb = hs.virtual_hierarchy_prolongators()
    u = u_lv0
    for P in P_hb:
        u = P @ u
    f_hb = HSplineFunc(hs, u)
    # compare it to the original function
    assert np.allclose(f0.grid_eval(X), f_hb.grid_eval(X))

    #### prolongators for THB-splines ####
    hs.truncate = True
    # prolongate f to the finest space (hs itself)
    P_thb = hs.virtual_hierarchy_prolongators()
    u = u_lv0
    for P in P_thb:
        u = P @ u
    f_thb = HSplineFunc(hs, u)
    # compare it to the original function
    assert np.allclose(f0.grid_eval(X), f_thb.grid_eval(X))

def test_project_L2():
    def f(x, y): return x**2 - 4*x*y + y**3
    X = 2 * (np.linspace(0, 1, 20),)
    from pyiga import approx

    # HB-splines
    hs = create_example_hspace(p=3, dim=2, n0=4, disparity=np.inf, num_levels=3)
    u = approx.project_L2(hs, f, f_physical=True)   # geo=None
    u_func = HSplineFunc(hs, u)
    assert np.allclose(utils.grid_eval(f, X), u_func.grid_eval(X))

    # THB-splines
    hs = create_example_hspace(p=3, dim=2, n0=4, disparity=np.inf, truncate=True, num_levels=3)
    u = approx.project_L2(hs, f, f_physical=True)   # geo=None
    u_func = HSplineFunc(hs, u)
    assert np.allclose(utils.grid_eval(f, X), u_func.grid_eval(X))

def test_boundary_HSpace():
    hs = create_example_hspace(p=3, dim=3, n0=6, num_levels=3)
    u_vec_3D = rand(hs.numdofs)
    u_HS_3D = HSplineFunc(hs, u_vec_3D)
    grid_3D = 3 * (np.linspace(0, 1, 20),)
    grid_2D = 2 * (np.linspace(0, 1, 20),)

    def restrict_grid_to_boundary(bdspec):
        bdgrid = list(grid_3D)
        bdgrid[bdspec[0]] = np.array([0.]) if bdspec[1] == 0 else np.array([1.])
        return bdgrid

    bdspecs = ['left', 'right', 'top', 'bottom', 'front', 'back']
    for bdspec in bdspecs:
        bd_HSpace, bd_mapping = hs.boundary(bdspec)
        u_vec_2D = u_vec_3D[bd_mapping]
        u_HS_2D = HSplineFunc(bd_HSpace, u_vec_2D)
        bdgrid = restrict_grid_to_boundary(bspline._parse_bdspec(bdspec, hs.dim))
        assert np.allclose(np.squeeze(u_HS_3D.grid_eval(bdgrid)), u_HS_2D.grid_eval(grid_2D))

def test_comparison():
    hs0 = create_example_hspace(p=3, dim=3, n0=6, num_levels=0)
    hs1 = create_example_hspace(p=3, dim=3, n0=6, num_levels=1)
    hs2 = create_example_hspace(p=3, dim=3, n0=6, num_levels=2)
    hs3 = create_example_hspace(p=3, dim=3, n0=6, num_levels=3)

    assert hs0 == hs0.copy()
    assert hs1 == hs1.copy()
    assert hs2 == hs2.copy()
    assert hs3 == hs3.copy()

    assert hs0.is_subspace_of(hs1)
    assert hs1.is_subspace_of(hs2)
    assert hs2.is_subspace_of(hs3)
    assert hs0.is_subspace_of(hs2)
    assert hs1.is_subspace_of(hs3)
    assert hs0.is_subspace_of(hs3)

    assert not hs1.is_subspace_of(hs0)
    assert not hs2.is_subspace_of(hs1)
    assert not hs3.is_subspace_of(hs2)
    assert not hs2.is_subspace_of(hs0)
    assert not hs3.is_subspace_of(hs1)
    assert not hs3.is_subspace_of(hs0)

    hs0_ = hs3.get_virtual_space(0)
    hs1_ = hs3.get_virtual_space(1)
    hs2_ = hs3.get_virtual_space(2)
    hs3_ = hs3.get_virtual_space(3)

    assert hs0 == hs0_
    assert hs1 == hs1_
    assert hs2 == hs2_
    assert hs3 == hs3_

def test_prolongate_to_HSpace():
    hs_fine = create_example_hspace(p=3, dim=2, n0=8, num_levels=5)
    hs_coarse = hs_fine.copy()
    for i in reversed(range(hs_fine.numlevels)):
        hs_fine.refine_region(i, lambda *X: X[0] <= X[1])
    u_coarse_vec = rand(hs_coarse.numdofs)
    u_coarse = HSplineFunc(hs_coarse, u_coarse_vec)
    P = hs_coarse.prolongate_to(hs_fine, check_nestedness=True, check_nestedness_kv=True)
    u_fine_vec = P @ u_coarse_vec
    u_fine = HSplineFunc(hs_fine, u_fine_vec)
    grid = 2 * (np.linspace(0, 1, 20),)
    assert np.allclose(u_fine.grid_eval(grid), u_coarse.grid_eval(grid))
