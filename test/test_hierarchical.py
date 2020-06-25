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


def run_hierarchical_assemble(p, dim, n0, disparity):
    from .test_localmg import create_example_hspace
    hs = create_example_hspace(p, dim, n0, disparity, num_levels=3)

    # assemble full tensor-product linear system on each level for simplicity
    kvs = tuple(hs.knotvectors(lv) for lv in range(hs.numlevels))
    As = [assemble.stiffness(kv) for kv in kvs]

    # number of active and deactivated dofs per level
    na = tuple(len(ii) for ii in hs.active_indices())

    # I_hb[k]: maps HB-coefficients to TP coefficients on level k
    I_hb = [hs.represent_fine(lv=k) for k in range(hs.numlevels)]

    # compute dofs interacting with active dofs on each level
    neighbors = hs.cell_supp_indices(remove_dirichlet=False)
    # interactions on the same level are handled separately, so remove them
    for k in range(hs.numlevels):
        neighbors[k][k] = []

    # dofs to be assembled for interlevel contributions - all level k dofs which
    # are required to represent the coarse functions which interact with level k
    to_assemble = []
    for k in range(hs.numlevels):
        to_assemble.append(set())
        for lv in range(max(0, k - hs.disparity), k):
            to_assemble[-1] |= set(hs.hmesh.function_babys(lv, neighbors[k][lv], k))
    # convert them to raveled form
    to_assemble = hs._ravel_indices(to_assemble)

    # compute neighbors as matrix indices
    neighbors = [hs._ravel_indices(idx) for idx in neighbors]
    neighbors = hs.raveled_to_virtual_matrix_indices(neighbors)

    # new indices per level as local tensor product indices
    new_loc = hs.active_indices()
    # new indices per level as global matrix indices
    new = [np.arange(sum(na[:k]), sum(na[:k+1])) for k in range(hs.numlevels)]

    # the diagonal block consisting of interactions on the same level
    A_hb_new = [As[k][new_loc[k]][:,new_loc[k]]
            for k in range(hs.numlevels)]
    # the off-diagonal blocks which describe interactions with coarser levels
    A_hb_interlevel = [(I_hb[k][to_assemble[k]][:, neighbors[k]].T
                        @ As[k][to_assemble[k]][:, new_loc[k]]
                        @ I_hb[k][new_loc[k]][:, new[k]])
                       for k in range(hs.numlevels)]
    #A_hb_interlevel = [(I_hb[k][:, neighbors[k]].T @ As[k] @ I_hb[k][:, new[k]])
    #        for k in range(hs.numlevels)]

    # assemble the matrix from the levelwise contributions
    A = scipy.sparse.lil_matrix((hs.numdofs, hs.numdofs))

    for k in range(hs.numlevels):
        # store the k-th diagonal block
        A[np.ix_(new[k], new[k])] = A_hb_new[k]
        A[np.ix_(neighbors[k], new[k])] = A_hb_interlevel[k]
        A[np.ix_(new[k], neighbors[k])] = A_hb_interlevel[k].T

    # compute matrix on the finest level for comparison
    A_hb = (I_hb[-1].T @ As[-1] @ I_hb[-1])
    return scipy.linalg.norm((A - A_hb).A)

def test_hierarchical_assemble():
    error = run_hierarchical_assemble(p=4, dim=2, n0=4, disparity=1)
    assert error < 1e-12
