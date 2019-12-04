from pyiga.solvers import *
from pyiga import bspline, assemble

def test_fastdiag_solver():
    kvs = [
        bspline.make_knots(4, 0.0, 1.0, 3),
        bspline.make_knots(3, 0.0, 1.0, 4),
        bspline.make_knots(2, 0.0, 1.0, 5)
    ]
    # compute Dirichlet matrices
    KM = [(assemble.stiffness(kv)[1:-1, 1:-1].A, assemble.mass(kv)[1:-1, 1:-1].A) for kv in kvs]
    solver = fastdiag_solver(KM)

    def multikron(*Xs):
        return reduce(np.kron, Xs)
    A = ( multikron(KM[0][0], KM[1][1], KM[2][1]) +
          multikron(KM[0][1], KM[1][0], KM[2][1]) +
          multikron(KM[0][1], KM[1][1], KM[2][0]) )
    f = np.random.rand(A.shape[0])
    assert np.allclose(f, solver.dot(A.dot(f)))

def test_gauss_seidel():
    from numpy.random import rand
    A = abs(rand(10,10)) + np.eye(10) # try to make it not too badly conditioned
    b = rand(10)

    for sweep in ('forward', 'backward', 'symmetric'):
        x1 = rand(10)
        x2 = x1.copy()

        gauss_seidel(scipy.sparse.csr_matrix(A), x1, b, iterations=2, sweep=sweep)
        gauss_seidel(A, x2, b, iterations=2, sweep=sweep)
        assert abs(x1-x2).max() < 1e-12

def test_gauss_seidel_indexed():
    from numpy.random import rand
    A = abs(rand(10,10)) + np.eye(10) # try to make it not too badly conditioned
    b = rand(10)
    indices = [3, 6, 9]

    for sweep in ('forward', 'backward', 'symmetric'):
        x1 = rand(10)
        x2 = x1.copy()

        gauss_seidel(scipy.sparse.csr_matrix(A), x1, b, iterations=2, indices=indices, sweep=sweep)
        gauss_seidel(A, x2, b, iterations=2, indices=indices, sweep=sweep)
        assert abs(x1-x2).max() < 1e-12
