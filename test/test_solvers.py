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
