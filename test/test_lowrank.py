from pyiga.lowrank import *
from numpy.random import rand

def test_matrixgenerator():
    X = np.zeros((4,7))
    mgen = MatrixGenerator.from_array(X)
    assert np.allclose(X, mgen.full())
    assert np.allclose(X[2,:], mgen.row(2))
    assert np.allclose(X[:,5], mgen.column(5))
    assert np.allclose(X[1,2], mgen.entry(1,2))

def test_aca():
    n,k = 50, 3
    X = np.zeros((n,n))
    for i in range(k):
        X += np.outer(rand(n), rand(n))
    X_aca = aca(X, tol=0, maxiter=3, verbose=0)
    assert np.allclose(X, X_aca)
