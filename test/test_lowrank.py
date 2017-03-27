from pyiga.lowrank import *
from numpy.random import rand

def test_aca():
    n,k = 50, 3
    X = np.zeros((n,n))
    for i in range(k):
        X += np.outer(rand(n), rand(n))
    X_gen = MatrixGenerator(X.shape[0], X.shape[1], lambda i,j: X[i,j])
    X_aca = aca(X_gen, tol=0, maxiter=3, verbose=0)
    assert np.allclose(X, X_aca)
