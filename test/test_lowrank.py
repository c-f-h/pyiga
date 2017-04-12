from pyiga.lowrank import *
from numpy.random import rand

def test_aca():
    n,k = 50, 3
    X = np.zeros((n,n))
    for i in range(k):
        X += np.outer(rand(n), rand(n))
    X_aca = aca(X, tol=0, maxiter=3, verbose=0)
    assert np.allclose(X, X_aca)
