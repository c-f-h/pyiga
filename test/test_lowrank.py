from pyiga.lowrank import *
from numpy.random import rand

def test_matrixgenerator():
    X = rand(4,7)
    mgen = MatrixGenerator.from_array(X)
    assert np.allclose(X, mgen.full())
    assert np.allclose(X[2,:], mgen.row(2))
    assert np.allclose(X[:,5], mgen.column(5))
    assert np.allclose(X[1,2], mgen.entry(1,2))

def test_tensorgenerator():
    X = rand(3,4,5)
    tgen = TensorGenerator.from_array(X)
    assert np.allclose(X, tgen.full())
    assert np.allclose(X[1,2,3], tgen.entry((1,2,3)))
    assert np.allclose(X[2,:,1], tgen.fiber_at((2,0,1), axis=1))
    assert np.allclose(X[:,3,:], tgen.matrix_at((0,3,0), axes=(0,2)).full())

def test_aca():
    n,k = 50, 3
    X = np.zeros((n,n))
    for i in range(k):
        X += np.outer(rand(n), rand(n))
    X_aca = aca(X, tol=0, maxiter=3, verbose=0)
    assert np.allclose(X, X_aca)
