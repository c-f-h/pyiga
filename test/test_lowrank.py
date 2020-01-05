from pyiga.lowrank import *
from pyiga import tensor
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
    X_aca = aca(X, tol=0, maxiter=k, verbose=0)
    assert np.allclose(X, X_aca)
    # compute approximation in low-rank form
    crosses = aca_lr(X, tol=0, maxiter=k, verbose=0)
    assert len(crosses) == 3
    T = tensor.CanonicalTensor.from_terms(crosses)
    assert np.allclose(X, T.asarray())
    # check that approximation terminates correctly
    crosses = aca_lr(X, tol=0, verbose=0)
    assert len(crosses) <= 4    # due to rounding error, may require one more
    T = tensor.CanonicalTensor.from_terms(crosses)
    assert np.allclose(X, T.asarray())

def test_aca3d():
    n,k = 10, 3
    X = np.zeros((n,n,n))
    for i in range(k):
        X += rand(n,1,1) * rand(1,n,1) * rand(1,1,n)
    X_aca = aca_3d(TensorGenerator.from_array(X), tol=0, maxiter=k, verbose=0)
    assert np.allclose(X, X_aca)
    # test automatic termination and low-rank tensor output
    X_aca_lr = aca_3d(TensorGenerator.from_array(X), tol=0, lr=True, verbose=0)
    assert np.allclose(X, X_aca_lr.asarray())
