from pyiga.lowrank import *
from pyiga import tensor
from numpy.random import rand

def test_tensorgenerator():
    X = rand(3,4,5)
    tgen = TensorGenerator.from_array(X)
    assert np.allclose(X, tgen.asarray())
    assert np.allclose(X[1,2,3], tgen.entry((1,2,3)))
    assert np.allclose(X[:,3,:], tgen.matrix_at((0,3,0), axes=(0,2)).asarray())
    ## slicing notation
    assert np.array_equal(tgen[1,2,3], X[1,2,3])
    assert np.array_equal(tgen[2,:,1], X[2,:,1])
    assert np.array_equal(tgen[:,3,:], X[:,3,:])
    assert np.array_equal(tgen[::-1], X[::-1])
    assert np.array_equal(tgen[:, 3:0:-2, 2], X[:, 3:0:-2, 2])
    assert np.array_equal(tgen[1:,2:,4:], X[1:,2:,4:])
    assert np.array_equal(tgen[-1,-2,-3:], X[-1,-2,-3:])
    i = [1,3]
    assert np.array_equal(tgen[1,i,2], X[1,i,2])

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
    assert len(crosses) <= 5    # due to rounding error, may require some more
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
