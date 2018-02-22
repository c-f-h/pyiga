from pyiga.tensor import *

from numpy.random import rand
from scipy.sparse.linalg import aslinearoperator

def test_modek_tprod():
    X = rand(3,3,3)
    A = rand(3,3)
    Y = np.zeros((3,3,3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                Y[i,j,k] = np.dot(A[i,:], X[:,j,k])
    assert np.allclose(Y, modek_tprod(A, 0, X))
    # test modek_tprod with LinearOperator
    assert np.allclose(modek_tprod(A, 1, X),
            modek_tprod(aslinearoperator(A), 1, X))

def test_tuckerprod():
    U = [rand(10,n) for n in range(3,6)]
    C = rand(3,4,5)
    for i in range(3):
        U2 = U[:]
        U2[i] = np.eye(i+3)
        X1 = apply_tprod(U2, C)
        U2[i] = None
        X2 = apply_tprod(U2, C)
        assert np.allclose(X1, X2)

def _random_tucker(shape, R):
    Us = tuple(rand(n,R) for n in shape)
    d = len(Us)
    return TuckerTensor(Us, rand(*(d * (R,))))

def test_tucker():
    X = rand(3,4,5)
    T = hosvd(X)
    assert np.allclose(X, T.asarray())
    assert np.allclose(T.asarray(), T.orthogonalize().asarray())
    assert np.allclose(np.linalg.norm(X), T.norm())
    assert np.allclose(T.asarray(), T.copy().asarray())
    ###
    X = _random_tucker((3,4,5), 2)
    # orthogonalize
    assert np.allclose(X.asarray(), X.orthogonalize().asarray())
    # add and sub
    Y = _random_tucker((3,4,5), 3)
    assert np.allclose((X + Y).asarray(), X.asarray() + Y.asarray())
    assert np.allclose((X - Y).asarray(), X.asarray() - Y.asarray())
    # compression
    XX = (X + X).compress()
    assert XX.R == X.R and np.allclose(XX.asarray(), 2*X.asarray())
    # als1
    x = als1(X)
    y = als1(X.asarray())
    assert np.allclose(outer(*x), outer(*y))
    # conversion
    X = _random_canonical((3,4,5), 2)
    Y = TuckerTensor.from_tensor(X)
    assert np.allclose(X.asarray(), Y.asarray())

def test_join_tucker():
    A = _random_tucker((3,4,5), 2)
    B = _random_tucker((3,4,5), 3)
    Us, XA, XB = join_tucker_bases(A, B)
    assert np.allclose(A.asarray(), TuckerTensor(Us,XA).asarray())
    assert np.allclose(B.asarray(), TuckerTensor(Us,XB).asarray())

def test_truncate():
    """Check that a rank 1 tensor is exactly represented
    by the 1-truncation of the HOSVD."""
    # rank 1 tensor
    X = outer(rand(3), rand(4), rand(5))
    T = hosvd(X)
    assert find_truncation_rank(T.X, 1e-12) == (1,1,1)
    T1 = T.truncate(1)
    assert np.allclose(X, T1.asarray())

def test_truncate2():
    """Check that Tucker truncation error is the Frobenius norm
    of the residual core tensor."""
    X = rand(5,5,5)
    T = hosvd(X)
    k = 3
    Tk = T.truncate(k)
    E = X - Tk.asarray()
    Cdk = T.X
    Cdk[:k,:k,:k] = 0
    assert np.allclose(fro_norm(E), fro_norm(Cdk))

def test_outer():
    x, y, z, = rand(3), rand(4), rand(5)
    X = outer(x, y, z)
    Y = x[:, None, None] * y[None, :, None] * z[None, None, :]
    assert np.allclose(X, Y)

def _random_canonical(shape, R):
    Xs = tuple(rand(n,R) for n in shape)
    return CanonicalTensor(Xs)

def test_canonical():
    X,Y,Z = tuple(np.zeros((5,2)) for _ in range(3))
    for i in range(2):
        X[i,i] = Y[i,i] = Z[i,i] = 2.0
    A = CanonicalTensor((X,Y,Z))
    assert A.ndim == 3
    assert A.shape == (5,5,5)
    assert A.R == 2
    assert np.allclose(A.asarray(), A.copy().asarray())
    B = A.asarray()
    assert B.shape == A.shape
    C = np.zeros((2,2,2))
    np.fill_diagonal(C, 8.0)
    B[:2, :2, :2] -= C
    assert np.allclose(B, 0.0)
    Z = CanonicalTensor.zeros((3,4,5))
    assert fro_norm(Z.asarray()) == 0.0
    ###
    # norm
    A = _random_canonical((3,4,5), 2)
    assert np.allclose(A.norm(), np.linalg.norm(A.asarray()))
    # als1
    x = als1(A)
    y = als1(A.asarray())
    assert np.allclose(outer(*x), outer(*y))
    # add and sub
    B = _random_canonical(A.shape, 3)
    assert np.allclose((A + B).asarray(), A.asarray() + B.asarray())
    assert np.allclose((A - B).asarray(), A.asarray() - B.asarray())

def test_als1():
    xs = rand(3), rand(4), rand(5)
    X = outer(*xs)
    ys = als1(X)
    from numpy.linalg import norm
    assert all(np.allclose(x / norm(x), y / norm(y))
            for (x,y) in zip(xs, ys))

def test_als():
    ### canonical
    A = _random_canonical((3,4,5), 2)
    B = als(A, R=2)
    assert np.allclose(A.asarray(), B.asarray())
    ### full tensor
    C = als(A.asarray(), R=2)
    assert np.allclose(A.asarray(), C.asarray())
    ### Tucker
    A = _random_tucker((3,4,5), 2)
    # diagonalize core tensor
    A.X[:] = 0.0
    A.X[0,0,0] = A.X[1,1,1] = 1.0
    B = als(A, R=2)
    assert np.allclose(A.asarray(), B.asarray())
