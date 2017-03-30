from pyiga.tucker import *

from numpy.random import rand

def test_modek_tprod():
    X = rand(3,3,3)
    A = rand(3,3)
    Y = np.zeros((3,3,3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                Y[i,j,k] = np.dot(X[:,j,k], A[:,i])
    assert np.allclose(Y, modek_tprod(X, 0, A))

def test_tucker():
    X = rand(3,4,5)
    T = hosvd(X)
    assert np.allclose(X, tucker_prod(*T))

def test_truncate():
    """Check that a rank 1 tensor is exactly represented
    by the 1-truncation of the HOSVD."""
    # rank 1 tensor
    X = np.outer(rand(3),rand(4))[:,:,None] * rand(5)[None,None,:]
    T = hosvd(X)
    assert find_truncation_rank(T[0], 1e-12) == (1,1,1)
    T1 = truncate(T, 1)
    assert np.allclose(X, tucker_prod(*T1))

def test_truncate2():
    """Check that Tucker truncation error is the Frobenius norm
    of the residual core tensor."""
    X = rand(5,5,5)
    T = hosvd(X)
    k = 3
    Tk = truncate(T, k)
    E = X - tucker_prod(*Tk)
    Cdk = T[0]
    Cdk[:k,:k,:k] = 0
    assert np.allclose(np.sum(E*E), np.sum(Cdk*Cdk))

def test_truncation_rank():
    X = np.zeros((4,3,7))
    X[:2,:3,:4] = rand(2,3,4)
    assert find_truncation_rank(X, 1e-12) == (2,3,4)
