from pyiga.mlmatrix import *
from numpy.random import rand

def _random_banded(n, bw):
    return scipy.sparse.spdiags(rand(2*bw+1, n), np.arange(-bw,bw+1), n, n)

def test_mlstructure():
    bs, bw = (5,5), (2,2)
    S = MLStructure.multi_banded(bs, bw)
    A = _random_banded(bs[0], bw[0])
    A2 = scipy.sparse.kron(A, A)
    assert np.array_equal(S.nonzero(), A2.nonzero())

def test_mlbanded_1d():
    bs = (20,)
    bw = (3,)
    S = MLStructure.multi_banded(bs, bw)
    A = _random_banded(bs[0], bw[0]).A
    X = MLMatrix(structure=S, matrix=A)
    A2 = X.asmatrix()
    assert np.allclose(A, A2.A)
    x = rand(A.shape[1])
    assert np.allclose(A.dot(x), X.dot(x))

def test_mlbanded_2d():
    bs = (9, 12)
    bw = (2, 3)
    S = MLStructure.multi_banded(bs, bw)

    A, B = (_random_banded(n, p).A for (n,p) in zip(bs, bw))
    # rowwise vectorizations of A and B
    vecA, vecB = (X.ravel()[np.flatnonzero(X.ravel())] for X in (A,B))
    # reordering of Kronecker product is outer product of vecs
    M = MLMatrix(structure=S, data=np.outer(vecA, vecB))
    assert M.shape == (9*12, 9*12)
    assert M.nnz == vecA.size * vecB.size
    # test asmatrix()
    X = np.kron(A, B)
    assert np.allclose(X, M.asmatrix().A)
    # test reorder()
    Y = np.kron(B, A)
    assert np.allclose(Y, M.reorder((1,0)).asmatrix().A)
    # test matvec
    x = rand(M.shape[1])
    assert np.allclose(X.dot(x), M.dot(x))
    # test matrix constructor
    M2 = MLMatrix(structure=S, matrix=X)
    assert np.allclose(X, M.asmatrix().A)
    M2 = MLMatrix(structure=S, matrix=scipy.sparse.csr_matrix(X))
    assert np.allclose(X, M.asmatrix().A)

def test_mlbanded_3d():
    bs = (8, 7, 6)
    bw = (3, 2, 2)
    S = MLStructure.multi_banded(bs, bw)
    S1 = MLStructure.multi_banded(bs[:2], bw[:2])
    S2 = MLStructure.multi_banded(bs[2:], bw[2:])
    S12 = S1.join(S2)
    assert S.bs == S12.bs
    assert S.slice(0,2).bs == S1.bs

    A, B, C = (_random_banded(n, p).A for (n,p) in zip(bs, bw))
    # rowwise vectorizations of A, B, C
    vecA, vecB, vecC = (X.ravel()[np.flatnonzero(X.ravel())] for X in (A,B,C))
    # reordering of Kronecker product is outer product of vecs
    M = MLMatrix(structure=S,
            data=vecA[:,None,None]*vecB[None,:,None]*vecC[None,None,:])
    assert M.shape == (8*7*6, 8*7*6)
    assert M.nnz == vecA.size * vecB.size * vecC.size
    # test asmatrix()
    X = np.kron(np.kron(A, B), C)
    assert np.allclose(X, M.asmatrix().A)
    # test reorder()
    Y = np.kron(np.kron(C, A), B)
    assert np.allclose(Y, M.reorder((2,0,1)).asmatrix().A)
    # test matvec
    x = rand(M.shape[1])
    assert np.allclose(X.dot(x), M.dot(x))
    # test matrix constructor
    M2 = MLMatrix(structure=S, matrix=X)
    assert np.allclose(X, M.asmatrix().A)
    M2 = MLMatrix(structure=S, matrix=scipy.sparse.csr_matrix(X))
    assert np.allclose(X, M.asmatrix().A)

def test_tofrom_seq():
    for i in range(3*4*5):
        assert to_seq(from_seq(i, (3,4,5)), (3,4,5)) == i

def test_tofrom_multilevel():
    bs = np.array(((3,3), (4,4), (5,5)))  # block sizes for each level
    for i in range(3*3 + 4*4 + 5*5):
        for j in range(3*3 + 4*4 + 5*5):
            assert reindex_from_multilevel(reindex_to_multilevel(i, j, bs), bs) == (i,j)

def test_banded_sparsity():
    n = 10
    bw = 2

    X = np.zeros((n,n), dtype=int)
    for i in range(n):
        for j in range(n):
            if abs(i-j) <= bw:
                X[i,j] = 1
    assert np.array_equal(np.flatnonzero(X),
                          compute_banded_sparsity(n, bw))
    assert np.array_equal(np.transpose(np.nonzero(X)),
                          compute_banded_sparsity_ij(n, bw))

def test_reorder():
    n1, n2 = 6, 7
    A1 = _random_banded(n1, 3).A
    A2 = _random_banded(n2, 4).A
    A = np.kron(A1, A2)
    AR = reorder(A, n1, n1)     # shape: (n1*n1) x (n2*n2)
    print(AR.shape)
    for i in range(n1*n1):
        for j in range(n2*n2):
            ii, jj = reindex_from_reordered(i, j, n1, n1, n2, n2)
            assert AR[i, j] == A[ii, jj]
