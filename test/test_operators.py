from pyiga.operators import *

from numpy.random import rand
import scipy.linalg

def _test_oper(A, B):
    assert A.shape == B.shape
    n = A.shape[1]
    x = rand(n)
    assert np.allclose(A.dot(x), B.dot(x))
    x = rand(n,1)
    assert np.allclose(A.dot(x), B.dot(x))
    x = rand(n,3)
    assert np.allclose(A.dot(x), B.dot(x))

def test_null():
    Z = np.zeros((7,3))
    _test_oper(NullOperator(Z.shape), Z)
    _test_oper(NullOperator(Z.shape).T, Z.T)

def test_identity():
    I = np.eye(7)
    _test_oper(IdentityOperator(7), I)
    _test_oper(IdentityOperator(7).T, I)

def test_diagonal():
    diag = rand(10)
    diag_op = DiagonalOperator(diag)
    D = np.diag(diag)
    _test_oper(diag_op, D)
    _test_oper(diag_op.T, D)

def test_blockdiag():
    A = rand(2,3)
    B = rand(4,4)
    C = rand(3,1)
    X = scipy.linalg.block_diag(A, B, C)
    Y = BlockDiagonalOperator(A, B, C)
    _test_oper(X, Y)
    _test_oper(X.T, Y.T)

def test_block():
    A,B = rand(3,3), rand(3,4)
    C,D = rand(2,3), rand(2,4)
    blocks = [[A,B],[C,D]]
    X = BlockOperator(blocks)
    Y = np.block(blocks)
    _test_oper(X, Y)
    _test_oper(X.T, Y.T)

def test_subspace():
    I = np.eye(4)
    P1, P2 = I[:, :2], I[:, 2:]
    B1, B2 = rand(2,2),rand(2,2)
    X = SubspaceOperator((P1,P2), (B1,B2))
    _test_oper(X,     scipy.linalg.block_diag(B1,   B2))
    _test_oper(X.T,   scipy.linalg.block_diag(B1.T, B2.T))
    _test_oper(X.T.T, scipy.linalg.block_diag(B1,   B2))

def test_solver():
    A = rand(3,3)
    _test_oper(make_solver(A), np.linalg.inv(A))
    B = A + A.T + 3*np.eye(3)
    _test_oper(make_solver(B, symmetric=True), np.linalg.inv(B))
    _test_oper(make_solver(B, spd=True), np.linalg.inv(B))
    A = scipy.sparse.csr_matrix(A)
    _test_oper(make_solver(A), np.linalg.inv(A.toarray()))
    B = scipy.sparse.csr_matrix(B)
    _test_oper(make_solver(B, symmetric=True), np.linalg.inv(B.toarray()))
    _test_oper(make_solver(B, spd=True), np.linalg.inv(B.toarray()))

def test_kronecker():
    A = rand(2,3)
    B = rand(4,5)
    X = KroneckerOperator(A, B)
    Y = np.kron(A, B)
    _test_oper(X, Y)
    _test_oper(X.T, Y.T)

def test_kron_solver():
    A = rand(3,3)
    B = rand(4,4)
    _test_oper(make_kronecker_solver(A, B), np.linalg.inv(np.kron(A, B)))
    A = scipy.sparse.csr_matrix(A)
    B = scipy.sparse.csr_matrix(B)
    _test_oper(make_kronecker_solver(A, B), np.linalg.inv(np.kron(A.toarray(), B.toarray())))
