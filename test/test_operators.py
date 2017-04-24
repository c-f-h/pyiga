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

def test_diagonal():
    diag = rand(10)
    diag_op = DiagonalOperator(diag)
    D = np.diag(diag)
    _test_oper(diag_op, D)

def test_blockdiag():
    A = rand(2,3)
    B = rand(4,4)
    C = rand(3,1)
    X = scipy.linalg.block_diag(A, B, C)
    Y = BlockDiagonalOperator(A, B, C)
    _test_oper(X, Y)

def test_solver():
    A = rand(3,3)
    _test_oper(make_solver(A), np.linalg.inv(A))
    B = A + A.T + 3*np.eye(3)
    _test_oper(make_solver(B, symmetric=True), np.linalg.inv(B))
    A = scipy.sparse.csr_matrix(A)
    _test_oper(make_solver(A), np.linalg.inv(A.A))
    B = scipy.sparse.csr_matrix(B)
    _test_oper(make_solver(B, symmetric=True), np.linalg.inv(B.A))

def test_kron_solver():
    A = rand(3,3)
    B = rand(4,4)
    _test_oper(make_kronecker_solver(A, B), np.linalg.inv(np.kron(A, B)))
    A = scipy.sparse.csr_matrix(A)
    B = scipy.sparse.csr_matrix(B)
    _test_oper(make_kronecker_solver(A, B), np.linalg.inv(np.kron(A.A, B.A)))
