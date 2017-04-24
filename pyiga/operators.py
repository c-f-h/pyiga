"""Classes and functions for creating custom instances of :class:`scipy.sparse.linalg.LinearOperator`."""
import numpy as np
import scipy.sparse.linalg

from . import kronecker


def DiagonalOperator(diag):
    """Return a `LinearOperator` which acts like a diagonal matrix
    with the given diagonal."""
    diag = np.squeeze(diag)
    assert diag.ndim == 1, 'Diagonal must be a vector'
    N = diag.shape[0]
    def matvec(x):
        if x.ndim == 1:
            return diag * x
        else:
            return diag[:,None] * x
    return scipy.sparse.linalg.LinearOperator(
        shape=(N,N),
        matvec=matvec,
        rmatvec=matvec,
        matmat=matvec,
        dtype=diag.dtype
    )


def KroneckerOperator(*ops):
    """Return a `LinearOperator` which efficiently implements the
    application of the Kronecker product of the given input operators.
    """
    # assumption: all operators are square
    sz = np.prod([A.shape[0] for A in ops])
    if all(isinstance(A, np.ndarray) for A in ops):
        applyfunc = lambda x: kronecker._apply_kronecker_dense(ops, x)
        return scipy.sparse.linalg.LinearOperator(shape=(sz,sz),
                matvec=applyfunc, matmat=applyfunc)
    else:
        ops = [scipy.sparse.linalg.aslinearoperator(B) for B in ops]
        applyfunc = lambda x: kronecker._apply_kronecker_linops(ops, x)
        return scipy.sparse.linalg.LinearOperator(shape=(sz,sz),
                matvec=applyfunc, matmat=applyfunc)


class BaseBlockOperator(scipy.sparse.linalg.LinearOperator):
    def __init__(self, shape, ops, ran_out, ran_in):
        self.ops = ops
        self.ran_out = ran_out
        self.ran_in = ran_in
        scipy.sparse.linalg.LinearOperator.__init__(self, ops[0].dtype, shape)
    
    def _matvec(self, x):
        y = np.zeros(self.shape[0])
        if x.ndim == 2:
            x = x[:,0]
        for i in range(len(self.ops)):
            y[self.ran_out[i]] += self.ops[i].dot(x[self.ran_in[i]])
        return y

    def _matmat(self, x):
        y = np.zeros((self.shape[0], x.shape[1]))
        for i in range(len(self.ops)):
            y[self.ran_out[i]] += self.ops[i].dot(x[self.ran_in[i]])
        return y


def BlockDiagonalOperator(*ops):
    """Return a `LinearOperator` with block diagonal structure, with the given
    operators on the diagonal.
    """
    K = len(ops)
    sizes_i = [op.shape[0] for op in ops]
    sizes_j = [op.shape[1] for op in ops]
    runsizes_i = [0] + list(np.cumsum(sizes_i))
    runsizes_j = [0] + list(np.cumsum(sizes_j))
    ranges_i = [range(runsizes_i[k], runsizes_i[k+1]) for k in range(K)]
    ranges_j = [range(runsizes_j[k], runsizes_j[k+1]) for k in range(K)]
    shape = (runsizes_i[-1], runsizes_j[-1])
    return BaseBlockOperator(shape, ops, ranges_i, ranges_j)


def make_solver(B, symmetric=False):
    """Construct a linear solver for the (dense or sparse) square matrix B.
    
    Returns a LinearOperator.
    """
    if scipy.sparse.issparse(B):
        spLU = scipy.sparse.linalg.splu(B.tocsc(), permc_spec='NATURAL')
        return scipy.sparse.linalg.LinearOperator(B.shape, spLU.solve)
    else:
        if symmetric:
            chol = scipy.linalg.cho_factor(B, check_finite=False)
            return scipy.sparse.linalg.LinearOperator(B.shape,
                lambda x: scipy.linalg.cho_solve(chol, x, check_finite=False))
        else:
            LU = scipy.linalg.lu_factor(B, check_finite=False)
            return scipy.sparse.linalg.LinearOperator(B.shape,
                lambda x: scipy.linalg.lu_solve(LU, x, check_finite=False))


def make_kronecker_solver(*Bs): #, symmetric=False): # kw arg doesn't work in Py2
    """Given a list of square matrices `Bs`, returns an operators which efficiently applies
    the inverse of their Kronecker product.
    """
    Binvs = tuple(make_solver(B) for B in Bs)
    return KroneckerOperator(*Binvs)

