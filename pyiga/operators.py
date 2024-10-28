"""Classes and functions for creating custom instances of :class:`scipy.sparse.linalg.LinearOperator`."""
import numpy as np
import scipy.sparse.linalg
from sksparse.cholmod import cholesky
from builtins import range   # Python 2 compatibility

from . import kronecker

HAVE_MKL = True
try:
    import pyMKL
except:
    HAVE_MKL = False


class NullOperator(scipy.sparse.linalg.LinearOperator):
    """Null operator of the given shape which always returns zeros. Used as placeholder."""
    def __init__(self, shape, dtype=np.float64):
        scipy.sparse.linalg.LinearOperator.__init__(self, shape=shape, dtype=dtype)

    def _matvec(self, x):
        return np.zeros(self.shape[0], dtype=self.dtype)
    def _matmat(self, x):
        return np.zeros((self.shape[0], x.shape[1]), dtype=self.dtype)
    def _transpose(self):
        return NullOperator((self.shape[1], self.shape[0]), dtype=self.dtype)

class IdentityOperator(scipy.sparse.linalg.LinearOperator):
    """Identity operator of size `n`."""
    def __init__(self, n, dtype=np.float64):
        scipy.sparse.linalg.LinearOperator.__init__(self, shape=(n,n), dtype=dtype)
    def _matvec(self, x):
        return x
    def _matmat(self, x):
        return x
    def _transpose(self):
        return self

class DiagonalOperator(scipy.sparse.linalg.LinearOperator):
    """A :class:`LinearOperator` which acts like a diagonal matrix with the given diagonal."""
    def __init__(self, diag):
        diag = np.squeeze(diag)
        assert diag.ndim == 1, 'Diagonal must be a vector'
        N = diag.shape[0]
        self.diag = diag
        scipy.sparse.linalg.LinearOperator.__init__(self, shape=(N,N), dtype=diag.dtype)

    def _matvec(self, x):
        if x.ndim == 1:
            return self.diag * x
        else:
            return self.diag[:,None] * x

    def _matmat(self, x):
        return self._matvec(x)

    def _transpose(self):
        return self


class KroneckerOperator(scipy.sparse.linalg.LinearOperator):
    """A :class:`LinearOperator` which efficiently implements the
    application of the Kronecker product of the given input operators.
    """
    def __init__(self, *ops):
        self.ops = ops
        sz     = np.prod([A.shape[1] for A in ops])
        sz_out = np.prod([A.shape[0] for A in ops])
        alldense = all(isinstance(A, np.ndarray) for A in ops)
        allsquare = all(A.shape[0] == A.shape[1] for A in ops)
        if alldense or not allsquare:
            self.applyfunc = kronecker._apply_kronecker_dense
        else:   # use implementation for square LinearOperators; TODO: is this faster??
            self.applyfunc = kronecker._apply_kronecker_linops
        scipy.sparse.linalg.LinearOperator.__init__(self, dtype=ops[0].dtype, shape=(sz_out,sz))

    def _matvec(self, x):
        return self.applyfunc(self.ops, x)

    def _matmat(self, x):
        return self.applyfunc(self.ops, x)

    def _transpose(self):
        return KroneckerOperator(*(B.T for B in self.ops))

    def _adjoint(self):
        return KroneckerOperator(*(B.H for B in self.ops))


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

    def _transpose(self):
        shape_T = (self.shape[1], self.shape[0])
        return BaseBlockOperator(shape_T, tuple(op.T for op in self.ops),
                self.ran_in, self.ran_out)

    def _adjoint(self):
        shape_T = (self.shape[1], self.shape[0])
        return BaseBlockOperator(shape_T, tuple(op.H for op in self.ops),
                self.ran_in, self.ran_out)


def _sizes_to_ranges(sizes):
    """Convert an iterable of sizes into a list of consecutive ranges of these sizes."""
    sizes = list(sizes)
    runsizes = [0] + list(np.cumsum(sizes))
    return [range(runsizes[k], runsizes[k+1]) for k in range(len(sizes))]


def BlockDiagonalOperator(*ops):
    """Return a :class:`LinearOperator` with block diagonal structure, with the given
    operators on the diagonal.
    """
    ranges_i = _sizes_to_ranges(op.shape[0] for op in ops)
    ranges_j = _sizes_to_ranges(op.shape[1] for op in ops)
    shape = (ranges_i[-1].stop, ranges_j[-1].stop)
    return BaseBlockOperator(shape, ops, ranges_i, ranges_j)


def BlockOperator(ops):
    """Construct a block operator.

    Args:
        ops (list): a rectangular list of lists of operators or matrices.
            All operators in a given row should have the same height
            (output dimension).
            All operators in a given column should have the same width
            (input dimension).
            Empty blocks should use :class:`NullOperator` as a placeholder.

    Returns:
        LinearOperator: a block structured linear operator. Its height is the
        total height of one input column of operators, and its width is the
        total width of one input row.

    See also :func:`numpy.block`, which has analogous functionality for
    dense matrices.
    """
    M, N = len(ops), len(ops[0])
    ranges_i = _sizes_to_ranges(ops[i][0].shape[0] for i in range(M))
    ranges_j = _sizes_to_ranges(ops[0][j].shape[1] for j in range(N))
    shape = (ranges_i[-1].stop, ranges_j[-1].stop)

    ops_list, ranges_i_list, ranges_j_list = [], [], []
    for i in range(M):
        assert len(ops[i]) == N, "All rows must have equal length"
        for j in range(N):
            op = ops[i][j]
            if op is None or isinstance(op, NullOperator):
                continue
            else:
                assert op.shape == (len(ranges_i[i]), len(ranges_j[j])), \
                    "Operator at position (%d,%d) has wrong shape" % (i,j)
                ops_list.append(op)
                ranges_i_list.append(ranges_i[i])
                ranges_j_list.append(ranges_j[j])
    if ops_list:
        return BaseBlockOperator(shape, ops_list, ranges_i_list, ranges_j_list)
    else:
        return NullOperator(shape)


class SubspaceOperator(scipy.sparse.linalg.LinearOperator):
    r"""Implements an abstract additive subspace correction operator.

    Args:
        subspaces (seq): a list of `k` prolongation matrices
            :math:`P_j \in \mathbb R^{n \times n_j}`
        Bs (seq): a list of `k` square matrices or instances of :class:`LinearOperator`
            :math:`B_j \in \mathbb R^{n_j \times n_j}`

    Returns:
        LinearOperator: operator with shape :math:`n \times n` that implements the action

        .. math::
            Lx = \sum_{j=1}^k P_j B_j P_j^T x
    """
    def __init__(self, subspaces, Bs):
        subspaces, Bs = tuple(subspaces), tuple(Bs)
        assert len(subspaces) == len(Bs)
        assert len(Bs) > 0, "No operators given"
        n = subspaces[0].shape[0]
        self.subspaces = subspaces
        self.Bs = Bs
        self._is_transpose = False
        super().__init__(shape=(n,n), dtype=Bs[0].dtype)

    def _matvec(self, x):
        if x.ndim > 1:
            x = np.squeeze(x)
        y = np.zeros(len(x))
        if self._is_transpose:
            for j in range(len(self.subspaces)):
                P_j = self.subspaces[j]
                y += P_j.dot(self.Bs[j].T.dot(P_j.T.dot(x)))
        else:
            for j in range(len(self.subspaces)):
                P_j = self.subspaces[j]
                y += P_j.dot(self.Bs[j].dot(P_j.T.dot(x)))
        return y

    def _transpose(self):
        Y = SubspaceOperator(self.subspaces, self.Bs)
        Y._is_transpose = not self._is_transpose
        # shape stays the same since we are square
        return Y


class PardisoSolverWrapper(scipy.sparse.linalg.LinearOperator):
    """Wraps a PARDISO solver object and frees up the memory when deallocated."""
    def __init__(self, shape, dtype, solver):
        self.solver = solver
        scipy.sparse.linalg.LinearOperator.__init__(self, shape=shape, dtype=dtype)
    def _matvec(self, x):
        return self.solver.solve(x)
    def _matmat(self, x):
        return self.solver.solve(x)
    def __del__(self):
        self.solver.clear()
        self.solver = None


def make_solver(B, symmetric=False, spd=False):
    """Return a :class:`LinearOperator` that acts as a linear solver for the
    (dense or sparse) square matrix `B`.

    If `B` is symmetric, passing ``symmetric=True`` may try to take advantage of this.
    If `B` is symmetric and positive definite, pass ``spd=True``.
    """
    if spd:
        symmetric = True

    if scipy.sparse.issparse(B):
        if HAVE_MKL:
            # use MKL Pardiso
            mtype = 11   # real, nonsymmetric
            if symmetric:
                mtype = 2 if spd else -2
            solver = pyMKL.pardisoSolver(B, mtype)
            solver.factor()
            return PardisoSolverWrapper(B.shape, B.dtype, solver)
        else:
            if symmetric:
                chol = cholesky(B.tocsc())
                return scipy.sparse.linalg.LinearOperator(B.shape, dtype=B.dtype, matvec=chol.solve_A, matmat=chol.solve_A)
            else:
                # use SuperLU (unless scipy uses UMFPACK?) -- really slow!
                spLU = scipy.sparse.linalg.splu(B.tocsc(), permc_spec='NATURAL')
                return scipy.sparse.linalg.LinearOperator(B.shape, dtype=B.dtype,
                        matvec=spLU.solve, matmat=spLU.solve)
    else:
        if symmetric:
            chol = scipy.linalg.cho_factor(B, check_finite=False)
            solve = lambda x: scipy.linalg.cho_solve(chol, x, check_finite=False)
            return scipy.sparse.linalg.LinearOperator(B.shape, dtype=B.dtype,
                    matvec=solve, matmat=solve)
        else:
            LU = scipy.linalg.lu_factor(B, check_finite=False)
            solve = lambda x: scipy.linalg.lu_solve(LU, x, check_finite=False)
            return scipy.sparse.linalg.LinearOperator(B.shape, dtype=B.dtype,
                    matvec=solve, matmat=solve)


def make_kronecker_solver(*Bs): #, symmetric=False): # kw arg doesn't work in Py2
    """Given several square matrices, return an operator which efficiently applies
    the inverse of their Kronecker product.
    """
    Binvs = tuple(make_solver(B) for B in Bs)
    return KroneckerOperator(*Binvs)

