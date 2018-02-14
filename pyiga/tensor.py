r"""Functions and classes for manipulating tensors in full, canonical, and
Tucker format, and for tensor approximation.

A **full tensor** is simply represented as a :class:`numpy.ndarray`.
Additional tensor formats are implemented in the following classes:

* :class:`CanonicalTensor`
* :class:`TuckerTensor`
"""
import numpy as np
import numpy.linalg
import scipy.linalg


def _modek_tensordot_sparse(B, X, k):
    # This does the same as the np.tensordot() operation used below in
    # `apply_tprod`, but works for sparse matrices and LinearOperators.
    nk = X.shape[k]
    assert nk == B.shape[1]

    # bring the k-th axis to the front
    Xk = np.rollaxis(X, k, 0)
    shp = Xk.shape

    # matricize and apply operator B
    Xk = Xk.reshape((nk, -1))
    Yk = B.dot(Xk)
    if Yk.shape[0] != nk:   # size changed?
        shp = (Yk.shape[0],) + shp[1:]
    # reshape back, new axis is in first position
    return np.reshape(Yk, shp)


def apply_tprod(ops, A):
    """Apply tensor product of operators to ndarray `A`.

    Args:
        ops (seq): a list of matrices, sparse matrices, or LinearOperators
        A (ndarray): a tensor
    Returns:
        ndarray: a new tensor with the same number of axes as `A` that is
        the result of applying the tensor product operator
        ``ops[0] x ... x ops[-1]`` to `A`

    This does essentially the same as :func:`pyiga.kronecker.apply_kronecker`,
    but assumes that A is an ndarray with the proper number of dimensions
    rather than its matricization.

    The initial dimensions of `A` must match the sizes of the
    operators, but `A` is allowed to have an arbitrary number of
    trailing dimensions. ``None`` is a valid operator and is
    treated like the identity."""
    n = len(ops)
    for i in reversed(range(n)):
        if ops[i] is not None:
            if isinstance(ops[i], np.ndarray):
                A = np.tensordot(ops[i], A, axes=([1],[n-1]))
            else:
                A = _modek_tensordot_sparse(ops[i], A, n-1)
        else:   # None means identity
            A = np.rollaxis(A, n-1, 0)   # bring this axis to the front
    return A


def fro_norm(X):
    """Compute the Frobenius norm of multi-dimensional array `X`."""
    return np.linalg.norm(X.ravel(order='K'))

def matricize(X, k):
    """Return the mode-`k` matricization of the ndarray `X`."""
    nk = X.shape[k]
    return np.reshape(np.swapaxes(X, 0,k), (nk,-1), order='C')

def modek_tprod(B, k, X):
    """Compute the mode-`k` tensor product of the ndarray `X` with the matrix
    or operator `B`.

    Args:
        B: an `ndarray`, sparse matrix, or `LinearOperator` of size `m x nk`
        k (int): the mode along which to multiply `X`
        X (ndarray): tensor with ``X.shape[k] == nk``

    Returns:
        ndarray: the mode-`k` tensor product of size `(n1, ... nk-1, m, nk+1, ..., nN)`
    """
    if isinstance(B, np.ndarray):
        Y = np.tensordot(X, B, axes=((k,1)))
        return np.rollaxis(Y, -1, k) # put last (new) axis back into k-th position
    else:
        Y = _modek_tensordot_sparse(B, X, k)
        return np.moveaxis(Y, 0, k) # put first (new) axis back into k-th position


def hosvd(X):
    """Compute higher-order SVD (Tucker decomposition).

    Args:
        X (ndarray): a full tensor of arbitrary size
    Returns:
        :class:`TuckerTensor`: a Tucker tensor which represents `X` with the
        core tensor having the same shape as `X` and the factor matrices `Uk`
        being square and orthogonal.
    """
    # left singular vectors for each matricization
    U = [scipy.linalg.svd(matricize(X,k), full_matrices=False, check_finite=False)[0]
            for k in range(X.ndim)]
    C = apply_tprod(tuple(Uk.T for Uk in U), X)   # core tensor (same size as X)
    return TuckerTensor(U, C)

def find_best_truncation_axis(X):
    """Find the axis along which truncating the last slice causes the smallest error."""
    errors = [np.linalg.norm(np.swapaxes(X, i, 0)[-1].ravel())
              for i in range(X.ndim)]
    i = np.argmin(errors)
    return i, errors[i]

def find_truncation_rank(X, tol=1e-12):
    """A greedy algorithm for finding a good truncation rank for a HOSVD core tensor."""
    total_err_squ = 0.0
    tolsq = tol**2
    while X.size > 0:
        ax,err = find_best_truncation_axis(X)
        total_err_squ += err**2
        if total_err_squ > tolsq:
            break
        else:
            # truncate one slice off axis ax
            sl = X.ndim * [slice(None)]
            sl[ax] = slice(None, -1)
            X = X[sl]
    return X.shape


def outer(*xs):
    """Outer product of an arbitrary number of vectors.

    Args:
        xs: `d` input vectors `(x1, ..., xd)` with lengths `n1, ..., nd`
    Returns:
        ndarray: the outer product as an `ndarray` with `d` dimensions
    """
    if len(xs) == 1:
        return xs[0]
    else:
        return outer(*xs[:-1])[..., None] * xs[-1][None, ...]

def dot_rank1(xs, ys):
    """Compute the inner (Frobenius) product of two rank 1 tensors."""
    return np.prod(tuple(np.dot(xs[j], ys[j]) for j in range(len(xs))))



def als1(B, tol=1e-15):
    """Compute best rank 1 approximation to tensor `B` using Alternating Least Squares.

    Returns:
        A tuple of vectors `(x1, ..., xd)` such that ``outer(x1, ..., xd)`` is
        the approximate best rank 1 approximation to `B`.
    """
    d = B.ndim
    # use random row vectors as starting values
    xs = [np.random.rand(1,n) for n in B.shape]

    while True:
        delta = 1.0
        for k in range(d):
            ys = xs.copy()
            ys[k] = None
            xk = apply_tprod(ys, B).ravel() / np.prod([np.sum(xs[l]*xs[l]) for l in range(d) if l != k])
            delta = delta * np.linalg.norm(xk - xs[k][0])
            xs[k][0, :] = xk
        if delta < tol:
            break
    return tuple(x[0] for x in xs)  # return xs as 1D vectors


class CanonicalTensor:
    """A tensor in CP (canonical/PARAFAC) format, i.e., a sum of rank 1 tensors.

    For a tensor of order `d`, `Xs` should be a tuple of `d` matrices.  Their
    number of columns should be identical and determines the rank `R` of the
    tensor.  The number of rows of the `j`-th matrix determines the size of the
    tensor along the `j`-th axis.

    The tensor is given by the sum, for `r` up to `R`, of the outer products of the
    `r`-th columns of the matrices `Xs`.
    """
    def __init__(self, Xs):
        # ensure Xs are matrices
        self.Xs = tuple(X[:,None] if X.ndim==1 else X for X in Xs)
        self.ndim = len(self.Xs)
        self.shape = tuple(X.shape[0] for X in self.Xs)
        self.R = self.Xs[0].shape[1]
        assert all(X.shape[1] == self.R for X in self.Xs), 'invalid matrix shape'

    def asarray(self):
        """Convert canonical tensor to a full `ndarray`."""
        X = np.zeros(self.shape)
        for r in range(self.R):
            X += outer(*tuple(X[:,r] for X in self.Xs))
        return X

    def norm(self):
        """Compute the Frobenius norm of the tensor."""
        def term(j):
            return tuple(X[:,j] for X in self.Xs)
        return np.sqrt(
            sum(dot_rank1(term(i), term(j))
                for i in range(self.R)
                for j in range(self.R)))


class TuckerTensor:
    """A *d*-dimensional tensor in **Tucker format** is given as a list of *d* basis matrices

    .. math::
        U_k \in \mathbb R^{n_k \times m_k}, \qquad k=1,\ldots,d

    and a (typically small) core coefficient tensor

    .. math::
        X \in \mathbb R^{m_1 \times \ldots \times m_d}.

    When expanded (using :func:`TuckerTensor.asarray`), a Tucker tensor turns into a full
    tensor

    .. math::
        A \in \mathbb R^{n_1 \times \ldots \times n_d}.

    One way to compute a Tucker tensor approximation from a full tensor is to first
    compute the HOSVD using :func:`hosvd` and then truncate it using
    :func:`TuckerTensor.truncate` to the rank estimated by :func:`find_truncation_rank`.
    """
    def __init__(self, Us, X):
        self.Us = tuple(Us)
        self.X = X
        self.ndim = len(Us)
        assert self.ndim == X.ndim, 'Incompatible sizes'
        self.shape = tuple(U.shape[0] for U in self.Us)

    def asarray(self):
        """Convert Tucker tensor to a full `ndarray`."""
        return apply_tprod(self.Us, self.X)

    def orthogonalize(self):
        """Compute an equivalent Tucker representation of the current tensor
        where the matrices `U` have orthonormal columns.
        """
        QR = tuple(np.linalg.qr(U) for U in self.Us)
        Rinv = tuple(scipy.linalg.solve_triangular(R, np.eye(R.shape[1])) for (_,R) in QR)
        return TuckerTensor(tuple(Q for (Q,_) in QR),
                apply_tprod(Rinv, self.X))

    def norm(self):
        """Compute the Frobenius norm of the tensor."""
        return fro_norm(self.orthogonalize().X)

    def truncate(self, k):
        """Truncate a Tucker tensor `T` to the given rank `k`."""
        N = self.ndim
        if np.isscalar(k):
            slices = N * (slice(None,k),)
        else:
            assert len(k) == N
            slices = tuple(slice(None, ki) for ki in k)
        return TuckerTensor(tuple(self.Us[i][:,slices[i]] for i in range(N)), self.X[slices])

