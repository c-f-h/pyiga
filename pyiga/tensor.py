r"""Functions for manipulating tensors in full and in Tucker format,
and for tensor approximation.

A **full tensor** is simply represented as a :class:`numpy.ndarray`.

A *d*-dimensional tensor in **Tucker format** is given as a list of *d* basis matrices

.. math::
    U_k \in \mathbb R^{n_k \times m_k}, \qquad k=1,\ldots,d

and a (typically small) core coefficient tensor

.. math::
    X \in \mathbb R^{m_1 \times \ldots \times m_d}.

When expanded (using :func:`tucker_prod`), a Tucker tensor turns into a full
tensor

.. math::
    A \in \mathbb R^{n_1 \times \ldots \times n_d}.

One way to compute a Tucker tensor approximation from a full tensor is to first
compute the HOSVD using :func:`hosvd` and then truncate it using
:func:`truncate` to the rank estimated by :func:`find_truncation_rank`.
"""
import numpy as np
import numpy.linalg
import scipy.linalg


def _modek_tensordot_sparse(B, X, k):
    # This does the same as the np.tensordot() operation used below in
    # `apply_tprod`, but works for sparse matrices and LinearOperators.
    nk = X.shape[k]
    assert nk == B.shape[1]

    Xk = np.rollaxis(X, k, 0)
    shp = Xk.shape

    Xk = Xk.reshape((nk, -1))
    Yk = B.dot(Xk)
    if Yk.shape[0] != nk:   # size changed?
        shp = (Yk.shape[0],) + shp[1:]
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
    """Compute the mode-`k` tensor product of the ndarray `X` with the matrix `B`.

    Args:
        B (ndarray): a 2D array of size `m x nk`.
        k (int): the mode along which to multiply `X`
        X (ndarray): tensor with ``X.shape[k] == nk``

    Returns:
        ndarray: the mode-`k` tensor product of size `(n1, ... nk-1, m, nk+1, ..., nN)`
    """
    Y = np.tensordot(X, B, axes=((k,1)))
    return np.rollaxis(Y, -1, k) # put last (new) axis back into k-th position

def tucker_prod(Uk, X):
    """Convert the Tucker tensor `(Uk,X)` to a full tensor (`ndarray`) by multiplying
    each mode of `X` by the corresponding matrix in `Uk`."""
    return apply_tprod(Uk, X)

def hosvd(X):
    """Compute higher-order SVD (Tucker decomposition).

    The result is a tuple `((U0,U1,...,Un), C)`, where the
    core tensor `C` has the same shape as `X` and the
    `Uk` are square, orthogonal matrices of size `X.shape[k]`."""
    # left singular vectors for each matricization
    U = [scipy.linalg.svd(matricize(X,k), full_matrices=False, check_finite=False)[0]
            for k in range(X.ndim)]
    C = tucker_prod(tuple(Uk.T for Uk in U), X)   # core tensor (same size as X)
    return (U, C)

def truncate(T, k):
    """Truncate a Tucker tensor `T` to the given rank `k`."""
    Uk, C = T
    N = C.ndim
    if np.isscalar(k):
        slices = N * (slice(None,k),)
    else:
        assert len(k) == N
        slices = tuple(slice(None, ki) for ki in k)
    return (tuple(Uk[i][:,slices[i]] for i in range(N)), C[slices])

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

