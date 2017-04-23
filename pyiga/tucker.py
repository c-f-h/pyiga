r"""Functions for manipulating tensors in Tucker format and computing the HOSVD.

A *d*-dimensional Tucker tensor is given as a list of *d* basis matrices

.. math::
    U_k \in \mathbb R^{n_k \times m_k}, \qquad k=1,\ldots,d

and a (typically small) core coefficient tensor

.. math::
    X \in \mathbb R^{m_1 \times \ldots \times m_d}.

When expanded, it represents a full tensor

.. math::
    A \in \mathbb R^{n_1 \times \ldots \times n_d}.
"""
import numpy as np
import numpy.linalg
import scipy.linalg

from . import kronecker

def matricize(X, k):
    """Return the mode-`k` matricization of the ndarray `X`."""
    nk = X.shape[k]
    return np.reshape(np.swapaxes(X, 0,k), (nk,-1), order='C')

def modek_tprod(X, k, B):
    """Compute the mode-`k` tensor product of the ndarray X with the matrix `B`.

    Args:
        X (ndarray): tensor with ``X.shape[k] == nk``
        k (int): the mode along which to multiply `X`
        B (ndarray): a 2D array of size `m x nk`.

    Returns:
        ndarray: the mode-`k` tensor product of size `(n1, ... nk-1, m, nk+1, ..., nN)`
    """
    Y = np.tensordot(X, B, axes=((k,1)))
    return np.rollaxis(Y, -1, k) # put last (new) axis back into k-th position

def tucker_prod(Uk, X):
    """Convert the Tucker tensor `(Uk,X)` to a full tensor (`ndarray`) by multiplying
    each mode of `X` by the corresponding matrix in `Uk`."""
    return kronecker.apply_tprod(Uk, X)

def hosvd(X):
    """Compute higher-order SVD (Tucker decomposition).

    The result is a tuple (C, (U0,U1,...,Un)), where the
    core tensor C has the same shape as X and the
    Uk are square, orthogonal matrices of size X.shape[k]."""
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

