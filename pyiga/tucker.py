import numpy as np
import numpy.linalg

def matricize(X, k):
    """Return the mode-`k` matricization of the ndarray `X`."""
    nk = X.shape[k]
    return np.reshape(np.swapaxes(X, 0,k), (nk,-1), order='C')

def modek_tprod(X, k, B):
    """Compute the mode-`k` tensor product of the ndarray X with the matrix `B`.

    Args:
        X (ndarray): tensor with ``X.shape[k] == nk``
        k (int): the mode along which to multiply `X`
        B (ndarray): a 2D array of size `nk x m`.

    Returns:
        ndarray: the mode-`k` tensor product of size `(n1, ... nk-1, m, nk+1, ..., nN)`
    """
    Y = np.tensordot(X, B, axes=((k,0)))
    return np.rollaxis(Y, -1, k) # put last (new) axis back into k-th position

def hosvd(X):
    """Compute higher-order SVD (Tucker decomposition).

    The result is a tuple (C, (U0,U1,...,Un)), where the
    core tensor C has the same shape as X and the
    Uk are square, orthogonal matrices of size X.shape[k]."""
    # left singular vectors for each matricization
    U = [np.linalg.svd(matricize(X,k), full_matrices=False)[0]
            for k in range(X.ndim)]
    C = tucker_prod(X, U)   # core tensor (same size as X)
    return (C, tuple(Uk.T for Uk in U))

def tucker_prod(X, Uk):
    """Convert the Tucker tensor `(X,Uk)` to a full tensor (`ndarray`) by multiplying
    each mode of `X` by the corresponding matrix in `Uk`."""
    assert len(Uk) == X.ndim
    Y = X
    for i in range(len(Uk)):
        if Uk[i] is not None:   # None is equivalent to identity
            Y = np.tensordot(Y, Uk[i], axes=(0,0))  # tensordot brings the new axis to the end
        else:   # None, do not change this dimension
            Y = np.rollaxis(Y, 0, len(Uk))   # bring this axes to the end
    return Y

def truncate(T, k):
    """Truncate a Tucker tensor `T` to the given rank `k`."""
    C, Uk = T
    N = C.ndim
    if np.isscalar(k):
        slices = N * (slice(None,k),)
    else:
        assert len(k) == N
        slices = tuple(slice(None, ki) for ki in k)
    return (C[slices], tuple(Uk[i][slices[i],:] for i in range(N)))

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

