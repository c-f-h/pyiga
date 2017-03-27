import numpy as np
import numpy.linalg

def matricize(X, k):
    nk = X.shape[k]
    return np.reshape(np.swapaxes(X, 0,k), (nk,-1), order='C')

def modek_tprod(X, k, B):
    """X is an ndarray with X.shape[k] = nk and
    B is a matrix of size nk x m.

    The result is the mode-k tensor product of
    size (n1, ... nk-1, j, nk+1, ..., nN)."""
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
    assert len(Uk) == X.ndim
    Y = X
    for i in range(len(Uk)):
        Y = modek_tprod(Y, i, Uk[i])
    return Y

def truncate_tucker(T, k):
    C, Uk = T
    N = C.ndim
    if np.isscalar(k):
        slices = N * (slice(None,k),)
    else:
        assert len(k) == N
        slices = tuple(slice(None, ki) for ki in k)
    return (C[slices], tuple(Uk[i][slices[i],:] for i in range(N)))
