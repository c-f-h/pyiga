import numpy as np
import scipy.sparse.linalg

from . import tensor

def apply_kronecker(ops, x):
    """Apply the Kronecker product of a sequence of square matrices or linear operators."""
    if all(isinstance(A, np.ndarray) for A in ops):
        return _apply_kronecker_dense(ops, x)
    else:
        ops = [scipy.sparse.linalg.aslinearoperator(B) for B in ops]
        return _apply_kronecker_linops(ops, x)


def _apply_kronecker_linops(ops, x):
    """Apply the Kronecker product of a sequence of square linear operators."""
    assert len(ops) >= 1, "Empty Kronecker product"
    
    if len(ops) == 1:
        return ops[0] * x
    
    # assumption: all operators are square
    sz = np.prod([A.shape[0] for A in ops])
    assert sz == x.shape[0], "Wrong size for input matrix"
    
    orig_shape = x.shape
    # make sure input x is 2D ndarray
    if (len(orig_shape) == 1):
        x = x.reshape((orig_shape[0], 1))
    
    n = x.shape[1] # number of input vectors
    
    # algorithm relies on column-major matrices
    q0 = np.empty(x.shape, order='F')
    q0[:] = x
    q1 = np.empty(x.shape, order='F')
    
    for i in reversed(range(len(ops))):
        sz_i = ops[i].shape[1]
        r_i = sz // sz_i  # result is always integer
        
        q0 = q0.reshape((sz_i, n * r_i), order='F')
        q1.resize((r_i, n * sz_i))
        
        if n == 1:
            q1[:] = (ops[i] * q0).T
        else:
            for k in range(n):
                # apply op to coefficients for k-th rhs vector
                temp = ops[i] * q0[:, k*r_i : (k+1)*r_i] # sz_i x r_i
                q1[:, k*sz_i : (k+1)*sz_i] = temp.T
        
        q0, q1 = q1, q0   # swap input and output
    
    return q0.reshape(orig_shape, order='F')


def _apply_kronecker_dense(ops, x):
    shape = tuple(op.shape[0] for op in ops)
    assert x.ndim == 1 or x.ndim == 2, \
        'Only vectors or matrices allowed as right-hand sides'
    if x.ndim == 2 and x.shape[1] > 1:
        m = x.shape[1]
        shape = shape + (m,)
    X = x.reshape(shape)
    Y = tensor.apply_tprod(ops, X)
    return Y.reshape(x.shape)

