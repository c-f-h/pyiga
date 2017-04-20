import numpy as np
import scipy.sparse.linalg

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
    Y = apply_tprod(ops, X)
    return Y.reshape(x.shape)


def KroneckerOperator(*ops):
    """Return a `LinearOperator` which efficiently implements the
    application of the Kronecker product of the given input operators.
    """
    # assumption: all operators are square
    sz = np.prod([A.shape[0] for A in ops])
    if all(isinstance(A, np.ndarray) for A in ops):
        applyfunc = lambda x: _apply_kronecker_dense(ops, x)
        return scipy.sparse.linalg.LinearOperator(shape=(sz,sz),
                matvec=applyfunc, matmat=applyfunc)
    else:
        ops = [scipy.sparse.linalg.aslinearoperator(B) for B in ops]
        applyfunc = lambda x: _apply_kronecker_linops(ops, x)
        return scipy.sparse.linalg.LinearOperator(shape=(sz,sz),
                matvec=applyfunc, matmat=applyfunc)


def apply_tprod(ops, A):
    """Apply tensor product of operators to ndarray `A`.

    Args:
        ops (seq): a list of matrices
        A (ndarray): a tensor
    Returns:
        ndarray: a new tensor with the same number of axes as `A` that is
        the result of applying the tensor product operator
        ``ops[0] x ... x ops[-1]`` to `A`

    This does essentially the same as :func:`apply_kronecker`, but assumes
    that A is an ndarray with the proper number of dimensions rather
    than its matricization.
    See also :func:`pyiga.tucker.tucker_prod`, which is almost
    equivalent, but uses transposed matrices.

    The initial dimensions of `A` must match the sizes of the
    operators, but `A` is allowed to have an arbitrary number of
    trailing dimensions. ``None`` is a valid operator and is
    treated like the identity."""
    # this works only for dense matrices as operators
    n = len(ops)
    for i in reversed(range(n)):
        if ops[i] is not None:
            A = np.tensordot(ops[i], A, axes=([1],[n-1]))
        else:   # None means identity
            A = np.rollaxis(A, n-1, 0)   # bring this axis to the front
    return A


def DiagonalOperator(diag):
    """Return a `LinearOperator` which acts like a diagonal matrix
    with the given diagonal."""
    diag = np.squeeze(diag)
    assert diag.ndim == 1, 'Diagonal must be a vector'
    N = diag.shape[0]
    matvec = lambda x: diag * x
    return scipy.sparse.linalg.LinearOperator(
        shape=(N,N),
        matvec=matvec,
        rmatvec=matvec,
        matmat=lambda x: diag[:,None] * x,
        dtype=diag.dtype
    )
