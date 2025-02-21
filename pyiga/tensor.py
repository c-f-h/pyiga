r"""Functions and classes for manipulating tensors in full, canonical, and
Tucker format, and for tensor approximation.

A **full tensor** is simply represented as a :class:`numpy.ndarray`.
Additional tensor formats are implemented in the following classes:

* :class:`CanonicalTensor`
* :class:`TuckerTensor`

In addition, arbitrary tensors can be composed into sums or tensor products
using the following classes:

* :class:`TensorSum`
* :class:`TensorProd`

Below, whenever we refer generically to "a tensor", we mean either an `ndarray`
or an instance of any of these tensor classes.

All tensor classes have members ``ndim``, ``shape``, and ``ravel`` which have the same
meaning as for an `ndarray`.
Any tensor can be expanded to a full `ndarray` using :func:`asarray`.
In addition, most tensor classes have overloaded operators for adding and
subtracting tensors in their native format.

All tensors can be sliced using the standard numpy ``[]`` indexing syntax. The
result is a tensor in the same format, except for the case where all axes have
a single scalar index, in which case the entry at the corresponding index is
returned as a scalar value.

Linear operators on tensors, themselves represented in suitable low-rank formats,
are described by

* :class:`CanonicalOperator`

--------------
Module members
--------------
"""
import numpy as np
import numpy.linalg
import scipy.linalg

from functools import reduce
import operator
from . import utils


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

def _normalize_indices(I, shape):
    if not isinstance(I, tuple):
        I = (I,)
    d = len(shape)
    if len(I) < d:
        I = I + (d - len(I)) * (slice(None),)
    elif len(I) > d:
        raise ValueError('got {} indices but have only {} axes'.format(len(I), d))

    I_new = []
    shape_new = []
    singleton = []

    for k in range(d):
        ik = I[k]
        if np.isscalar(ik):
            i = range(shape[k])[ik] # raises error if invalid index
            r = range(i, i+1)
            shape_new.append(1)
            singleton.append(k)
        elif isinstance(ik, slice):
            r = range(shape[k])[ik]
            shape_new.append(len(r))
        else:   # should be a list/array of indices
            r = np.arange(shape[k])[ik]
            shape_new.append(len(r))
        I_new.append(r)

    return tuple(I_new), tuple(shape_new), tuple(singleton)


def apply_tprod(ops, A):
    """Apply multi-way tensor product of operators to tensor `A`.

    Args:
        ops (seq): a list of matrices, sparse matrices, or LinearOperators
        A (tensor): the tensor to apply the multi-way tensor product to
    Returns:
        a new tensor with the same number of axes as `A` that is the result of
        applying the tensor product operator ``ops[0] x ... x ops[-1]`` to `A`.
        The return type is typically the same type as `A`.

    The initial dimensions of `A` must match the sizes of the
    operators, but `A` is allowed to have an arbitrary number of
    trailing dimensions. ``None`` is a valid operator and is
    treated like the identity.

    An interpretation of this operation is that the Kronecker product of the
    matrices `ops` is applied to the vectorization of the tensor `A`.
    """
    if hasattr(A, 'nway_prod'):
        # tensor classes provide their own implementation
        return A.nway_prod(ops)
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
    """Compute the Frobenius norm of the tensor `X`."""
    if hasattr(X, 'norm'):
        return X.norm()
    else:
        return np.linalg.norm(X.ravel())

def asarray(X):
    """Return the tensor `X` as a full ndarray."""
    if hasattr(X, 'asarray'):
        return X.asarray()
    else:
        return np.asanyarray(X)

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

def _find_best_truncation_axis(X):
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
        ax,err = _find_best_truncation_axis(X)
        total_err_squ += err**2
        if total_err_squ > tolsq:
            break
        else:
            # truncate one slice off axis ax
            sl = X.ndim * [slice(None)]
            sl[ax] = slice(None, -1)
            X = X[tuple(sl)]
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

def array_outer(*xs):
    """Outer product of an arbitrary number of ndarrays.

    Args:
        xs: an arbitrary number of input ndarrays
    Returns:
        ndarray: the outer product of the inputs. Its shape is the
        concatenation of the shapes of the inputs.
    """
    if len(xs) == 1:
        return xs[0]
    else:
        return np.multiply.outer(array_outer(*xs[:-1]), xs[-1])

def pad(X, pad_width):
    """Pad a tensor with zero rows in each direction.

    Args:
        pad_width (list): a list of `(before,after)` tuples, the same
            length as dimensions of `X`, which specifices how many
            zeros to prepend/append in each direction. `None` is
            admissible and is equivalent to `(0,0)`.
    Returns:
        the padded tensor
    """
    assert len(pad_width) == X.ndim, 'invalid length of pad_width'
    P = []
    for j in range(X.ndim):
        if pad_width[j] is None:
            P.append(None)
        else:
            nj = X.shape[j]
            B = scipy.sparse.lil_matrix((nj + pad_width[j][0] + pad_width[j][1], nj))
            B[pad_width[j][0] : nj + pad_width[j][0]] = scipy.sparse.eye(nj)
            P.append(B)
    return apply_tprod(P, X)

################################################################################
## Approximation algorithms
################################################################################


def _dot_rank1(xs, ys):
    """Compute the inner (Frobenius) product of two rank 1 tensors."""
    return np.prod(tuple(np.dot(xs[j], ys[j]) for j in range(len(xs))))

def _apply_lowrank(Ts, xs):
    """Apply a sum of rank 1 operators to a rank 1 tensor."""
    return list(
            tuple(T[j].dot(xs[j])
                for j in range(len(T)))
            for T in Ts)

def _multi_kron(As):
    """Kronecker product of an arbitrary number of matrices."""
    return As[0] if len(As)==1 else np.kron(As[0], _multi_kron(As[1:]))


def als1(A, tol=1e-15):
    """Compute best rank 1 approximation to tensor `A` using Alternating Least Squares.

    Args:
        A (tensor): the tensor to be approximated
        tol (float): tolerance for the stopping criterion
    Returns:
        A tuple of vectors `(x1, ..., xd)` such that ``outer(x1, ..., xd)`` is
        the approximate best rank 1 approximation to `A`.
    """
    d = A.ndim
    # use random row vectors as starting values
    xs = [np.random.rand(1,n) for n in A.shape]

    while True:
        delta = 1.0
        for k in range(d):
            ys = xs[:]      # copy list
            ys[k] = None
            xk = apply_tprod(ys, A).ravel() / np.prod([np.sum(xs[l]*xs[l]) for l in range(d) if l != k])
            delta = delta * np.linalg.norm(xk - xs[k][0])
            xs[k][0, :] = xk
        if delta < tol:
            break
    return tuple(x[0] for x in xs)  # return xs as 1D vectors


def _without_k(L, k):
    """Drop the k-th component of a list."""
    return L[:k] + L[k+1:]


def als(A, R, tol=1e-10, maxiter=10000, startval=None):
    """Compute best rank `R` approximation to tensor `A` using Alternating Least Squares.

    Args:
        A (tensor): the tensor to be approximated
        R (int): the desired rank
        tol (float): tolerance for the stopping criterion
        maxiter (int): maximum number of iterations
        startval: starting tensor for iteration. By default, a random rank `R`
            tensor is used. A :class:`CanonicalTensor` with rank `R` may be
            supplied for `startval` instead.
    Returns:
        :class:`CanonicalTensor`: a rank `R` approximation to `A`; generally close
        to the best rank `R` approximation if the algorithm converged to a small
        enough tolerance.
    """
    if startval is None:
        xs = [np.random.rand(R,n) for n in A.shape]
    else:
        if isinstance(startval, CanonicalTensor):
            assert startval.R == R, 'starting value has wrong rank'
            startval = startval.Xs
        xs = [x.T for x in startval]
        assert all(x.shape == (R,n) for (x,n) in zip(xs, A.shape)), \
                'starting value has wrong shape'

    d = A.ndim
    A_norm = fro_norm(A)
    ys = d * [None]     # temporary
    # precompute matrices x_j x_j^T
    xxT = [xs[j].dot(xs[j].T) for j in range(d)]

    for it in range(maxiter):
        delta = 0.0
        for k in range(d):
            C = np.empty((R, A.shape[k]))
            for r in range(R):
                for j in range(d):
                    ys[j] = xs[j][r:r+1,:]
                ys[k] = None
                C[r, :] = apply_tprod(ys, A).ravel()

            # entrywise product of the matrices (x_j x_j^T) (size R x R) for all j != k
            Gamma = np.prod(_without_k(xxT, k), axis=0)

            delta = delta + fro_norm(-C + Gamma.dot(xs[k]))**2
            xs[k] = np.linalg.solve(Gamma, C)
            # update x[k] x[k]^T
            xxT[k] = xs[k].dot(xs[k].T)
        if (np.sqrt(delta) / A_norm) < tol:
            break
    return CanonicalTensor((x.T for x in xs))


def grou(B, R, tol=1e-12, return_errors=False):
    """Canonical tensor approximation by Greedy Rank One Updates.

    References:
        https://doi.org/10.1016/j.cam.2019.03.002

    Args:
        B (tensor): the tensor to be approximated
        R (int): the desired canonical rank for the approximation
        tol (double): the desired absolute error tolerance
        return_errors (bool): whether to return the error history as
            a second return value
    Returns:
        The computed approximation as a :class:`CanonicalTensor`.
        If `return_errors` is true, instead returns a tuple containing
        the tensor and a list of the error history over the iterations.
    """
    E = asarray(B)
    terms = []
    errors = []

    for j in range(R):
        xs = als1(E)
        terms.append(xs)
        E = E - outer(*xs)
        err = fro_norm(E)
        errors.append(err)
        if err < tol:
            break
    X = CanonicalTensor.from_terms(terms)
    return (X, errors) if return_errors else X


def als1_ls(A, B, tol=1e-15, maxiter=10000, spd=False):
    """Compute rank 1 approximation to the solution of a linear system by Alternating Least Squares."""
    if spd:
        return _als1_ls_spd(A, B, tol=tol, maxiter=maxiter)
    d = B.ndim
    rankA = len(A)
    xs = list(np.random.rand(B.shape[j]) for j in range(d))

    # precompute the sparse matrices Ai^A A_j for each coordinate axis
    AitAj = [[[ (A[i][k].T.dot(A[j][k])).tocsr()
                for j in range(rankA)]
                for i in range(rankA)]
                for k in range(d)]

    for it in range(maxiter):
        delta = 1.0
        for k in range(d):
            ys = _apply_lowrank([_without_k(Ar,k) for Ar in A], _without_k(xs, k))

            # compute left-hand side matrix
            #ZtZ = sum(dot_rank1(ys[i], ys[j]) * A[i][k].T.dot(A[j][k])
            #          for j in range(rankA) for i in range(rankA))
            ZtZ = reduce(operator.add,
                         (_dot_rank1(ys[i], ys[j]) * AitAj[k][i][j]
                             for j in range(rankA) for i in range(rankA)))

            # compute right-hand side
            b = np.zeros(B.shape[k])
            for j in range(rankA):
                # zs = ((A_1 x_1)^T, ..., A_k^T, ..., (A_d x_d)^T)
                zs = [y[None, :] for y in ys[j]]
                zs = zs[:k] + [A[j][k].T] + zs[k:]
                b += apply_tprod(zs, B).ravel()

            # solve least squares problem
            xk = scipy.sparse.linalg.spsolve(ZtZ, b)
            delta *= np.linalg.norm(xs[k] - xk)
            xs[k] = xk

        if delta < tol:
            break
    return xs


def _als1_ls_spd(A, B, tol=1e-15, maxiter=10000):
    """Compute rank 1 approximation to the solution of a linear system by Alternating Least Squares,
    assuming A is spd. Computes the best rank 1 approximation to the solution in the energy norm."""
    d = B.ndim
    rankA = len(A)
    xs = list(np.random.rand(B.shape[j]) for j in range(d))

    for it in range(maxiter):
        delta = 1.0
        for k in range(d):
            Ak = scipy.sparse.csr_matrix(A[0][k].shape)

            for r in range(rankA):
                w = 1.0
                for m in range(d):
                    if m != k:
                        w *= A[r][m].dot(xs[m]).dot(xs[m])
                Ak += w * A[r][k]

            # compute right-hand side
            xts = [x[None,:] for x in xs]
            xts[k] = None
            b = apply_tprod(xts, B).ravel()

            # solve least squares problem
            xk = scipy.sparse.linalg.spsolve(Ak, b)
            delta *= np.linalg.norm(xs[k] - xk)
            xs[k] = xk

        if delta < tol:
            break
    return xs

def als1_ls_structured(A, B, tol=1e-15, maxiter=10000):
    """Compute rank 1 approximation to the solution of a linear system by Alternating Least Squares.

    Faster version of :func:`als1_ls`, but works only if all the matrices in the
    operator `A` have identical sparsity structure.
    """
    d = B.ndim
    rankA = len(A)
    xs = list(np.random.rand(B.shape[j]) for j in range(d))

    # precompute the sparse matrices Ai^A A_j for each coordinate axis
    AitAj = [[[ (A[i][k].T.dot(A[j][k])).tocsr()
                for j in range(rankA)]
                for i in range(rankA)]
                for k in range(d)]

    for it in range(maxiter):
        delta = 1.0
        for k in range(d):
            ys = _apply_lowrank([_without_k(Ar,k) for Ar in A], _without_k(xs, k))

            # compute left-hand side matrix
            ZtZ = AitAj[k][0][0].copy()
            ZtZ.data[:] = 0.0
            for i in range(rankA):
                for j in range(rankA):
                    ZtZ.data += _dot_rank1(ys[i], ys[j]) * AitAj[k][i][j].data

            # compute right-hand side
            b = np.zeros(B.shape[k])
            for j in range(rankA):
                # zs = ((A_1 x_1)^T, ..., A_k^T, ..., (A_d x_d)^T)
                zs = [y[None, :] for y in ys[j]]
                zs = zs[:k] + [A[j][k].T] + zs[k:]
                b += apply_tprod(zs, B).ravel()

            # solve least squares problem
            xk = scipy.sparse.linalg.spsolve(ZtZ, b)
            delta *= np.linalg.norm(xs[k] - xk)
            xs[k] = xk

        if delta < tol:
            break
    return xs


def gta(A, R, tol=1e-12, rtol=1e-12, return_errors=False):
    """Greedy Tucker approximation of the tensor `A`.

    References:
        https://doi.org/10.1016/j.cam.2019.03.002

    Args:
        A (tensor): the tensor to be approximated
        R (int): the desired multilinear rank of the approximation
        tol (double): target absolute error tolerance
        rtol (double): target relative error tolerance
        return_errors (bool): whether to return the error history as
            a second return value
    Returns:
        The computed approximation as a :class:`TuckerTensor`.
        If `return_errors` is true, instead returns a tuple containing
        the tensor and a list of the error history over the iterations.
    """
    if isinstance(A, np.ndarray):
        A = TensorSum(A) # make sure it's a tensor object so A-T works
    us = als1(A)
    U = [u[:,None] / np.linalg.norm(u) for u in us]
    d = A.ndim
    A_norm = fro_norm(A)

    errors = []

    for k in range(R):
        # compute projection of A into the space spanned by U
        X = asarray(apply_tprod(tuple(u.T for u in U), A))
        T = TuckerTensor(U, X)

        E = A - T
        err = fro_norm(E)
        errors.append(err)

        if k == R-1 or err < tol or err < rtol*A_norm:
            break

        vs = als1(E)

        for j in range(d):
            # orthonormalize vs[j]
            y = vs[j] - U[j].dot( U[j].T.dot( vs[j] ))
            ny = np.linalg.norm(y)
            if ny < 1e-14:
                continue    # skip almost zero vectors
            U[j] = np.column_stack((U[j], y / ny))
    return (T, errors) if return_errors else T


def _tucker_sum(terms):
    terms = list(terms)
    A = terms[0]
    for B in terms[1:]:
        U, X1, X2 = join_tucker_bases(A, B)
        A = TuckerTensor(U, X1 + X2)
        if any(u.shape[1] > u.shape[0] for u in A.Us):
            A = A.compress()
    return A

def gta_ls(A, F, R, tol=1e-12, verbose=0, gs=None, spd=False):
    """Greedy Tucker approximation of the solution of a linear system `A U = F`.

    References:
        https://doi.org/10.1016/j.cam.2019.03.002

    Args:
        A (list): the linear operator in low Kronecker rank format given as a
            list of tuples. Each tuple represents a Kronecker product operator and
            contains `d` matrices or linear operators; the operator is considered
            as the Kronecker product of these operators
        F (tensor): the right-hand side of the linear system as a (possibly
            low-rank) tensor
        R (int): the desired multilinear rank of the approximation (number of iterations)
        tol (double): desired reduction of the initial residual
        verbose (int): 0 = no printed output, 1 = moderate detail, 2 = full detail
        gs (int): if this is not None, then this many Gauss-Seidel iterations are used on
            the core linear system instead of direct solution; see the paper for details
        spd (bool): pass True if `A` is a symmetric positive definite operator; uses a
            more efficient and accurate rank 1 approximation algorithm (see the corresponding
            parameter of :func:`als1_ls`)

    Returns:
        the computed approximation as a :class:`TuckerTensor`
    """
    res0_norm = fro_norm(F)

    # start with rank one approximation
    us = als1_ls(A, F, tol=tol, spd=spd)
    U = [u[:,None] / np.linalg.norm(u) for u in us]
    d = F.ndim
    rankA = len(A)
    X = np.zeros(d * (0,))

    for it in range(R):
        # construct reduced linear system in tensor product basis U
        A_U = reduce(operator.add,
                     (_multi_kron([U[k].T.dot(A[j][k].dot(U[k])) for k in range(d)])
                      for j in range(rankA)))
        F_U = apply_tprod([u.T for u in U], F).ravel()
        shpX = tuple(U[k].shape[1] for k in range(d))

        ## solve reduced linear system #########################################
        if gs is not None and A_U.shape[0] > 500:
            # extend previous coefficients with 0 and do Gauss-Seidel iteration
            pad_size = tuple((0, U[k].shape[1] - X.shape[k]) for k in range(d))
            zz = np.pad(X, pad_size, 'constant').ravel()
            from .solvers import gauss_seidel
            gauss_seidel(A_U, zz, F_U, iterations=gs)
        else:
            # do a full direct solve
            zz = np.linalg.solve(A_U, F_U)
        X = zz.reshape(shpX)
        ########################################################################

        UX = TuckerTensor(U, X)

        # stopping criterion: number of iterations
        if it == R - 1:
            return UX

        # compute new residual: first, apply A to UX
        #A_UX = reduce(operator.add, (apply_tprod(Aj, UX) for Aj in A))
        A_UX = _tucker_sum(apply_tprod(Aj, UX) for Aj in A)

        if False:
            if verbose >= 2: print('Compressing A(UX)', A_UX.R, '-> ', end='')
            A_UX = A_UX.compress(rtol=1e-3)
            if verbose >= 2: print(A_UX.R)

        # compute the residual
        Rk = F - A_UX

        if True:
            # recompress it
            if verbose >= 2: print('Compressing residual', Rk.R, '-> ', end='')
            Rk = Rk.compress(rtol=1e-2)
            if verbose >= 2: print(Rk.R)

        # stopping criterion: reduction of initial residual
        res = fro_norm(Rk)
        if verbose >= 2: print('Residual norm:', res)
        if res < tol * res0_norm:
            if verbose >= 1:
                print(it, 'iterations, residual reduction =', res / res0_norm)
            return UX

        # compute rank 1 approximation to A^(-1) Rk
        vs = als1_ls(A, Rk, tol=tol)

        # update tensor product basis U
        for j in range(d):
            # orthonormalize vs
            y = vs[j] - U[j].dot( U[j].T.dot(vs[j]) )
            norm_orth = np.linalg.norm(y)
            #norm_full = np.linalg.norm(vs[j])
            #if norm_orth > 1e-2 * norm_full:
                # only add if not almost orthogonal to old space
            U[j] = np.column_stack((U[j], y / norm_orth))


################################################################################
## Tensor classes
################################################################################

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

    def __repr__(self):
        return '<%s shape=%s R=%s>' % (self.__class__.__name__, self.shape, self.R)

    @staticmethod
    def zeros(shape):
        """Construct a zero canonical tensor with the given shape."""
        return CanonicalTensor(np.zeros((n,0)) for n in shape)

    @staticmethod
    def ones(shape):
        """Construct a constant canonical tensor with all entries one and the given shape."""
        return CanonicalTensor(np.ones((n,1)) for n in shape)

    @staticmethod
    def from_terms(terms):
        """Construct a canonical tensor from a list of rank 1 terms, represented
        as tuples of vectors.
        """
        terms = list(terms)
        d = len(terms[0])
        return CanonicalTensor(
                tuple(np.column_stack([terms[j][k] for j in range(len(terms))])
                    for k in range(d)))

    @staticmethod
    def from_tensor(A):
        """Convert `A` from other tensor formats to canonical format."""
        if isinstance(A, TuckerTensor):
            terms = []
            for index in np.ndindex(*A.R):
                a = A.X[index]
                if abs(a) > 1e-15:
                    xs = tuple(U[:,j] for (U,j) in zip(A.Us, index))
                    terms.append((a * xs[0],) + xs[1:])
            if terms:
                return CanonicalTensor.from_terms(terms)
            else:
                return CanonicalTensor.zeros(A.shape)
        else:
            raise TypeError('conversion from %s to canonical not implemented' % type(A))

    def copy(self):
        """Create a deep copy of this tensor."""
        return CanonicalTensor((X.copy() for X in self.Xs))

    def asarray(self):
        """Convert canonical tensor to a full `ndarray`."""
        X = np.zeros(self.shape)
        for r in range(self.R):
            X += outer(*tuple(X[:,r] for X in self.Xs))
        return X

    def terms(self):
        """Return the rank one components as a list of tuples."""
        for j in range(self.R):
            yield tuple(X[:,j] for X in self.Xs)

    def norm(self):
        """Compute the Frobenius norm of the tensor."""
        return np.sqrt(
            sum(_dot_rank1(ti, tj)
                for ti in self.terms()
                for tj in self.terms()))

    def nway_prod(self, Bs):
        """Implements :func:`apply_tprod` for canonical tensors.

        Returns:
            :class:`CanonicalTensor`: the result in canonical format
        """
        Bs = tuple(Bs)
        if len(Bs) < self.ndim:
            Bs = Bs + (self.ndim - len(Bs)) * (None,)
        elif len(Bs) > self.ndim:
            raise ValueError('more operators than tensor axes (%d vs %d)'
                    % (len(Bs), self.ndim))

        Xs = []
        for j in range(self.ndim):
            if Bs[j] is not None:
                Xs.append(Bs[j].dot(self.Xs[j]))
            else:
                Xs.append(np.array(self.Xs[j]))
        return CanonicalTensor(Xs)

    def ravel(self):
        """Return the vectorization of this tensor."""
        return self.asarray().ravel()

    def __neg__(self):
        A = self.copy()
        return CanonicalTensor((-A.Xs[0],) + A.Xs[1:])

    def __add__(self, T2):
        assert self.shape == T2.shape, 'incompatible shapes'
        if isinstance(T2, CanonicalTensor):
            return CanonicalTensor(
                    (np.hstack((X1,X2)) for (X1,X2) in zip(self.Xs, T2.Xs)))
        elif isinstance(T2, TuckerTensor):
            return TuckerTensor.from_tensor(self) + T2
        elif isinstance(T2, np.ndarray):
            return self.asarray() + T2
        else:
            raise TypeError('cannot add CanonicalTensor and %s' % type(T2))

    def __sub__(self, T2):
        return self + (-T2)

    def squeeze(self, axis=None):
        """Eliminate singleton axes. Equivalent to :func:`numpy.squeeze`."""
        if axis is None:
            axis = tuple(i for i in range(self.ndim) if self.shape[i] == 1)
        else:
            if np.isscalar(axis):
                axis = (axis,)
            if not all(self.shape[i] == 1 for i in axis):
                raise ValueError('all given axes must be singletons!')
        if len(axis) == 0:
            return self
        elif len(axis) == self.ndim:    # singleton tensor
            return self.ravel()[0]      # - return the single entry
        else:
            remaining = sorted(set(range(self.ndim)) - set(axis))
            Xs = tuple(self.Xs[i] for i in remaining)
            # each singleton X has shape 1 x R - multiply them all up
            factors = self.Xs[axis[0]].copy()
            for i in axis[1:]:
                factors *= self.Xs[i]
            # incorporate the factors into the first remaining X
            Xs = (Xs[0] * factors,) + Xs[1:]
            return CanonicalTensor(Xs)

    def __getitem__(self, I):
        I, shp, singl = _normalize_indices(I, self.shape)
        A = CanonicalTensor(X[Ik] for (X,Ik) in zip(self.Xs, I))
        assert A.shape == shp
        return A.squeeze(axis=singl)


class TuckerTensor:
    r"""A *d*-dimensional tensor in **Tucker format** is given as a list of *d* basis matrices

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
    Such a rank compression is implemented in :func:`TuckerTensor.compress`.
    """
    def __init__(self, Us, X):
        self.Us = tuple(Us)
        self.X = X
        self.ndim = len(self.Us)
        assert self.ndim == X.ndim, 'Incompatible sizes'
        self.shape = tuple(U.shape[0] for U in self.Us)
        self.R = self.X.shape

    def __repr__(self):
        return '<%s shape=%s R=%s>' % (self.__class__.__name__, self.shape, self.R)

    @staticmethod
    def zeros(shape):
        """Construct a zero Tucker tensor with the given shape."""
        return TuckerTensor.from_tensor(CanonicalTensor.zeros(shape))

    @staticmethod
    def ones(shape):
        """Construct a constant Tucker tensor with all entries one and the given shape."""
        return TuckerTensor.from_tensor(CanonicalTensor.ones(shape))

    @staticmethod
    def from_tensor(A):
        """Convert `A` from other tensor formats to Tucker format."""
        if isinstance(A, CanonicalTensor):
            X = np.zeros(A.ndim * (A.R,))
            np.fill_diagonal(X, 1.0)
            return TuckerTensor(A.Xs, X)
        elif isinstance(A, TuckerTensor):
            return A
        else:
            # trivial full-rank Tucker representation
            U = tuple(np.eye(n) for n in A.shape)
            return TuckerTensor(U, asarray(A))

    def copy(self):
        """Create a deep copy of this tensor."""
        return TuckerTensor((U.copy() for U in self.Us), self.X.copy())

    def asarray(self):
        """Convert Tucker tensor to a full `ndarray`."""
        return apply_tprod(self.Us, self.X)

    def orthogonalize(self):
        """Compute an equivalent Tucker representation of the current tensor
        where the matrices `U` have orthonormal columns.

        Returns:
            :class:`TuckerTensor`: the orthonormalized Tucker tensor
        """
        QR = tuple(scipy.linalg.qr(U, mode='economic', check_finite=False)
                for U in self.Us)
        Qs = tuple(Q for (Q,_) in QR)
        Rs = tuple(R for (_,R) in QR)
        return TuckerTensor(Qs, apply_tprod(Rs, self.X))

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

    def compress(self, tol=1e-15, rtol=1e-15):
        """Approximate this Tucker tensor by another one of smaller rank, up to
        an absolute error tolerance `tol` or a relative error tolerance `rtol`.

        Returns:
            the approximation as a :class:`TuckerTensor`
        """
        # first, orthogonalize the basis
        T = self.orthogonalize()
        tol = max(tol, fro_norm(T.X) * rtol)
        # we could now simply truncate T.X, but we get better results by computing its HOSVD first
        TXSVD = hosvd(T.X)
        TX2 = TXSVD.truncate(find_truncation_rank(TXSVD.X, tol))
        return TX2.nway_prod(T.Us)

    def nway_prod(self, Bs):
        """Implements :func:`apply_tprod` for Tucker tensors.

        Returns:
            :class:`TuckerTensor`: the result in Tucker format
        """
        Bs = tuple(Bs)
        if len(Bs) < self.ndim:
            Bs = Bs + (self.ndim - len(Bs)) * (None,)
        elif len(Bs) > self.ndim:
            raise ValueError('more operators than tensor axes (%d vs %d)'
                    % (len(Bs), self.ndim))

        Us = []
        for j in range(self.ndim):
            if Bs[j] is not None:
                Us.append(Bs[j].dot(self.Us[j]))
            else:
                Us.append(np.array(self.Us[j]))
        return TuckerTensor(Us, self.X)

    def ravel(self):
        """Return the vectorization of this tensor."""
        return self.asarray().ravel()

    def __add__(self, T2):
        assert T2.shape == self.shape, 'incompatible shapes'
        if isinstance(T2, TuckerTensor):
            U, X1, X2 = join_tucker_bases(self, T2)
            return TuckerTensor(U, X1 + X2)
        elif isinstance(T2, CanonicalTensor):
            return self + TuckerTensor.from_tensor(T2)
        elif isinstance(T2, np.ndarray):
            return self.asarray() + T2
        else:
            raise TypeError('cannot add TuckerTensor and %s' % type(T2))

    def __sub__(self, T2):
        assert T2.shape == self.shape, 'incompatible shapes'
        if isinstance(T2, TuckerTensor):
            U, X1, X2 = join_tucker_bases(self, T2)
            return TuckerTensor(U, X1 - X2)
        else:
            return self + (-T2)

    def __neg__(self):
        return TuckerTensor((U.copy() for U in self.Us), -self.X)

    def squeeze(self, axis=None):
        """Eliminate singleton axes. Equivalent to :func:`numpy.squeeze`."""
        if axis is None:
            axis = tuple(i for i in range(self.ndim) if self.shape[i] == 1)
        else:
            if np.isscalar(axis):
                axis = (axis,)
            if not all(self.shape[i] == 1 for i in axis):
                raise ValueError('all given axes must be singletons!')
        if len(axis) == 0:
            return self
        elif len(axis) == self.ndim:    # singleton tensor
            return self.ravel()[0]      # - return the single entry
        else:
            remaining = sorted(set(range(self.ndim)) - set(axis))
            factors = self.ndim * [None]
            for i in axis:
                factors[i] = self.Us[i]     # has shape 1 x R_i
            X = apply_tprod(factors, self.X).squeeze(axis)
            return TuckerTensor(tuple(self.Us[i] for i in remaining), X)

    def __getitem__(self, I):
        I, shp, singl = _normalize_indices(I, self.shape)
        T = TuckerTensor(tuple(U[Ik] for (U,Ik) in zip(self.Us, I)), self.X)
        assert T.shape == shp
        return T.squeeze(axis=singl)


def join_tucker_bases(T1, T2):
    """Represent the two Tucker tensors `T1` and `T2` in a joint basis.

    Returns:
        tuple: `(U,X1,X2)` such that ``T1 == TuckerTensor(U,X1)`` and
        ``T2 == TuckerTensor(U,X2)``. The basis `U` is the concatenation
        of the bases of `T1` and `T2`.
    """
    assert T1.shape == T2.shape
    # join basis matrices
    U = tuple(np.hstack((U1, U2))
            for (U1,U2) in zip(T1.Us, T2.Us))
    # pad X1 and X2 with zeros
    R1, R2 = T1.X.shape, T2.X.shape
    X1 = np.pad(T1.X, tuple((0,n) for n in R2), 'constant')
    X2 = np.pad(T2.X, tuple((n,0) for n in R1), 'constant')
    return U, X1, X2


class TensorSum:
    """Represents the abstract sum of an arbitrary number of tensors with identical shapes."""
    def __init__(self, *Xs):
        self.Xs = tuple(Xs)
        assert self.Xs, 'cannot form sum of empty list of tensors'
        self.ndim = self.Xs[0].ndim
        self.shape = self.Xs[0].shape
        assert all(X.shape == self.shape for X in self.Xs), 'tensors must have identical shape'

    def __repr__(self):
        return '<%s shape=%s nterms=%s>' % (self.__class__.__name__, self.shape, len(self.Xs))

    def asarray(self):
        """Convert sum of tensors to a full `ndarray`."""
        A = np.array(asarray(self.Xs[0]))
        for X in self.Xs[1:]:
            A += asarray(X)
        return A

    def ravel(self):
        """Return the vectorization of this tensor."""
        return self.asarray().ravel()

    def nway_prod(self, Bs):
        """Implements :func:`apply_tprod` for sums of tensors.

        Returns:
            :class:`TensorSum`: the result as a sum of tensors
        """
        return TensorSum(*(apply_tprod(Bs, X) for X in self.Xs))

    def __add__(self, T2):
        return TensorSum(*(self.Xs + (T2,)))

    def __sub__(self, T2):
        return TensorSum(*(self.Xs + (-T2,)))

    def __neg__(self):
        return TensorSum(*(-X for X in self.Xs))

    def __getitem__(self, I):
        Ys = tuple(X[I] for X in self.Xs)
        if all(np.isscalar(Y) for Y in Ys):
            return sum(Ys)
        else:
            return TensorSum(*Ys)


class TensorProd:
    """Represents the abstract tensor product of an arbitrary number of tensors."""
    def __init__(self, *Xs):
        self.Xs = tuple(Xs)
        shp = ()
        self.slices = []
        for X in self.Xs:
            start = len(shp)
            shp = shp + X.shape
            end = len(shp)
            self.slices.append(slice(start, end))
        self.ndim = len(shp)
        self.shape = shp

    def __repr__(self):
        return '<%s shape=%s nterms=%s>' % (self.__class__.__name__, self.shape, len(self.Xs))

    def asarray(self):
        """Convert sum of tensors to a full `ndarray`."""
        As = tuple(asarray(X) for X in self.Xs)
        return array_outer(*As)

    def ravel(self):
        """Return the vectorization of this tensor."""
        return self.asarray().ravel()

    def nway_prod(self, Bs):
        """Implements :func:`apply_tprod` for tensor products.

        Returns:
            :class:`TensorProd`: the result as a tensor product
        """
        return TensorProd(
                *(apply_tprod(Bs[sl], X) for (sl,X) in zip(self.slices, self.Xs)))

    def __add__(self, T2):
        return TensorSum(self, T2)

    def __sub__(self, T2):
        return TensorSum(self, -T2)

    def __neg__(self):
        return TensorProd(*((-self.Xs[0],) + self.Xs[1:]))

    def __getitem__(self, I):
        if not isinstance(I, tuple):
            I = (I,)
        if len(I) > self.ndim:
            raise ValueError('too many indices')
        if len(I) < self.ndim:
            I = I + (self.ndim - len(I)) * (slice(None),)
        Ys = tuple(X[I[sl]] for (X,sl) in zip(self.Xs, self.slices))
        if all(np.isscalar(Y) for Y in Ys):
            return np.prod(Ys)
        else:
            return TensorProd(*Ys)

################################################################################
## Linear operators on tensors
################################################################################

class CanonicalOperator:
    r"""Represents a linear operator on tensors which is described as a sum
    of rank one operators (Kronecker products), i.e.,

    .. math::
        \mathcal A = \sum_{r=1}^{R} A^1_r \otimes\cdots\otimes A^d_r.

    The argument `terms` is a list of length `R` of `d`-tuples containing the
    matrices :math:`A^k_r`.

    Attributes:
        R (int): Kronecker rank of the operator
        shape (tuple): a pair where `shape[1]` is the shape of input tensors
            accepted by this operator and `shape[0]` is the shape of output
            tensors produced
        ndim (int): the number of dimensions, i.e., `d` in the formula above
    """
    def __init__(self, terms):
        self.terms = list(terms)
        self.R = len(self.terms)
        d = len(self.terms[0])
        assert all(self.terms[r][j].shape == self.terms[0][j].shape
                   for r in range(self.R)
                   for j in range(d)), 'inconsistent shapes'
        shapeout = tuple(A.shape[0] for A in self.terms[0])
        shapein  = tuple(A.shape[1] for A in self.terms[0])
        self.shape = (shapeout, shapein)
        self.ndim = len(shapein)

    def __repr__(self):
        return '<%s %s -> %s R=%s>' % (self.__class__.__name__, self.shape[1], self.shape[0], self.R)

    @staticmethod
    def eye(ns, format='dia'):
        """Represent the identity as a tensor product of identity matrices with sizes
        given by the tuple of integers `ns`.
        """
        return CanonicalOperator([[ scipy.sparse.eye(n, format=format) for n in ns ]])

    def asmatrix(self, format='csr'):
        """Return the raveled form of this operator as a sparse matrix in the given format."""
        X = utils.multi_kron_sparse(self.terms[0], format=format)
        for j in range(1, self.R):
            X += utils.multi_kron_sparse(self.terms[j], format=format)
        return X

    @property
    def T(self):
        """Return the transpose of this operator as a :class:`CanonicalOperator`."""
        return CanonicalOperator([
            tuple(B.T for B in term) for term in self.terms
        ])

    def __add__(self, other):
        assert isinstance(other, CanonicalOperator), 'can only add CanonicalOperators to each other'
        assert self.shape == other.shape, 'incompatible shapes'
        return CanonicalOperator(self.terms + other.terms)

    def __neg__(self):
        return CanonicalOperator([
            (-t[0],) + t[1:] for t in self.terms
        ])

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        assert isinstance(other, CanonicalOperator)
        assert self.shape[1] == other.shape[0], 'incompatible shapes'
        def _alldot(t1, t2):
            return tuple(tj1.dot(tj2) for (tj1,tj2) in zip(t1,t2))
        return CanonicalOperator([
            _alldot(t1,t2) for t1 in self.terms for t2 in other.terms
        ])

    def kron(self, other):
        """Construct a new :class:`CanonicalOperator` as the Kronecker product of this and `other`."""
        return CanonicalOperator([
            tuple(t1) + tuple(t2) for t1 in self.terms for t2 in other.terms
        ])

    def apply(self, X):
        """Return the result of applying this operator to a tensor `X`."""
        assert X.shape == self.shape[1], 'wrong shape of input tensor'
        return reduce(operator.add, (apply_tprod(t, X) for t in self.terms))

    def __matmul__(self, X):
        if isinstance(X, CanonicalOperator):
            return self * X
        else:
            return self.apply(X)

    def slice(self, limits):
        return CanonicalOperator([
            tuple(A[l[0]:l[1], l[0]:l[1]] for (A,l) in zip(term, limits))
            for term in self.terms
        ])
