"""Linear solvers."""
import numpy as np
import numpy.linalg
import scipy.linalg
from .operators import make_solver, KroneckerOperator, DiagonalOperator

from functools import reduce


def _asdense(X):
    try:
        return X.A
    except:
        return X


def fastdiag_solver(KM):
    """The fast diagonalization solver as described in [Sangalli, Tani 2016].

    Args:
        KM: a sequence of length `d` (dimension of the problem) containing pairs
            of symmetric matrices `(K_i, M_i)`

    Returns:
        A `LinearOperator` which realizes the inverse of the generalized Laplacian
        matrix described by the input matrices.
    """
    dim = len(KM)
    n = tuple(K.shape[0] for (K,_) in KM)
    EV = [scipy.linalg.eigh(_asdense(K), _asdense(M)) for (K,M) in KM]

    diags = []
    for d in range(dim):
        D = [np.ones(n[j]) for j in range(dim)]
        D[d] = EV[d][0]  # eigenvalues
        diags.append(reduce(np.kron, D))
    diag = sum(diags)

    l_op = KroneckerOperator(*tuple(U   for (_,U) in EV))
    r_op = KroneckerOperator(*tuple(U.T for (_,U) in EV))

    return l_op * DiagonalOperator(1.0 / diag) * r_op


## Smoothers

def gauss_seidel(A, x, b, iterations=1, indices=None, sweep='forward'):
    """Perform Gauss-Seidel relaxation on the linear system `Ax=b`, updating `x` in place.

    Args:
        A: the matrix; either sparse or dense, but should be in CSR format if sparse
        x: the current guess for the solution
        b: the right-hand side
        iterations: the number of iterations to perform
        indices: if given, relaxation is only performed on this list of indices
        sweep: the direction; either 'forward', 'backward', or 'symmetric'
    """
    if sweep == 'symmetric':
        for i in range(iterations):
            gauss_seidel(A, x, b, iterations=1, indices=indices, sweep='forward')
            gauss_seidel(A, x, b, iterations=1, indices=indices, sweep='backward')
        return

    if sweep not in ('forward', 'backward'):
        raise ValueError("valid sweep directions are 'forward', 'backward', and 'symmetric'")

    if scipy.sparse.issparse(A):
        from . import relaxation_cy
        if not scipy.sparse.isspmatrix_csr(A):
            import warnings
            warnings.warn('matrix for Gauss-Seidel is not CSR; converting (performance warning)', RuntimeWarning)
            A = scipy.sparse.csr_matrix(A)

        if indices is not None:
            indices = np.asanyarray(indices, dtype=np.intc)
            reverse = (sweep == 'backward')
            for i in range(iterations):
                relaxation_cy.gauss_seidel_indexed(A.indptr, A.indices, A.data, x, b, indices, reverse)
        else:
            N = A.shape[0]
            start,end,step = ((0,N,1) if sweep=='forward' else (N-1,-1,-1))
            for i in range(iterations):
                relaxation_cy.gauss_seidel(A.indptr, A.indices, A.data, x, b, start, end, step)

    else:
        if indices is None:
            indices = range(A.shape[0])
        if sweep == 'backward':
            indices = list(reversed(indices))

        for k in range(iterations):
            for i in indices:
                z = A[i].dot(x)
                a = A[i,i]
                z -= a * x[i]
                x[i] = (b[i] - z) / a


def OperatorSmoother(S):
    r"""A smoother which applies an arbitrary operator `S` to the residual
    and uses the result as an update, i.e.,

    .. math::
        u \leftarrow u + S(f - Au).
    """
    def apply(A, u, f):
        u += S.dot(f - A.dot(u))
    return apply

def GaussSeidelSmoother(iterations=1, sweep='forward'):
    """Gauss-Seidel smoother.

    By default, `iterations` is 1. The direction to be used is specified by
    `sweep` and may be either 'forward', 'backward', or 'symmetric'."""
    def apply(A, u, f):
        gauss_seidel(A, u, f, iterations=iterations, sweep=sweep)
    return apply

def SequentialSmoother(smoothers):
    """Smoother which applies several smoothers in sequence."""
    def apply(A, u, f):
        for S in smoothers:
            S(A, u, f)
    return apply


## Multigrid

def twogrid(A, f, P, smoother, u0=None, tol=1e-8, smooth_steps=2, maxiter=1000):
    """Generic two-grid method with arbitrary smoother.

    Args:
        A: stiffness matrix on fine grid
        f: right-hand side
        P: prolongation matrix from coarse to fine grid
        smoother: a function with arguments `(A,u,f)` which applies one smoothing iteration in-place to `u`
        u0: starting value; 0 if not given
        tol: desired reduction relative to initial residual
        smooth_steps: number of smoothing steps
        maxiter: maximum number of iterations

    Returns:
        ndarray: the computed solution to the equation `Au = f`
    """
    A_c = (P.T.dot(A).dot(P)) #.A
    A_c_inv = make_solver(A_c)

    u = np.array(u0) if u0 else np.zeros(A.shape[0])
    res0 = np.linalg.norm(f - A.dot(u))
    numiter = 0

    while True:
        for _ in range(smooth_steps):
            smoother(A, u, f)

        # coarse-grid correction
        r = f - A.dot(u)
        res = np.linalg.norm(r)
        u += P.dot(A_c_inv * P.T.dot(r))

        numiter += 1

        if res < tol * res0:
            break
        elif res > 20 * res0:
            print('Diverged')
            break
        elif numiter > maxiter:
            print('too many iterations, aborting. reduction =', res/res0)
            break
    print(numiter, 'iterations')
    return u

def local_mg_step(hs, A, f, Ps, lv_inds, smoother='symmetric_gs', smooth_steps=2):
    # Utility function for solve_hmultigrid which returns a function step(x)
    # which realizes one V-cycle of the local multigrid method.
    assert smoother in ('gs', 'forward_gs', 'backward_gs', 'symmetric_gs', 'exact'), 'Invalid smoother'
    As = [A]
    for P in reversed(Ps):
        As.append(P.T.dot(As[-1]).dot(P).tocsr())
    As.reverse()

    Bs = [] # exact solvers

    exact_levels = range(hs.numlevels) if smoother=='exact' else [0]
    for lv in exact_levels:
        lv_ind = lv_inds[lv]
        Bs.append(make_solver(As[lv][lv_ind][:, lv_ind], spd=True))

    def step(lv, x, f):
        if lv == 0:
            x1 = x.copy()
            lv_ind = lv_inds[lv]
            x1[lv_ind] = Bs[0].dot(f[lv_ind])
            return x1
        else:
            x1 = x.copy()
            P = Ps[lv-1]
            A = As[lv]
            n_lv = A.shape[0]
            lv_ind = lv_inds[lv]

            # pre-smoothing
            if smoother == "gs":
                # Gauss-Seidel smoothing
                gauss_seidel(A, x1, f, indices=lv_ind, iterations=smooth_steps, sweep='forward')
            elif smoother == "forward_gs":
                # forward Gauss-Seidel smoothing
                gauss_seidel(A, x1, f, indices=lv_ind, iterations=smooth_steps, sweep='forward')
            elif smoother == "backward_gs":
                # backward Gauss-Seidel smoothing
                gauss_seidel(A, x1, f, indices=lv_ind, iterations=smooth_steps, sweep='backward')
            elif smoother == "symmetric_gs":
                # Gauss-Seidel smoothing
                gauss_seidel(A, x1, f, indices=lv_ind, iterations=smooth_steps, sweep='symmetric')
            elif smoother == "exact":
                # exact solve
                r_fine = (f - A.dot(x1))[lv_ind]
                x1[lv_ind] += Bs[lv].dot(r_fine)

            # coarse grid correction
            r = f - A.dot(x1)
            r_c = P.T.dot(r)
            x1 += P.dot(step(lv-1, np.zeros_like(r_c), r_c))

            # post-smoothing
            if smoother == "gs":
                # Gauss-Seidel smoothing
                gauss_seidel(A, x1, f, indices=lv_ind, iterations=smooth_steps, sweep='backward')
            elif smoother == "forward_gs":
                # Gauss-Seidel smoothing
                gauss_seidel(A, x1, f, indices=lv_ind, iterations=smooth_steps, sweep='forward')
            elif smoother == "backward_gs":
                # Gauss-Seidel smoothing
                gauss_seidel(A, x1, f, indices=lv_ind, iterations=smooth_steps, sweep='backward')
            elif smoother == "symmetric_gs":
                # Gauss-Seidel smoothing
                gauss_seidel(A, x1, f, indices=lv_ind, iterations=smooth_steps, sweep='symmetric')
            elif smoother == "exact":
                pass # exact solve - no post-smoothing
            return x1
    return lambda x: step(hs.numlevels-1, x, f)

def iterative_solve(step, A, f, x0=None, active_dofs=None, tol=1e-8, maxiter=5000):
    """Solve the linear system Ax=f using a basic iterative method.

    Args:
        step (callable): a function which performs the update x_old -> x_new for
            the iterative method
        A: matrix or linear operator describing the linear system of equations
        f (ndarray): the right-hand side
        x0: the starting vector; 0 is used if not specified
        active_dofs (list or ndarray): list of active dofs on which the residual
            is computed. Useful for eliminating Dirichlet dofs without changing
            the matrix. If not specified, all dofs are active.
        tol (float): the desired reduction in the Euclidean norm of the residual
            relative to the starting residual
        maxiter (int): the maximum number of iterations

    Returns:
        a pair `(x, iterations)` containing the solution and the number of
        iterations performed. If `maxiter` was reached without convergence, the
        returned number of iterations is infinite.
    """
    if active_dofs is None:
        active_dofs = slice(A.shape[0])    # all dofs are active
    if x0 is None:
        x = np.zeros(A.shape[0])
        res0 = f
    else:
        x = x0
        res0 = f - A @ x
    res0 = scipy.linalg.norm(res0[active_dofs])
    iterations = 0
    while True:
        x = step(x)
        r = f - A @ x       # compute new residual
        res = scipy.linalg.norm(r[active_dofs])
        iterations += 1
        if res / res0 < tol:
            return x, iterations
        elif iterations >= maxiter:
            print("Warning: iterative solver did not converge in {} iterations".format(iterations))
            return x, np.inf

def solve_hmultigrid(hs, A, f, strategy='cell_supp', smoother='gs', smooth_steps=2, truncate=False, tol=1e-8, maxiter=5000):
    """Solve a linear scalar problem in a hierarchical spline space using local multigrid.

    Args:
        hs: the :class:`.HSpace` which describes the hierarchical spline space
        A: the matrix describing the discretization of the problem
        f: the right-hand side vector
        strategy (string): how to choose the smoothing sets. Valid options are

            - ``"new"``: only the new dofs per level
            - ``"trunc"``: all dofs which interact via the truncation operator
            - ``"cell_supp"``: all dofs whose support intersects that of the
              new ones (support extension)
            - ``"func_supp"``

        smoother (string): the multigrid smoother to use. Valid options are

            - ``"gs"``: forward Gauss-Seidel for pre-smoothing, backward
              Gauss-Seidel for post-smoothing
            - ``"forward_gs"``: always use forward Gauss-Seidel
            - ``"backward_gs"``: always use backward Gauss-Seidel
            - ``"symmetric_gs"``: use complete symmetric Gauss-Seidel sweep for
              both pre- and post-smoothing
            - ``"exact"``: use an exact direct solver as a pre-smoother (no
              post-smoothing)

        smooth_steps (int): the number of pre- and post-smoothing steps
        truncate (bool): if True, the linear system is interpreted as a
            THB-spline discretization rather than an HB-spline one
        tol (float): the desired reduction in the residual
        maxiter (int): the maximum number of iterations

    Returns:
        a pair `(x, iterations)` containing the solution and the number of
        iterations performed. If `maxiter` was reached without convergence, the
        returned number of iterations is infinite.
    """
    Ps = hs.virtual_hierarchy_prolongators(truncate=truncate)
    # determine non-Dirichlet dofs (for residual computation)
    non_dir_dofs = hs.non_dirichlet_dofs()
    mg_step = local_mg_step(hs, A, f, Ps, hs.indices_to_smooth(strategy), smoother)
    return iterative_solve(mg_step, A, f, active_dofs=non_dir_dofs, tol=tol, maxiter=maxiter)
