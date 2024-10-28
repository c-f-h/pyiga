"""Solvers for linear, nonlinear, and time-dependent problems."""
import numpy as np
import scipy.linalg
from .operators import make_solver, KroneckerOperator, DiagonalOperator
from . import utils, algebra

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

def solve_hmultigrid(hs, A, f, strategy='cell_supp', smoother='gs', smooth_steps=2, tol=1e-8, maxiter=5000):
    """Solve a linear scalar problem in a hierarchical spline space using local multigrid.

    Args:
        hs: the :class:`.HSpace` which describes the HB- or THB-spline space
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
        tol (float): the desired reduction in the residual
        maxiter (int): the maximum number of iterations

    Returns:
        a pair `(x, iterations)` containing the solution and the number of
        iterations performed. If `maxiter` was reached without convergence, the
        returned number of iterations is infinite.
    """
    Ps = hs.virtual_hierarchy_prolongators()
    # determine non-Dirichlet dofs (for residual computation)
    non_dir_dofs = hs.non_dirichlet_dofs()
    mg_step = local_mg_step(hs, A, f, Ps, hs.indices_to_smooth(strategy), smoother)
    return iterative_solve(mg_step, A, f, active_dofs=non_dir_dofs, tol=tol, maxiter=maxiter)


## Nonlinear problems

class NoConvergenceError(Exception):
    def __init__(self, method, num_iter, last_iterate):
        self.method = method
        self.num_iter = num_iter
        self.last_iterate = last_iterate

def newton(F, J, x0, atol=1e-6, rtol=1e-6, maxiter=100, freeze_jac=1):
    """Solve the nonlinear problem F(x) == 0 using Newton iteration.

    Args:
        F (function):      function computing the residual of the nonlinear equation
        J (function):      function computing the Jacobian matrix of `F`
        x0 (ndarray):      the initial guess as a vector
        atol (float):      absolute tolerance for the norm of the residual
        rtol (float):      relative tolerance with respect to the initial residual
        maxiter (int):     the maximum number of iterations
        freeze_jac (int):  if >1, the Jacobian is only updated every `freeze_jac` steps

    Returns:
        ndarray: a vector `x` which approximately satisfies F(x) == 0
    """
    x = np.array(x0)
    res = F(x)
    target = max(atol, rtol * np.linalg.norm(res))
    for num_it in range(maxiter):
        if np.linalg.norm(res) < target:    # converged?
            return x
        if num_it % freeze_jac == 0:  # update Jacobian only every freeze_jac steps
            jac = J(x)
            jac_inv = make_solver(jac)
        x -= jac_inv.dot(res)
        res = F(x)
    raise NoConvergenceError('newton', maxiter, x)
    
def pcg(A, f, x0 = None, P = 1, rtol = 1e-5, atol = 0.0, maxiter = 100, output = False):    
    """Solve the the linear system Ax = f by conjugated gradient method.
    
    Args:
        A (LinearOperator or ndarray or sparse matrix):  the symmetric and positive definite matrix of the linear system
        f (ndarray):                                     the right-hand side vector of the system
        x0 (ndarray):                                    initial guess for the solution, by default the zero vector
        P (LinearOperator or ndarray or sparse matrix):  preconditioner to use for the system. by default the identity map.
        rtol (float), atol (float) :                     iteration stops if relative error of residual and initial residual has reached rtol or if absolute error has reached atol
        maxiter (int) :                                  maximum number of iterations if the stopping criterion is not met
        output (boolean) :                               information to be printed after iteration stops
    """
    maxiter = int(maxiter)
    
    if not callable(A):
        Afun = lambda x : A@x
    else:
        Afun = A
        
    if not isinstance(f,np.ndarray):
        f_ = f.A.ravel()
    else:
        f_ = f.ravel()
        
    if x0 is not None:
        if not isinstance(x0, np.ndarray):
            x = x0.A.ravel()
        else:
            x = x0.ravel() 
    else:
        x = np.zeros(len(f_))
        
    if not callable(P):
        if isinstance(P, np.ndarray) or scipy.sparse.issparse(P):
            assert P.shape==2*(len(f),), 'dimension mismatch'
            Pfun = lambda x: P@x
        else:
            Pfun = lambda x : x
    else:
        Pfun = P
        # splu_pfun = sp.linalg.splu(pfuns,permc_spec='COLAMD')
        # pfun = lambda x : splu_pfun.solve(x)
    # print('Cond about',condest(pfuns@Afuns))
    # x, it, delta, gamma, d = solvers_cy.pyx_pcg(Afun, f, x0, Pfun, tol, maxiter)
    r = f_ - Afun(x)
    h = Pfun(r)
    rho = h@r
    err = np.sqrt(rho)
    err0 = np.sqrt(Pfun(f_)@f_)
    d = h
    
    delta = np.zeros(maxiter+1, dtype=float)
    gamma = np.zeros(maxiter,   dtype=float)
    
    if err < max(rtol * err0, atol):
        #L = algebra.LanczosMatrix(delta[:1], gamma[:0])
        delta[0] = (Afun(d)@d)/rho
        if output:
            print('pcg with preconditioned condition number '+ str('\N{greek small letter kappa}')+ ' ~ ' + str(1.) + ' stopped after ' + str(0) + ' iterations with relres ' + str(err/err0))
        return x, 0 , delta[0], delta[0], err

    #while err > max(rtol * err0, atol) and it < maxiter:
    for it in range(maxiter):
        z = Afun(d)
        alpha = rho/(z@d)
        delta[it]+=1/alpha
        x += alpha*d
        r -= alpha*z
        h = Pfun(r)
        rho_old = rho
        rho = h@r
        err = np.sqrt(rho)
        if err < max(rtol * err0, atol):
            break
        beta = rho/rho_old
        d = h + beta*d
        gamma[it] = -np.sqrt(beta)/alpha
        delta[it+1] = beta/alpha
        if it==maxiter-1:
            delta[it+1] += (Afun(d)@d)/rho
        
    #print(delta,gamma)
    eigs = scipy.linalg.eigvalsh_tridiagonal(delta[:(it+1)],gamma[:it])
    m = min(abs(eigs))
    M = max(abs(eigs))
    #L = algebra.LanczosMatrix(delta[:(it+1)], gamma[:(it)])
    # m = L.minEigenvalue()
    # M = L.maxEigenvalue()
    cond = abs(M/m)
    
    if output:
        print('pcg with preconditioned condition number ' + str('\N{greek small letter kappa}')+ ' ~ ' + str(cond) + ' stopped after ' + str(it+1) + ' iterations with relres ' + str(err/err0))
    return x, it+1 , m, M, err


## Time stepping

def dirk_step(A, M, F, J, x, tau, data=None, Fx=None):
    # A: Butcher tableau (including b vector, optionally b_hat vector)
    # M: mass matrix or None
    # F: right-hand side
    # x: current iterate
    # tau: stepsize
    if M is None:
        M = scipy.sparse.eye(x.shape[0])
    if data is None:
        data = dict()
    s = A.shape[1]      # number of stages
    b = A[s, :]
    is_sa = np.allclose(b, A[s-1, :])   # stiffly accurate?
    ys = []
    Fy = []         # list of right-hand sides F(y[i])
    for i in range(s):
        # add up contributions below the diagonal
        a_ii = A[i,i]

        if a_ii == 0: # and (M is None or i == 0):       # explicit step
            # if diagonal is 0 and i == 0, then y_i = x
            # if diagonal is 0 and M=I, we don't need to solve a linear system
            assert i == 0
            ys.append(x)
            if Fx is not None:
                Fy.append(Fx)
            else:
                Fy.append(F(x))
        else:
            # set up right-hand side (constant term in F)
            terms = tau * sum(A[i,j] * Fy[j] for j in range(i))
            rhs = M @ x + terms

            last_Fz = None      # remember F(y_i) to avoid one evaluation of F
            def newton_F(z):
                nonlocal last_Fz
                last_Fz = F(z)
                return M @ z - tau * a_ii * last_Fz - rhs
            def newton_J(z):
                return M - tau * a_ii * J(z)

            # solve the nonlinear system
            x_start = x if (i==0) else ys[-1]
            y_i = newton(newton_F, newton_J, x_start, atol=1e-4, freeze_jac=2)
            ys.append(y_i)
            Fy.append(last_Fz)

    # caching accessor for solver for mass matrix
    # NB: assumes M nonsingular - doesn't work for DAEs!
    def get_Minv():
        if 'M_inv' in data:
            return data['M_inv']
        else:
            M_inv = make_solver(M, spd=True)
            data['M_inv'] = M_inv
            return M_inv

    if is_sa:
        x_new = ys[s-1]
        F_x_new = Fy[s-1]
    else:
        x_new = get_Minv() @ (M @ x + tau * sum(b[i] * Fy[i] for i in range(s)))
        F_x_new = None

    if A.shape[0] == s + 2:   # embedded RK scheme?
        b_hat = A[s + 1, :]
        x_est = get_Minv() @ (M @ x + tau * sum(b_hat[i] * Fy[i] for i in range(s)))
        return x_new, x_est, F_x_new
    else:
        return x_new, F_x_new

def _constant_step_method(stepper):
    def _method(M, F, J, x, tau, t_end, *, t0=0.0, progress=False):
        """
        Args:
            M (matrix): the mass matrix
            F (function): the right-hand side
            J (function): function computing the Jacobian of `F`
            x (vector): the initial value
            tau0 (float): the initial time step
            t_end (float): the final time up to which to integrate
            t0 (float): the initial time; 0 by default
            progress (bool): whether to show a progress bar

        Returns:
            A tuple `(times, solutions)`, where `times` is a list of increasing
            times in the interval `(t0, t_end)`, and `solutions` is a list of
            vectors which contains the computed solutions at these times.
        """
        times = [t0]
        solutions = [x]
        Fx = None
        data = dict()
        from math import ceil
        num_iter = int(ceil((t_end - t0) / tau))
        tqdm = utils.progress_bar(progress)
        for i in tqdm(range(num_iter)):
            try:
                x, Fx = stepper(M, F, J, x, tau, data, Fx=Fx)
            except NoConvergenceError:
                # if Newton failed to converge, return partial results
                print('Nonlinear solve failed; returning partial results')
                return times, solutions
            t = t0 + (i + 1) * tau
            times.append(t)
            solutions.append(x)
        return times, solutions
    return _method

def _adaptive_step_method(stepper, err_order, const_method):
    def _method(M, F, J, x, tau0, t_end, tol, *, t0=0.0, step_factor=0.9, progress=False):
        """
        Args:
            M (matrix): the mass matrix
            F (function): the right-hand side
            J (function): function computing the Jacobian of `F`
            x (vector): the initial value
            tau0 (float): the initial time step
            t_end (float): the final time up to which to integrate
            tol (float): error tolerance for choosing the adaptive time step;
                if `None`, use constant time steps
            t0 (float): the initial time; 0 by default
            step_factor (float): the safety factor for choosing the step size
            progress (bool): whether to show a progress bar

        Returns:
            A tuple `(times, solutions)`, where `times` is a list of increasing
            times in the interval `(t0, t_end)`, and `solutions` is a list of
            vectors which contains the computed solutions at these times.
        """
        if tol is None:
            return const_method(M, F, J, x, tau0, t_end, t0=t0)
        times = [t0]
        solutions = [x]
        Fx = None
        tau = tau0
        data = dict()
        tqdm = utils.progress_bar(progress)
        with tqdm(total=t_end-t0) as pbar:
            t = t0
            while t < t_end:
                try:
                    xnew, xhat, Fxnew = stepper(M, F, J, x, tau, data, Fx=Fx)
                    d = tol + tol * abs(x)
                    r = np.linalg.norm((xhat - xnew) / d) / np.sqrt(len(x))
                    if r == 0: r = 1e-15

                    if r <= 1:
                        # successful step
                        t += tau
                        x = xnew
                        Fx = Fxnew
                        times.append(t)
                        solutions.append(x)
                        # update the progress bar
                        pbar.update(tau)
                        pbar.set_postfix({'tau': tau})

                    # update step size for next step
                    fac = step_factor * r**(-1 / err_order)
                    fac = min(5.0, max(0.2, fac))
                    tau *= fac

                except NoConvergenceError:
                    # Newton failed to converge; reject step
                    tau *= 0.5

        return times, solutions
    return _method

def dirk_method(A, name, displayname):
    def stepper(*args, **kwargs):
        return dirk_step(A, *args, **kwargs)
    f = _constant_step_method(stepper)
    f.__name__ = f.__qualname__ = name
    f.__doc__ = ('Solve a time-dependent problem using the {} method.\n'
        .format(displayname) + f.__doc__)
    return f

def adaptive_dirk_method(A, err_order, name, displayname):
    # define non-adaptive fallback
    const_method = dirk_method(A[:-1, :], name, displayname)

    def stepper(*args, **kwargs):
        return dirk_step(A, *args, **kwargs)
    f = _adaptive_step_method(stepper, err_order, const_method)
    f.__name__ = f.__qualname__ = name
    f.__doc__ = ('Solve a time-dependent problem using the {} method.\n'
        .format(displayname) + f.__doc__)
    return f

def coeffs_sdirk3():
    # Skvortsov 2006; Alexander 1977
    gamma = 0.435866521508
    b2 = 1/4 * (5 - 20*gamma + 6*gamma**2)
    A = np.array([
        [gamma,       0.0,    0.0],
        [(1-gamma)/2, gamma,  0.0],
        [1-b2-gamma,  b2,     gamma],
        ##########
        [1-b2-gamma,  b2,     gamma],
    ])
    return A

def coeffs_sdirk3_b():
    # Nørsett's three-stage, 4th order DIRK method
    # NB: not stiffly accurate
    xi = 0.128886400515
    A = np.array([
        [xi,        0.0,   0.0],
        [1/2 - xi,   xi,   0.0],
        [2*xi, 1 - 4*xi,    xi],
        ##########
        [1 / (6*(2*xi-1)**2), 2 * (6*xi**2 - 6*xi +1) / (3*(2*xi-1)**2), 1 / (6*(2*xi-1)**2)]
    ])
    return A

def coeffs_sdirk21():
    # Ellsiepen; order 2, embedded rule of order 1
    alpha = 1 - np.sqrt(2)/2
    alp_hat = 2 - 5/4 * np.sqrt(2)
    A = np.array([
        [alpha,     0.0],
        [1 - alpha, alpha],
        ##################
        [1 - alpha, alpha],
        [1 - alp_hat, alp_hat]
    ])
    return A, 1

def coeffs_dirk34():
    # 4 stages, order 3, L-stable, stiffly accurate
    # embedded rule has order 2
    a21 = a22 = a33 = a44 = 0.1558983899988677
    a32 = 1.072486270734370
    a31 = 1 - a32 - a22
    a42 = 0.7685298292769537
    a43 = 0.09666483609791597
    A = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [a21, a22, 0.0, 0.0],
        [a31, a32, a33, 0.0],
        [0.0, a42, a43, a44],
        ####################
        [0.0, a42, a43, a44],
        [a31, a32, a33, 0.0],
    ])
    return A, 2

def coeffs_esdirk23():
    # Jorgensen et al 2018
    # https://arxiv.org/pdf/1803.01613.pdf
    # 3 stages, order 2, A- and L-stable, stiffly accurate
    # embedded method has order 3 (not stable)
    gamma = (2 - np.sqrt(2)) / 2
    A = np.array([
        [0.0,         0.0,         0.0],
        [gamma,       gamma,       0.0],
        [(1-gamma)/2, (1-gamma)/2, gamma],
        ##########
        [(1-gamma)/2, (1-gamma)/2, gamma],
        [(6*gamma-1)/(12*gamma), 1/(12*gamma*(1-2*gamma)), (1-3*gamma)/(3*(1-2*gamma))],
    ])
    return A, 3

def coeffs_esdirk34():
    # Jorgensen et al 2018
    # https://arxiv.org/pdf/1803.01613.pdf
    # 4 stages, order 3, A- and L-stable, stiffly accurate
    # embedded method has order 4 (not stable)
    a21 = 0.43586652150845899942
    a31 = 0.14073777472470619619
    a32 = -0.1083655513813208000
    gam = 0.43586652150845899942
    b = [
      0.10239940061991099768,
      -0.3768784522555561061,
      0.83861253012718610911,
      gam
    ]
    b_hat = [
      0.15702489786032493710,
      0.11733044137043884870,
      0.61667803039212146434,
      0.10896663037711474985
    ]
    A = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [a21, gam, 0.0, 0.0],
        [a31, a32, gam, 0.0],
        b,
        ####################
        b,
        b_hat
    ])
    return A, 4


crank_nicolson = dirk_method(np.array([
    [0.0, 0.0],
    [0.5, 0.5],
    ##########
    [0.5, 0.5]
]), 'crank_nicolson', 'Crank-Nicolson')

sdirk3 = dirk_method(coeffs_sdirk3(), 'sdirk3', 'SDIRK3 Runge-Kutta')
sdirk3_b = dirk_method(coeffs_sdirk3_b(), 'sdirk3_b', 'SDIRK3 (alternate) Runge-Kutta')
sdirk21 = adaptive_dirk_method(*coeffs_sdirk21(), 'sdirk21', 'SDIRK21 (Ellsiepen) Runge-Kutta')
dirk34 = adaptive_dirk_method(*coeffs_dirk34(), 'dirk34', 'DIRK34 Runge-Kutta')

# methods from https://arxiv.org/pdf/1803.01613.pdf
esdirk23 = adaptive_dirk_method(*coeffs_esdirk23(), 'esdirk23', 'ESDIRK23 Runge-Kutta')
esdirk34 = adaptive_dirk_method(*coeffs_esdirk34(), 'esdirk34', 'ESDIRK34 Runge-Kutta')

## Rosenbrock methods

# All of these are referenced in http://dx.doi.org/10.1016/j.cma.2009.10.005

def rosenbrock_step(A, Gamma, b, b_hat, M, F, J, x, tau, data, Fx=None):
    gamma = Gamma[0,0]    # assume all entries on the diagonal are the same

    jac = J(x)   # Jacobian at initial time
    C = M - tau * gamma * jac
    C_inv = make_solver(C)

    ks = []
    s = A.shape[0]
    for i in range(s):
        y_i = x + tau * sum(A[i,j] * ks[j] for j in range(i))
        rhs = F(y_i)
        if i > 0:
            w_i = sum(Gamma[i,j] * ks[j] for j in range(i))
            rhs += tau * jac.dot(w_i)
        k_i = C_inv.dot(rhs)
        ks.append(k_i)
    x_new = x + tau * sum(b[i] * ks[i] for i in range(s))

    if b_hat is not None:
        x_est = x + tau * sum(b_hat[i] * ks[i] for i in range(s))
        return x_new, x_est, None
    else:
        return x_new, None

def coeffs_ros3p():
    A = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])

    gam = 0.7886751347
    g21 = -1.0
    g31 = -0.7886751347
    g32 = -1.077350269

    Gamma = np.array([
        [gam, 0.0, 0.0],
        [g21, gam, 0.0],
        [g31, g32, gam],
    ])

    b = np.array([2/3, 0, 1/3])
    b_hat = np.array([1/3, 1/3, 1/3])
    return A, Gamma, b, b_hat, 2

def coeffs_ros3pw():
    a21 = 1.5773502691896257e+00
    a31 = 5.0000000000000000e-01
    a32 = 0.0000000000000000e+00

    gam = 7.8867513459481287e-01
    g21 = -1.5773502691896257e+00
    g31 = -6.7075317547305480e-01
    g32 = -1.7075317547305482e-01

    b1 = 1.0566243270259355e-01
    b2 = 4.9038105676657971e-02
    b3 = 8.4529946162074843e-01

    b_hat1 = -1.7863279495408180e-01
    b_hat2 =  3.3333333333333333e-01
    b_hat3 =  8.4529946162074843e-01

    A = np.array([
        [0.0, 0.0, 0.0],
        [a21, 0.0, 0.0],
        [a31, a32, 0.0],
    ])
    Gamma = np.array([
        [gam, 0.0, 0.0],
        [g21, gam, 0.0],
        [g31, g32, gam],
    ])
    b = np.array([b1, b2, b3])
    b_hat = np.array([b_hat1, b_hat2, b_hat3])

    return A, Gamma, b, b_hat, 2

def coeffs_rowdaind2():
    a21 = 0.5
    a31 = 0.28
    a32 = 0.72
    a41 = 0.28
    a42 = 0.72
    a43 = 0.0

    A = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [a21, 0.0, 0.0, 0.0],
        [a31, a32, 0.0, 0.0],
        [a41, a42, a43, 0.0],
    ])

    gam = 0.3
    g21 = -1.121794871794876e-1
    g31 = 2.54
    g32 = -3.84
    g41 = 29.0/75.0
    g42 = -0.72
    g43 = 1.0/30.0
    Gamma = np.array([
        [gam, 0.0, 0.0, 0.0],
        [g21, gam, 0.0, 0.0],
        [g31, g32, gam, 0.0],
        [g41, g42, g43, gam],
    ])

    b1 = 2.0/3.0
    b2 = 0.0
    b3 = 1.0/30.0
    b4 = 0.3

    b_hat1 = 4.799002800355166e-1
    b_hat2 = 5.176203811215082e-1
    b_hat3 = 2.479338842975209e-3
    b_hat4 = 0.0

    b = np.array([b1, b2, b3, b4])
    b_hat = np.array([b_hat1, b_hat2, b_hat3, b_hat4])

    return A, Gamma, b, b_hat, 2

def coeffs_rodasp():
    gamma = 0.25

    a21 = 0.75
    a31 = 8.6120400814152190E-2
    a32 = 0.1238795991858478
    a41 = 0.7749345355073236
    a42 = 0.1492651549508680
    a43 = -0.2941996904581916
    a51 = 5.308746682646142
    a52 = 1.330892140037269
    a53 = -5.374137811655562
    a54 = -0.2655010110278497
    a61 = -1.764437648774483
    a62 = -0.4747565572063027
    a63 = 2.369691846915802
    a64 = 0.6195023590649829
    a65 = 0.25
    A = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [a21, 0.0, 0.0, 0.0, 0.0, 0.0],
        [a31, a32, 0.0, 0.0, 0.0, 0.0],
        [a41, a42, a43, 0.0, 0.0, 0.0],
        [a51, a52, a53, a54, 0.0, 0.0],
        [a61, a62, a63, a64, a65, 0.0],
    ])

    b21 = 0.0
    b31 = -0.049392
    b32 = -0.014112
    b41 = -0.4820494693877561
    b42 = -0.1008795555555556
    b43 =  0.9267290249433117

    b51 = -1.764437648774483
    b52 = -0.4747565572063027
    b53 =  2.369691846915802
    b54 =  0.6195023590649829

    b61 = -8.0368370789113464E-2
    b62 = -5.6490613592447572E-2
    b63 =  0.4882856300427991
    b64 =  0.5057162114816189
    b65 = -0.1071428571428569

    B = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [b21, 0.0, 0.0, 0.0, 0.0, 0.0],
        [b31, b32, 0.0, 0.0, 0.0, 0.0],
        [b41, b42, b43, 0.0, 0.0, 0.0],
        [b51, b52, b53, b54, 0.0, 0.0],
        [b61, b62, b63, b64, b65, 0.0],
    ])

    np.fill_diagonal(B, gamma)
    Gamma = B - A

    b     = np.array([b61, b62, b63, b64, b65, gamma])
    b_hat = np.array([b51, b52, b53, b54, gamma, 0])
    return A, Gamma, b, b_hat, 3

def coeffs_rosi2p1():
    a21 = 5.0000000000000000e-1
    a31 = 5.5729261836499822e-1
    a32 = 1.9270738163500176e-1
    a41 =-3.0084516445435860e-1
    a42 = 1.8995581939026787e+0
    a43 =-5.9871302944832006e-1

    A = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [a21, 0.0, 0.0, 0.0],
        [a31, a32, 0.0, 0.0],
        [a41, a42, a43, 0.0],
    ])

    gam = 4.3586652150845900e-1
    g21 =-5.0000000000000000e-1
    g31 =-6.4492162993321323e-1
    g32 = 6.3491801247597734e-2
    g41 = 9.3606009252719842e-3
    g42 =-2.5462058718013519e-1
    g43 =-3.2645441930944352e-1

    Gamma = np.array([
        [gam, 0.0, 0.0, 0.0],
        [g21, gam, 0.0, 0.0],
        [g31, g32, gam, 0.0],
        [g41, g42, g43, gam],
    ])

    b1  = 5.2900072579103834e-2
    b2  = 1.3492662311920438e+0
    b3  =-9.1013275270050265e-1
    b4  = 5.0796644892935516e-1

    bh1 = 1.4974465479289098e-1
    bh2 = 7.0051069041421810e-1
    bh3 = 0.0000000000000000e+0
    bh4 = 1.4974465479289098e-1

    b = np.array([b1, b2, b3, b4])
    b_hat = np.array([bh1, bh2, bh3, bh4])

    return A, Gamma, b, b_hat, 2

def rosenbrock_method(A, Gamma, b, name, displayname):
    def stepper(*args, **kwargs):
        return rosenbrock_step(A, Gamma, b, None, *args, **kwargs)
    f = _constant_step_method(stepper)
    f.__name__ = f.__qualname__ = name
    f.__doc__ = ('Solve a time-dependent problem using the {} method.\n'
        .format(displayname) + f.__doc__)
    return f

def adaptive_rosenbrock_method(A, Gamma, b, b_hat, err_order, name, displayname):
    # define non-adaptive fallback
    const_method = rosenbrock_method(A, Gamma, b, name, displayname)

    def stepper(*args, **kwargs):
        return rosenbrock_step(A, Gamma, b, b_hat, *args, **kwargs)
    f = _adaptive_step_method(stepper, err_order, const_method)
    f.__name__ = f.__qualname__ = name
    f.__doc__ = ('Solve a time-dependent problem using the {} method.\n'
        .format(displayname) + f.__doc__)
    return f

ros3p = adaptive_rosenbrock_method(*coeffs_ros3p(), 'ros3p', 'ROS3P Rosenbrock')
ros3pw = adaptive_rosenbrock_method(*coeffs_ros3pw(), 'ros3pw', 'ROS3PW Rosenbrock')
rowdaind2 = adaptive_rosenbrock_method(*coeffs_rowdaind2(), 'rowdaind2', 'ROWDAIND2 Rosenbrock')
rodasp = adaptive_rosenbrock_method(*coeffs_rodasp(), 'rodasp', 'RODASP Rosenbrock')
rosi2p1 = adaptive_rosenbrock_method(*coeffs_rosi2p1(), 'rosi2p1', 'ROSI2P1 Rosenbrock')

# import sys
# import time
# import numpy as np
# import numba as nb
# import scipy.sparse as sps
# from numba import float64,float32,int64

# import importlib.util
# spam_spec = importlib.util.find_spec("sksparse")
# found = spam_spec is not None

# from scipy.sparse.linalg import splu

# def fastBlockInverse2(Mh):
#     spluMh = splu(Mh)
#     L = spluMh.L; U = spluMh.U

#     Pv2 = spluMh.perm_r
#     Pv3 = spluMh.perm_c

#     P2 = sps.csc_matrix((np.ones(Pv2.size),(np.r_[0:Pv2.size],Pv2)), shape = (Pv2.size,Pv2.size))
#     P3 = sps.csc_matrix((np.ones(Pv3.size),(np.r_[0:Pv3.size],Pv3)), shape = (Pv3.size,Pv3.size))
    
#     L = L.tocsc()
#     UT = (U.T).tocsc()
    
    
#     #####################################################################################
#     # Find indices where the blocks begin/end
#     #####################################################################################
#     tm = time.time()
    
#     L_diag = L.diagonal(k=-1) # Nebendiagonale anfangen
#     block_ends_L = np.r_[np.argwhere(abs(L_diag)==0)[:,0],L.shape[0]-1]
    
#     for i in range(L.shape[0]):
#         L_diag = np.r_[L.diagonal(k=-(i+2)),np.zeros(i+2)]
        
#         for j in range(i+1):
#             arg = np.argwhere(abs(L_diag[block_ends_L-j])>0)[:,0]
#             block_ends_L = np.delete(block_ends_L,arg).copy()
            
#         if np.linalg.norm(L_diag)==0: break
    
#     block_ends_L = np.r_[0,block_ends_L+1]
    
#     #####################################################################################
    
    
#     #####################################################################################
#     # Find indices where the blocks begin/end
#     #####################################################################################
    
#     UT_diag = UT.diagonal(k=-1) # Nebendiagonale anfangen
#     block_ends_UT = np.r_[np.argwhere(abs(UT_diag)==0)[:,0],UT.shape[0]-1]
    
#     for i in range(UT.shape[0]):
#         UT_diag = np.r_[UT.diagonal(k=-(i+2)),np.zeros(i+2)]
        
#         for j in range(i+1):
#             arg = np.argwhere(abs(UT_diag[block_ends_UT-j])>0)[:,0]
#             block_ends_UT = np.delete(block_ends_UT,arg).copy()
            
#         if np.linalg.norm(UT_diag)==0: break
    
#     block_ends_UT = np.r_[0,block_ends_UT+1]
    
#     #####################################################################################
    
#     tm = time.time()
#     data_iUT,indices_iUT,indptr_iUT = createIndicesInversion(UT.data,UT.indices,UT.indptr,block_ends_UT)
#     iUT = sps.csc_matrix((data_iUT, indices_iUT, indptr_iUT), shape = UT.shape)
    
#     data_iL,indices_iL,indptr_iL = createIndicesInversion(L.data,L.indices,L.indptr,block_ends_L)
#     iL = sps.csc_matrix((data_iL, indices_iL, indptr_iL), shape = L.shape)
    
#     iMh = P3@(iUT.T@iL)@P2.T
#     iMh.data = iMh.data*(np.abs(iMh.data)>1e-13)
#     iMh.eliminate_zeros()
    
#     return iMh#P3@(iUT.T@iL)@P2.T

# if found == True:
#     from sksparse.cholmod import cholesky
#     def fastBlockInverse(Mh):
        
#         cholMh = cholesky(Mh)
#         N = cholMh.L()
#         Pv = cholMh.P()
#         P = sps.csc_matrix((np.ones(Pv.size),(np.r_[0:Pv.size],Pv)), shape = (Pv.size,Pv.size))
#         N = N.tocsc()
        
#         #####################################################################################
#         # Find indices where the blocks begin/end
#         #####################################################################################
        
#         tm = time.time()
        
#         N_diag = N.diagonal(k=-1) # Nebendiagonale anfangen
#         block_ends = np.r_[np.argwhere(abs(N_diag)==0)[:,0],N.shape[0]-1]
        
#         for i in range(N.shape[0]):
#             N_diag = np.r_[N.diagonal(k=-(i+2)),np.zeros(i+2)]
            
#             for j in range(i+1):
#                 arg = np.argwhere(abs(N_diag[block_ends-j])>0)[:,0]
#                 block_ends = np.delete(block_ends,arg).copy()
                
#             if np.linalg.norm(N_diag)==0: break
        
#         block_ends = np.r_[0,block_ends+1]
        
        
#         #####################################################################################
#         # Inversion of the blocks, 2nd try.
#         #####################################################################################
        
#         tm = time.time()
#         data_iN,indices_iN,indptr_iN = createIndicesInversion(N.data,N.indices,N.indptr,block_ends)
#         iN = sps.csc_matrix((data_iN, indices_iN, indptr_iN), shape = N.shape)
#         iMh = P.T@(iN.T@iN)@P
#         return iMh



# @nb.njit(cache = True, parallel = True, fastmath = False)
# def createIndicesInversion(dataN,indicesN,indptrN,block_ends) -> (float64[:],int64[:],int64[:]):

#     block_lengths = block_ends[1:]-block_ends[0:-1]
    
#     sbl = np.sum(block_lengths)+1
#     sbl2 = np.sum(block_lengths**2)
    
#     blicum = np.zeros(block_lengths.size+1, dtype = np.int64)
#     bli2cum = np.zeros(block_lengths.size+1, dtype = np.int64)
    
#     for z in range(block_lengths.size):
#         blicum[z+1] = blicum[z] + block_lengths[z]
#         bli2cum[z+1] = bli2cum[z] + block_lengths[z]**2
        
#     C = np.zeros(sbl2)
#     indptr_iN = np.zeros(sbl, dtype = int64)
#     indices_iN = np.zeros(sbl2, dtype = int64)
    
#     blis = 0; blis2 = 0
    
#     for i in range(block_lengths.size):
        
#         blis = blicum[i]
#         blis2 = bli2cum[i]
        
#         bli = block_lengths[i]
#         bei = block_ends[i]
        
#         blis2p1 = blis2 + bli**2
        
#         CC = np.zeros(shape = (bli,bli), dtype = np.float64)
        
#         for k in range(bli):
#             in_k = np.arange(indptrN[bei+k],indptrN[bei+k+1])
#             for _,jj in enumerate(in_k):
#                 CC[k,indicesN[jj]-bei] = dataN[jj]
                                
#             indptr_iN[k+blis+1] = blis2+bli*(k+1)
#             indices_iN[blis2+bli*np.repeat(k,bli)+np.arange(0,bli)] = np.arange(bei,bei+bli)
        
#         iCCflat = np.linalg.inv(CC).flatten()
#         C[blis2:blis2p1] = iCCflat
#     return C,indices_iN,indptr_iN