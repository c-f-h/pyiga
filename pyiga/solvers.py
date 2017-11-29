"""Linear solvers."""
import numpy as np
import numpy.linalg
from .operators import make_solver

## Smoothers

def OperatorSmoother(A, S):
    """A smoother which applies an arbitrary operator `S` to the residual
    and uses the result as an update, i.e.,

    .. math::
        u \leftarrow S(f - Au).
    """
    def apply(u, f):
        u += S.dot(f - A.dot(u))
    return apply

def GaussSeidelSmoother(A, iterations=1, sweep='forward'):
    """Gauss-Seidel smoother.

    By default, `iterations` is 1. The direction to be used is specified by
    `sweep` and may be either 'forward', 'backward', or 'symmetric'."""
    from .relaxation import gauss_seidel
    def apply(u, f):
        gauss_seidel(A, u, f, iterations=iterations, sweep=sweep)
    return apply

def SequentialSmoother(smoothers):
    """Smoother which applies several smoothers in sequence."""
    def apply(u, f):
        for S in smoothers:
            S(u, f)
    return apply


## Multigrid

def twogrid(A, f, P, smoother, u0=None, tol=1e-8, smooth_steps=2, maxiter=1000):
    """Generic two-grid method with arbitrary smoother.

    Args:
        A: stiffness matrix on fine grid
        f: right-hand side
        P: prolongation matrix from coarse to fine grid
        smoother: a function with arguments `(u,f)` which applies one smoothing iteration in-place to `u`
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
            smoother(u, f)

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
