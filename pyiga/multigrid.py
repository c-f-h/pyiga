"""Multigrid solvers."""
import numpy as np
import numpy.linalg
from .operators import make_solver

tol = 1e-8

def twogrid(A, f, P, smoother, u0=None, tol=1e-8, smooth_steps=2, maxiter=1000):
    """Generic two-grid method with arbitrary operator smoother.

    Args:
        A: stiffness matrix on fine grid
        f: right-hand side
        P: prolongation matrix from coarse to fine grid
        smoother: linear operator to use in smoothing iteration
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

    def apply_smoother(u):
        r = f - A.dot(u)
        u += smoother.dot(r)

    while True:
        for _ in range(smooth_steps):
            apply_smoother(u)

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
