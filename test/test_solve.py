# integration tests for solving PDEs

import numpy as np
from pyiga import bspline, geometry, assemble, solvers, approx

def test_poisson_2d():
    kvs = 2 * (bspline.make_knots(3, 0.0, 1.0, 10),)
    geo = geometry.quarter_annulus()

    def g(x, y):    # exact solution / boundary data
        return np.cos(x + y) + np.exp(y - x)
    def f(x, y):    # right-hand side (-Laplace of g)
        return 2 * (np.cos(x + y) - np.exp(y - x))

    dir_boundaries = [ 'left', 'right', 'top', 'bottom' ]    # entire boundary

    # compute Dirichlet values from function g
    bcs = assemble.combine_bcs(
            assemble.compute_dirichlet_bc(kvs, geo, bcside, g)
            for bcside in dir_boundaries
    )

    # compute right-hand side from function f
    rhs = assemble.inner_products(kvs, f, f_physical=True, geo=geo).ravel()
    A = assemble.stiffness(kvs, geo=geo)
    LS = assemble.RestrictedLinearSystem(A, rhs, bcs)

    u_sol = solvers.make_solver(LS.A, spd=True).dot(LS.b)
    u = LS.complete(u_sol)
    u_ex = approx.project_L2(kvs, g, f_physical=True, geo=geo).ravel()

    rms_err = np.sqrt(np.mean((u - u_ex)**2))
    assert rms_err < 5e-5       # error: about 4.83e-05
