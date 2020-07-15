#!/usr/bin/env python
# coding: utf-8

import numpy as np
import scipy.linalg
from pyiga import bspline, assemble, hierarchical, solvers, vform, geometry, utils

from .test_hierarchical import create_example_hspace

def run_local_multigrid(p, dim, n0, disparity, smoother, strategy, tol):
    hs = create_example_hspace(p, dim, n0, disparity, num_levels=3)
    dir_dofs = hs.dirichlet_dofs()

    # assemble full tensor-product linear system on each level for simplicity

    def rhs(*x):
        return 1.0

    # assemble and solve the HB-spline problem
    hdiscr = hierarchical.HDiscretization(hs, vform.stiffness_vf(dim=2),
            {'geo': geometry.unit_square(), 'f': rhs})
    A_hb = hdiscr.assemble_matrix()
    f_hb = hdiscr.assemble_rhs()

    LS_hb = assemble.RestrictedLinearSystem(A_hb, f_hb,
            (dir_dofs, np.zeros_like(dir_dofs)))
    u_hb = scipy.sparse.linalg.spsolve(LS_hb.A, LS_hb.b)
    u_hb0 = LS_hb.complete(u_hb)

    # assemble and solve the THB-spline problem
    hdiscr = hierarchical.HDiscretization(hs, vform.stiffness_vf(dim=2),
            {'geo': geometry.unit_square(), 'f': rhs},
            truncate=True)
    A_thb = hdiscr.assemble_matrix()
    f_thb = hdiscr.assemble_rhs()

    LS_thb = assemble.RestrictedLinearSystem(A_thb, f_thb,
            (dir_dofs, np.zeros_like(dir_dofs)))
    u_thb = scipy.sparse.linalg.spsolve(LS_thb.A, LS_thb.b)
    u_thb0 = LS_thb.complete(u_thb)

    # iteration numbers of the local MG method in the (T)HB basis
    P_hb = hs.virtual_hierarchy_prolongators()
    P_thb = [
            hs.truncate_one_level(k, num_rows=P_hb[k].shape[0], inverse=True) @ P_hb[k]
            for k in range(hs.numlevels - 1)]
    inds = hs.indices_to_smooth(strategy)
    iter_hb  = num_iterations(solvers.local_mg_step(hs, A_hb, f_hb, P_hb, inds, smoother), u_hb0, tol=tol)
    iter_thb = num_iterations(solvers.local_mg_step(hs, A_thb, f_thb, P_thb, inds, smoother), u_thb0, tol=tol)

    winner = "HB" if (iter_hb <= iter_thb) else "THB"
    linestr = f'{strategy} ({smoother}) '.ljust(2*22)
    print(linestr + f'{iter_hb:5}    {iter_thb:5}    {winner:6}')
    return (iter_hb, iter_thb)

def num_iterations(step, sol, tol=1e-8):
    x = np.zeros_like(sol)
    for iterations in range(1, 20000):
        x = step(x)

        if scipy.linalg.norm(x-sol) < tol:
            return iterations
    return np.inf

################################################################################

def test_localmg():
    tol = 1e-8
    dim = 2
    n0 = 6

    print("---------------------------------------------------------")
    print("dim =", dim, "n0 =", n0)
    print("---------------------------------------------------------")

    # "exact", "symmetric_gs", "forward_gs", "backward_gs", "symmetric_gs"
    smoother = "symmetric_gs"

    p = 3

    results = dict()
    for disparity in (np.inf, 1):
        results[disparity] = []
        linestr = (f"[p = {p} disparity = {disparity}]").ljust(2*22)
        print(linestr + "{:8s} {:8s} {:8s}".format("HB", "THB", "Winner"))
        # available strategies: "new", "trunc", "func_supp", "cell_supp", "global"
        for strategy in ("new", "trunc", "func_supp", "cell_supp"):
            results[disparity].append(
                    run_local_multigrid(p, dim, n0, disparity, smoother, strategy, tol)
            )

    assert np.array_equal(results[np.inf],
            [ ( 107, 118 ),
              (  49,  19 ),
              (  49,  15 ),
              (  41,  15 ) ])

    assert np.array_equal(results[1],
            [ ( 105, 104 ),
              (  59,  23 ),
              (  59,  23 ),
              (  61,  22 ) ])

def test_solve_hmultigrid():
    # test the built-in solve_hmultigrid function in pyiga.solvers
    hs = create_example_hspace(p=3, dim=2, n0=10, disparity=1, num_levels=3)

    # assemble and solve the HB-spline problem
    hdiscr = hierarchical.HDiscretization(hs, vform.stiffness_vf(dim=2),
            {'geo': geometry.unit_square(), 'f': lambda *x: 1.0})
    A_hb = hdiscr.assemble_matrix()
    f_hb = hdiscr.assemble_rhs()

    dir_dofs = hs.dirichlet_dofs()
    LS_hb = assemble.RestrictedLinearSystem(A_hb, f_hb,
            (dir_dofs, np.zeros_like(dir_dofs)))
    u_hb = scipy.sparse.linalg.spsolve(LS_hb.A, LS_hb.b)
    u_hb0 = LS_hb.complete(u_hb)

    u_mg, iters = solvers.solve_hmultigrid(hs, A_hb, f_hb, tol=1e-8) # use default parameters for smoother and strategy
    assert np.allclose(u_hb0, u_mg)
