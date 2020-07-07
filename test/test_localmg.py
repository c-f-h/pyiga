#!/usr/bin/env python
# coding: utf-8

import numpy as np
import scipy.linalg
from pyiga import bspline, assemble, hierarchical, solvers, vform, geometry, utils

def local_mg_step(hs, A, f_in, Ps, lv_inds, smoother='symmetric_gs'):
    assert smoother in ("forward_gs", "backward_gs", "symmetric_gs", "exact"), "Invalid smoother."
    As = [A]
    for P in reversed(Ps):
        As.append(P.T.dot(As[-1]).dot(P).tocsr())
    As.reverse()
    Rs = [P.T for P in Ps]

    Bs = [] # exact solvers

    for lv in range(hs.numlevels):
        lv_ind = lv_inds[lv]
        Bs.append(solvers.make_solver(As[lv][lv_ind][:, lv_ind], spd=True))

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
            if smoother == "forward_gs":
                # Gauss-Seidel smoothing
                solvers.gauss_seidel(A, x1, f, indices=lv_ind, iterations=2, sweep='forward')
            elif smoother == "backward_gs":
                # Gauss-Seidel smoothing
                solvers.gauss_seidel(A, x1, f, indices=lv_ind, iterations=2, sweep='backward')
            elif smoother == "symmetric_gs":
                # Gauss-Seidel smoothing
                solvers.gauss_seidel(A, x1, f, indices=lv_ind, iterations=1, sweep='symmetric')
            elif smoother == "exact":
                # exact solve
                r_fine = (f - A.dot(x1))[lv_ind]
                x1[lv_ind] += Bs[lv].dot(r_fine)

            # coarse grid correction
            r = f - A.dot(x1)
            r_c = Rs[lv-1].dot(r)
            aux = step(lv-1, np.zeros_like(r_c), r_c)
            x1 += P.dot(aux)

            # post-smoothing
            if smoother == "forward_gs":
                # Gauss-Seidel smoothing
                solvers.gauss_seidel(A, x1, f, indices=lv_ind, iterations=2, sweep='backward')
            elif smoother == "backward_gs":
                # Gauss-Seidel smoothing
                solvers.gauss_seidel(A, x1, f, indices=lv_ind, iterations=2, sweep='forward')
            elif smoother == "symmetric_gs":
                # Gauss-Seidel smoothing
                solvers.gauss_seidel(A, x1, f, indices=lv_ind, iterations=1, sweep='symmetric')
            elif smoother == "exact":
                # exact solve
                #r_fine = (f - A.dot(x1))[lv_ind]
                #x1[lv_ind] += Bs[lv].dot(r_fine)
                pass
            return x1
    return lambda x: step(hs.numlevels-1, x, f_in)

from .test_hierarchical import create_example_hspace

def run_local_multigrid(p, dim, n0, disparity, smoother, strategy, tol):
    hs = create_example_hspace(p, dim, n0, disparity, num_levels=3)

    # assemble full tensor-product linear system on each level for simplicity

    def rhs(*x):
        return 1.0

    # assemble and solve the HB-spline problem
    hdiscr = hierarchical.HDiscretization(hs, vform.stiffness_vf(dim=2), {'geo': geometry.unit_square()})
    A_hb = hdiscr.assemble_matrix()
    f_hb = hdiscr.assemble_rhs(rhs)

    LS_hb = assemble.RestrictedLinearSystem(A_hb, f_hb,
            (hs.smooth_dirichlet[-1], np.zeros_like(hs.smooth_dirichlet[-1])))
    u_hb = scipy.sparse.linalg.spsolve(LS_hb.A, LS_hb.b)
    u_hb0 = LS_hb.complete(u_hb)

    # assemble and solve the THB-spline problem
    hdiscr = hierarchical.HDiscretization(hs, vform.stiffness_vf(dim=2), {'geo': geometry.unit_square()},
            truncate=True)
    A_thb = hdiscr.assemble_matrix()
    f_thb = hdiscr.assemble_rhs(rhs)

    LS_thb = assemble.RestrictedLinearSystem(A_thb, f_thb,
            (hs.smooth_dirichlet[-1], np.zeros_like(hs.smooth_dirichlet[-1])))
    u_thb = scipy.sparse.linalg.spsolve(LS_thb.A, LS_thb.b)
    u_thb0 = LS_thb.complete(u_thb)

    # iteration numbers of the local MG method in the (T)HB basis
    P_hb = hs.virtual_hierarchy_prolongators()
    P_thb = [
            hs.truncate_one_level(k, num_rows=P_hb[k].shape[0], inverse=True) @ P_hb[k]
            for k in range(hs.numlevels - 1)]
    inds = hs.indices_to_smooth(strategy)
    spek_hb  = num_iterations(local_mg_step(hs, A_hb, f_hb, P_hb, inds, smoother), u_hb0, tol=tol)
    spek_thb = num_iterations(local_mg_step(hs, A_thb, f_thb, P_thb, inds, smoother), u_thb0, tol=tol)

    winner = "HB" if (spek_hb <= spek_thb) else "THB"
    linestr = f'{strategy} ({smoother}) '.ljust(2*22)
    print(linestr + f'{spek_hb:5}    {spek_thb:5}    {winner:6}')
    return (spek_hb, spek_thb)

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
