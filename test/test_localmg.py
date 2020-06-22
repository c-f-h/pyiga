#!/usr/bin/env python
# coding: utf-8

import numpy as np
import scipy.linalg
from pyiga import bspline, assemble, hierarchical, solvers

def virtual_hierarchy_prolongators(hs):
    # compute tensor product prolongators
    Ps = tuple(hs.tp_prolongation(lv, kron=True) for lv in range(hs.numlevels-1))

    # indices of active and deactivated basis functions per level
    IA = hs.active_indices()
    ID = hs.deactivated_indices()
    # indices of all functions in the refinement region per level
    IR = tuple(np.concatenate((iA,iD)) for (iA,iD) in zip(IA,ID))

    # number of active and deactivated dofs per level
    na = tuple(len(ii) for ii in IA)
    nr = tuple(len(ii) for ii in IR)

    prolongators = []
    prolongators_THB = []
    for lv in range(hs.numlevels - 1):
        n_coarse = sum(na[:lv+1])
        P_hb = scipy.sparse.bmat((
          (scipy.sparse.eye(n_coarse), None),
          (None,                       Ps[lv][IR[lv+1]][:, ID[lv]])
        ), format='csc')
        prolongators.append(P_hb)

        # map HB-coefficients to THB-coefficients 0:na[0]->n_coarse, IA[0]-> 1:na[1]->nr[lv+1], IA[1]->IR[lv+1]
        if lv > 0:
            P = scipy.sparse.bmat([
                [Ps[lv] @ P, Ps[lv][:, IA[lv]]]
            ], format='csc')
        else:
            P = Ps[lv][:, IA[lv]]

        T_h2t = scipy.sparse.bmat([
            [scipy.sparse.eye(n_coarse), None],
            [P[IR[lv+1]],      scipy.sparse.eye(nr[lv+1])]
        ], format='csc')

        # P_thb: maps coarse coefficients to THB coefficients
        P_thb = T_h2t @ P_hb
        prolongators_THB.append(P_thb)

    return prolongators, prolongators_THB

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

def create_example_hspace(p, dim, n0, disparity, num_levels=3):
    hs = hierarchical.HSpace(dim * (bspline.make_knots(p, 0.0, 1.0, n0),))
    hs.disparity = disparity
    #hs.bdspecs = []
    hs.bdspecs = [(0,0), (0,1), (1,0), (1,1)] if dim==2 else [(0,0),(0,1)]
    # perform local refinement
    delta = 0.5
    for lv in range(num_levels):
        hs.refine_region(lv, lambda *X: min(X) > 1 - delta**(lv+1))
    return hs

def run_local_multigrid(p, dim, n0, disparity, smoother, strategy, tol):
    hs = create_example_hspace(p, dim, n0, disparity, num_levels=3)

    # assemble full tensor-product linear system on each level for simplicity
    kvs = tuple(hs.knotvectors(lv) for lv in range(hs.numlevels))
    #As = [assemble.stiffness(kv)+assemble.mass(kv) for kv in kvs]
    As = [assemble.stiffness(kv) for kv in kvs]

    def rhs(*x): 
        return 1.0

    fs = [assemble.inner_products(kv, rhs).ravel() for kv in kvs]
    prolongators, prolongators_THB = virtual_hierarchy_prolongators(hs)
    
    # assemble and solve the HB-spline problem

    # I_hb: maps HB-coefficients to fine coefficients
    I_hb = hs.represent_fine()

    A_hb = (I_hb.T @ As[-1] @ I_hb).tocsr()
    f_hb = I_hb.T @ fs[-1]

    LS_hb = assemble.RestrictedLinearSystem(A_hb, f_hb, (hs.smooth_dirichlet[-1], np.zeros_like(hs.smooth_dirichlet[-1])))
    A_hb_D = LS_hb.A.A
    u_hb = scipy.linalg.solve(A_hb_D, LS_hb.b)
    u_hb0 = LS_hb.complete(u_hb)
    
    # compute THB-stiffness matrix

    # I_thb: maps THB coeffs to fine coeffs
    I_thb = hs.represent_fine(truncate=True)

    A_thb = I_thb.T @ As[-1] @ I_thb
    f_thb = I_thb.T @ fs[-1]

    LS_thb = assemble.RestrictedLinearSystem(A_thb, f_thb, (hs.smooth_dirichlet[-1], np.zeros_like(hs.smooth_dirichlet[-1])))
    A_thb_D = LS_thb.A.A
    u_thb = scipy.linalg.solve(A_thb_D, LS_thb.b)
    u_thb0 = LS_thb.complete(u_thb)

    # iteration numbers of the local MG method in the (T)HB basis
    inds = hs.indices_to_smooth(strategy)
    spek_hb  = num_iterations(local_mg_step(hs, A_hb, f_hb, prolongators, inds, smoother), u_hb0, tol=tol)
    spek_thb = num_iterations(local_mg_step(hs, A_thb, f_thb, prolongators_THB, inds, smoother), u_thb0, tol=tol)

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
