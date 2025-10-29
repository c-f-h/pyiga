# -*- coding: utf-8 -*-
"""Methods for approximating functions in spline spaces."""
from . import bspline
from . import assemble
from . import tensor
from . import operators
from . import utils
from . import hierarchical

import sys
import numpy as np
import scipy.sparse.linalg

def interpolate(kvs, func, geo=None, nodes=None):
    """Perform interpolation in a spline space.

    Returns the coefficients for the interpolant of the function `f` in the
    tensor product B-spline basis `kvs`.

    By default, `f` is assumed to be defined in the parameter domain. If a
    geometry is passed in `geo`, interpolation is instead done in physical
    coordinates.

    `nodes` should be a tensor grid (i.e., a sequence of one-dimensional
    arrays) in the parameter domain specifying the interpolation nodes. If not
    specified, the Gréville abscissae are used.

    It is possible to pass an array of function values for `f` instead of a
    function; they should have the proper shape and correspond to the function
    values at the `nodes`. In this case, `geo` is ignored.
    """
    if isinstance(kvs, bspline.KnotVector):
        kvs = (kvs,)
    if nodes is None:
        nodes = [kv.greville() for kv in kvs]

    # evaluate f at interpolation nodes?
    if isinstance(func, np.ndarray):
        # check that leading dimensions match the number of dofs
        if np.shape(func)[:len(kvs)] != tuple(kv.numdofs for kv in kvs):
            raise ValueError('array func has wrong shape')
        rhs = func
    else:
        if geo is not None:
            rhs = utils.grid_eval_transformed(func, nodes, geo)
        else:
            rhs = utils.grid_eval(func, nodes)

    Cinvs = [operators.make_solver(bspline.collocation(kvs[i], nodes[i]))
                for i in range(len(kvs))]
    return tensor.apply_tprod(Cinvs, rhs)

def interpolate_normal(kvs, bdspec, func, func_deriv, geo=None, nodes=None):
    print(2)
    ax, sd = bspline._parse_bdspec(bdspec,2)[0]
    sup = kvs[ax].support()
    bkvs = assemble.boundary_kv(kvs, bdspec)
    dofs = assemble.boundary_dofs(kvs, bdspec, ravel = True, layer=1)
    if nodes is None:
        nodes = [kv.greville() for kv in bkvs]
        nodes = nodes[:ax] + [np.array([sup[0] if sd==0 else sup[-1]])] + nodes[ax+1:]
    
    dim = len(kvs)
    m = np.prod([len(nodes) for nodes in nodes])
    n = len(dofs)
        
    if isinstance(func, np.ndarray):
        # check that leading dimensions match the number of dofs
        if np.shape(func)[:len(kvs)] != tuple(kv.numdofs for kv in kvs):
            raise ValueError('array f has wrong shape')
        rhs1 = func
    else:
        if geo is not None:
            rhs1 = utils.grid_eval_transformed(func, nodes, geo)
        else:
            rhs1 = utils.grid_eval(func, nodes)

    if isinstance(func_deriv, np.ndarray):
        # check that leading dimensions match the number of dofs
        if np.shape(func_deriv)[:len(kvs)] != tuple(kv.numdofs for kv in kvs):
            raise ValueError('array f has wrong shape')
        rhs2 = func_deriv
    else:
        if geo is not None:
            rhs2 = utils.grid_eval_transformed(func_deriv, nodes, geo)
        else:
            rhs2 = utils.grid_eval(func_deriv, nodes)

    C, D = bspline.collocation_derivs_tp(kvs, nodes, derivs=1)
        
    C = C[0].tocsr()[:,dofs]
    for i in range(dim):
        D[i] = D[i].tocsr()[:,dofs]
    N=geo.boundary(bdspec).grid_outer_normal(nodes[:ax]+nodes[ax+1:], reference_point=geo.center()).reshape(m,dim)
    
    J=geo.grid_jacobian(nodes).reshape(m,dim,dim)
       
    invJ=np.array([np.linalg.inv(jac) for jac in J[:]]) ###TODO: compute inverse manually for 2D and not call np.linalg.inv
    
    NC=scipy.sparse.csr_matrix((m, n))
    for i in range(dim):
        NC_ = scipy.sparse.csr_matrix((m, n))
        for j in range(dim):
            NC_ += scipy.sparse.spdiags(invJ[:,j,i], 0, m, m)*D[dim-1-j]
        NC += scipy.sparse.spdiags(N[:,i], 0, m, m)*NC_

    A = scipy.sparse.vstack([C, NC])
    return operators.make_solver(A)@np.r_[rhs1,rhs2]
    

def _project_L2_hspace(hs, f, f_physical=False, geo=None):
    from . import vform, geometry
    if geo is None:
        geo = geometry.identity(hs.knotvectors(0))
    M = assemble.assemble(vform.mass_vf(hs.dim), hs, geo=geo)
    rhs = assemble.assemble(vform.L2functional_vf(hs.dim, physical=f_physical),
            hs, geo=geo, f=f)
    return operators.make_solver(M, spd=True).dot(rhs)

def project_L2(kvs, f, f_physical=False, geo=None):
    """Perform :math:`L_2`-projection into a spline space.

    Returns the coefficients for the :math:`L_2`-projection of the function `f`
    into the tensor product B-spline basis `kvs`. Optionally, a geometry
    transform `geo` can be specified to compute the projection in a physical
    domain.

    By default, `f` is assumed to be defined in the parameter domain. If it is
    given in physical coordinates, pass `f_physical=True`. This requires `geo`
    to be specified.

    This function also supports projection into a hierarchical spline space by
    passing a :class:`.HSpace` object in place of `kvs`.
    """
    if isinstance(kvs, hierarchical.HSpace):
        return _project_L2_hspace(kvs, f, f_physical, geo)
    elif isinstance(kvs, bspline.KnotVector):
        kvs = (kvs,)
    Minvs = [operators.make_solver(assemble.mass(kv), spd=True) for kv in kvs]
    rhs = assemble.inner_products(kvs, f, f_physical=f_physical, geo=geo)
    if geo is None:
        assert not f_physical, 'Cannot use physical coordinates without geometry'
        # in the parameter domain, we simply apply the Kronecker product of the M^{-1}
        return tensor.apply_tprod(Minvs, rhs)
    else:
        # in the physical domain, use the Kronecker product as a preconditioner
        M = assemble.mass(kvs, geo=geo)
        b = rhs.ravel()
        assert b.shape[0] == M.shape[1], 'L2 projection with geometry only implemented for scalar functions'
        x, info = scipy.sparse.linalg.cg(M, b, rtol=1e-12, atol=1e-12,
                maxiter=100, M=operators.KroneckerOperator(*Minvs))
        if info:
            print('WARNING: L2 projection - CG did not converge:', info, file=sys.stderr)
        return x.reshape(rhs.shape)
