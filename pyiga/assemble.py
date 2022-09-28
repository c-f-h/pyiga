# -*- coding: utf-8 -*-
"""Assembling functions for B-spline IgA.

This module contains functions to assemble stiffness matrices and load vectors
for isogeometric discretizations. Furthermore, it provides support for
computing and eliminating Dirichlet dofs from a linear system.


.. _gauss-asms:

Assembling general problems
---------------------------

General variational forms can be assembled using the following function.
See the section :doc:`/guide/vforms` for further details.

.. autofunction:: assemble
.. autofunction:: assemble_vf
.. autofunction:: assemble_entries

.. autoclass:: Assembler
    :members:


Tensor product Gauss quadrature assemblers
------------------------------------------

Standard Gauss quadrature assemblers for mass and stiffness matrices.
They take one or two arguments:

- `kvs` (list of :class:`.KnotVector`):
  Describes the tensor product B-spline basis for which to assemble
  the matrix. One :class:`KnotVector` per coordinate direction.
  In the 1D case, a single :class:`.KnotVector` may be passed
  directly.
- `geo` (:class:`.BSplineFunc` or :class:`.NurbsFunc`; optional):
  Geometry transform, mapping from the parameter domain to the
  physical domain. If omitted, assume the identity map; a fast
  Kronecker product implementation is used in this case.

.. autofunction:: mass
.. autofunction:: stiffness


.. _fast-asms:

Fast low-rank assemblers
------------------------

Fast low-rank assemblers based on the paper
"A Black-Box Algorithm for Fast Matrix Assembly in Isogeometric Analysis".
They may achieve significant speedups over the classical Gauss assemblers,
in particular for fine discretizations and higher spline degrees.
They only work well if the geometry transform is rather smooth so that the
resulting matrix has relatively low (numerical) Kronecker rank.

They take the following additional arguments:

- `tol`: the stopping accuracy for the Adaptive Cross Approximation (ACA)
  algorithm
- `maxiter`: the maximum number of ACA iterations
- `skipcount`: terminate after finding this many successive near-zero pivots
- `tolcount`: terminate after finding this many successive pivots below the
  desired accuracy
- `verbose`: the amount of output to display. `0` is silent, `1` prints
  basic information, `2` prints detailed information

.. autofunction:: mass_fast
.. autofunction:: stiffness_fast


Right-hand sides
----------------

.. autofunction:: inner_products


Boundary and initial conditions
-------------------------------

.. autofunction:: compute_dirichlet_bcs
.. autofunction:: compute_dirichlet_bc
.. autofunction:: compute_initial_condition_01
.. autofunction:: combine_bcs

.. autoclass:: RestrictedLinearSystem
    :members:


Multipatch problems
-------------------

.. autoclass:: Multipatch
    :members:


Integration
-----------

.. autofunction:: integrate

"""
import numpy as np
import scipy
import scipy.sparse
import itertools
import math

from . import bspline
from . import assemble_tools
from . import assemblers
from . import fast_assemble_cy
from . import tensor
from . import operators
from . import utils
from . import geometry

from .quadrature import make_iterated_quadrature, make_tensor_quadrature
from .mlmatrix import MLStructure

################################################################################
# 1D assembling routines
################################################################################

def _assemble_element_matrices(nspans, nqp, vals1, vals2, qweights):
    assert nspans * nqp == vals1.shape[1]
    assert nspans * nqp == vals2.shape[1]
    assert qweights.shape == (vals1.shape[1],)
    n_act1,n_act2 = vals1.shape[0],vals2.shape[0]
    elMats = np.empty((nspans, n_act1, n_act2)) # contains one n_act1 x n_act2 element matrix per span
    for k in range(nspans):
        f1 = vals1[:,nqp*k:nqp*(k+1)]       # n_act1 x nqp
        f2 = vals2[:,nqp*k:nqp*(k+1)]       # n_act2 x nqp
        w = qweights[nqp*k:nqp*(k+1)]       # nqp-vector of weights
        elMats[k, :, :] = np.dot( f1, (f2 * w).transpose() )
    return elMats       # n_act1*nspans x n_act2

def _create_coo_1d_from_kv(kv):
    n_act1 = n_act2 = kv.p + 1
    nspans = kv.numspans
    grid = np.mgrid[:n_act1, :n_act2] # 2 x n_act1 x n_act2 array which indexes element matrix
    I_ref = grid[0].ravel()          # slowly varying index, first basis
    J_ref = grid[1].ravel()          # fast varying index, second basis

    first_act = kv.first_active(kv.mesh_span_indices())
    first_act = np.repeat(first_act, n_act1*n_act2)
    I = first_act + np.tile(I_ref, nspans)
    J = first_act + np.tile(J_ref, nspans)
    return (I, J)

def _create_coo_1d_custom(nspans, n_act1, n_act2, first_act1, first_act2):
    """Create COO indices for two sequentially numbered bases over `nspans` knot spans"""
    grid = np.mgrid[:n_act1, :n_act2] # 2 x n_act1 x n_act2 array which indexes element matrix
    I_ref = grid[0].ravel()          # slowly varying index, first basis
    J_ref = grid[1].ravel()          # fast varying index, second basis

    I = np.repeat(first_act1, n_act1*n_act2) + np.tile(I_ref, nspans)
    J = np.repeat(first_act2, n_act1*n_act2) + np.tile(J_ref, nspans)
    return (I, J)

def _assemble_matrix_custom(nspans, nqp, vals1, vals2, I, J, qweights):
    elMats = _assemble_element_matrices(nspans, nqp, vals1, vals2, qweights)
    return scipy.sparse.coo_matrix((elMats.ravel(), (I, J))).tocsr()

def bsp_mass_1d(knotvec, weightfunc=None):
    """Assemble the mass matrix for the B-spline basis over the given knot vector.

    Optionally, a weight function can be passed; by default, it is assumed to be 1.
    """
    return bsp_mixed_deriv_biform_1d(knotvec, 0, 0, weightfunc=weightfunc)

def bsp_stiffness_1d(knotvec, weightfunc=None):
    """Assemble the Laplacian stiffness matrix for the B-spline basis over the given knot vector.

    Optionally, a weight function can be passed; by default, it is assumed to be 1.
    """
    return bsp_mixed_deriv_biform_1d(knotvec, 1, 1, weightfunc=weightfunc)

def bsp_mixed_deriv_biform_1d(knotvec, du, dv, nqp=None, weightfunc=None):
    """Assemble the matrix for a(u,v)=(weight*u^(du),v^(dv)) for the B-spline basis over the given knot vector"""
    nspans = knotvec.numspans
    # default: use that q-term Gauss quadrature is exact up to poly degree 2q-1
    if nqp is None: nqp = int(math.ceil((2 * knotvec.p - du - dv + 1) / 2.0))
    q = make_iterated_quadrature(knotvec.mesh, nqp)
    derivs = bspline.active_deriv(knotvec, q[0], max(du, dv))
    qweights = q[1]
    if weightfunc is not None:
        qweights *= utils.grid_eval(weightfunc, (q[0],))
    I,J = _create_coo_1d_from_kv(knotvec)
    return _assemble_matrix_custom(nspans, nqp, derivs[dv, :, :], derivs[du, :, :], I, J, qweights)

def bsp_mixed_deriv_biform_1d_asym(knotvec1, knotvec2, du, dv, quadgrid=None, nqp=None):
    """Assemble the matrix for a(u,v)=(u^(du),v^(dv)) relating the two B-spline
    bases. By default, uses the first knot vector for quadrature.

    `knotvec1` is the space of trial functions, having `du` derivatives applied to them.
    `knotvec2` is the space of test functions, having `dv` derivatives applied to them.

    The resulting matrix has size `knotvec2.numdofs × knotvec1.numdofs`.
    """
    if quadgrid is None:
        quadgrid = knotvec1.mesh

    # create iterated Gauss quadrature rule for each interval
    if nqp is None:
        nqp = int(math.ceil((knotvec1.p + knotvec2.p - du - dv + 1) / 2.0))
    nspans = len(quadgrid) - 1
    q = make_iterated_quadrature(quadgrid, nqp)
    assert len(q[0]) == nspans * nqp

    # evaluate derivatives of basis functions at quadrature nodes
    derivs1 = bspline.active_deriv(knotvec1, q[0], du)[du, :, :]
    derivs2 = bspline.active_deriv(knotvec2, q[0], dv)[dv, :, :]

    first_points = q[0][::nqp]
    assert len(first_points) == nspans
    # map first_active_at over first quadrature points to get first active basis function index
    first_act1 = np.vectorize(knotvec1.first_active_at, otypes=(int,))(first_points)
    first_act2 = np.vectorize(knotvec2.first_active_at, otypes=(int,))(first_points)
    I,J = _create_coo_1d_custom(nspans, derivs2.shape[0], derivs1.shape[0], first_act2, first_act1)

    return _assemble_matrix_custom(nspans, nqp, derivs2, derivs1, I, J, q[1])

def bsp_mass_1d_asym(knotvec1, knotvec2, quadgrid=None):
    """Assemble a mass matrix relating two B-spline bases. By default, uses the first knot vector for quadrature."""
    return bsp_mixed_deriv_biform_1d_asym(knotvec1, knotvec2, 0, 0, quadgrid=quadgrid)

def bsp_stiffness_1d_asym(knotvec1, knotvec2, quadgrid=None):
    """Assemble a stiffness matrix relating two B-spline bases. By default, uses the first knot vector for quadrature."""
    return bsp_mixed_deriv_biform_1d_asym(knotvec1, knotvec2, 1, 1, quadgrid=quadgrid)

################################################################################
# 2D/3D assembling routines (rely on Cython module)
################################################################################

def bsp_mass_2d(knotvecs, geo=None, format='csr'):
    if geo is None:
        (kv1, kv2) = knotvecs
        M1 = bsp_mass_1d(kv1)
        M2 = bsp_mass_1d(kv2)
        return scipy.sparse.kron(M1, M2, format=format)
    else:
        return assemble_entries(
                assemblers.MassAssembler2D(knotvecs, geo),
                symmetric=True, format=format)

def bsp_stiffness_2d(knotvecs, geo=None, format='csr'):
    if geo is None:
        (kv1, kv2) = knotvecs
        M1 = bsp_mass_1d(kv1)
        M2 = bsp_mass_1d(kv2)
        K1 = bsp_stiffness_1d(kv1)
        K2 = bsp_stiffness_1d(kv2)
        return scipy.sparse.kron(K1, M2, format=format) + scipy.sparse.kron(M1, K2, format=format)
    else:
        return assemble_entries(
                assemblers.StiffnessAssembler2D(knotvecs, geo),
                symmetric=True, format=format)

def bsp_mass_3d(knotvecs, geo=None, format='csr'):
    if geo is None:
        M = [bsp_mass_1d(kv) for kv in knotvecs]
        def k(A,B):
            return scipy.sparse.kron(A, B, format=format)
        return k(M[0], k(M[1], M[2]))
    else:
        return assemble_entries(
                assemblers.MassAssembler3D(knotvecs, geo),
                symmetric=True, format=format)

def bsp_stiffness_3d(knotvecs, geo=None, format='csr'):
    if geo is None:
        MK = [(bsp_mass_1d(kv), bsp_stiffness_1d(kv)) for kv in knotvecs]
        def k(A,B):
            return scipy.sparse.kron(A, B, format=format)
        M12 = k(MK[1][0], MK[2][0])
        K12 = k(MK[1][1], MK[2][0]) + k(MK[1][0], MK[2][1])
        return k(MK[0][1], M12) + k(MK[0][0], K12)
    else:
        return assemble_entries(
                assemblers.StiffnessAssembler3D(knotvecs, geo),
                symmetric=True, format=format)

################################################################################
# Assembling right-hand sides
################################################################################

def inner_products(kvs, f, f_physical=False, geo=None):
    """Compute the :math:`L_2` inner products between each basis
    function in a tensor product B-spline basis and the function `f`
    (i.e., the load vector).

    Args:
        kvs (seq): a sequence of :class:`.KnotVector`,
            representing a tensor product basis
        f: a function or :class:`.BSplineFunc` object
        f_physical (bool): whether `f` is given in physical coordinates.
            If `True`, `geo` must be passed as well.
        geo: a :class:`.BSplineFunc` or :class:`.NurbsFunc` which describes
            the integration domain; if not given, the integrals are
            computed in the parameter domain

    Returns:
        ndarray: the inner products as an array of size
        `kvs[0].ndofs × kvs[1].ndofs × ... × kvs[-1].ndofs`.
        Each entry corresponds to the inner product of the
        corresponding basis function with `f`.
        If `f` is not scalar, then each of its components is treated separately
        and the corresponding dimensions are appended to the end of the return
        value.
    """
    if isinstance(kvs, bspline.KnotVector):
        kvs = (kvs,)
    # compute quadrature rules
    nqp = max(kv.p for kv in kvs) + 1
    gaussgrid, gaussweights = make_tensor_quadrature([kv.mesh for kv in kvs], nqp)

    # evaluate function f on grid or transformed grid
    if f_physical:
        assert geo is not None, 'inner_products in physical domain requires geometry'
        fvals = utils.grid_eval_transformed(f, gaussgrid, geo)
    else:
        fvals = utils.grid_eval(f, gaussgrid)

    # multiply function values with quadrature weights
    fvals = tensor.apply_tprod(
              [operators.DiagonalOperator(gw) for gw in gaussweights], fvals)
    # if geometry was specified, multiply by abs(det(jac))
    if geo is not None:
        geo_jac = geo.grid_jacobian(gaussgrid)
        geo_det = np.abs(assemble_tools.determinants(geo_jac))
        # if f is not scalar, we simply add trivial dimensions on to the end
        extra_dims = fvals.ndim - geo_det.ndim
        if extra_dims > 0:
            geo_det.shape = geo_det.shape + (extra_dims * (1,))
        fvals *= geo_det
    # apply transposed spline collocation matrices (sum over Gauss nodes)
    Ct = [bspline.collocation(kvs[i], gaussgrid[i]).T
            for i in range(len(kvs))]
    return tensor.apply_tprod(Ct, fvals)

################################################################################
# Incorporating essential boundary conditions
################################################################################

def slice_indices(ax, idx, shape, ravel=False, flip=None):
    """Return dof indices for a slice of a tensor product basis with size
    `shape`. The slice is taken across index `idx` on axis `ax`.

    The indices are returned either as a `N × dim` array of multiindices or,
    with `ravel=True`, as an array of sequential (raveled) indices.
    """
    shape = tuple(shape)
    if idx < 0:
        idx += shape[ax]     # wrap around
    axdofs = [range(n) for n in shape]
    if flip is not None:
        flip = tuple(flip)
        flip = flip[:ax] + (False,) + flip[ax:]     # insert trivial axis
        for i, flp in enumerate(flip):
            if flp:
                axdofs[i] = reversed(axdofs[i])
    axdofs[ax] = [idx]
    multi_indices = np.array(list(itertools.product(*axdofs)))
    if ravel:
        multi_indices = np.ravel_multi_index(multi_indices.T, shape)
    return multi_indices

def boundary_dofs(kvs, bdspec, ravel=False, flip=None, k=0):
    """Indices of the dofs which lie on the given boundary of the tensor
    product basis `kvs`. Output format is as for :func:`slice_indices`.
    """
    bdax, bdside = bspline._parse_bdspec(bdspec, len(kvs))
    idx = (np.arange(k+1) if bdside==0 else np.arange(-1,-k-2,-1))
    N = tuple(kv.numdofs for kv in kvs)
    dofs=[slice_indices(bdax, idx, N, ravel=ravel, flip=flip) for idx in idx]
    return np.concatenate(dofs)

def boundary_kv(kvs, bdspec, flip=None):
    if flip is None:
        flip=(len(kvs)-1)*(False,)
    (ax,_) = bdspec
    bkvs = (bspline.KnotVector(1-np.flip(kv.kv),kv.p) if flp else kv for kv, flp in zip(kvs[:ax]+kvs[(ax+1):],flip))
    return tuple(bkvs)

def corner_dofs(kvs, cspec, k=[0,0], ravel=False):
    N = tuple(kv.numdofs for kv in kvs)
    assert len(k) == len(kvs), "dimension error."
    dim = len(kvs)
    
    #2D hardcoded for now
    if cspec == (0,0):
        bdspecs = [(0,0), (1,0)]  #bottom and left
    elif cspec == (0,1):
        bdspecs = [(0,0), (1,1)]  #bottom and right
    elif cspec == (1,0):
        bdspecs = [(0,1), (1,0)]  #top and left
    elif cspec == (1,1):
        bdspecs = [(0,1), (1,1)]  #top and right
    else:
        assert False, 'Wrong input specification.'
        
    bdaxes, bdsides = [bdspec[0] for bdspec in bdspecs], [bdspec[1] for bdspec in bdspecs]
    idx = [np.arange(k_+1) if bdsides[i]==0 else np.arange(-1,-k_-2,-1) for i,k_ in enumerate(k)]
    dofs=[slice_indices(bdaxes[0], i, N, ravel=ravel) for i in idx[0]]
    dofs=[dofs_[idx[1]] for dofs_ in dofs]
    return np.concatenate(dofs)

def boundary_cells(kvs, bdspec, ravel=False):
    """Indices of the cells which lie on the given boundary of the tensor
    product basis `kvs`. Output format is as for :func:`slice_indices`.
    """
    bdax, bdside = bspline._parse_bdspec(bdspec, len(kvs))
    idx = (0 if bdside==0 else -1)
    N = tuple(kv.numspans for kv in kvs)
    return slice_indices(bdax, idx, N, ravel=ravel)

def _drop_nans(indices, values):
    isnan = np.isnan(values)
    if np.any(isnan):
        notnan_idx = np.nonzero(np.logical_not(isnan))[0]
        return indices[notnan_idx], values[notnan_idx]
    else:
        return indices, values

def compute_dirichlet_bc(kvs, geo, bdspec, dir_func):
    """Compute indices and values for a Dirichlet boundary condition using
    interpolation.

    Args:
        kvs: a tensor product B-spline basis
        geo (:class:`.BSplineFunc` or :class:`.NurbsFunc`): the geometry transform
        bdspec: a pair `(axis, side)`. `axis` denotes the axis along
            which the boundary condition lies, and `side` is either
            0 for the "lower" boundary or 1 for the "upper" boundary.
            Alternatively, one of the following six strings can be
            used for `bdspec`:

            ===================  ==================
            value                Meaning
            ===================  ==================
            ``"left"``           `x` low
            ``"right"``          `x` high
            ``"bottom"``         `y` low
            ``"top"``            `y` high
            ``"front"``          `z` low
            ``"back"``           `z` high
            ===================  ==================
        dir_func: a function which will be interpolated to obtain the
            Dirichlet boundary values. Assumed to be given in physical
            coordinates. If it is vector-valued, one Dirichlet dof is
            computed per component, and they are numbered according to
            the "blocked" matrix layout. If `dir_func` is a scalar value, a
            constant function with that value is assumed.

    Returns:
        A pair of arrays `(indices, values)` which denote the indices of the
        dofs within the tensor product basis which lie along the Dirichlet
        boundary and their computed values, respectively.
    """
    bdspec = bspline._parse_bdspec(bdspec, len(kvs))
    bdax, bdside = bdspec

    # get basis for the boundary face
    bdbasis = list(kvs)
    assert len(bdbasis) == geo.sdim, 'Invalid dimension of geometry'
    del bdbasis[bdax]

    # get boundary geometry and interpolate dir_func
    bdgeo = geo.boundary(bdspec)
    from .approx import interpolate
    if np.isscalar(dir_func):
        const_value = dir_func
        dir_func = lambda *x: const_value
    dircoeffs = interpolate(bdbasis, dir_func, geo=bdgeo)

    # compute sequential indices for eliminated dofs
    N = tuple(kv.numdofs for kv in kvs)
    bdindices = slice_indices(bdax, 0 if bdside==0 else -1, N, ravel=True)

    extra_dims = dircoeffs.ndim - len(bdbasis)
    if extra_dims == 0:
        return _drop_nans(bdindices, dircoeffs.ravel())
    elif extra_dims == 1:
        # vector function; assume blocked vector discretization
        numcomp = dircoeffs.shape[-1]
        NN = np.prod(N)
        idx, val = combine_bcs(
            (bdindices + j*NN, dircoeffs[..., j].ravel())
                for j in range(numcomp))
        return _drop_nans(idx, val)
    else:
        raise ValueError('invalid dimension of Dirichlet coefficients: %s' % dircoeffs.shape)

def compute_dirichlet_bcs(kvs, geo, bdconds):
    """Compute indices and values for Dirichlet boundary conditions on
    several boundaries at once.

    Args:
        kvs: a tensor product B-spline basis
        geo (:class:`.BSplineFunc` or :class:`.NurbsFunc`): the geometry transform
        bdconds: a list of `(bdspec, dir_func)` pairs, where `bdspec`
            specifies the boundary to apply a Dirichlet boundary condition to
            and `dir_func` is the function providing the Dirichlet values. For
            the exact meaning, refer to :func:`compute_dirichlet_bc`.
            As a shorthand, it is possible to pass a single pair ``("all",
            dir_func)`` which applies Dirichlet boundary conditions to all
            boundaries.
    Returns:
        A pair `(indices, values)` suitable for passing to
        :class:`RestrictedLinearSystem`.
    """
    if len(bdconds) == 2 and bdconds[0] == 'all':
        dir_func = bdconds[1]
        bdconds = [((ax, bd), dir_func)
                for ax in range(len(kvs))
                for bd in (0,1)]
    return combine_bcs(
            compute_dirichlet_bc(kvs, geo, bdspec, g)
            for (bdspec, g) in bdconds
    )

def compute_initial_condition_01(kvs, geo, bdspec, g0, g1, physical=True):
    r"""Compute indices and values for an initial condition including function
    value and derivative for a space-time discretization using interpolation.
    This only works for a space-time cylinder with constant (in time) geometry.
    To be precise, the space-time geometry transform `geo` should have the form

    .. math:: G(\vec x, t) = (\widetilde G(\vec x), t).

    Args:
        kvs: a tensor product B-spline basis
        geo (:class:`.BSplineFunc` or :class:`.NurbsFunc`): the geometry transform of
            the space-time cylinder
        bdspec: a pair `(axis, side)`. `axis` denotes the time axis of `geo`,
            and `side` is either 0 for the "lower" boundary or 1 for the
            "upper" boundary.
        g0: a function which will be interpolated to obtain the initial
            function values
        g1: a function which will be interpolated to obtain the initial
            derivatives.
        physical (bool): whether the functions `g0` and `g1` are given in
            physical (True) or parametric (False) coordinates. Physical
            coordinates are assumed by default.

    Returns:
        A pair of arrays `(indices, values)` which denote the indices of the
        dofs within the tensor product basis which lie along the initial face
        of the space-time cylinder and their computed values, respectively.
    """
    bdspec = bspline._parse_bdspec(bdspec, len(kvs))
    bdax, bdside = bdspec

    bdbasis = list(kvs)
    del bdbasis[bdax]

    bdgeo = geo.boundary(bdspec) if physical else None
    from .approx import interpolate
    coeffs01 = np.stack((  # coefficients for 0th and 1st derivatives, respectively
        interpolate(bdbasis, g0, geo=bdgeo).ravel(),
        interpolate(bdbasis, g1, geo=bdgeo).ravel()
    ))

    # compute 2x2 matrix which maps the two boundary coefficients to 0-th and 1-st derivative
    # at the boundary (only two basis functions have contributions there!)
    if bdside == 0:
        bdcolloc = bspline.active_deriv(kvs[bdax], 0.0, 1)[:2, :2] # first two basis functions
    else:
        bdcolloc = bspline.active_deriv(kvs[bdax], 1.0, 1)[:2, -2:] # last two basis functions

    # note: this only works for a space-time cylinder with constant geometry!
    coll_coeffs = np.linalg.solve(bdcolloc, coeffs01)

    # compute indices for the two boundary slices
    N = tuple(kv.numdofs for kv in kvs)
    firstidx = (0 if bdside==0 else -2)
    bdindices = np.concatenate((
        slice_indices(bdax, firstidx,   N, ravel=True),
        slice_indices(bdax, firstidx+1, N, ravel=True)
    ))

    return bdindices, coll_coeffs.ravel()


def combine_bcs(bcs):
    """Given a sequence of `(indices, values)` pairs such as returned by
    :func:`compute_dirichlet_bc`, combine them into a single pair
    `(indices, values)`.

    Dofs which occur in more than one `indices` array take their
    value from an arbitrary corresponding `values` array.
    """
    bcs = list(bcs)
    indices = np.concatenate([ind for ind,_ in bcs])
    values  = np.concatenate([val for _,val in bcs])
    assert indices.shape == values.shape, 'Inconsistent BC sizes'

    uidx, lookup = np.unique(indices, return_index=True)
    return uidx, values[lookup]


class RestrictedLinearSystem:
    """Represents a linear system with some of its dofs eliminated.

    Args:
        A: the full matrix
        b: the right-hand side (may be 0)
        bcs: a pair of arrays `(indices, values)` which contain the
            indices and values, respectively, of dofs to be eliminated
            from the system
        elim_rows: for Petrov-Galerkin discretizations, the equations to be
            eliminated from the linear system may not match the dofs to be
            eliminated. In this case, an array of indices of rows to be
            eliminated may be passed in this argument.

    Once constructed, the restricted linear system can be accessed through
    the following attributes:

    Attributes:
        A: the restricted matrix
        b: the restricted and updated right-hand side
    """
    def __init__(self, A, b, bcs, elim_rows=None):
        indices, values = bcs
        if np.isscalar(b):
            b = np.broadcast_to(b, A.shape[0])
        if np.isscalar(values):
            values = np.broadcast_to(values, indices.shape[0])
        self.values = values

        I = scipy.sparse.eye(A.shape[1], format='csr')
        # compute mask which contains non-eliminated dofs
        mask = np.ones(A.shape[1], dtype=bool)
        mask[list(indices)] = False

        # TODO/BUG: this may require the indices to be in increasing order?
        self.R_free = I[mask]
        self.R_elim = I[np.logical_not(mask)]

        if elim_rows is not None:
            # if the rows to be eliminated differ from the dofs to be fixed,
            # build a separate set of matrices
            elim_rows = sorted(elim_rows)
            I = scipy.sparse.eye(A.shape[0], format='csr')
            maskv = np.ones(A.shape[0], dtype=bool)
            maskv[elim_rows] = False
            self.R_free_v = I[maskv]
            self.R_elim_v = I[np.logical_not(maskv)]
        else:
            self.R_free_v = self.R_free
            self.R_elim_v = self.R_elim

        self.A = self.restrict_matrix(A)
        self.b = self.restrict_rhs(b - A.dot(self.R_elim.T.dot(values)))

    def restrict(self, u):
        """Given a vector `u` containing all dofs, return its restriction to the free dofs."""
        return self.R_free.dot(u)

    def restrict_rhs(self, f):
        """Given a right-hand side vector `f`, return its restriction to the non-eliminated rows.

        If `elim_rows` was not passed, this is equivalent to :meth:`restrict`.
        """
        return self.R_free_v.dot(f)

    def restrict_matrix(self, B):
        """Given a matrix `B` which operates on all dofs, return its restriction to the free dofs."""
        if not scipy.sparse.issparse(B):
            # the code below only works for sparse matrices
            B = scipy.sparse.csr_matrix(B)
        return self.R_free_v.dot(B).dot(self.R_free.T)

    def extend(self, u):
        """Given a vector `u` containing only the free dofs, pad it with zeros to all dofs."""
        return self.R_free.T.dot(u)

    def complete(self, u):
        """Given a solution `u` of the restricted linear system, complete it
        with the values of the eliminated dofs to a solution of the original
        system.
        """
        return self.extend(u) + self.R_elim.T.dot(self.values)

################################################################################
# Integration
################################################################################

def integrate(kvs, f, f_physical=False, geo=None):
    """Compute the integral of the function `f` over the geometry
    `geo` or a simple tensor product domain.

    Args:
        kvs (seq): a sequence of :class:`.KnotVector`;
            determines the parameter domain and the quadrature rule
        f: a function or :class:`.BSplineFunc` object
        f_physical (bool): whether `f` is given in physical coordinates.
            If `True`, `geo` must be passed as well.
        geo: a :class:`.BSplineFunc` or :class:`.NurbsFunc` which describes
            the integration domain; if not given, the integral is
            computed in the parameter domain

    Returns:
        float: the integral of `f` over the specified domain
    """
    if isinstance(kvs, bspline.KnotVector):
        kvs = (kvs,)
    # compute quadrature rules
    nqp = max(kv.p for kv in kvs) + 1
    gaussgrid, gaussweights = make_tensor_quadrature([kv.mesh for kv in kvs], nqp)

    # evaluate function f on grid or transformed grid
    if f_physical:
        assert geo is not None, 'integrate in physical domain requires geometry'
        fvals = utils.grid_eval_transformed(f, gaussgrid, geo)
    else:
        fvals = utils.grid_eval(f, gaussgrid)

    # multiply function values with quadrature weights
    fvals = tensor.apply_tprod(
              [operators.DiagonalOperator(gw) for gw in gaussweights], fvals)
    # if geometry was specified, multiply by abs(det(jac))
    if geo is not None:
        geo_jac = geo.grid_jacobian(gaussgrid)
        geo_det = np.abs(assemble_tools.determinants(geo_jac))
        fvals *= geo_det
    # sum over all coordinate axes (leave vector components intact, if any)
    return fvals.sum(axis=tuple(range(len(kvs))))

################################################################################
# Driver routines for assemblers
################################################################################

def assemble_entries(asm, symmetric=False, format='csr', layout='blocked'):
    """Given an instance `asm` of an assembler class, assemble all entries and return
    the resulting matrix or vector.

    Args:
        asm: an instance of an assembler class, e.g. one compiled using
            :func:`pyiga.compile.compile_vform`
        symmetric (bool): (matrices only) exploit symmetry of the matrix to
            speed up the assembly
        format (str): (matrices only) the sparse matrix format to use; default 'csr'
        layout (str): (vector-valued problems only): the layout of the generated
            matrix or vector. Valid options are:

            - 'blocked': the matrix is laid out as a `k_1 x k_2` block matrix,
              where `k_1` and `k_2` are the number of components of the test
              and trial functions, respectively
            - 'packed': the interactions of the components are packed together,
              i.e., each entry of the matrix is a small `k_1 x k_2` block

    Returns:
        ndarray or sparse matrix:

        - if the assembler has arity=1: an ndarray of vector entries whose
          shape is given by the number of degrees of freedom per coordinate
          direction. For vector-valued problem, an additional final axis is
          added which has the number of components as its length.
        - if the assembler has arity=2: a sparse matrix in the given `format`
    """
    is_vector_valued = hasattr(asm, 'num_components')
    if asm.arity == 1:
        result = asm.assemble_vector()
        if is_vector_valued and layout == 'blocked':
            # bring the last (component) axis to the front
            result = np.moveaxis(result, -1, 0)
        return result
    if is_vector_valued:
        return assemble_entries_vec(asm, symmetric=symmetric, format=format, layout=layout)
    kvs0, kvs1 = asm.kvs

    S = MLStructure.from_kvs(kvs0, kvs1)                # set up sparsity structure
    IJ = S.nonzero(lower_tri=symmetric)                 # compute locations of nonzero entries
    entries = asm.multi_entries(np.column_stack(IJ))    # compute the entries
    A = scipy.sparse.coo_matrix((entries, IJ), shape=S.shape).tocsr()

    # if symmetric, add the transpose of the lower triangular part
    if symmetric:
        I, J = IJ
        off_diag = np.nonzero(I != J)[0]    # indices of off-diagonal entries
        A_upper = scipy.sparse.coo_matrix((entries[off_diag], (J[off_diag], I[off_diag])), shape=S.shape)
        A += A_upper

    return A.asformat(format)

def _coo_to_csr_indices(IJ, shape):
    X = scipy.sparse.coo_matrix((np.arange(len(IJ[0])), IJ), shape=shape)
    X = X.tocsr()
    return X.indices, X.indptr, X.data

def assemble_entries_vec(asm, symmetric=False, format='csr', layout='blocked'):
    assert layout in ('packed', 'blocked')

    kvs0, kvs1 = asm.kvs
    dim = len(kvs0)
    nc = asm.num_components()[::-1]  # reverse axes (u = kv0 = columns)
    S_base = MLStructure.from_kvs(kvs0, kvs1)   # structure for the block indices
    S = S_base.join(MLStructure.dense(nc))      # final matrix structure (for 'packed')

    if layout == 'packed' and format == 'bsr':
        IJ = S_base.nonzero(lower_tri=symmetric)          # compute locations of nonzero blocks
        blocks = asm.multi_blocks(np.column_stack(IJ))    # compute the blocks
        # convert the block-COO coordinates IJ to BSR coordinates
        indices, indptr, permut = _coo_to_csr_indices(IJ, S_base.shape)
        A = scipy.sparse.bsr_matrix((blocks[permut], indices, indptr), shape=S.shape, blocksize=nc)

        if symmetric:
            assert nc[0] == nc[1], 'matrix with nonsymmetric block size cannot be symmetric'
            I, J = IJ
            off_diag = np.nonzero(I != J)[0]        # only blocks off the diagonal
            indices, indptr, permut = _coo_to_csr_indices((J[off_diag], I[off_diag]), S_base.shape)
            blocks_T = np.swapaxes(blocks[off_diag][permut], -1, -2)  # transpose the blocks
            A_upper = scipy.sparse.bsr_matrix((blocks_T, indices, indptr), shape=S.shape, blocksize=nc)
            A += A_upper

        return A.asformat(format)

    else:
        X = S.make_mlmatrix()

        if dim == 1:
            X.data = assemble_tools.generic_assemble_core_vec_1d(asm, X.structure.bidx[:dim], symmetric)
        elif dim == 2:
            X.data = assemble_tools.generic_assemble_core_vec_2d(asm, X.structure.bidx[:dim], symmetric)
        elif dim == 3:
            X.data = assemble_tools.generic_assemble_core_vec_3d(asm, X.structure.bidx[:dim], symmetric)
        else:
            # NB: dim-independent, but does not take advantage of symmetry
            IJ = S_base.nonzero()          # compute locations of nonzero blocks
            blocks = asm.multi_blocks(np.column_stack(IJ))    # compute the blocks
            X.data = blocks.reshape(X.datashape)

        if layout == 'blocked':
            axes = (dim,) + tuple(range(dim))   # bring last axis to the front
            X = X.reorder(axes)

        if format == 'mlb':
            return X
        else:
            return X.asmatrix(format)

def assemble_vf(vf, kvs, symmetric=False, format='csr', layout='blocked', args=None, **kwargs):
    """Compile the given variational form (:class:`.VForm`) into a matrix or vector.

    Any named inputs defined in the vform must be given in the `args` dict or
    as keyword arguments. For the meaning of the remaining arguments, refer to
    :func:`assemble_entries`.
    """
    if args is None:
        args = dict()
    args.update(kwargs)
    return assemble(vf, kvs, symmetric=symmetric, format=format, layout=layout, args=args)

def _assemble_hspace(problem, hs, args, bfuns=None, symmetric=False, format='csr', layout='blocked'):
    if isinstance(problem, str):
        from . import vform
        problem = vform.parse_vf(problem, hs.knotvectors(0), args=args, bfuns=bfuns)
    from .hierarchical import HDiscretization
    # TODO: vector-valued problems
    if problem.arity == 2:
        hdiscr = HDiscretization(hs, problem, asm_args=args)
        return hdiscr.assemble_matrix(symmetric=symmetric).asformat(format)
    elif problem.arity == 1:
        hdiscr = HDiscretization(hs, None, asm_args=args)
        return hdiscr.assemble_functional(problem)

def assemble(problem, kvs, args=None, bfuns=None, boundary=None, symmetric=False, format='csr', layout='blocked', **kwargs):
    """Assemble a matrix or vector in a function space.

    Args:
        problem: the description of the variational form to assemble. It can be
            passed in a number of formats (see :doc:`/guide/vforms` for details):

            - string: a textual description of the variational form
            - :class:`.VForm`: an abstract description of the variational form
            - assembler class (result of compiling a :class:`.VForm` using
              :func:`pyiga.compile.compile_vform`)
            - assembler object (result of instantiating an assembler class with
              a concrete space and input functions) -- in this case `kvs` and
              `args` are ignored

        kvs: the space or spaces in which to assemble the problem. Either:

            - a tuple of :class:`.KnotVector` instances, describing a tensor product
              spline space
            - if the variational form requires more than one space (e.g., for a
              Petrov-Galerkin discretization), then a tuple of such tuples,
              each describing one tensor product spline space (usually one for
              the trial space and one for the test space)
            - an :class:`.HSpace` instance for problems in hierarchical spline spaces

        args (dict): a dictionary which provides named inputs for the assembler. Most
            problems will require at least a geometry map; this can be given in
            the form ``{'geo': geo}``, where ``geo`` is a geometry function
            defined using the :mod:`pyiga.geometry` module. Further values used
            in the `problem` description must be passed here.

            For convenience, any additional keyword arguments to this function are
            added to the `args` dict automatically.

        bfuns: a list of used basis functions. By default, scalar basis functions 'u'
            and 'v' are assumed, and the arity of the variational form is determined
            automatically based on whether one or both of these functions are used.
            Otherwise, `bfuns` should be a list of tuples `(name, components, space)`,
            where `name` is a string, `components` is an integer describing the number
            of components the basis function has, and `space` is an integer referring
            to which input space the function lives in. Shorter tuples are valid, in
            which case the components default to `components=1` (scalar basis
            function) and `space=0` (all functions living in the same, first, input
            space).

            This argument is only used if `problem` is given as a string.

    For the meaning of the remaining arguments and the format of the output,
    refer to :func:`assemble_entries`.
    """
    if args is None:
        args = dict()
    args.update(kwargs)     # add additional keyword args

    from .hierarchical import HSpace
    if isinstance(kvs, HSpace):
        return _assemble_hspace(problem, kvs, bfuns=bfuns, symmetric=symmetric,
                format=format, layout=layout, args=args)
    else:
        asm = instantiate_assembler(problem, kvs, args, bfuns, boundary)
        return assemble_entries(asm, symmetric=symmetric, format=format, layout=layout)

def _Jac_to_boundary_matrix(bdspec, dim):
    """Compute a `dim x (dim-1)` matrix which restricts the Jacobian for a volumetric geometry map
    to the Jacobian for the boundary `bdspec`. Signs are chosen such that the resulting normal
    vector points outside assuming that the patch has positive orientation (`det(J) > 0`).
    """
    ax, side = bdspec
    ax = dim - 1 - ax       # last axis refers to x coordinate
    I = np.eye(dim)
    # change the signs so that normal vector for lower limit has correct orientation
    I[:, 0::2] *= -1        # this is sufficient for 2D and 3D; higher dimensions not worked out!
    B = np.hstack((I[:, :ax], I[:, ax+1:]))
    if side != 0:           # for the upper limit, we need to flip the normal vector
        B[:, 0] *= -1
    return B

def instantiate_assembler(problem, kvs, args, bfuns, boundary=None, updatable=[]):
    from . import vform

    # parse string to VForm
    if isinstance(problem, str):
        problem = vform.parse_vf(problem, kvs, args=args, bfuns=bfuns, boundary=bool(boundary), updatable=updatable)

    num_spaces = 1          # by default, only one space

    # compile VForm to assembler class
    if isinstance(problem, vform.VForm):
        num_spaces = problem.num_spaces()
        from . import compile
        problem = compile.compile_vform(problem)

    # instantiate assembler class
    if isinstance(problem, type):
        # check that all named inputs have been passed and extract the used args
        # (it is valid to specify additional, non-used args)
        used_args = dict()

        if boundary:
            # parse the bdspec and pass both `boundary` and the `Jac_to_boundary` matrix
            # to the assembler
            bdspec = bspline._parse_bdspec(boundary, len(kvs))
            used_args['boundary'] = bdspec
            args['Jac_to_boundary'] = _Jac_to_boundary_matrix(bdspec, len(kvs))

        for inp in itertools.chain(problem.inputs().keys(), problem.parameters().keys()):
            if inp not in args:
                raise ValueError("required input parameter '%s' missing" % inp)
            used_args[inp] = args[inp]

        if num_spaces <= 1:
            asm = problem(kvs, **used_args)
        else:
            assert num_spaces == 2, 'no more than two spaces allowed'
            asm = problem(kvs[0], kvs[1], **used_args)

        return asm

    # if we land here, problem has invalid type
    raise TypeError("invalid type for 'problem': {}".format(type(problem)))

class Assembler:
    """A high-level interface to an assembler class.

    Usually, you will simply call :func:`assemble` to assemble a matrix or
    vector. When assembling a sequence of problems with changing parameters, it
    may be more efficient to instantiate the assembler class only once and
    inform it of the changing inputs via the `updatable` argument. You may then
    call :meth:`update` to change these input fields and assemble the problem
    via :meth:`.assemble`.  Instead of calling :meth:`update` explicitly, you
    can also simply pass the fields to be updated as additional keyword
    arguments to :meth:`.assemble`.

    `updatable` is a list of names of assembler arguments which are to be
    considered updatable. The other arguments have the same meaning as for
    :func:`assemble`.
    """
    def __init__(self, problem, kvs, args=None, bfuns=None, boundary=None, symmetric=False, updatable=[], **kwargs):
        if args is None:
            args = dict()
        args.update(kwargs)
        self.symmetric = bool(symmetric)
        self.updatable = tuple(updatable)
        self.asm = instantiate_assembler(problem, kvs, args, bfuns, boundary, self.updatable)
        if not all(upd_name in self.asm.inputs().keys() for upd_name in self.updatable):
            raise ValueError('Assembler received an updatable argument which is not an assembler input')

    def update(self, **kwargs):
        """Update all input fields given as `name=func` keyword arguments.

        All fields updated in this manner must have been specified in the list
        of `updatable` arguments when creating the :class:`Assembler` object.
        """
        if not hasattr(self.asm, 'update'):
            raise RuntimeError('assembler object is not updatable')
        if not all(name in self.updatable for name in kwargs.keys()):
            raise RuntimeError('update() received an argument which was not specified as updatable')
        self.asm.update(**kwargs)

    def assemble(self, format='csr', layout='blocked', **upd_fields):
        """Assemble the problem.

        Any additional keyword arguments are passed to :meth:`update`.
        """
        if upd_fields:
            self.update(**upd_fields)
        return assemble_entries(self.asm, symmetric=self.symmetric, format=format, layout=layout)

################################################################################
# Convenience functions
################################################################################

def int_to_bdspec(i):
    return (i//2,i%2)

def bdspec_to_int(i):
    return 2*i[0]+i[1]

def _detect_dim(kvs):
    if isinstance(kvs, bspline.KnotVector):
        return 1, kvs
    else:
        d = len(kvs)
        # if dim==1, unpack the tuple and return only the kv
        return d, (kvs[0] if d==1 else kvs)

def mass(kvs, geo=None, format='csr'):
    """Assemble a mass matrix for the given basis (B-spline basis
    or tensor product B-spline basis) with an optional geometry transform.
    """
    dim, kvs = _detect_dim(kvs)
    if geo:
        assert geo.dim == dim, "Geometry has wrong dimension"
    if dim == 1:
        assert geo is None, "Geometry map not supported for 1D assembling"
        return bsp_mass_1d(kvs)
    elif dim == 2:
        return bsp_mass_2d(kvs, geo, format)
    elif dim == 3:
        return bsp_mass_3d(kvs, geo, format)
    else:
        assert False, "Dimensions higher than 3 are currently not implemented."

def stiffness(kvs, geo=None, format='csr'):
    """Assemble a stiffness matrix for the given basis (B-spline basis
    or tensor product B-spline basis) with an optional geometry transform.
    """
    dim, kvs = _detect_dim(kvs)
    if geo:
        assert geo.dim == dim, "Geometry has wrong dimension"
    if dim == 1:
        assert geo is None, "Geometry map not supported for 1D assembling"
        return bsp_stiffness_1d(kvs)
    elif dim == 2:
        return bsp_stiffness_2d(kvs, geo, format)
    elif dim == 3:
        return bsp_stiffness_3d(kvs, geo, format)
    else:
        assert False, "Dimensions higher than 3 are currently not implemented."

def divdiv(kvs, geo=None, layout='blocked', format='csr'):
    dim, kvs = _detect_dim(kvs)
    if geo is None:
        geo = geometry.unit_cube(dim=dim)   # TODO: fast assembling for div-div?
    if dim == 2:
        asm = assemblers.DivDivAssembler2D(kvs, geo)
    elif dim == 3:
        asm = assemblers.DivDivAssembler3D(kvs, geo)
    else:
        assert False, 'dimension %d not implemented' % dim
    return assemble_entries_vec(asm, symmetric=True, layout=layout, format=format)

def mass_fast(kvs, geo=None, tol=1e-10, maxiter=100, skipcount=3, tolcount=3, verbose=2):
    """Assemble a mass matrix for the given tensor product B-spline basis with
    an optional geometry transform, using the fast low-rank assembling
    algorithm.
    """
    if geo is None:
        # the default assemblers use Kronecker product assembling if no geometry present
        return mass(kvs)
    dim, kvs = _detect_dim(kvs)
    assert geo.dim == dim, "Geometry has wrong dimension"
    if dim == 1:
        assert False, "Geometry map not supported for 1D assembling"
    elif dim == 2:
        asm = assemblers.MassAssembler2D(kvs, geo)
    elif dim == 3:
        asm = assemblers.MassAssembler3D(kvs, geo)
    else:
        assert False, "Dimensions higher than 3 are currently not implemented."
    return fast_assemble_cy.fast_assemble(asm, kvs, tol, maxiter, skipcount, tolcount, verbose)

def stiffness_fast(kvs, geo=None, tol=1e-10, maxiter=100, skipcount=3, tolcount=3, verbose=2):
    """Assemble a stiffness matrix for the given tensor product B-spline basis
    with an optional geometry transform, using the fast low-rank assembling
    algorithm.
    """
    if geo is None:
        # the default assemblers use Kronecker product assembling if no geometry present
        return stiffness(kvs)
    dim, kvs = _detect_dim(kvs)
    assert geo.dim == dim, "Geometry has wrong dimension"
    if dim == 1:
        assert False, "Geometry map not supported for 1D assembling"
    elif dim == 2:
        asm = assemblers.StiffnessAssembler2D(kvs, geo)
    elif dim == 3:
        asm = assemblers.StiffnessAssembler3D(kvs, geo)
    else:
        assert False, "Dimensions higher than 3 are currently not implemented."
    return fast_assemble_cy.fast_assemble(asm, kvs, tol, maxiter, skipcount, tolcount, verbose)

################################################################################
# Multipatch
################################################################################

# helper functions for interface detection
def _bb_rect(G):
    # geo bounding box as a Rectangle
    bb = G.bounding_box()
    return scipy.spatial.Rectangle(
        tuple(bb_i[0] for bb_i in bb),
        tuple(bb_i[1] for bb_i in bb))

def _check_geo_match(G1, G2, grid=4):
    # check if the two geos match with any possible flip
    if G1.sdim != G2.sdim or G1.dim != G2.dim:
        return False, (None, None)
    if not np.allclose(G1.support, G2.support):
        return False, (None, None)
    grid = [np.linspace(s[0], s[1], grid) for s in G1.support]
    X1 = G1.grid_eval(grid)

    for k, perm in enumerate(itertools.permutations(np.arange(G1.sdim))):
        all_flips = itertools.product(*(G2.sdim * [(False, True)]))
        for flip in all_flips:  # try all 2^d possible flips
            flipped_grid = list(grid)
            for (i, f) in enumerate(flip):
                if f: flipped_grid[i] = np.ascontiguousarray(np.flip(flipped_grid[i]))
            X2 = G2.grid_eval(flipped_grid).transpose(perm + (G2.sdim,))
            if np.allclose(X1, X2):
                if G1.sdim == 1 or k==0:
                    return True, (None, flip)
                print(i)
                print(perm)
                return True, (perm, flip)
    return False, (None, None)

def _find_matching_boundaries(G1, G2):
    # find all interfaces which match between G1 and G2
    assert G1.sdim == G2.sdim and G1.dim == G2.dim
    all_bds = list(itertools.product(range(G1.sdim), (0,1)))
    matches = []
    for bdspec1 in all_bds:
        bd1 = G1.boundary(bdspec1)
        for bdspec2 in all_bds:
            bd2 = G2.boundary(bdspec2)
            match, conn_info = _check_geo_match(bd1, bd2)
            if match:
                matches.append((bdspec1, bdspec2, conn_info))
    return matches

def detect_interfaces(patches):
    """Automatically detect matching interfaces between patches.

    Args:
        patches: a list of patches in the form `(kvs, geo)`

    Returns:
        A pair `(connected, interfaces)`, where `connected` is a `bool`
        describing whether the detected patch graph is connected, and
        `interfaces` is a list of the detected interfaces, each entry of
        which is suitable for passing to :meth:`join_boundaries`.
    """
    interfaces = []
    bbs = [_bb_rect(geo) for (_, geo) in patches]
    diams = [bb.max_distance_rectangle(bb) for bb in bbs]

    # set up a graph of patch connectivity for later checking
    import networkx as nx
    patch_graph = nx.Graph()
    patch_graph.add_nodes_from(range(len(patches)))

    for p1 in range(len(patches)):
        for p2 in range(p1 + 1, len(patches)):
            mindist = bbs[p1].min_distance_rectangle(bbs[p2])
            maxdiam = max(diams[p1], diams[p2])
            if mindist < 1e-10 * maxdiam:    # do the bounding boxes touch?
                matches = _find_matching_boundaries(patches[p1][1], patches[p2][1])
                for (bd1, bd2, conn_info) in matches:
                    interfaces.append((p1, bd1, p2, bd2, conn_info))
                if matches:
                    patch_graph.add_edge(p1, p2)

    return nx.is_connected(patch_graph), interfaces


class Multipatch:
    """Represents a multipatch structure, consisting of a number of patches
    together with their discretizations and the information about shared dofs
    between patches. Currently, only conforming patches are supported.

    Args:
        patches: a list of `(kvs, geo)` pairs, each of which describes a patch.
            Here `kvs` is a tuple of :class:`.KnotVector` instances describing
            the tensor product spline space used for discretization and `geo`
            is the geometry map.
        automatch (bool): if True, attempt to automatically determine the
            matching interfaces between all patches and finalize then. If
            False, the user has to manually join the patches by calling
            :meth:`join_boundaries` as often as needed, followed by
            :meth:`finalize`.
    """
    def __init__(self, patches, automatch=False):
        """Initialize a multipatch structure."""
        self.patches = patches
        # number of tensor product dofs per patch
        self.N = [bspline.numdofs(kvs) for (kvs,_) in self.patches]
        # offset to the dofs of the i-th patch
        self.N_ofs = np.concatenate(([0], np.cumsum(self.N)))
        # per patch, a dict of local-to-shared indices
        self.shared_per_patch = [dict() for i in range(len(self.patches))]
        # per shared dof, a set of local dofs (patch, index)
        self.shared_dofs = []

        if automatch:
            connected, interfaces = detect_interfaces(self.patches)
            if not connected:
                print('WARNING: patch graph is not connected - ' +
                        'interface detection may have failed')
            for intf in interfaces:
                self.join_boundaries(*intf)
            self.finalize()

    @property
    def numpatches(self):
        """Number of patches in the multipatch structure."""
        return len(self.patches)

    @property
    def numdofs(self):
        """Number of dofs after eliminating shared dofs.

        May only be called after :func:`finalize`.
        """
        return self.M_ofs[-1] + len(self.shared_dofs)

    def join_dofs(self, p1, I1, p2, I2):
        """Join the dofs `I1` of patch `p1` with the dofs `I2` of patch `p2`."""
        assert len(I1) == len(I2), 'dof arrays must have the same length'
        assert p1 != p2, 'patches must be different'

        def add_to_shared(sd, p, i):
            # add dof (p, i) to the shared dof `sd`
            self.shared_per_patch[p][i] = sd
            self.shared_dofs[sd].add((p, i))

        for (i1, i2) in zip(I1, I2):
            if i1 in self.shared_per_patch[p1]:
                sd = self.shared_per_patch[p1][i1]
                add_to_shared(sd, p2, i2)
            elif i2 in self.shared_per_patch[p2]:
                sd = self.shared_per_patch[p2][i2]
                add_to_shared(sd, p1, i1)
            else:
                # none of them are shared yet - create a new shared dof
                sd = self._new_shared_dof()
                add_to_shared(sd, p1, i1)
                add_to_shared(sd, p2, i2)

    def join_boundaries(self, p1, bdspec1, p2, bdspec2, flip=None):
        """Join the dofs lying along boundary `bdspec1` of patch `p1` with
        those lying along boundary `bdspec2` of patch `p2`.

        See :func:`compute_dirichlet_bc` for the format of the boundary
        specification.

        If `flip` is given, it should be a sequence of booleans indicating for
        each coordinate axis of the boundary if the coordinates of `p2` have to
        be flipped along that axis.
        """
        P1, P2 = self.patches[p1], self.patches[p2]
        dofs1 = boundary_dofs(P1[0], bdspec1, ravel=True)
        dofs2 = boundary_dofs(P2[0], bdspec2, ravel=True, flip=flip)
        self.join_dofs(p1, dofs1, p2, dofs2)

    def _new_shared_dof(self):
        i = len(self.shared_dofs)
        self.shared_dofs.append(set())
        return i

    def finalize(self):
        """After all shared dofs have been declared using
        :meth:`join_boundaries` or :meth:`join_dofs`, call this function to set
        up the internal data structures.
        """
        num_shared = [len(spp) for spp in self.shared_per_patch]
        # number of local dofs per patch
        self.M = [n - s for (n, s) in zip(self.N, num_shared)]
        # local-to-global offset per patch
        self.M_ofs = np.concatenate(([0], np.cumsum(self.M)))

    def patch_to_global_idx(self, p):
        """Return an array which maps local tensor product indices for patch
        `p` to global indices.
        """
        tpdofs = np.arange(self.N[p])   # local TP indices
        # construct array with one column for local and one for corresponding
        # shared indices
        sdofs = np.array([l_s for l_s in self.shared_per_patch[p].items()])
        # which TP indices are local (non-shared)?
        local_dofs = np.setdiff1d(tpdofs, sdofs[:,0], assume_unique=True)

        # reuse tpdofs for the output
        m_ofs = self.M_ofs[p]   # offset to global indices for this patch
        tpdofs[local_dofs] = np.arange(m_ofs, m_ofs + local_dofs.shape[0])
        tpdofs[sdofs[:,0]] = self.M_ofs[-1] + sdofs[:,1]
        return tpdofs

    def patch_to_global(self, p, j_global=False):
        """Compute a sparse binary matrix which maps dofs local to patch `p` to
        the corresponding global dofs.

        Args:
            p (int): the index of the patch
            j_global (bool): if False, the matrix has only as many columns as
                `p` has dofs; if True, the number of columns is the sum of the
                number of local dofs over all patches

        Returns:
            a CSR sparse matrix
        """
        shape = (self.numdofs, self.N_ofs[-1] if j_global else self.N[p])
        n_ofs = self.N_ofs[p] if j_global else 0
        I = self.patch_to_global_idx(p)
        J = np.arange(n_ofs, n_ofs + self.N[p])
        X = scipy.sparse.coo_matrix((np.ones(len(I)), (I,J)), shape=shape)
        return X.tocsr()

    def global_to_patch(self, p):
        """Compute a sparse binary matrix which maps global dofs to local dofs
        in patch `p`. This is just the transpose of :meth:`patch_to_global` and
        also its left-inverse.

        Args:
            p (int): the index of the patch

        Returns:
            a sparse matrix
        """
        return self.patch_to_global(p).T

    def assemble_system(self, problem, rhs, args=None, bfuns=None,
            symmetric=False, format='csr', layout='blocked', **kwargs):
        """Assemble both the system matrix and the right-hand side vector
        for a variational problem over the multipatch geometry.

        Here `problem` represents a bilinear form and `rhs` a linear functional.
        See :func:`assemble` for the precise meaning of the arguments.

        Returns:
            A pair `(A, b)` consisting of the sparse system matrix and the
            right-hand side vector.
        """
        n = self.numdofs
        A = scipy.sparse.csr_matrix((n, n)).asformat(format)
        b = np.zeros(n)
        if args is None:
            args = dict()
        for p in range(self.numpatches):
            X = self.patch_to_global(p)
            kvs, geo = self.patches[p]
            args.update(geo=geo)
            # TODO: vector-valued problems
            A_p = assemble(problem, kvs, args=args, bfuns=bfuns,
                    symmetric=symmetric, format=format, layout=layout,
                    **kwargs)
            A += X @ A_p @ X.T
            b_p = assemble(rhs, kvs, args=args, bfuns=bfuns,
                    symmetric=symmetric, format=format, layout=layout,
                    **kwargs).ravel()
            b += X @ b_p
        return A, b

    def compute_dirichlet_bcs(self, bdconds):
        """Performs the same operation as the global function
        :func:`compute_dirichlet_bcs`, but for a multipatch problem.

        The sequence `bdconds` should contain triples of the form `(patch,
        bdspec, dir_func)`.

        Returns:
            A pair `(indices, values)` suitable for passing to
            :class:`RestrictedLinearSystem`.
        """
        bcs = []
        p2g = dict()        # cache the patch-to-global indices for efficiency
        for (p, bdspec, g) in bdconds:
            kvs, geo = self.patches[p]
            bc = compute_dirichlet_bc(kvs, geo, bdspec, g)
            if p not in p2g:
                p2g[p] = self.patch_to_global_idx(p)
            idx = p2g[p]    # maps local dofs to global dofs
            bcs.append((idx[bc[0]], bc[1]))
        return combine_bcs(bcs)
