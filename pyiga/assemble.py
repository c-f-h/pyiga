"""Matrix assembling functions for B-spline IgA.

This module contains functions to assemble mass and stiffness matrices
for IgA with tensor product B-spline functions.


.. _gauss-asms:

Tensor product Gauss quadrature assemblers
------------------------------------------

Standard Gauss quadrature assemblers for mass and stiffness matrices.
They take one or two arguments:

- `kvs` (list of :class:`pyiga.bspline.KnotVector`):
  Describes the tensor product B-spline basis for which to assemble
  the matrix. One :class:`KnotVector` per coordinate direction.
  In the 1D case, a single :class:`pyiga.bspline.KnotVector` may be passed
  directly.
- `geo` (:class:`pyiga.geometry.BSplinePatch`; optional):
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

"""
import numpy as np
import scipy
import scipy.sparse

from . import bspline
from . import assemble_tools
from . import kronecker
from . import operators
from . import utils

from .quadrature import make_iterated_quadrature

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

def _create_coo_1d(nspans, n_act1, n_act2=None):
    """Create COO indices for two sequentially numbered bases over `nspans` knot spans"""
    if n_act2 is None:
        n_act2 = n_act1     # if only one given, assume symmetry
    grid = np.mgrid[:n_act1, :n_act2] # 2 x n_act1 x n_act2 array which indexes element matrix
    I_ref = grid[0].ravel()          # slowly varying index, first basis
    J_ref = grid[1].ravel()          # fast varying index, second basis

    # first active basis function = index of knot span (independent of spline degree)
    first_act = np.repeat(range(nspans), n_act1*n_act2)
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

def _assemble_matrix(nspans, nqp, vals1, vals2, qweights):
    n_act1 = vals1.shape[0]
    n_act2 = vals2.shape[0]
    elMats = _assemble_element_matrices(nspans, nqp, vals1, vals2, qweights)
    (I, J) = _create_coo_1d(nspans, n_act1, n_act2)
    return scipy.sparse.coo_matrix((elMats.ravel(), (I, J))).tocsr()

def _assemble_matrix_custom(nspans, nqp, vals1, vals2, I, J, qweights):
    n_act1 = vals1.shape[0]
    n_act2 = vals2.shape[0]
    elMats = _assemble_element_matrices(nspans, nqp, vals1, vals2, qweights)
    return scipy.sparse.coo_matrix((elMats.ravel(), (I, J))).tocsr()

def bsp_mass_1d(knotvec):
    "Assemble the mass matrix for the B-spline basis over the given knot vector"""
    nspans = knotvec.numspans
    nqp = knotvec.p + 1
    q = make_iterated_quadrature(knotvec.kv[knotvec.p:-knotvec.p], nqp)
    vals = bspline.active_ev(knotvec, q[0])
    return _assemble_matrix(nspans, nqp, vals, vals, q[1])

def bsp_mass_1d_asym(knotvec1, knotvec2, quadgrid=None):
    """Assemble a mass matrix relating two B-spline bases. By default, uses the first knot vector for quadrature."""
    if quadgrid is None:
        quadgrid = np.unique(knotvec1.kv)

    # create iterated Gauss quadrature rule for each interval
    nqp = max(knotvec1.p, knotvec2.p) + 1
    nspans = len(quadgrid) - 1
    q = make_iterated_quadrature(quadgrid, nqp)
    assert len(q[0]) == nspans * nqp

    # evaluate basis functions at quadrature nodes
    vals1 = bspline.active_ev(knotvec1, q[0])
    vals2 = bspline.active_ev(knotvec2, q[0])

    first_points = q[0][::nqp]
    assert len(first_points) == nspans
    # map first_active_at over first quadrature points to get first active basis function index
    first_act1 = np.vectorize(knotvec1.first_active_at, otypes=(np.int,))(first_points)
    first_act2 = np.vectorize(knotvec2.first_active_at, otypes=(np.int,))(first_points)
    I,J = _create_coo_1d_custom(nspans, vals1.shape[0], vals2.shape[0], first_act1, first_act2)

    return _assemble_matrix_custom(nspans, nqp, vals1, vals2, I, J, q[1])

def bsp_mass_1d_weighted(knotvec, weightfunc):
    nspans = knotvec.numspans
    nqp = knotvec.p
    q = make_iterated_quadrature(knotvec.kv[knotvec.p:-knotvec.p], nqp)
    vals = bspline.active_ev(knotvec, q[0])
    weights = q[1] * weightfunc(q[0])
    return _assemble_matrix(nspans, nqp, vals, vals, weights)

def bsp_stiffness_1d(knotvec):
    "Assemble the Laplacian stiffness matrix for the B-spline basis over the given knot vector"""
    nspans = knotvec.numspans
    nqp = knotvec.p
    q = make_iterated_quadrature(knotvec.kv[knotvec.p:-knotvec.p], nqp)
    derivs = bspline.active_deriv(knotvec, q[0], 1)[1, :, :]
    return _assemble_matrix(nspans, nqp, derivs, derivs, q[1])

def bsp_stiffness_1d_asym(knotvec1, knotvec2, quadgrid=None):
    """Assemble a stiffness matrix relating two B-spline bases. By default, uses the first knot vector for quadrature."""
    if quadgrid is None:
        quadgrid = np.unique(knotvec1.kv)

    # create iterated Gauss quadrature rule for each interval
    nqp = max(knotvec1.p, knotvec2.p) + 1
    nspans = len(quadgrid) - 1
    q = make_iterated_quadrature(quadgrid, nqp)
    assert len(q[0]) == nspans * nqp

    # evaluate derivatives of basis functions at quadrature nodes
    derivs1 = bspline.active_deriv(knotvec1, q[0], 1)[1, :, :]
    derivs2 = bspline.active_deriv(knotvec2, q[0], 1)[1, :, :]

    first_points = q[0][::nqp]
    assert len(first_points) == nspans
    # map first_active_at over first quadrature points to get first active basis function index
    first_act1 = np.vectorize(knotvec1.first_active_at, otypes=(np.int,))(first_points)
    first_act2 = np.vectorize(knotvec2.first_active_at, otypes=(np.int,))(first_points)
    I,J = _create_coo_1d_custom(nspans, derivs1.shape[0], derivs2.shape[0], first_act1, first_act2)

    return _assemble_matrix_custom(nspans, nqp, derivs1, derivs2, I, J, q[1])

################################################################################
# 2D/3D assembling routines (rely on Cython module)
################################################################################

def bsp_mass_2d(knotvecs, geo=None):
    if geo is None:
        (kv1, kv2) = knotvecs
        M1 = bsp_mass_1d(kv1)
        M2 = bsp_mass_1d(kv2)
        return scipy.sparse.kron(M1, M2, format='csr')
    else:
        return assemble_tools.mass_2d(knotvecs, geo)

def bsp_stiffness_2d(knotvecs, geo=None):
    if geo is None:
        (kv1, kv2) = knotvecs
        M1 = bsp_mass_1d(kv1)
        M2 = bsp_mass_1d(kv2)
        K1 = bsp_stiffness_1d(kv1)
        K2 = bsp_stiffness_1d(kv2)
        return scipy.sparse.kron(K1, M2, format='csr') + scipy.sparse.kron(M1, K2, format='csr')
    else:
        return assemble_tools.stiffness_2d(knotvecs, geo)

def bsp_mass_3d(knotvecs, geo=None):
    if geo is None:
        M = [bsp_mass_1d(kv) for kv in knotvecs]
        def k(A,B):
            return scipy.sparse.kron(A, B, format='csr')
        return k(M[0], k(M[1], M[2]))
    else:
        return assemble_tools.mass_3d(knotvecs, geo)

def bsp_stiffness_3d(knotvecs, geo=None):
    if geo is None:
        MK = [(bsp_mass_1d(kv), bsp_stiffness_1d(kv)) for kv in knotvecs]
        def k(A,B):
            return scipy.sparse.kron(A, B, format='csr')
        M12 = k(MK[1][0], MK[2][0])
        K12 = k(MK[1][1], MK[2][0]) + k(MK[1][0], MK[2][1])
        return k(MK[0][1], M12) + k(MK[0][0], K12)
    else:
        return assemble_tools.stiffness_3d(knotvecs, geo)

################################################################################
# Assembling right-hand sides
################################################################################

def inner_products(kvs, f):
    """Compute the :math:`L_2` inner products between each basis
    function in a tensor product B-spline basis and the function `f`
    (i.e., the load vector).

    Args:
        kvs (seq): a list of :class:`pyiga.bspline.KnotVector`,
                   representing a tensor product basis
        f: a function or :class:`pyiga.geometry.BSplinePatch` object

    Returns:
        ndarray: the inner products as an array of size
        `kvs[0].ndofs x kvs[1].ndofs x ... x kvs[-1].ndofs`.
        Each entry corresponds to the inner product of the
        corresponding basis function with `f`.
    """
    if isinstance(kvs, bspline.KnotVector):
        kvs = (kvs,)
    # compute quadrature rules
    nqp = max(kv.p for kv in kvs) + 1
    gauss = [make_iterated_quadrature(kv.mesh, nqp) for kv in kvs]
    # evaluate function f on grid
    fvals = utils.grid_eval(f, [g[0] for g in gauss])
    # multiply function values with quadrature weights
    fvals = kronecker.apply_tprod(
              [operators.DiagonalOperator(g[1]) for g in gauss], fvals)
    # multiply with spline collocation matrices
    Ct = [bspline.collocation(kvs[i], gauss[i][0]).T
            for i in range(len(kvs))]
    return kronecker.apply_tprod(Ct, fvals)

################################################################################
# Convenience functions
################################################################################

def _detect_dim(kvs):
    if isinstance(kvs, bspline.KnotVector):
        return 1
    else:
        return len(kvs)

def mass(kvs, geo=None):
    """Assemble a mass matrix for the given basis (B-spline basis
    or tensor product B-spline basis) with an optional geometry transform.
    """
    dim = _detect_dim(kvs)
    if geo:
        assert geo.dim == dim, "Geometry has wrong dimension"
    if dim == 1:
        assert geo is None, "Geometry map not supported for 1D assembling"
        return bsp_mass_1d(kvs)
    elif dim == 2:
        return bsp_mass_2d(kvs, geo)
    elif dim == 3:
        return bsp_mass_3d(kvs, geo)
    else:
        assert False, "Dimensions higher than 3 are currently not implemented."

def stiffness(kvs, geo=None):
    """Assemble a stiffness matrix for the given basis (B-spline basis
    or tensor product B-spline basis) with an optional geometry transform.
    """
    dim = _detect_dim(kvs)
    if geo:
        assert geo.dim == dim, "Geometry has wrong dimension"
    if dim == 1:
        assert geo is None, "Geometry map not supported for 1D assembling"
        return bsp_stiffness_1d(kvs)
    elif dim == 2:
        return bsp_stiffness_2d(kvs, geo)
    elif dim == 3:
        return bsp_stiffness_3d(kvs, geo)
    else:
        assert False, "Dimensions higher than 3 are currently not implemented."

def mass_fast(kvs, geo=None, tol=1e-10, maxiter=100, skipcount=3, tolcount=3, verbose=2):
    """Assemble a mass matrix for the given basis (B-spline basis
    or tensor product B-spline basis) with an optional geometry transform,
    using the fast low-rank assembling algorithm.
    """
    if geo is None:
        # the default assemblers use Kronecker product assembling if no geometry present
        return mass(kvs)
    dim = _detect_dim(kvs)
    assert geo.dim == dim, "Geometry has wrong dimension"
    if dim == 1:
        assert False, "Geometry map not supported for 1D assembling"
    elif dim == 2:
        return assemble_tools.fast_mass_2d(kvs, geo, tol, maxiter, skipcount, tolcount, verbose)
    elif dim == 3:
        return assemble_tools.fast_mass_3d(kvs, geo, tol, maxiter, skipcount, tolcount, verbose)
    else:
        assert False, "Dimensions higher than 3 are currently not implemented."

def stiffness_fast(kvs, geo=None, tol=1e-10, maxiter=100, skipcount=3, tolcount=3, verbose=2):
    """Assemble a stiffness matrix for the given basis (B-spline basis
    or tensor product B-spline basis) with an optional geometry transform,
    using the fast low-rank assembling algorithm.
    """
    if geo is None:
        # the default assemblers use Kronecker product assembling if no geometry present
        return stiffness(kvs)
    dim = _detect_dim(kvs)
    assert geo.dim == dim, "Geometry has wrong dimension"
    if dim == 1:
        assert False, "Geometry map not supported for 1D assembling"
    elif dim == 2:
        return assemble_tools.fast_stiffness_2d(kvs, geo, tol, maxiter, skipcount, tolcount, verbose)
    elif dim == 3:
        return assemble_tools.fast_stiffness_3d(kvs, geo, tol, maxiter, skipcount, tolcount, verbose)
    else:
        assert False, "Dimensions higher than 3 are currently not implemented."

