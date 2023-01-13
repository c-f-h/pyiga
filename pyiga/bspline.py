# -*- coding: utf-8 -*-
"""Functions and classes for B-spline basis functions.

"""

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import scipy.interpolate

from .tensor import apply_tprod

def _parse_bdspec(bdspec, dim):
    if bdspec == 'left':
        bd = ((dim - 1, 0),)
    elif bdspec == 'right':
        bd = ((dim - 1, 1),)
    elif bdspec == 'bottom':
        bd = ((dim - 2, 0),)
    elif bdspec == 'top':
        bd = ((dim - 2, 1),)
    elif bdspec == 'front':
        bd = ((dim - 3, 0),)
    elif bdspec == 'back':
        bd = ((dim - 3, 1),)
    else:
        bd=bdspec
        if not all((side in (0,1) for _, side in bd)):
            raise ValueError('invalid bdspec ' + str(bd))
        if any(( ax < 0 or ax >= dim for ax, _ in bd)):
            raise ValueError('invalid bdspec %s for space of dimension %d'
                % (bdspec, dim))
    return bd

def is_sub_space(kv1,kv2):
    """Checks if the spline space induced by the knot vector kv1 is a subspace of the corresponding spline space induced by kv2 in some subinterval of kv1.
    
    Currently only covers cases with the same spline degree `p`.
    
    """
    assert kv1.p == kv2.p
    
    a1, b1 = kv1.support()
    a2, b2 = kv2.support()
    
    if a2 < a1 or b2 > b1:
        return False

    return all([any(np.isclose(k,kv2.mesh)) for k in kv1.mesh if a2 <= k <= b2])


def multi_indices(N, k=0):
    assert N>0 and k>=0, "N must be positive and k non-negative."
    if N==1:
        yield (k,)
    else:
        for i in range(k,-1,-1):
            for j in multi_indices(N-1,k-i):
                yield (i,) + j

class KnotVector:
    """Represents an open B-spline knot vector together with a spline degree.

    Args:
        knots (ndarray): the 1D knot vector. Should be an open knot vector,
            i.e., the first and last knot should be repeated `p+1` times.
            Interior knots may be single or repeated up to `p` times.
        p (int): the spline degree.

    This class is commonly used to represent the B-spline basis
    over the given knot vector with the given spline degree.
    The B-splines are normalized in the sense that they satisfy a
    partition of unity property.

    Tensor product B-spline bases are typically represented simply as tuples of
    the univariate knot spans. E.g., ``(kv1, kv2)`` would represent the tensor
    product basis formed from the two B-spline bases over the
    :class:`KnotVector` instances ``kv1`` and ``kv2``.

    A more convenient way to create knot vectors is the :func:`make_knots` function.

    Attributes:
        kv (ndarray): vector of knots
        p (int): spline degree
    """

    def __init__(self, knots, p):
        """Construct an open B-spline knot vector with given `knots` and degree `p`."""
        self.kv = knots
        # sanity check: knots should be monotonically increasing
        assert np.all(self.kv[1:] - self.kv[:-1] >= 0), 'knots should be increasing'
        self.p = p
        self._mesh = None    # knots with duplicates removed (on demand)
        self._knots_to_mesh = None   # knot indices to mesh indices (on demand)

    def __str__(self):
        return '<KnotVector p=%d sz=%d>' % (self.p, self.kv.size)

    def __repr__(self):
        return 'KnotVector(%s, %s)' % (repr(self.kv), repr(self.p))

    def __eq__(self, other):
        if self.p == other.p and len(self.kv) == len(other.kv):
            if np.allclose(self.kv, other.kv, atol=1e-8, rtol=1e-8):
                return True
        return False

    @property
    def numknots(self):
        return self.kv.size

    @property
    def numdofs(self):
        """Number of basis functions in a B-spline basis defined over this knot vector"""
        return self.kv.size - self.p - 1

    @property
    def numspans(self):
        """Number of nontrivial intervals in the knot vector"""
        return self.mesh.size - 1

    def copy(self):
        """Return a copy of this knot vector."""
        return KnotVector(self.kv.copy(), self.p)

    def support(self, j=None):
        """Support of the knot vector or, if `j` is passed, of the j-th B-spline"""
        if j is None:
            return (self.kv[0], self.kv[-1])
        else:
            return (self.kv[j], self.kv[j+self.p+1])

    def support_idx(self, j):
        """Knot indices of support of j-th B-spline"""
        return (j, j+self.p+1)

    def _ensure_mesh(self):
        """Make sure that the _mesh and _knots_to_mesh arrays are set up"""
        if self._knots_to_mesh is None:
            self._mesh, self._knots_to_mesh = np.unique(self.kv, return_inverse=True)

    @property
    def mesh(self):
        """Return the mesh, i.e., the vector of unique knots in the knot vector."""
        self._ensure_mesh()
        return self._mesh

    def mesh_support_idx(self, j):
        """Return the first and last mesh index of the support of i"""
        self._ensure_mesh()
        supp = self.support_idx(j)
        return (self._knots_to_mesh[supp[0]], self._knots_to_mesh[supp[1]])

    def mesh_support_idx_all(self):
        """Compute an integer array of size `N × 2`, where N = self.numdofs, which
        contains for each B-spline the result of :func:`mesh_support_idx`.
        """
        self._ensure_mesh()
        n = self.numdofs
        startend = np.stack((np.arange(0,n), np.arange(self.p+1, n+self.p+1)), axis=1)
        return self._knots_to_mesh[startend]

    def mesh_span_indices(self):
        """Return an array of indices i such that kv[i] != kv[i+1], i.e., the indices
        of the nonempty spans. Return value has length self.numspans.
        """
        self._ensure_mesh()
        k2m = self._knots_to_mesh
        return np.where(k2m[1:] != k2m[:-1])[0]

    def findspan(self, u):
        """Returns an index i such that
         kv[i] <= u < kv[i+1]     (except for the boundary, where u <= kv[m-p] is allowed)
         and p <= i < len(kv) - 1 - p"""
        #if u >= self.kv[-self.p-1]:
        #    return self.kv.size - self.p - 2  # last interval
        #else:
        #    return self.kv.searchsorted(u, side='right') - 1
        return pyx_findspan(self.kv, self.p, u)

    def first_active(self, k):
        """Index of first active basis function in interval (kv[k], kv[k+1])"""
        return k - self.p

    def first_active_at(self, u):
        """Index of first active basis function in the interval which contains `u`."""
        return self.first_active(self.findspan(u))

    def greville(self):
        """Compute Gréville abscissae for this knot vector"""
        p = self.p
        if p == 0:
            return (self.kv[1:] + self.kv[:-1]) / 2     # cell middle points
        else:
            # running averages over p knots
            g = (np.convolve(self.kv, np.ones(p) / p))[p:-p]
            # due to rounding errors, some points may not be contained in the
            # support interval; clamp them manually to avoid problems later on
            return np.clip(g, self.kv[0], self.kv[-1])

    def refine(self, new_knots=None, mult=1):
        """Return the refinement of this knot vector by inserting `new_knots`,
        or performing uniform refinement if none are given."""
        if new_knots is None:
            mesh = self.mesh
            new_knots = (mesh[1:] + mesh[:-1]) / 2
            if mult>1:
                new_knots = np.hstack(mult*[new_knots,])
        kvnew = np.sort(np.concatenate((self.kv, new_knots)))
        return KnotVector(kvnew, self.p)

    def meshsize_avg(self):
        """Compute average length of the knot spans of this knot vector"""
        nspans = self.numspans
        support = abs(self.kv[-1] - self.kv[0])
        return support / nspans


def make_knots(p, a, b, n, mult=1):
    """Create an open knot vector of degree `p` over an interval `(a,b)` with `n` knot spans.

    This automatically repeats the first and last knots `p+1` times in order
    to create an open knot vector. Interior knots are single by default, i.e., have
    maximum continuity.

    Args:
        p (int): the spline degree
        a (float): the starting point of the interval
        b (float): the end point of the interval
        n (int): the number of knot spans to divide the interval into
        mult (int): the multiplicity of interior knots

    Returns:
        :class:`KnotVector`: the new knot vector
    """
    kv = np.concatenate(
            (np.repeat(a, p+1),
             np.repeat(np.arange(a, b, (b-a) / n)[1:], mult),
             np.repeat(b, p+1)))
    return KnotVector(kv, p)

def numdofs(kvs):
    """Convenience function which returns the number of dofs in a single knot vector
    or in a tensor product space represented by a tuple of knot vectors.
    """
    if isinstance(kvs, KnotVector):
        return kvs.numdofs
    else:
        return np.prod([kv.numdofs for kv in kvs])

################################################################################

def ev(knotvec, coeffs, u):
    """Evaluate a spline with given B-spline coefficients at all points `u`.

    Args:
        knotvec (:class:`KnotVector`): B-spline knot vector
        coeffs (`ndarray`): 1D array of coefficients, length `knotvec.numdofs`
        u (`ndarray`): 1D array of evaluation points

    Returns:
        `ndarray`: the vector of function values
    """
    assert len(coeffs) == knotvec.numdofs, 'Wrong size of coefficient vector'
    return scipy.interpolate.splev(u, (knotvec.kv, coeffs, knotvec.p))

def deriv(knotvec, coeffs, deriv, u):
    """Evaluate a derivative of the spline with given B-spline coefficients at all points `u`.

    Args:
        knotvec (:class:`KnotVector`): B-spline knot vector
        coeffs (`ndarray`): 1D array of coefficients, length `knotvec.numdofs`
        deriv (int): which derivative to evaluate
        u (`ndarray`): 1D array of evaluation points

    Returns:
        `ndarray`: the vector of function derivatives
    """
    assert len(coeffs) == knotvec.numdofs, 'Wrong size of coefficient vector'
    return scipy.interpolate.splev(u, (knotvec.kv, coeffs, knotvec.p), der=deriv)

################################################################################

# currently unused
#def _bspline_active_ev_single(knotvec, u):
#    """Evaluate all active B-spline basis functions at a single point `u`"""
#    kv, p = knotvec.kv, knotvec.p
#    left = np.empty(p)
#    right = np.empty(p)
#    result = np.empty(p+1)
#
#    span = knotvec.findspan(u)
#
#    result[0] = 1.0
#
#    for j in range(1, p+1):
#        left[j-1] = u - kv[span+1-j]
#        right[j-1] = kv[span+j] - u
#        saved = 0.0
#
#        for r in range(j):
#            temp = result[r] / (right[r] + left[j-r-1])
#            result[r] = saved + right[r] * temp
#            saved = left[j-r-1] * temp
#
#        result[j] = saved
#
#    return result

def active_ev(knotvec, u):
    """Evaluate all active B-spline basis functions at the points `u`.

    Returns an array of shape (p+1, u.size) if `u` is an array."""
    if np.isscalar(u):
        return active_ev(knotvec, np.array([u]))[:, 0]
    else:
        # use active_deriv(), which is implemented in Cython and much faster
        return active_deriv(knotvec, u, 0)[0, :]
        #result = np.empty((knotvec.p+1, u.size))
        #for i in range(u.size):
        #    result[:,i] = _bspline_active_ev_single(knotvec, u[i])
        #return result

# optimized in bspline_cy
#def _bspline_active_deriv_single(knotvec, u, numderiv):
#    """Evaluate all active B-spline basis functions and their derivatives
#    up to `numderiv` at a single point `u`"""
#    kv, p = knotvec.kv, knotvec.p
#    NDU   = np.empty((p+1, p+1))
#    left  = np.empty(p)
#    right = np.empty(p)
#    result = np.empty((numderiv+1, p+1))
#
#    span = knotvec.findspan(u)
#
#    NDU[0,0] = 1.0
#
#    for j in range(1, p+1):
#        # Compute knot splits
#        left[j-1]  = u - kv[span+1-j]
#        right[j-1] = kv[span+j] - u
#        saved = 0.0
#
#        for r in range(j):     # For all but the last basis functions of degree j (ndu row)
#            # Strictly lower triangular part: Knot differences of distance j
#            NDU[j, r] = right[r] + left[j-r-1]
#            temp = NDU[r, j-1] / NDU[j, r]
#            # Upper triangular part: Basis functions of degree j
#            NDU[r, j] = saved + right[r] * temp  # r-th function value of degree j
#            saved = left[j-r-1] * temp
#
#        # Diagonal: j-th (last) function value of degree j
#        NDU[j, j] = saved
#
#    # copy function values into result array
#    result[0, :] = NDU[:, -1]
#
#    a1 = np.empty(p+1)
#    a2 = np.empty(p+1)
#
#    for r in range(p+1):    # loop over basis functions
#        a1[0] = 1.0
#
#        fac = p        # fac = fac(p) / fac(p-k)
#
#        # Compute the k-th derivative of the r-th basis function
#        for k in range(1, numderiv+1):
#            rk = r - k
#            pk = p - k
#            d = 0.0
#
#            if r >= k:
#                a2[0] = a1[0] / NDU[pk+1, rk]
#                d = a2[0] * NDU[rk, pk]
#
#            j1 = 1 if rk >= -1  else -rk
#            j2 = k-1 if r-1 <= pk else p - r
#
#            for j in range(j1, j2+1):
#                a2[j] = (a1[j] - a1[j-1]) / NDU[pk+1, rk+j]
#                d += a2[j] * NDU[rk+j, pk]
#
#            if r <= pk:
#                a2[k] = -a1[k-1] / NDU[pk+1, r]
#                d += a2[k] * NDU[r, pk]
#
#            result[k, r] = d * fac
#            fac *= pk          # update fac = fac(p) / fac(p-k) for next k
#
#            # swap rows a1 and a2
#            (a1,a2) = (a2,a1)
#
#    return result

# optimized in bspline_cy
#def active_deriv(knotvec, u, numderiv):
#    """Evaluate all active B-spline basis functions and their derivatives
#    up to `numderiv` at the points `u`.
#
#    Returns an array with shape (numderiv+1, p+1) if `u` is scalar or
#    an array with shape (numderiv+1, p+1, u.size) otherwise.
#    """
#    if np.isscalar(u):
#        return _bspline_active_deriv_single(knotvec, u, numderiv)
#    else:
#        result = np.empty((numderiv+1, knotvec.p+1, u.size))
#        for i in range(u.size):
#            result[:,:,i] = _bspline_active_deriv_single(knotvec, u[i], numderiv)
#        return result


def _bspline_single_ev_single(knotvec, i, u):
    """Evaluate i-th B-spline at single point `u`"""
    kv, p = knotvec.kv, knotvec.p
    m = kv.size

    # special case: boundary conditions for first and last B-spline
    if (i == 0 and u == kv[0]) or (i == m-p-2 and u == kv[-1]):
        return 1.0

    # check if u within support of B-spline
    if u < kv[i] or u >= kv[i+p+1]:
        return 0.0

    # initialize zeroth-degree functions
    N = np.empty(p+1)
    for j in range(p+1):
        if u >= kv[i+j] and u < kv[i+j+1]:
            N[j] = 1.0
        else:
            N[j] = 0.0

    # compute triangular table
    for k in range(1, p+1):
        if N[0] == 0.0:
            saved = 0.0
        else:
            saved = ((u - kv[i]) * N[0]) / (kv[i+k] - kv[i])

        for j in range(0, p-k+1):
            kleft = kv[i+j+1]
            kright = kv[i+j+k+1]
            if N[j+1] == 0.0:
                N[j] = saved
                saved = 0.0
            else:
                temp = N[j+1] / (kright - kleft)
                N[j] = saved + (kright - u)*temp
                saved = (u - kleft) * temp

    return N[0]

def single_ev(knotvec, i, u):
    """Evaluate i-th B-spline at all points `u`"""
    if np.isscalar(u):
        return _bspline_single_ev_single(knotvec, i, u)
    else:
        result = np.empty(u.size)
        for j in range(u.size):
            result[j] = _bspline_single_ev_single(knotvec, i, u[j])
        return result

def tp_bsp_eval_pointwise(kvs, coeffs, points):
    """Evaluate a tensor-product B-spline function at an unstructured list of points.

    Args:
        kvs: tuple of :class:`KnotVector` instances representing a
            tensor-product B-spline basis
        coeffs (ndarray): coefficient array; see :class:`BSplineFunc` for details
        points: an array or sequence such that `points[i]` is an array containing
            the coordinates for dimension `i`, where `i = 0, ..., len(kvs) - 1`
            (in xyz order). All arrays must have the same shape.

    Returns:
        An `ndarray` containing the function values of the spline function at
        the `points`.
    """
    if not all(x.shape == points[0].shape for x in points):
        raise ValueError('All coordinate arrays should have the same shape')
    XY = tuple(points[d].ravel() for d in range(len(points)))
    sdim, n = len(XY), len(XY[0])   # source dimension, number of points
    # collocation info (indices and coefficients) for the evaluation nodes
    # (NB: axes are in zyx order)
    coll = [collocation_info(kvs[d], XY[1-d]) for d in range(sdim)]
    pp1 = tuple(kv.p + 1 for kv in kvs)

    # build einsum index string, e.g.: 'i,j,k,ijk...' for the sum
    #   sum_ijk (u_i * v_j * w_k * C_ijk)
    indices = tuple(chr(ord('i') + d) for d in range(sdim))  # i, j, k, ...
    einsum_str = ','.join(indices) + ',' + ''.join(indices) + '...'

    input_shape = points[0].shape
    output_shape = coeffs.shape[sdim:]
    result = np.empty((n,) + output_shape)

    for k in range(n):
        Is = tuple(coll[d][0][k] for d in range(sdim))    # index of first active basis function
        cs = tuple(coll[d][1][k] for d in range(sdim))    # coefficient vector for p+1 basis functions
        # construct slice of coefficient array containing the active basis functions (p+1 per dimension)
        slices = tuple(slice(ii, ii+pp) for (ii, pp) in zip(Is, pp1))
        #vals = tensor.apply_tprod(cs, coeffs[slices])
        # the following is equivalent but faster:
        vals = np.einsum(einsum_str, *cs, coeffs[slices])
        result[k] = vals
    result.shape = input_shape + output_shape   # bring result into proper shape
    return result

def tp_bsp_jac_pointwise(kvs, coeffs, points):
    """Evaluate the Jacobian of a tensor-product B-spline function at an
    unstructured list of points.

    Args:
        kvs: tuple of :class:`KnotVector` instances representing a
            tensor-product B-spline basis
        coeffs (ndarray): coefficient array; see :class:`BSplineFunc` for details
        points: an array or sequence such that `points[i]` is an array containing
            the coordinates for dimension `i`, where `i = 0, ..., len(kvs) - 1`
            (in xyz order). All arrays must have the same shape.

    Returns:
        An `ndarray` containing the Jacobians of the spline function at the
        `points`.
    """
    if not all(x.shape == points[0].shape for x in points):
        raise ValueError('All coordinate arrays should have the same shape')
    XY = tuple(points[d].ravel() for d in range(len(points)))
    sdim, n = len(XY), len(XY[0])   # source dimension, number of points
    # collocation info (indices and coefficients) for the evaluation nodes
    # (NB: axes are in zyx order)
    coll = [collocation_derivs_info(kvs[d], XY[1-d], derivs=1) for d in range(sdim)]
    pp1 = tuple(kv.p + 1 for kv in kvs)

    # build einsum index string, e.g.: 'i,j,k,ijk...' for the sum
    #   sum_ijk (u_i * v_j * w_k * C_ijk)
    indices = tuple(chr(ord('i') + d) for d in range(sdim))  # i, j, k, ...
    einsum_str = ','.join(indices) + ',' + ''.join(indices) + '...'

    input_shape = points[0].shape
    output_shape = coeffs.shape[sdim:] + (sdim,)    # last axis is derivative
    result = np.empty((n,) + output_shape)

    for k in range(n):
        Is = tuple(coll[d][0][k] for d in range(sdim))      # index of first active basis function
        cs = tuple(coll[d][1][0,k] for d in range(sdim))    # coefficient vector for values
        ds = tuple(coll[d][1][1,k] for d in range(sdim))    # coefficient vector for derivative

        # construct slice of coefficient array containing the active basis functions (p+1 per dimension)
        slices = tuple(slice(ii, ii+pp) for (ii, pp) in zip(Is, pp1))
        C_active = coeffs[slices]

        for i in range(sdim):
            ops = [(ds[j] if j==i else cs[j]) for j in range(sdim)] # deriv. in i-th direction
            vals = np.einsum(einsum_str, *ops, C_active)
            result[k, :, sdim - i - 1] = vals   # x-component is the last one
    result.shape = input_shape + output_shape   # bring result into proper shape
    return result

def tp_bsp_eval_with_jac_pointwise(kvs, coeffs, points):
    """Evaluate the values and Jacobians of a tensor-product B-spline
    function at an unstructured list of points.

    Args:
        kvs: tuple of :class:`KnotVector` instances representing a
            tensor-product B-spline basis
        coeffs (ndarray): coefficient array; see :class:`BSplineFunc` for details
        points: an array or sequence such that `points[i]` is an array containing
            the coordinates for dimension `i`, where `i = 0, ..., len(kvs) - 1`
            (in xyz order). All arrays must have the same shape.

    Returns:
        A pair of `ndarray`s: one for the values and one for the Jacobians.
    """
    if not all(x.shape == points[0].shape for x in points):
        raise ValueError('All coordinate arrays should have the same shape')
    XY = tuple(points[d].ravel() for d in range(len(points)))
    sdim, n = len(XY), len(XY[0])   # source dimension, number of points
    # collocation info (indices and coefficients) for the evaluation nodes
    # (NB: axes are in zyx order)
    coll = [collocation_derivs_info(kvs[d], XY[1-d], derivs=1) for d in range(sdim)]
    pp1 = tuple(kv.p + 1 for kv in kvs)

    # build einsum index string, e.g.: 'i,j,k,ijk...' for the sum
    #   sum_ijk (u_i * v_j * w_k * C_ijk)
    indices = tuple(chr(ord('i') + d) for d in range(sdim))  # i, j, k, ...
    einsum_str = ','.join(indices) + ',' + ''.join(indices) + '...'

    input_shape = points[0].shape
    val_shape = coeffs.shape[sdim:]
    jac_shape = coeffs.shape[sdim:] + (sdim,)    # last axis is derivative
    result_val = np.empty((n,) + val_shape)
    result_jac = np.empty((n,) + jac_shape)

    for k in range(n):
        Is = tuple(coll[d][0][k] for d in range(sdim))      # index of first active basis function
        cs = tuple(coll[d][1][0,k] for d in range(sdim))    # coefficient vector for values
        ds = tuple(coll[d][1][1,k] for d in range(sdim))    # coefficient vector for derivative

        # construct slice of coefficient array containing the active basis functions (p+1 per dimension)
        slices = tuple(slice(ii, ii+pp) for (ii, pp) in zip(Is, pp1))
        C_active = coeffs[slices]

        # evaluate function value
        result_val[k] = np.einsum(einsum_str, *cs, C_active)

        # evaluate Jacobian
        for i in range(sdim):
            ops = [(ds[j] if j==i else cs[j]) for j in range(sdim)] # deriv. in i-th direction
            vals = np.einsum(einsum_str, *ops, C_active)
            result_jac[k, :, sdim - i - 1] = vals   # x-component is the last one

    # bring results into proper shape
    result_val.shape = input_shape + val_shape
    result_jac.shape = input_shape + jac_shape
    return result_val, result_jac

################################################################################

def collocation(kv, nodes):
    """Compute collocation matrix for B-spline basis at the given interpolation nodes.

    Args:
        kv (:class:`KnotVector`): the B-spline knot vector
        nodes (array): array of nodes at which to evaluate the B-splines

    Returns:
        A Scipy CSR matrix with shape `(len(nodes), kv.numdofs)` whose entry at
        `(i,j)` is the value of the `j`-th B-spline evaluated at `nodes[i]`.
    """
    nodes = np.ascontiguousarray(nodes)
    indices, values = collocation_info(kv, nodes)
    m, n = nodes.size, kv.numdofs       # collocation matrix size

    # compute I, J indices:
    # I: p + 1 entries per row
    I = np.repeat(np.arange(m), kv.p + 1)
    # J: arange(indices[k], indices[k] + p + 1) per row
    J = (indices[:, None] + np.arange(kv.p + 1)[None, :]).ravel()

    return scipy.sparse.coo_matrix((values.ravel(), (I,J)), shape=(m,n)).tocsr()

def collocation_tp(kvs, gridaxes):
    """Compute collocation matrix for Tensor product B-spline basis at the given interpolation grid"""
    dim=len(kvs)
    assert len(gridaxes)==dim,"Input has wrong dimension."
    if not all(np.ndim(ax) == 1 for ax in gridaxes):
            gridaxes = tuple(np.squeeze(ax) for ax in gridaxes)
            assert all(ax.ndim == 1 for ax in gridaxes), \
                "Grid axes should be one-dimensional"
    
    Colloc = [collocation(kvs[d],gridaxes[d]) for d in range(dim)]
    C = Colloc[0]
    for d in range(1,dim):
        C = scipy.sparse.kron(C,Colloc[d])
    return C

def collocation_info(kv, nodes):
    """Return two arrays: one containing the index of the first active B-spline
    per evaluation node, and one containing, per node, the coefficient vector
    of length `p+1` for the linear combination of basis functions which yields
    the point evaluation at that node.

    Corresponds to a row-wise representation of the collocation matrix (see
    :func:`collocation`).
    """
    nodes = np.ascontiguousarray(nodes) # pyx_findspans requires a contiguous array
    values = active_ev(kv, nodes) # (p+1) x n
    #indices = [kv.first_active_at(u) for u in nodes]
    indices = pyx_findspans(kv.kv, kv.p, nodes) - kv.p        # faster version
    return indices, np.asarray(values.T)

def collocation_derivs(kv, nodes, derivs=1):
    """Compute collocation matrix and derivative collocation matrices for B-spline
    basis at the given interpolation nodes.

    Returns a list of derivs+1 sparse CSR matrices with shape (nodes.size, kv.numdofs)."""
    nodes = np.array(nodes, copy=False)
    m = nodes.size
    n = kv.numdofs
    p = kv.p
    indices, values = collocation_derivs_info(kv, nodes, derivs)

    # compute I, J indices:
    # I: p + 1 entries per row
    I = np.repeat(np.arange(m), p + 1)
    # J: arange(indices[k], indices[k] + p + 1) per row
    J = (indices[:, None] + np.arange(p + 1)[None, :]).ravel()

    return [scipy.sparse.coo_matrix((values[d].ravel(), (I,J)), shape=(m,n)).tocsr()
            for d in range(derivs + 1)]

def collocation_derivs_tp(kvs, gridaxes, derivs=1):
    """Compute collocation matrix and derivative collocation matrices for Tensor product B-spline
    basis at the given grid.

    Returns a list of derivs+1 lists containing collocation matrices for derivatives of order derivs as sparse CSR matrices with shape (m,n) where me is the number of gridpoints
    and n the  overall number of basis functions related to kvs."""
    dim=len(kvs)
    assert len(gridaxes)==dim,"Input has wrong dimension."
    if not all(np.ndim(ax) == 1 for ax in gridaxes):
            gridaxes = tuple(np.squeeze(ax) for ax in gridaxes)
            assert all(ax.ndim == 1 for ax in gridaxes), \
                "Grid axes should be one-dimensional"
    
    Colloc = [collocation_derivs(kvs[d],gridaxes[d],derivs=derivs) for d in range(dim)]
    D = [[] for k in range(derivs+1)]
    for k in range(derivs+1):
        for ind in multi_indices(dim,k):
            C=Colloc[0][ind[0]]
            for d in range(1,dim):
                C = scipy.sparse.kron(C,Colloc[d][ind[d]])
            D[k].append(C)
    return D

def collocation_derivs_info(kv, nodes, derivs=1):
    """Similar to :func:`collocation_info`, but the second array also contains
    coefficients for computing derivatives up to order `derivs`.  It has shape
    `(derivs + 1) x len(nodes) x (p + 1)`.

    Corresponds to a row-wise representation of the matrices computed by
    :func:`collocation_derivs`.
    """
    nodes = np.ascontiguousarray(nodes) # pyx_findspans requires a contiguous array
    values = active_deriv(kv, nodes, derivs)    # (derivs+1) x (p+1) x n
    indices = pyx_findspans(kv.kv, kv.p, nodes) - kv.p
    return indices, np.asarray(values).swapaxes(-2, -1) # (derivs+1) x n x (p+1)

def interpolate(kv, func, nodes=None):
    """Interpolate function in B-spline basis at given nodes (or Gréville abscissae by default)"""
    if nodes is None:
        nodes = kv.greville()
    else:
        nodes = np.array(nodes, copy=False)
    C = collocation(kv, nodes)
    vals = func(nodes)
    return scipy.sparse.linalg.spsolve(C, vals)

################################################################################

def load_vector(kv, f):
    """Compute the load vector (L_2 inner products of basis functions with `f`)."""
    from .quadrature import make_iterated_quadrature
    nqp = kv.p + 1
    q = make_iterated_quadrature(kv.mesh, nqp)
    C = collocation(kv, q[0])
    fvals = q[1] * f(q[0])  # values weighted with quadrature weights
    return C.T.dot(fvals)

def project_L2(kv, f):
    """Compute the B-spline coefficients for the L_2-projection of `f`."""
    from .assemble import bsp_mass_1d
    lv = load_vector(kv, f)
    M = bsp_mass_1d(kv)
    return scipy.sparse.linalg.spsolve(M, lv)

################################################################################

def prolongation(kv1, kv2):
    """Compute prolongation matrix between B-spline bases.

    Given two B-spline bases, where the first spans a subspace of the second
    one, compute the matrix which maps spline coefficients from the first
    basis to the coefficients of the same function in the second basis.

    Args:
        kv1 (:class:`KnotVector`): source B-spline basis knot vector
        kv2 (:class:`KnotVector`): target B-spline basis knot vector

    Returns:
        csr_matrix: sparse matrix which prolongs coefficients from `kv1` to `kv2`
    """
    g = kv2.greville()
    C1 = collocation(kv1, g).A
    C2 = collocation(kv2, g)
    P = scipy.sparse.linalg.spsolve(C2, C1)
    # prune matrix
    P[np.abs(P) < 1e-15] = 0.0
    return scipy.sparse.csr_matrix(P)

def prolongation_tp(kvs1, kvs2):
    """Compute prolongation matrix between Tensor product B-spline bases.

    Given two Tensor product B-spline bases, where the first spans a subspace of the second
    one, compute the matrix which maps spline coefficients from the first
    basis to the coefficients of the same function in the second basis.

    Args:
        kvs1 (tuple(:class:`KnotVector`)): tuple of source B-spline basis knot vectors
        kvs2 (tuple(:class:`KnotVector`)): tuple of target B-spline basis knot vectors

    Returns:
        csr_matrix: sparse matrix which prolongs coefficients from `kvs1` to `kvs2`
    """
    assert len(kvs1)==len(kvs2), "Number of dimensions does not match!"
    dim = len(kvs1)
    
    Prol = [prolongation(kvs1[d],kvs2[d]) for d in range(dim)]
    P = Prol[0]
    for d in range(1,dim):
        P = scipy.sparse.kron(P, Prol[d])
    P=scipy.sparse.csr_matrix(P)
    P.eliminate_zeros()
    return P
    
def knot_insertion(kv, u):
    """Return a sparse matrix of size `(n+1) x n`, with `n = kv.numdofs´, which
    maps coefficients from `kv` to a new knot vector obtained by inserting the
    new knot `u` into `kv`.
    """
    n, p = kv.numdofs, kv.p
    k = kv.findspan(u)

    P = scipy.sparse.lil_matrix((n+1, n))

    # coefficients outside the affected area are left unchanged
    for i in range(k - p + 1):
        P[i, i] = 1.0
    for i in range(k + 1, n + 1):
        P[i, i-1] = 1.0

    knots = kv.kv
    for i in reversed(range(k - p + 1, k + 1)):
        a = (u - knots[i]) / (knots[i + p] - knots[i])
        P[i, i - 1] = 1 - a
        P[i, i]     = a

    return P.tocsr()

################################################################################

class _BaseGeoFunc:
    def __call__(self, *x):
        return self.eval(*x)

    def is_scalar(self):
        """Returns True if the function is scalar-valued."""
        return len(self.output_shape()) == 0

    def is_vector(self):
        """Returns True if the function is vector-valued."""
        return len(self.output_shape()) == 1

    def bounding_box(self, grid=1):
        """Compute a bounding box for the image of this geometry.

        By default, only the corners are taken into account. By choosing
        `grid > 1`, a finer grid can be used (for non-convex geometries).

        Returns:
            a tuple of `(lower,upper)` limits per dimension (in XY order)
        """
        supp = self.support
        grid = [np.linspace(s[0], s[1], grid+1) for s in supp]
        X = self.grid_eval(grid)
        X.shape = (-1, self.dim)
        return tuple((X[:, d].min(), X[:, d].max()) for d in range(self.dim))

    def find_inverse(self, x, tol=1e-8):
        """Find the coordinates in the parameter domain which correspond to the
        physical point `x`.
        """
        import scipy.optimize

        supp = np.transpose(self.support)   # two rows (min/max), columns are coordinates

        result = scipy.optimize.least_squares(
            lambda xi: self(*xi) - x,       # components of cost function
            np.mean(supp, axis=0),          # starting value (center of parameter domain)
            bounds=supp,                    # don't leave parameter domain
            method='dogbox',
            ftol=tol, xtol=tol, gtol=1e-15
        )
        if result.success and np.sqrt(result.cost) < tol:
            return result.x
        else:
            raise ValueError('Could not find coordinates for desired point %s' % (x,))

    def boundary(self, bdspec, flip = None):
        """Return one side of the boundary as a :class:`.UserFunction`.

        Args:
            bdspec: the side of the boundary to return; see :func:`.compute_dirichlet_bc`

        Returns:
            :class:`.UserFunction`: representation of the boundary side;
            has `sdim` reduced by 1 and the same `dim` as this function
        """
        from .geometry import _BoundaryFunction
        return _BoundaryFunction(self, bdspec, flip=flip)

class _BaseSplineFunc(_BaseGeoFunc):
    def eval(self, *x):
        """Evaluate the function at a single point of the parameter domain.

        Args:
            *x: the point at which to evaluate the function, in xyz order
        """
        def as_array(t):
            if np.isscalar(t):
                return np.asarray([t], dtype=float)
            else:
                return np.asanyarray(t, dtype=float)
        coords = tuple(reversed(x))     # XYZ->ZYX order
        singletons = tuple(i for i in range(self.sdim) if np.isscalar(coords[i]))
        coords = tuple(as_array(t) for t in reversed(x))
        y = self.grid_eval(coords).squeeze(axis=singletons)
        if y.shape == ():
            y = y.item()
        return y

class BSplineFunc(_BaseSplineFunc):
    """Any function that is given in terms of a tensor product B-spline basis with coefficients.

    Arguments:
        kvs (seq): tuple of `d` :class:`KnotVector`.
        coeffs (ndarray): coefficient array

    `kvs` represents a tensor product B-spline basis, where the *i*-th
    :class:`KnotVector` describes the B-spline basis in the *i*-th
    coordinate direction.

    `coeffs` is the array of coefficients with respect to this tensor product basis.
    The length of its first `d` axes must match the number of degrees of freedom
    in the corresponding :class:`KnotVector`.
    Trailing axes, if any, determine the output dimension of the function.
    If there are no trailing dimensions or only a single one of length 1,
    the function is scalar-valued.

    For convenience, if `coeffs` is a vector, it is reshaped to the proper
    size for the tensor product basis. The result is a scalar-valued function.

    Attributes:
        kvs (seq): the knot vectors representing the tensor product basis
        coeffs (ndarray): the coefficients for the function or geometry
        sdim (int): dimension of the parameter domain
        dim (int): dimension of the output of the function
    """
    def __init__(self, kvs, coeffs):
        if isinstance(kvs, KnotVector):
            kvs = (kvs,)
        self.kvs = tuple(kvs)
        self.sdim = len(kvs)    # source dimension

        N = tuple(kv.numdofs for kv in kvs)
        coeffs = np.asanyarray(coeffs)
        if coeffs.ndim == 1:
            assert coeffs.shape[0] == np.prod(N), "Wrong length of coefficient vector"
            coeffs = coeffs.reshape(N)
        assert N == coeffs.shape[:self.sdim], "Wrong shape of coefficients"
        self.coeffs = coeffs

        # determine target dimension
        dim = coeffs.shape[self.sdim:]
        if len(dim) == 0:
            dim = 1
        elif len(dim) == 1:
            dim = dim[0]
        self.dim = dim

        self._support_override = None

    def output_shape(self):
        return self.coeffs.shape[self.sdim:]

    def grid_eval(self, gridaxes):
        """Evaluate the function on a tensor product grid.

        Args:
            gridaxes (seq): list of 1D vectors describing the tensor product grid.

        .. note::

            The gridaxes should be given in reverse order, i.e.,
            the x axis last.

        Returns:
            ndarray: array of function values; shape corresponds to input grid.
        """
        assert len(gridaxes) == self.sdim, "Input has wrong dimension"
        # make sure axes are one-dimensional
        if not all(np.ndim(ax) == 1 for ax in gridaxes):
            gridaxes = tuple(np.squeeze(ax) for ax in gridaxes)
            assert all(ax.ndim == 1 for ax in gridaxes), \
                "Grid axes should be one-dimensional"
        colloc = [collocation(self.kvs[i], gridaxes[i]) for i in range(self.sdim)]
        return apply_tprod(colloc, self.coeffs)

    def grid_jacobian(self, gridaxes):
        """Evaluate the Jacobian on a tensor product grid.

        Args:
            gridaxes (seq): list of 1D vectors describing the tensor product grid.

        .. note::

            The gridaxes should be given in reverse order, i.e.,
            the x axis last.

        Returns:
            ndarray: array of Jacobians (:attr:`dim` × :attr:`sdim`); shape
            corresponds to input grid.  For scalar functions, the output is a
            vector of length :attr:`sdim` (the gradient) per grid point.
        """
        assert len(gridaxes) == self.sdim, "Input has wrong dimension"

        colloc = [collocation_derivs(self.kvs[i], gridaxes[i], derivs=1) for i in range(self.sdim)]

        grad_components = []
        for i in reversed(range(self.sdim)):  # x-component is the last one
            ops = [colloc[j][1 if j==i else 0] for j in range(self.sdim)] # deriv. in i-th direction
            grad_components.append(apply_tprod(ops, self.coeffs))   # shape: shape(grid) x self.dim
        return np.stack(grad_components, axis=-1)   # shape: shape(grid) x self.dim x self.sdim
    
    def grid_outer_normal(self, gridaxes):
        gridaxes = list(gridaxes)
        N = [len(grid) for grid in gridaxes]
        #gridaxes.insert(self.axis, np.array([self.fixed_coord]))
        jacs = self.grid_jacobian(gridaxes)
        if self.dim==2 and self.sdim==1:     # line integral
            x = jacs
            #di=-1 if self.axis != self.side else 1
            x[:,0]=-x[:,0]
            x[:,[0,1]]=x[:,[1,0]]
            return x/np.linalg.norm(x,axis=1)[:,None]
        elif self.dim==3 and self.sdim==2:   # surface integral
            #di=-1 if (self.axis+self.side)%2==0 else 1
            x, y = jacs[:,:,:,0], jacs[:,:,:,1]
            un=np.cross(x, y).reshape(N[0],N[1],3,1)
            return un/np.linalg.norm(un,axis=2)[:,:,None]
        else:
            assert False, 'do not know how to compute normal vector for Jacobian shape {}'.format(jacs.shape)

    def grid_hessian(self, gridaxes):
        """Evaluate the Hessian matrix of a scalar or vector function on a tensor product grid.

        Args:
            gridaxes (seq): list of 1D vectors describing the tensor product grid.

        .. note::

            The gridaxes should be given in reverse order, i.e.,
            the x axis last.

        Returns:
            ndarray: array of the components of the Hessian; symmetric part
            only, linearized. I.e., in 2D, it contains per grid point a
            3-component vector corresponding to the derivatives `(d_xx, d_xy,
            d_yy)`, and in 3D, a 6-component vector with the derivatives
            `(d_xx, d_xy, d_xz, d_yy, d_yz, d_zz)`. If the input function is
            vector-valued, one such Hessian vector is computed per component
            of the function.

            Thus, the output is an array of shape `grid_shape x num_comp x num_hess`,
            where `grid_shape` is the shape of the tensor grid described by the
            `gridaxes`, `num_comp` is the number of components of the function,
            and `num_hess` is the number of entries in the symmetric part of the
            Hessian as described above. The axis corresponding to `num_comp` is
            elided if the input function is scalar.
        """
        assert np.isscalar(self.dim), 'Hessian only implemented for scalar and vector functions'
        assert len(gridaxes) == self.sdim, "Input has wrong dimension"
        colloc = [collocation_derivs(self.kvs[i], gridaxes[i], derivs=2) for i in range(self.sdim)]

        d = self.sdim
        n_hess = ((d+1)*d) // 2         # number of components in symmetric part of Hessian
        N = tuple(len(g) for g in gridaxes)     # shape of tensor grid

        # determine size of output array
        if self.dim == 1:
            out_shape = N + (n_hess,)
        else:
            out_shape = N + (self.dim, n_hess)
        hess = np.empty(out_shape, dtype=self.coeffs.dtype)

        i_hess = 0
        for i in reversed(range(self.sdim)):  # x-component is the last one
            for j in reversed(range(i+1)):
                # compute vector of derivative indices
                D = self.sdim * [0]
                D[i] += 1
                D[j] += 1
                ops = [colloc[k][D[k]] for k in range(self.sdim)] # derivatives in directions i,j

                if self.dim == 1:   # scalar function
                    hess[..., i_hess] = apply_tprod(ops, self.coeffs) # D_i D_j (self)
                else:               # vector function
                    for k in range(self.dim):
                        hess[..., k, i_hess] = apply_tprod(ops, self.coeffs[..., k])    # D_i D_j (self[k])
                i_hess += 1
        return hess   # shape: shape(grid) x self.dim x n_hess

    def pointwise_eval(self, points):
        """Evaluate the B-spline function at an unstructured list of points.

        Args:
            points: an array or sequence such that `points[i]` is an array containing
                the coordinates for dimension `i`, where `i = 0, ..., sdim - 1`
                (in xyz order). All arrays must have the same shape.

        Returns:
            An `ndarray` containing the function values at the `points`.
        """
        return tp_bsp_eval_pointwise(self.kvs, self.coeffs, points)

    def pointwise_jacobian(self, points):
        """Evaluate the Jacobian of the B-spline function at an unstructured list of points.

        Args:
            points: an array or sequence such that `points[i]` is an array containing
                the coordinates for dimension `i`, where `i = 0, ..., sdim - 1`
                (in xyz order). All arrays must have the same shape.

        Returns:
            An `ndarray` containing the Jacobian matrices at the `points`,
            i.e., a matrix of size `dim x sdim` per evaluation point.
        """
        return tp_bsp_jac_pointwise(self.kvs, self.coeffs, points)

    def transformed_jacobian(self, geo):
        """Create a function which evaluates the physical (transformed) gradient of the current
        function after a geometry transform."""
        return PhysicalGradientFunc(self, geo)

    def boundary(self, bdspec):
        """Return one side of the boundary as a :class:`BSplineFunc`.

        Args:
            bdspec: the side of the boundary to return; see :func:`.compute_dirichlet_bc`

        Returns:
            :class:`BSplineFunc`: representation of the boundary side;
            has :attr:`sdim` reduced by 1 and the same :attr:`dim` as this function
        """
        if self._support_override:
            # if we have reduced support, the boundary may not be
            # interpolatory; return a custom function
            return _BaseGeoFunc.boundary(self, bdspec)
        
        bdspec = _parse_bdspec(bdspec, self.sdim)
        axis, sides = tuple(ax for ax, _ in bdspec), tuple(-idx for _, idx in bdspec)

        assert all([0 <= ax < self.sdim for ax in axis]), 'Invalid axis'
        slices = self.sdim * [slice(None)]
        for ax, idx in zip(axis, sides):
            slices[ax] = idx
        coeffs = self.coeffs[tuple(slices)]
        kvs = list(self.kvs)
        for ax in sorted(axis,reverse=True):
            del kvs[ax]
        return BSplineFunc(kvs, coeffs)

    @property
    def support(self):
        """Return a sequence of pairs `(lower,upper)`, one per source dimension,
        which describe the extent of the support in the parameter space."""
        if self._support_override:
            return self._support_override
        else:
            return tuple(kv.support() for kv in self.kvs)

    @support.setter
    def support(self, new_support):
        new_support = tuple(new_support)
        assert len(new_support) == self.sdim, 'wrong number of dimensions'
        assert all(len(supp_k) == 2 for supp_k in new_support), 'each entry should be a pair (lower,upper)'
        self._support_override = new_support

    def copy(self):
        """Return a copy of this geometry."""
        return BSplineFunc(
                tuple(kv.copy() for kv in self.kvs),
                self.coeffs.copy())

    def translate(self, offset):
        """Return a version of this geometry translated by the specified offset."""
        return BSplineFunc(self.kvs, self.coeffs + offset)

    def scale(self, factor):
        """Scale all control points either by a scalar factor or componentwise by
        a vector and return the resulting new function.
        """
        return BSplineFunc(self.kvs, self.coeffs * factor)

    def apply_matrix(self, A):
        """Apply a matrix to each control point of this function and return the result.

        `A` should either be a single matrix or an array of matrices, one for each
        control point. Standard numpy broadcasting rules apply.
        """
        assert self.is_vector(), 'Can only apply matrices to vector-valued functions'
        C = np.matmul(A, self.coeffs[..., None])
        assert C.shape[-1] == 1  # this should have created a new singleton axis
        return BSplineFunc(self.kvs, np.squeeze(C, axis=-1))

    def rotate_2d(self, angle):
        """Rotate a geometry with :attr:`dim` = 2 by the given angle and return the result."""
        assert self.dim == 2, 'Must be 2D vector function'
        s, c = np.sin(angle), np.cos(angle)
        R = np.array([
            [c, -s],
            [s, c]
        ])
        return self.apply_matrix(R)
    
    def rotate_3d(self, angle, n):
        """Rotate a geometry with :attr:`dim` = 3 by the given angle around a given line generated by n and return the result."""
        assert self.dim == 3, 'Must be 3D vector function'
        (n1,n2,n3) = n = n/np.linalg.norm(n)
        s, c = np.sin(angle), np.cos(angle)
        R = np.array([
            [n1**2*(1-c) + c   , n1*n2*(1-c) - n3*s, n1*n3*(1-c) + n2*s],
            [n1*n2*(1-c) + n3*s, n2**2*(1-c) + c   , n2*n3*(1-c) - n1*s],
            [n1*n3*(1-c) - n2*s, n2*n3*(1-c) + n1*s, n3**2*(1-c) + c   ]
        ])
        return self.apply_matrix(R)

    def perturb(self, noise):
        """Create a copy of this function where all coefficients are randomly perturbed
        by noise of the given magnitude."""
        return BSplineFunc(self.kvs,
            self.coeffs + 2*noise*(np.random.random_sample(self.coeffs.shape) - 0.5))

    def cylinderize(self, z0=0.0, z1=1.0, support=(0.0, 1.0)):
        """Create a patch with one additional space dimension by
        linearly extruding along the new axis from `z0` to `z1`.

        By default, the new knot vector will be defined over the
        interval (0, 1). A different interval can be specified through
        the `support` parameter.
        """
        from .geometry import tensor_product, line_segment
        return tensor_product(line_segment(z0, z1, support=support), self)

    def as_nurbs(self):
        """Return a NURBS version of this function with constant weights equal to 1."""
        from .geometry import NurbsFunc
        return NurbsFunc(self.kvs, self.coeffs.copy(), np.ones(self.coeffs.shape[:self.sdim]))

    def as_vector(self):
        """Convert a scalar function to a 1D vector function."""
        if self.is_vector():
            return self
        else:
            assert self.is_scalar()
            return BSplineFunc(self.kvs, self.coeffs[..., np.newaxis])

    def __getitem__(self, I):
        return BSplineFunc(self.kvs, self.coeffs[..., I])


class PhysicalGradientFunc(_BaseGeoFunc):
    """A class for function objects which evaluate physical (transformed) gradients of
    scalar functions with geometry transforms.
    """
    def __init__(self, func, geo):
        assert func.dim == 1, 'Transformed gradients only implemented for scalar functions'
        self.func = func
        self.geo = geo
        self.dim = self.sdim = func.sdim
        self.support = func.support

    def output_shape(self):
        return self.func.output_shape() + (self.sdim,)

    def grid_eval(self, gridaxes):
        geojac = self.geo.grid_jacobian(gridaxes)
        geojacinvT = np.linalg.inv(geojac).swapaxes(-2, -1)

        u_grad = self.func.grid_jacobian(gridaxes)
        return np.matmul(geojacinvT, u_grad[..., None])[..., 0]

################################################################################

from .bspline_cy import *

