# -*- coding: utf-8 -*-
"""Functions and classes for B-spline basis functions.

"""

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import scipy.interpolate

from .tensor import apply_tprod

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
        self.p = p
        self._mesh = None    # knots with duplicates removed (on demand)
        self._knots_to_mesh = None   # knot indices to mesh indices (on demand)

    def __str__(self):
        return '<KnotVector p=%d sz=%d>' % (self.p, self.kv.size)

    def __repr__(self):
        return 'KnotVector(%s, %s)' % (repr(self.kv), repr(self.p))

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
        if u >= self.kv[-self.p-1]:
            return self.kv.size - self.p - 2  # last interval
        else:
            return np.searchsorted(self.kv, u, side='right') - 1

    def first_active(self, k):
        """Index of first active basis function in interval (kv[k], kv[k+1])"""
        return k - self.p

    def first_active_at(self, u):
        """Index of first active basis function in the interval which contains `u`."""
        return self.first_active(self.findspan(u))

    def greville(self):
        """Compute Gréville abscissae for this knot vector"""
        p = self.p
        # running averages over p knots
        return (np.convolve(self.kv, np.ones(p) / p))[p:-p]

    def refine(self):
        """Returns the uniform refinement of this knot vector"""
        mesh = self.mesh
        midpoints = (mesh[1:] + mesh[:-1]) / 2
        kvnew = np.sort(np.concatenate((self.kv, midpoints)))
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

def _bspline_active_ev_single(knotvec, u):
    """Evaluate all active B-spline basis functions at a single point `u`"""
    kv, p = knotvec.kv, knotvec.p
    left = np.empty(p)
    right = np.empty(p)
    result = np.empty(p+1)

    span = knotvec.findspan(u)

    result[0] = 1.0

    for j in range(1, p+1):
        left[j-1] = u - kv[span+1-j]
        right[j-1] = kv[span+j] - u
        saved = 0.0

        for r in range(j):
            temp = result[r] / (right[r] + left[j-r-1])
            result[r] = saved + right[r] * temp
            saved = left[j-r-1] * temp

        result[j] = saved

    return result

def active_ev(knotvec, u):
    """Evaluate all active B-spline basis functions at the points `u`.

    Returns an array of shape (p+1, u.size) if `u` is an array."""
    if np.isscalar(u):
        return _bspline_active_ev_single(knotvec, u)
    else:
        result = np.empty((knotvec.p+1, u.size))
        for i in range(u.size):
            result[:,i] = _bspline_active_ev_single(knotvec, u[i])
        return result


def _bspline_active_deriv_single(knotvec, u, numderiv):
    """Evaluate all active B-spline basis functions and their derivatives
    up to `numderiv` at a single point `u`"""
    kv, p = knotvec.kv, knotvec.p
    NDU   = np.empty((p+1, p+1))
    left  = np.empty(p)
    right = np.empty(p)
    result = np.empty((numderiv+1, p+1))

    span = knotvec.findspan(u)

    NDU[0,0] = 1.0

    for j in range(1, p+1):
        # Compute knot splits
        left[j-1]  = u - kv[span+1-j]
        right[j-1] = kv[span+j] - u
        saved = 0.0

        for r in range(j):     # For all but the last basis functions of degree j (ndu row)
            # Strictly lower triangular part: Knot differences of distance j
            NDU[j, r] = right[r] + left[j-r-1]
            temp = NDU[r, j-1] / NDU[j, r]
            # Upper triangular part: Basis functions of degree j
            NDU[r, j] = saved + right[r] * temp  # r-th function value of degree j
            saved = left[j-r-1] * temp

        # Diagonal: j-th (last) function value of degree j
        NDU[j, j] = saved

    # copy function values into result array
    result[0, :] = NDU[:, -1]

    a1 = np.empty(p+1)
    a2 = np.empty(p+1)

    for r in range(p+1):    # loop over basis functions
        a1[0] = 1.0

        fac = p        # fac = fac(p) / fac(p-k)

        # Compute the k-th derivative of the r-th basis function
        for k in range(1, numderiv+1):
            rk = r - k
            pk = p - k
            d = 0.0

            if r >= k:
                a2[0] = a1[0] / NDU[pk+1, rk]
                d = a2[0] * NDU[rk, pk]

            j1 = 1 if rk >= -1  else -rk
            j2 = k-1 if r-1 <= pk else p - r

            for j in range(j1, j2+1):
                a2[j] = (a1[j] - a1[j-1]) / NDU[pk+1, rk+j]
                d += a2[j] * NDU[rk+j, pk]

            if r <= pk:
                a2[k] = -a1[k-1] / NDU[pk+1, r]
                d += a2[k] * NDU[r, pk]

            result[k, r] = d * fac
            fac *= pk          # update fac = fac(p) / fac(p-k) for next k

            # swap rows a1 and a2
            (a1,a2) = (a2,a1)

    return result

def active_deriv(knotvec, u, numderiv):
    """Evaluate all active B-spline basis functions and their derivatives
    up to `numderiv` at the points `u`.

    Returns an array with shape (numderiv+1, p+1) if `u` is scalar or
    an array with shape (numderiv+1, p+1, u.size) otherwise.
    """
    if np.isscalar(u):
        return _bspline_active_deriv_single(knotvec, u, numderiv)
    else:
        result = np.empty((numderiv+1, knotvec.p+1, u.size))
        for i in range(u.size):
            result[:,:,i] = _bspline_active_deriv_single(knotvec, u[i], numderiv)
        return result


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

################################################################################

def collocation(kv, nodes):
    """Compute collocation matrix for B-spline basis at the given interpolation nodes"""
    nodes = np.array(nodes, copy=False)
    m = nodes.size
    n = kv.numdofs
    p = kv.p
    I, J, V = [], [], []
    values = active_ev(kv, nodes) # (p+1) x n
    indices = [kv.first_active_at(u) for u in nodes]
    for k in range(m):
        V.extend(values[:, k])
        I.extend( (p+1) * [k] )
        J.extend( range(indices[k], indices[k] + p  + 1) )
    return scipy.sparse.coo_matrix((V, (I,J)), shape=(m,n)).tocsr()

def collocation_derivs(kv, nodes, derivs=1):
    """Compute collocation matrix and derivative collocation matrices for B-spline
    basis at the given interpolation nodes.

    Returns a list of derivs+1 sparse CSR matrices with shape (nodes.size, kv.numdofs)."""
    nodes = np.array(nodes, copy=False)
    m = nodes.size
    n = kv.numdofs
    p = kv.p
    I, J = [], []
    V = [[] for _ in range(derivs+1)]
    values = active_deriv(kv, nodes, derivs) # (derivs+1) x (p+1) x n
    indices = [kv.first_active_at(u) for u in nodes]
    for k in range(m):
        for d in range(derivs+1):
            V[d].extend(values[d, :, k])
        I.extend( (p+1) * [k] )
        J.extend( range(indices[k], indices[k] + p  + 1) )
    return [scipy.sparse.coo_matrix((vals, (I,J)), shape=(m,n)).tocsr()
            for vals in V]

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

################################################################################

class BSplineFunc:
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

    def is_scalar(self):
        """Returns True if the function is scalar-valued."""
        return len(self.coeffs.shape[self.sdim:]) == 0

    def is_vector(self):
        """Returns True if the function is vector-valued."""
        return len(self.coeffs.shape[self.sdim:]) == 1

    def eval(self, *x):
        """Evaluate the function at a single point of the parameter domain.

        Args:
            *x: the point at which to evaluate the function, in xyz order
        """
        coords = tuple(np.asarray([t]) for t in reversed(x))
        return self.grid_eval(coords)

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
        colloc = [collocation(self.kvs[i], gridaxes[i]).A for i in range(self.sdim)]
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
        colloc = [(C.A, Cd.A) for (C,Cd) in colloc]

        grad_components = []
        for i in reversed(range(self.sdim)):  # x-component is the last one
            ops = [colloc[j][1 if j==i else 0] for j in range(self.sdim)] # deriv. in i-th direction
            grad_components.append(apply_tprod(ops, self.coeffs))   # shape: shape(grid) x self.dim
        return np.stack(grad_components, axis=-1)   # shape: shape(grid) x self.dim x self.sdim

    def transformed_jacobian(self, geo):
        """Create a function which evaluates the physical (transformed) gradient of the current
        function after a geometry transform."""
        return PhysicalGradientFunc(self, geo)

    def boundary(self, axis, side):
        """Return one side of the boundary as a :class:`BSplineFunc`.

        Args:
            axis (int): the index of the axis along which to take the boundary.
            side (int): 0 for the "lower" or 1 for the "upper" boundary along
                the given axis

        Returns:
            :class:`BSplineFunc`: representation of the boundary side;
            has :attr:`sdim` reduced by 1 and the same :attr:`dim` as this function
        """
        assert 0 <= axis < self.sdim, 'Invalid axis'
        slices = self.sdim * [slice(None)]
        slices[axis] = (0 if side==0 else -1)
        coeffs = self.coeffs[tuple(slices)]
        kvs = list(self.kvs)
        del kvs[axis]
        return BSplineFunc(kvs, coeffs)

    @property
    def support(self):
        """Return a sequence of pairs `(lower,upper)`, one per source dimension,
        which describe the extent of the support in the parameter space."""
        return tuple(kv.support() for kv in self.kvs)

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


class PhysicalGradientFunc:
    """A class for function objects which evaluate physical (transformed) gradients of
    scalar functions with geometry transforms.
    """
    def __init__(self, func, geo):
        assert func.dim == 1, 'Transformed gradients only implemented for scalar functions'
        self.func = func
        self.geo = geo
        self.dim = self.sdim = func.sdim

    def grid_eval(self, gridaxes):
        geojac = self.geo.grid_jacobian(gridaxes)
        geojacinvT = np.linalg.inv(geojac).swapaxes(-2, -1)

        u_grad = self.func.grid_jacobian(gridaxes)
        return np.matmul(geojacinvT, u_grad[..., None])[..., 0]

################################################################################

from .bspline_cy import *

