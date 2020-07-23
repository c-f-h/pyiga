"""Classes and functions for creating and manipulating tensor product B-spline
and NURBS patches.

See :doc:`/guide/geometry` for examples on how to create custom geometries.
"""
import numpy as np
import numpy.random

from . import bspline
from . import utils
from .bspline import BSplineFunc
from .tensor import apply_tprod

import functools


class NurbsFunc:
    r"""Any function that is given in terms of a tensor product NURBS basis with
    coefficients and weights.

    Arguments:
        kvs (seq): tuple of `d` :class:`.KnotVector`\ s.
        coeffs (ndarray): coefficient array; see :class:`.BSplineFunc` for format.
            The constructor may modify `coeffs` during premultiplication!
        weights (ndarray): coefficients for weight function in the same format
            as `coeffs`. If `weights=None` is passed, the weights are assumed to
            be given as the last vector component of `coeffs` instead.
        premultiplied (bool): pass `True` if the coefficients are already
            premultiplied by the weights.

    Attributes:
        kvs (seq): the knot vectors representing the tensor product basis
        coeffs (ndarray): the premultiplied coefficients for the function,
            including the weights in the last vector component
        sdim (int): dimension of the parameter domain
        dim (int): dimension of the output of the function

    The evaluation functions have the same prototypes and behavior as those in
    :class:`.BSplineFunc`.
    """
    def __init__(self, kvs, coeffs, weights, premultiplied=False):
        if isinstance(kvs, bspline.KnotVector):
            kvs = (kvs,)
        self.kvs = tuple(kvs)
        self.sdim = len(self.kvs)    # source dimension

        N = tuple(kv.numdofs for kv in self.kvs)
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
        else:
            assert False, 'Tensor-valued NURBS functions not implemented'
        self.dim = dim

        if weights is None:
            assert self.dim > 1, 'Weights must be specified in the coeffs array'
            self.dim -= 1       # weights are already built into coeffs
        else:
            weights = np.asanyarray(weights)
            assert weights.shape == N, 'Wrong shape of weights array'
            if self.coeffs.shape == N:  # no trailing dimensions
                self.coeffs = np.stack((self.coeffs, weights), axis=-1)  # create new axis
            else:   # already have a trailing dimension
                self.coeffs = np.concatenate((self.coeffs, weights[..., None]), axis=-1)

        # pre-multiply coefficients by weights
        if not premultiplied:
            self.coeffs[..., :-1] *= self.coeffs[..., -1:]

    def is_vector(self):
        """Returns True if the function is vector-valued."""
        # NB: currently, NURBS can never be truly scalar due to how weights are stored
        return len(self.coeffs.shape[self.sdim:]) == 1

    def eval(self, *x):
        coords = tuple(np.asarray([t]) for t in reversed(x))
        y = self.grid_eval(coords).squeeze(axis=tuple(range(self.sdim)))
        if y.shape == ():
            y = y.item()
        return y

    def grid_eval(self, gridaxes):
        assert len(gridaxes) == self.sdim, "Input has wrong dimension"
        # make sure axes are one-dimensional
        if not all(np.ndim(ax) == 1 for ax in gridaxes):
            gridaxes = tuple(np.squeeze(ax) for ax in gridaxes)
            assert all(ax.ndim == 1 for ax in gridaxes), \
                "Grid axes should be one-dimensional"
        colloc = [bspline.collocation(self.kvs[i], gridaxes[i]) for i in range(self.sdim)]
        vals = apply_tprod(colloc, self.coeffs)
        return vals[..., :-1] / vals[..., -1:]       # divide by weight function

    def grid_jacobian(self, gridaxes):
        bsp = BSplineFunc(self.kvs, self.coeffs)
        val = bsp.grid_eval(gridaxes)
        V = val[..., :-1, None]
        W = val[..., -1:, None]
        jac = bsp.grid_jacobian(gridaxes)   # shape(grid) x (dim+1) x sdim
        Vjac = jac[..., :-1, :]
        Wjac = jac[..., -1:, :]
        return (Vjac * W - V * Wjac) / (W**2)   # use quotient rule for (V/W)'

    def grid_hessian(self, gridaxes):
        bsp = BSplineFunc(self.kvs, self.coeffs)
        val = bsp.grid_eval(gridaxes)
        V = val[..., :-1, None]             # shape(grid) x dim x 1
        W = val[..., -1:, None]             # shape(grid) x  1  x 1
        jac = bsp.grid_jacobian(gridaxes)   # shape(grid) x (dim+1) x sdim
        Vjac = jac[..., :-1, :]             # shape(grid) x dim x sdim
        Wjac = jac[..., -1:, :]             # shape(grid) x  1  x sdim
        # compute Jacobian of NURBS as above
        Njac = (Vjac * W - V * Wjac) / (W**2)   # shape(grid) x dim x sdim

        hess = bsp.grid_hessian(gridaxes)   # shape(grid) x (dim+1) x num_hess
        Vhess = hess[..., :-1, :]           # shape(grid) x dim x num_hess
        Whess = hess[..., -1:, :]           # shape(grid) x  1  x num_hess

        # first part of Hessian, already in linearized format
        Nhess1 = Vhess / W - (V * Whess) / (W**2)

        # second part of Hessian, in matrix format
        mat = (Njac[..., None, :] * Wjac[..., :, None]) / W[..., None]     # shape(grid) x dim x sdim x sdim
        mat += mat.swapaxes(-1, -2)             # symmetrize
        I,J = np.triu_indices(mat.shape[-1])    # indices for upper triangular part
        return Nhess1 - mat[..., I, J]          # linearize the symmetric part

    def boundary(self, axis, side):
        """Return one side of the boundary as a :class:`NurbsFunc`.

        Args:
            axis (int): the index of the axis along which to take the boundary.
            side (int): 0 for the "lower" or 1 for the "upper" boundary along
                the given axis

        Returns:
            :class:`NurbsFunc`: representation of the boundary side;
            has `sdim` reduced by 1 and the same `dim` as this function
        """
        assert 0 <= axis < self.sdim, 'Invalid axis'
        slices = self.sdim * [slice(None)]
        slices[axis] = (0 if side==0 else -1)
        coeffs = self.coeffs[tuple(slices)]
        kvs = list(self.kvs)
        del kvs[axis]
        return NurbsFunc(kvs, coeffs, weights=None, premultiplied=True)

    @property
    def support(self):
        """Return a sequence of pairs `(lower,upper)`, one per source dimension,
        which describe the extent of the support in the parameter space."""
        return tuple(kv.support() for kv in self.kvs)

    def coeffs_weights(self):
        """Return the non-premultiplied coefficients and weights as a pair of arrays."""
        W = self.coeffs[..., -1]
        return self.coeffs[..., :-1] / W[..., None], W.copy()

    def translate(self, offset):
        """Return a version of this geometry translated by the specified offset."""
        C, W = self.coeffs_weights()
        return NurbsFunc(self.kvs, C + offset, W)

    def scale(self, factor):
        """Scale all control points either by a scalar factor or componentwise by
        a vector, leave the weights unchanged, and return the resulting new function.
        """
        C, W = self.coeffs_weights()
        return NurbsFunc(self.kvs, C * factor, W)

    def apply_matrix(self, A):
        """Apply a matrix to each control point of this function, leave the weights
        unchanged, and return the result.

        `A` should either be a single matrix or an array of matrices, one for each
        control point. Standard numpy broadcasting rules apply.
        """
        assert self.is_vector(), 'Can only apply matrices to vector-valued functions'
        C, W = self.coeffs_weights()
        C = np.matmul(A, C[..., None])
        assert C.shape[-1] == 1  # this should have created a new singleton axis
        return NurbsFunc(self.kvs, np.squeeze(C, axis=-1), W)

    def rotate_2d(self, angle):
        """Rotate a geometry with :attr:`dim` = 2 by the given angle and return the result."""
        assert self.dim == 2, 'Must be 2D vector function'
        s, c = np.sin(angle), np.cos(angle)
        R = np.array([
            [c, -s],
            [s, c]
        ])
        return self.apply_matrix(R)

    def as_nurbs(self):
        return self


class UserFunction:
    """A function (supporting the same basic protocol as :class:`.BSplineFunc`) which is given
    in terms of a user-defined callable.

    Args:
        f (callable): a function of `d` variables; may be scalar or vector-valued
        support: a sequence of `d` pairs of the form `(lower,upper)` describing
            the support of the function (see :attr:`.BSplineFunc.support`)
        dim (int): the dimension of the function output; by default, is
            automatically determined by calling `f`
        jac (callable): optionally, a function evaluating the Jacobian matrix
            of the function

    The :attr:`sdim` attribute (see :attr:`.BSplineFunc.sdim`) is determined from the
    length of `support`.
    """
    def __init__(self, f, support, dim=None, jac=None):
        self.f = f
        self.support = tuple(support)
        self.jac = jac
        if dim is None:
            x0 = tuple(lo for (lo,hi) in reversed(support))
            dim = np.shape(f(*x0))
            if len(dim) == 0:
                dim = 1
            elif len(dim) == 1:
                dim = dim[0]
        self.dim = dim
        self.sdim = len(support)

    def grid_eval(self, grd):
        return utils.grid_eval(self.f, grd)

    def eval(self, *x):
        return self.f(*x)

    def grid_jacobian(self, grd):
        assert self.jac is not None, 'Jacobian not specified in UserFunction'
        return utils.grid_eval(self.jac, grd)

################################################################################
# Examples of 2D geometries
################################################################################

def unit_square(num_intervals=1):
    """Unit square with given number of intervals per direction.

    Returns:
        :class:`.BSplineFunc` 2D geometry
    """
    return unit_cube(dim=2, num_intervals=num_intervals)

def perturbed_square(num_intervals=5, noise=0.02):
    """Randomly perturbed unit square.

    Unit square with given number of intervals per direction;
    the control points are perturbed randomly according to the
    given noise level.

    Returns:
        :class:`.BSplineFunc` 2D geometry
    """
    return unit_square(num_intervals).perturb(noise)

def bspline_quarter_annulus(r1=1.0, r2=2.0):
    """A B-spline approximation of a quarter annulus in the first quadrant.

    Args:
        r1 (float): inner radius
        r2 (float): outer radius

    Returns:
        :class:`.BSplineFunc` 2D geometry
    """
    kvx = bspline.make_knots(1, 0.0, 1.0, 1)
    kvy = bspline.make_knots(2, 0.0, 1.0, 1)

    coeffs = np.array([
            [[ r1, 0.0],
             [ r2, 0.0]],
            [[ r1,  r1],
             [ r2,  r2]],
            [[0.0,  r1],
             [0.0,  r2]],
    ])
    return BSplineFunc((kvy,kvx), coeffs)

def quarter_annulus(r1=1.0, r2=2.0):
    """A NURBS representation of a quarter annulus in the first quadrant.

    Args:
        r1 (float): inner radius
        r2 (float): outer radius

    Returns:
        :class:`NurbsFunc` 2D geometry
    """
    kvx = bspline.make_knots(1, 0.0, 1.0, 1)
    kvy = bspline.make_knots(2, 0.0, 1.0, 1)

    coeffs = np.array([
            [[ r1, 0.0, 1.0],
             [ r2, 0.0, 1.0]],
            [[ r1,  r1, 1.0 / np.sqrt(2.0)],
             [ r2,  r2, 1.0 / np.sqrt(2.0)]],
            [[0.0,  r1, 1.0],
             [0.0,  r2, 1.0]],
    ])
    return NurbsFunc((kvy,kvx), coeffs, weights=None)

################################################################################
# Examples of 3D geometries
################################################################################

def unit_cube(dim=3, num_intervals=1):
    """The `dim`-dimensional unit cube with `num_intervals` intervals
    per coordinate direction.

    Returns:
        :class:`.BSplineFunc` geometry
    """
    return functools.reduce(tensor_product, dim * (line_segment(0.0, 1.0, intervals=num_intervals),))

def identity(extents):
    """Identity mapping (using linear splines) over a d-dimensional box
    given by `extents` as a list of (min,max) pairs or of :class:`.KnotVector`.

    Returns:
        :class:`.BSplineFunc` geometry
    """
    # if any inputs are KnotVectors, extract their supports
    extents = [
        ex.support() if isinstance(ex, bspline.KnotVector) else ex
        for ex in extents
    ]
    return functools.reduce(tensor_product,
            (line_segment(ex[0], ex[1], support=ex) for ex in extents))

def twisted_box():
    """A 3D volume that resembles a box with its right face twisted
    and bent upwards.

    Corresponds to gismo data file twistedFlatQuarterAnnulus.xml.

    Returns:
        :class:`.BSplineFunc` 3D geometry
    """
    kv1 = bspline.make_knots(1, 0.0, 1.0, 1)
    kv2 = bspline.make_knots(3, 0.0, 1.0, 1)
    kv3 = kv1

    coeffs = np.array([
      1   , 0   , 0   ,
      2   , 0   , 0   ,
      1   , 0.5 , 0   ,
      2   , 1.5 , 0   ,
      0.5 , 1   , 0.5 ,
      1.5 , 2   , 0.5 ,
      0   , 1   , 2   ,
      0   , 2   , 2   ,
      1   , 0   , 1   ,
      2   , 0   , 1   ,
      1   , 0.5 , 1   ,
      2   , 1.5 , 1   ,
      1   , 1   , 1.5 ,
      1.5 , 2   , 1.5 ,
      1   , 1   , 2   ,
      1   , 2   , 2   ,
    ]).reshape((2, 4, 2, 3))

    return BSplineFunc((kv1,kv2,kv3), coeffs)

################################################################################
# Functions for creating curves
################################################################################

def line_segment(x0, x1, support=(0.0, 1.0), intervals=1):
    """Return a :class:`.BSplineFunc` which describes the line between the
    vectors `x0` and `x1`.

    If specified, `support` describes the interval in which the function is
    supported; by default, it is the interval (0,1).

    If specified, `intervals` is the number of intervals in the underlying
    linear spline space. By default, the minimal spline space with 2 dofs is
    used.
    """
    if np.isscalar(x0): x0 = [x0]
    if np.isscalar(x1): x1 = [x1]
    assert len(x0) == len(x1), 'Vectors must have same dimension'
    # produce 1D arrays
    x0 = np.array(x0, dtype=float).ravel()
    x1 = np.array(x1, dtype=float).ravel()
    # interpolate linearly
    S = np.linspace(0.0, 1.0, intervals+1).reshape((intervals+1, 1))
    coeffs = (1-S) * x0 + S * x1
    return BSplineFunc(bspline.make_knots(1, support[0], support[1], intervals), coeffs)

def circular_arc(alpha, r=1.0):
    """Construct a circular arc with angle `alpha` and radius `r`.

    The arc is centered at the origin, starts on the positive `x` axis and
    travels in counterclockwise direction.
    The angle `alpha` must be between 0 and `pi`.
    """
    # formulas adapted from
    # https://www.cs.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/NURBS/RB-circles.html
    assert 0.0 < alpha < np.pi, 'Invalid angle'
    beta = np.pi/2 - alpha/2
    d = 1.0 / np.sin(beta)

    coeffs = r * np.array([
        [1.0, 0.0, 1.0],
        [d * np.cos(alpha/2), d * np.sin(alpha/2), np.sin(beta)],
        [np.cos(alpha), np.sin(alpha), 1.0]
    ])
    return NurbsFunc(bspline.make_knots(2, 0.0, 1.0, 1), coeffs, weights=None)

def circle(r=1.0):
    """Construct a circle with radius `r` using NURBS."""
    knots = np.array([0,0,0, 1./3., 1./3., 2./3., 2./3, 1,1,1])
    kv = bspline.KnotVector(knots, 2)

    pts = r * np.array([(np.cos(a), np.sin(a)) for a in np.linspace(0, 2*np.pi, 7)])
    pts[1] *= 2
    pts[3] *= 2
    pts[5] *= 2
    W = np.array([1, .5, 1, .5, 1, .5, 1])
    return NurbsFunc(kv, pts, weights=W)

################################################################################
# Operations on geometries
################################################################################

def _prepare_for_outer(Cs, sdims):
    """Bring the coefficient arrays (C1,C2)=Cs with source dimensions as given
    into a suitable form to apply outer sum or outer product on them.
    """
    SD1, SD2 = (np.atleast_1d(C.shape[:sdim]).astype(np.int_) for (C,sdim) in zip(Cs,sdims))
    VD1, VD2 = (np.atleast_1d(C.shape[sdim:]).astype(np.int_) for (C,sdim) in zip(Cs,sdims))
    shape1 = np.concatenate((SD1, np.ones_like(SD2), VD1))
    shape2 = np.concatenate((np.ones_like(SD1), SD2, VD2))
    return np.reshape(Cs[0], shape1), np.reshape(Cs[1], shape2)

def outer_sum(G1, G2):
    """Compute the outer sum of two :class:`.BSplineFunc` or :class:`.NurbsFunc` geometries.
    This means that given two input functions

    .. math:: G_1(y), G_2(x),

    it returns a new function

    .. math:: G(x,y) = G_1(y) + G_2(x).

    The returned function is a :class:`.NurbsFunc` if either input is and a
    :class:`.BSplineFunc` otherwise. It has source dimension
    (:attr:`.BSplineFunc.sdim`) equal to the sum of the source dimensions of
    the input functions.

    `G1` and `G2` should have the same image dimension (:attr:`.BSplineFunc.dim`),
    and the output then has the same as well. However, broadcasting according to
    standard Numpy rules is permissible; e.g., one function can be vector-valued
    and the other scalar-valued.

    The coefficients of the result are the pointwise sums of the coefficients of
    the input functions over a new tensor product spline space.
    """
    if isinstance(G1, NurbsFunc) or isinstance(G2, NurbsFunc):
        G1 = G1.as_nurbs()
        G2 = G2.as_nurbs()
        C1, W1 = G1.coeffs_weights()
        C2, W2 = G2.coeffs_weights()
        C1, C2 = _prepare_for_outer((C1, C2), (G1.sdim, G2.sdim))
        W1, W2 = _prepare_for_outer((W1, W2), (G1.sdim, G2.sdim))
        return NurbsFunc(G1.kvs + G2.kvs, C1 + C2, W1 * W2)
    else:
        assert isinstance(G1, BSplineFunc) and isinstance(G2, BSplineFunc)
        C1, C2 = _prepare_for_outer((G1.coeffs, G2.coeffs), (G1.sdim, G2.sdim))
        return BSplineFunc(G1.kvs + G2.kvs, C1 + C2)

def outer_product(G1, G2):
    """Compute the outer product of two :class:`.BSplineFunc` or :class:`.NurbsFunc` geometries.
    This means that given two input functions

    .. math:: G_1(y), G_2(x),

    it returns a new function

    .. math:: G(x,y) = G_1(y) G_2(x),

    where the multiplication is componentwise in the case of vector functions.
    The returned function is a :class:`.NurbsFunc` if either input is and a
    :class:`.BSplineFunc` otherwise. It has source dimension
    (:attr:`.BSplineFunc.sdim`) equal to the sum of the source dimensions of
    the input functions.

    `G1` and `G2` should have the same image dimension (:attr:`.BSplineFunc.dim`),
    and the output then has the same as well. However, broadcasting according to
    standard Numpy rules is permissible; e.g., one function can be vector-valued
    and the other scalar-valued.

    The coefficients of the result are the pointwise products of the coefficients of
    the input functions over a new tensor product spline space.
    """
    if isinstance(G1, NurbsFunc) or isinstance(G2, NurbsFunc):
        G1 = G1.as_nurbs()
        G2 = G2.as_nurbs()
        C1, W1 = G1.coeffs_weights()
        C2, W2 = G2.coeffs_weights()
        C1, C2 = _prepare_for_outer((C1, C2), (G1.sdim, G2.sdim))
        W1, W2 = _prepare_for_outer((W1, W2), (G1.sdim, G2.sdim))
        return NurbsFunc(G1.kvs + G2.kvs, C1 * C2, W1 * W2)
    else:
        assert isinstance(G1, BSplineFunc) and isinstance(G2, BSplineFunc)
        C1, C2 = _prepare_for_outer((G1.coeffs, G2.coeffs), (G1.sdim, G2.sdim))
        return BSplineFunc(G1.kvs + G2.kvs, C1 * C2)

def tensor_product(G1, G2, *Gs):
    r"""Compute the tensor product of two or more :class:`.BSplineFunc` or
    :class:`.NurbsFunc` functions.  This means that given two input functions

    .. math:: G_1(y), G_2(x),

    it returns a new function

    .. math:: G(x,y) = G_2(x) \times G_1(y),

    where :math:`\times` means that vectors are joined together.
    The resulting :class:`.BSplineFunc` or :class:`NurbsFunc` has source
    dimension (:attr:`.BSplineFunc.sdim`) equal to the sum of the source
    dimensions of the input functions, and target dimension
    (:attr:`.BSplineFunc.dim`) equal to the sum of the target dimensions of the
    input functions.
    """
    if Gs is not ():
        return tensor_product(G1, tensor_product(G2, *Gs))
    assert G1.is_vector() and G2.is_vector(), 'only implemented for vector-valued functions'

    Gs = (G1, G2)
    nurbs = any(isinstance(G, NurbsFunc) for G in Gs)

    if nurbs:
        Gs = tuple(G.as_nurbs() for G in Gs)
        G1, G2 = Gs
        CC1, W1 = G1.coeffs_weights()
        CC2, W2 = G2.coeffs_weights()
        Cs = (CC1, CC2)
        WW1, WW2 = _prepare_for_outer((W1, W2), (G1.sdim, G2.sdim))
        W = WW1 * WW2
    else:
        assert isinstance(G1, BSplineFunc) and isinstance(G2, BSplineFunc)
        Cs = tuple(G.coeffs for G in Gs)

    SD1, SD2 = (np.atleast_1d(C.shape[:G.sdim]) for (C,G) in zip(Cs,Gs))
    VD1, VD2 = (np.atleast_1d(C.shape[G.sdim:]) for (C,G) in zip(Cs,Gs))
    shape1 = np.concatenate((SD1, np.ones_like(SD2), VD1))
    shape2 = np.concatenate((np.ones_like(SD1), SD2, VD2))
    target_shape1 = np.concatenate((SD1, SD2, VD1))
    target_shape2 = np.concatenate((SD1, SD2, VD2))
    C1 = np.broadcast_to(np.reshape(Cs[0], shape1), target_shape1)
    C2 = np.broadcast_to(np.reshape(Cs[1], shape2), target_shape2)
    # NB: coefficients are in XY order, but coordinate axes in YX order!
    C = np.concatenate((C2,C1), axis=-1)

    if nurbs:
        return NurbsFunc(G1.kvs + G2.kvs, C, W)
    else:
        return BSplineFunc(G1.kvs + G2.kvs, C)
