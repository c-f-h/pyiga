"""Classes and functions for creating and manipulating tensor product B-spline
and NURBS patches."""
import numpy as np
import numpy.random

from . import bspline
from .tensor import apply_tprod

class BSplineFunc:
    """Any function that is given in terms of a tensor product B-spline basis with coefficients.

    Arguments:
        kvs (seq): tuple of `d` :class:`pyiga.bspline.KnotVector`.
        coeffs (ndarray): coefficient array

    `kvs` represents a tensor product B-spline basis, where the *i*-th
    :class:`pyiga.bspline.KnotVector` describes the B-spline basis in the *i*-th
    coordinate direction.

    `coeffs` is the array of coefficients with respect to this tensor product basis.
    The length of its first `d` axes must match the number of degrees of freedom
    in the corresponding :class:`pyiga.bspline.KnotVector`.
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
        if isinstance(kvs, bspline.KnotVector):
            kvs = (kvs,)
        self.kvs = tuple(kvs)
        self.sdim = len(kvs)    # source dimension

        N = tuple(kv.numdofs for kv in kvs)
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

    def eval(self, *x):
        """Evaluate the function at a single point of the parameter domain."""
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
        colloc = [bspline.collocation(self.kvs[i], gridaxes[i]).A for i in range(self.sdim)]
        return apply_tprod(colloc, self.coeffs)

    def grid_jacobian(self, gridaxes):
        """Evaluate the Jacobian on a tensor product grid.

        Args:
            gridaxes (seq): list of 1D vectors describing the tensor product grid.

        .. note::

            The gridaxes should be given in reverse order, i.e.,
            the x axis last.

        Returns:
            ndarray: array of Jacobians (`dim x sdim`); shape corresponds to input grid.
            For scalar functions, the output is a vector of length `sdim` (the gradient)
            per grid point.
        """
        assert len(gridaxes) == self.sdim, "Input has wrong dimension"
        colloc = [bspline.collocation_derivs(self.kvs[i], gridaxes[i], derivs=1) for i in range(self.sdim)]
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
            has `sdim` reduced by 1 and the same `dim` as this function
        """
        assert 0 <= axis < self.sdim, 'Invalid axis'
        slices = self.sdim * [slice(None)]
        slices[axis] = (0 if side==0 else -1)
        coeffs = self.coeffs[slices]
        kvs = list(self.kvs)
        del kvs[axis]
        return BSplineFunc(kvs, coeffs)

    @property
    def support(self):
        """Return a sequence of pairs `(lower,upper)`, one per source dimension,
        which describe the extent of the support in the parameter space."""
        return tuple(kv.support() for kv in self.kvs)


class BSplinePatch(BSplineFunc):
    """Represents a `d`-dimensional tensor product B-spline patch.
    Inherits from :class:`BSplineFunc`.

    Arguments:
        kvs (seq): tuple of `d` :class:`pyiga.bspline.KnotVector`.
        coeffs (ndarray): the control points. Array of shape `(n1, n2, ..., nd, d)`,
            where `ni` is the number of dofs in the basis given by the *i*-th
            :class:`pyiga.bspline.KnotVector`.

    `kvs` represents a tensor product B-spline basis, where the *i*-th
    :class:`pyiga.bspline.KnotVector` describes the B-spline basis in the *i*-th
    coordinate direction.

    `coeffs` are the control points, i.e., an array of coefficients with
    respect to this tensor product basis.
    The control point for the tensor product basis function `(i1, ..., id)`
    is given by ``coeffs[i1, ..., id, :]``.
    The j-th component of the geometry is
    represented by the coefficients ``coeffs[..., j]``.
    """

    def __init__(self, kvs, coeffs):
        """Construct a `d`-dimensional tensor product B-spline patch with the given knot vectors and coefficients."""
        BSplineFunc.__init__(self, kvs, coeffs)
        assert self.dim == self.sdim, "Wrong dimension: should be %s, not %s" % (self.sdim, self.dim)

    def extrude(self, z0=0.0, z1=1.0, support=(0.0, 1.0)):
        """Create a patch with one additional space dimension by
        linearly extruding along the new axis from `z0` to `z1`.

        By default, the new knot vector will be defined over the
        interval (0, 1). A different interval can be specified through
        the `support` parameter.
        """
        kvz = bspline.make_knots(1, support[0], support[1], 1)      # linear KV with a single interval

        pad0 = np.expand_dims(np.tile(z0, self.coeffs.shape[:-1]), axis=-1)
        pad1 = np.expand_dims(np.tile(z1, self.coeffs.shape[:-1]), axis=-1)
        # append the padding in the new direction and stack the two
        # new coefficient arrays along a new first axis
        newcoefs = np.stack(
                      (np.append(self.coeffs, pad0, axis=-1),
                       np.append(self.coeffs, pad1, axis=-1)),
                      axis=0)
        return BSplinePatch((kvz,) + self.kvs, newcoefs)

    def perturb(self, noise):
        """Create a copy of this patch where all coefficients are randomly perturbed
        by noise of the given magnitude."""
        return BSplinePatch(self.kvs,
            self.coeffs + 2*noise*(np.random.random_sample(self.coeffs.shape) - 0.5))


class NurbsFunc:
    """Any function that is given in terms of a tensor product NURBS basis with
    coefficients and weights.

    Arguments:
        kvs (seq): tuple of `d` :class:`pyiga.bspline.KnotVector`.
        coeffs (ndarray): coefficient array; see :class:`BSplineFunc` for format.
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
    :class:`BSplineFunc`.
    """
    def __init__(self, kvs, coeffs, weights, premultiplied=False):
        if isinstance(kvs, bspline.KnotVector):
            kvs = (kvs,)
        self.kvs = tuple(kvs)
        self.sdim = len(self.kvs)    # source dimension

        N = tuple(kv.numdofs for kv in self.kvs)
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
            assert weights.shape == N, 'Wrong shape of weights array'
            if self.coeffs.shape == N:  # no trailing dimensions
                self.coeffs = np.stack((self.coeffs, weights), axis=-1)  # create new axis
            else:   # already have a trailing dimension
                self.coeffs = np.concatenate((self.coeffs, weights[..., None]), axis=-1)

        # pre-multiply coefficients by weights
        if not premultiplied:
            self.coeffs[..., :-1] *= self.coeffs[..., -1:]

    def eval(self, *x):
        coords = tuple(np.asarray([t]) for t in reversed(x))
        return self.grid_eval(coords)

    def grid_eval(self, gridaxes):
        assert len(gridaxes) == self.sdim, "Input has wrong dimension"
        # make sure axes are one-dimensional
        if not all(np.ndim(ax) == 1 for ax in gridaxes):
            gridaxes = tuple(np.squeeze(ax) for ax in gridaxes)
            assert all(ax.ndim == 1 for ax in gridaxes), \
                "Grid axes should be one-dimensional"
        colloc = [bspline.collocation(self.kvs[i], gridaxes[i]).A for i in range(self.sdim)]
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
        coeffs = self.coeffs[slices]
        kvs = list(self.kvs)
        del kvs[axis]
        return NurbsFunc(kvs, coeffs, weights=None, premultiplied=True)

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
# Examples of 2D geometries
################################################################################

def unit_square(num_intervals=1):
    """Unit square with given number of intervals per direction.

    Returns:
        :class:`BSplinePatch` 2D geometry
    """
    return unit_cube(dim=2, num_intervals=num_intervals)

def perturbed_square(num_intervals=5, noise=0.02):
    """Randomly perturbed unit square.

    Unit square with given number of intervals per direction;
    the control points are perturbed randomly according to the
    given noise level.

    Returns:
        :class:`BSplinePatch` 2D geometry
    """
    return unit_square(num_intervals).perturb(noise)

def bspline_quarter_annulus(r1=1.0, r2=2.0):
    """A B-spline approximation of a quarter annulus in the first quadrant.

    Args:
        r1 (float): inner radius
        r2 (float): outer radius

    Returns:
        :class:`BSplinePatch` 2D geometry
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
    return BSplinePatch((kvy,kvx), coeffs)

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
        :class:`BSplinePatch` geometry
    """
    kv = bspline.make_knots(1, 0.0, 1.0, num_intervals)
    x = np.linspace(0.0, 1.0, num_intervals+1)
    XYZ = np.meshgrid(*(dim * (x,)), indexing='ij')
    coeffs = np.stack(reversed(XYZ), axis=-1)   # make X correspond to 1st axis
    return BSplinePatch(dim * (kv,), coeffs)

def twisted_box():
    """A 3D volume that resembles a box with its right face twisted
    and bent upwards.

    Corresponds to gismo data file twistedFlatQuarterAnnulus.xml.

    Returns:
        :class:`BSplinePatch` 3D geometry
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

    return BSplinePatch((kv1,kv2,kv3), coeffs)

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
