"""Classes and functions for creating and manipulating tensor product B-spline patches."""
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
        self.kvs = kvs
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
        if not all(ax.ndim == 1 for ax in gridaxes):
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
        """
        assert np.shape(gridaxes)[0] == self.sdim
        colloc = [bspline.collocation_derivs(self.kvs[i], gridaxes[i], derivs=1) for i in range(self.sdim)]
        colloc = [(C.A, Cd.A) for (C,Cd) in colloc]

        grad_components = []
        for i in reversed(range(self.sdim)):  # x-component is the last one
            ops = [colloc[j][1 if j==i else 0] for j in range(self.sdim)] # deriv. in i-th direction
            grad_components.append(apply_tprod(ops, self.coeffs))   # shape: shape(grid) x self.dim
        return np.stack(grad_components, axis=-1)   # shape: shape(grid) x self.dim x self.sdim


class BSplinePatch(BSplineFunc):
    """Represents a `d`-dimensional tensor product B-spline patch.

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

    def extrude(self, z0=0.0, z1=1.0):
        """Create a patch with one additional space dimension by
        linearly extruding along the new axis from `z0` to `z1`.
        """
        kvz = bspline.make_knots(1, 0.0, 1.0, 1)      # linear KV with a single interval

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

################################################################################
# Examples of 2D geometries
################################################################################

def unit_square(num_intervals=1):
    """Unit square with given number of intervals per direction.

    Returns:
        :class:`BSplinePatch` 2D geometry
    """
    kv = bspline.make_knots(1, 0.0, 1.0, num_intervals)
    #coeffs = np.array([
    #    [[ 0., 0.],
    #     [ 1., 0.]],
    #    [[ 0., 1.],
    #     [ 1., 1.]]
    #])
    x = np.linspace(0.0, 1.0, num_intervals+1)
    X,Y = np.meshgrid(x,x)
    coeffs = np.stack((X,Y), axis=-1)
    return BSplinePatch((kv,kv), coeffs)

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

################################################################################
# Examples of 3D geometries
################################################################################

def unit_cube(dim=3):
    """The `dim`-dimensional unit cube.

    Returns:
        :class:`BSplinePatch` geometry
    """
    coeffs = np.empty((2**dim, dim))

    for i in range(dim):
        coeffs[:,i] = (np.arange(2**dim) // (2**i)) % 2
    coeffs.shape = dim*(2,) + (dim,)

    kv = bspline.make_knots(1, 0.0, 1.0, 1)
    return BSplinePatch(dim*(kv,), coeffs)

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

