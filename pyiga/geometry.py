"""Classes and functions for creating and manipulating tensor product B-spline patches."""
import numpy as np
import numpy.random

from . import bspline
from .kronecker import apply_tprod

class BSplinePatch:
    """Represents a tensor product B-spline patch.

    Includes a tensor product B-spline basis in the form of a list
    of :class:`KnotVector` instances and an `ndarray` of coefficients.

    Attributes:
        kvs (seq): the knot vectors representing the tensor product basis
        coeffs (ndarray): the coefficients for the geometry.
            An array of shape (n1, n2, ..., nd, `dim`), where
            ni is the number of dofs in the i-th basis.
        dim (int): dimension of the space in which the geometry lies

    .. automethod:: __init__
    """

    def __init__(self, kvs, coeffs):
        """Construct a `d`-dimensional tensor product B-spline patch.

        Arguments:
            kvs (seq): tuple of `d` :class:`KnotVector`.
            coeffs (ndarray): an array of shape (n1, n2, ..., nd, d), where
                    ni is the number of dofs in the i-th :class:`KnotVector`.
        """
        self.kvs = kvs
        self.coeffs = coeffs
        self.dim = len(kvs)
        assert coeffs.ndim == self.dim + 1, "Wrong shape of coefficients"
        assert self.dim == self.coeffs.shape[-1], "Wrong shape of coefficients"
        for i in range(self.dim):
            assert self.kvs[i].numdofs == self.coeffs.shape[i], "Wrong shape of coefficients"

    def eval(self, *x):
        """Evaluate the geometry at a single point of the parameter domain.

        NB: For now, the components of x are in reverse order (e.g., zyx) to
        be consistent with the grid evaluation functions below.
        """
        coords = [ [t] for t in x ]
        return self.grid_eval(np.array(coords))

    def grid_eval(self, gridaxes):
        """Evaluate the patch on a tensor product grid.

        Args:
            gridaxes (seq): list of 1D vectors describing the tensor product grid.

        .. note::

            The gridaxes should be given in reverse order, i.e.,
            the x axis last.

        Returns:
            ndarray: array of function values; shape corresponds to input grid.
        """
        assert np.shape(gridaxes)[0] == self.dim
        colloc = [bspline.collocation(self.kvs[i], gridaxes[i]).A for i in range(self.dim)]
        return apply_tprod(colloc, self.coeffs)

    def grid_jacobian(self, gridaxes):
        """Evaluate the Jacobian on a tensor product grid.

        Args:
            gridaxes (seq): list of 1D vectors describing the tensor product grid.

        .. note::

            The gridaxes should be given in reverse order, i.e.,
            the x axis last.

        Returns:
            ndarray: array of Jacobians (`dim x dim`); shape corresponds to input grid.
        """
        assert np.shape(gridaxes)[0] == self.dim
        colloc = [bspline.collocation_derivs(self.kvs[i], gridaxes[i], derivs=1) for i in range(self.dim)]
        colloc = [(C.A, Cd.A) for (C,Cd) in colloc]

        grad_components = []
        for i in reversed(range(self.dim)):  # x-component is the last one
            ops = [colloc[j][1 if j==i else 0] for j in range(self.dim)] # deriv. in j-th direction
            grad_components.append(apply_tprod(ops, self.coeffs))
        return np.stack(grad_components, axis=-1)

    def extrude(self, z0=0.0, z1=1.0):
        """Create a patch with one additional space dimension by
        linearly extruding along the new axis from z0 to z1.
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
    geo = unit_square(num_intervals)
    geo.coeffs += 2*noise*(np.random.random_sample(geo.coeffs.shape) - 0.5)
    return geo

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

def unit_cube():
    """The 3D unit cube.

    Returns:
        :class:`BSplinePatch` 3D geometry
    """
    kv = bspline.make_knots(1, 0.0, 1.0, 1)
    coeffs = np.array([
       [[[ 0., 0., 0.],
         [ 1., 0., 0.]],
        [[ 0., 1., 0.],
         [ 1., 1., 0.]]],
       [[[ 0., 0., 1.],
         [ 1., 0., 1.]],
        [[ 0., 1., 1.],
         [ 1., 1., 1.]]]
    ])
    return BSplinePatch((kv,kv,kv), coeffs)

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

