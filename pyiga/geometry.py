import numpy as np

from . import bspline
from .kronecker import apply_tprod

class TensorBSplinePatch:
    def __init__(self, kvs, coeffs):
        """Construct a d-dimensional tensor product B-spline patch.

        Arguments:
            kvs: List of d KnotVectors.
            coeffs: an ndarray of shape (n1, n2, ..., nd, d), where
                    ni is the number of dofs in the i-th KnotVector.
        """
        self.kvs = kvs
        self.coeffs = coeffs
        self.dim = len(kvs)
        assert coeffs.ndim == self.dim + 1, "Wrong shape of coefficients"
        assert self.dim == self.coeffs.shape[-1], "Wrong shape of coefficients"
        for i in range(self.dim):
            assert self.kvs[i].numdofs() == self.coeffs.shape[i], "Wrong shape of coefficients"

    def eval(self, *x):
        """Evaluate the geometry at a single point of the parameter domain.

        NB: For now, the components of x are in reverse order (e.g., zyx) to
        be consistent with the grid evaluation functions below.
        """
        coords = [ [t] for t in x ]
        return self.grid_eval(np.array(coords))

    def grid_eval(self, gridaxes):
        """Evaluate the patch on a tensor product grid.
        
        Note that the gridaxes should be given in reverse order, i.e.,
        the x axis last."""
        assert np.shape(gridaxes)[0] == self.dim
        colloc = [bspline.collocation(self.kvs[i], gridaxes[i]).A for i in range(self.dim)]
        return apply_tprod(colloc, self.coeffs)

    def grid_jacobian(self, gridaxes):
        """Evaluate the Jacobian on a tensor product grid.
        
        Note that the gridaxes should be given in reverse order, i.e.,
        the x axis last."""
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
        return TensorBSplinePatch((kvz,) + self.kvs, newcoefs)

################################################################################
# Examples of 2D geometries
################################################################################

def unit_square(num_intervals=1):
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
    return TensorBSplinePatch((kv,kv), coeffs)

def bspline_quarter_annulus(r1=1.0, r2=2.0):
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
    return TensorBSplinePatch((kvy,kvx), coeffs)

################################################################################
# Examples of 3D geometries
################################################################################

def unit_cube():
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
    return TensorBSplinePatch((kv,kv,kv), coeffs)

def twisted_box():
    """A 3D volume that resembles a box with its right face twisted
    and bent upwards.

    Corresponds to gismo data file twistedFlatQuarterAnnulus.xml."""
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

    return TensorBSplinePatch((kv1,kv2,kv3), coeffs)

