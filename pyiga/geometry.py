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

def _nurbs_jacobian(val, jac):
    """Compute the NURBS Jacobians given an array of B-spline values and
    B-spline Jacobians.
    """
    V = val[..., :-1, None]     # function values
    W = val[..., -1:, None]     # weights
    Vjac = jac[..., :-1, :]     # derivatives of function values
    Wjac = jac[..., -1:, :]     # derivatives of weights
    return (Vjac * W - V * Wjac) / (W**2)   # use quotient rule for (V/W)'

class NurbsFunc(bspline._BaseSplineFunc):
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
    def __init__(self, kvs, coeffs, weights, premultiplied=False, support=None):
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
            self._isscalar = True
        elif len(dim) == 1:
            dim = dim[0]
            self._isscalar = False
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

        if support:
            self._support_override = tuple(support)
        else:
            self._support_override = None

    def output_shape(self):
        if self._isscalar:
            return ()
        else:
            shp = list(self.coeffs.shape[self.sdim:])
            shp[-1] -= 1
            return tuple(shp)

    def grid_eval(self, gridaxes):
        assert len(gridaxes) == self.sdim, "Input has wrong dimension"
        # make sure axes are one-dimensional
        if not all(np.ndim(ax) == 1 for ax in gridaxes):
            gridaxes = tuple(np.squeeze(ax) for ax in gridaxes)
            assert all(ax.ndim == 1 for ax in gridaxes), \
                "Grid axes should be one-dimensional"
        colloc = [bspline.collocation(self.kvs[i], gridaxes[i]) for i in range(self.sdim)]
        vals = apply_tprod(colloc, self.coeffs)
        f = vals[..., :-1] / vals[..., -1:]       # divide by weight function
        if self._isscalar:
            f = np.squeeze(f, -1)           # eliminate scalar axis
        return f

    def grid_jacobian(self, gridaxes):
        bsp = BSplineFunc(self.kvs, self.coeffs)
        val = bsp.grid_eval(gridaxes)
        jac = bsp.grid_jacobian(gridaxes)   # shape(grid) x (dim+1) x sdim
        J = _nurbs_jacobian(val, jac)
        if self._isscalar:
            J = np.squeeze(J, -2)           # eliminate scalar axis
        return J
    
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
        H = Nhess1 - mat[..., I, J]             # linearize the symmetric part
        if self._isscalar:
            H = np.squeeze(H, -2)           # eliminate scalar axis
        return H

    def pointwise_eval(self, points):
        """Evaluate the NURBS function at an unstructured list of points.

        Args:
            points: an array or sequence such that `points[i]` is an array containing
                the coordinates for dimension `i`, where `i = 0, ..., sdim - 1`
                (in xyz order). All arrays must have the same shape.

        Returns:
            An `ndarray` containing the function values at the `points`.
        """
        vals = bspline.tp_bsp_eval_pointwise(self.kvs, self.coeffs, points)
        f = vals[..., :-1] / vals[..., -1:]     # divide by weight function
        if self._isscalar:
            f = np.squeeze(f, -1)               # eliminate scalar axis
        return f

    def pointwise_jacobian(self, points):
        """Evaluate the Jacobian of the NURBS function at an unstructured list
        of points.

        Args:
            points: an array or sequence such that `points[i]` is an array containing
                the coordinates for dimension `i`, where `i = 0, ..., sdim - 1`
                (in xyz order). All arrays must have the same shape.

        Returns:
            An `ndarray` containing the Jacobian matrices at the `points`,
            i.e., a matrix of size `dim x sdim` per evaluation point.
        """
        val, jac = bspline.tp_bsp_eval_with_jac_pointwise(self.kvs, self.coeffs, points)
        J = _nurbs_jacobian(val, jac)
        if self._isscalar:
            J = np.squeeze(J, -2)           # eliminate scalar axis
        return J

    def boundary(self, bdspec, swap=None, flip = None):
        """Return one side of the boundary as a :class:`NurbsFunc`.

        Args:
            bdspec: the side of the boundary to return; see :func:`.compute_dirichlet_bc`

        Returns:
            :class:`NurbsFunc`: representation of the boundary side;
            has `sdim` reduced by 1 and the same `dim` as this function
        """
        if flip is None:
            flip = self.sdim*(False,)
        self.flip = tuple(flip)
        
        if self._support_override:
            # if we have reduced support, the boundary may not be
            # interpolatory; return a custom function
            return bspline._BaseGeoFunc.boundary(self, bdspec, flip=flip)

        bdspec = bspline._parse_bdspec(bdspec, self.sdim)
        axis, sides = tuple(ax for ax, _ in bdspec), tuple(-idx for _, idx in bdspec)
        assert all([0 <= ax < self.sdim for ax in axis]), 'Invalid axis'
        slices = self.sdim * [slice(None)]
        for ax, idx in zip(axis, sides):
            slices[ax] = idx
        coeffs = self.coeffs[tuple(slices)]
        kvs = list(self.kvs)
        for ax in sorted(axis, reverse=True):
            del kvs[ax]
        return NurbsFunc(kvs, coeffs, weights=None, premultiplied=True)

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
        self._support_override = tuple(new_support)

    def copy(self):
        """Return a copy of this geometry."""
        return NurbsFunc(
                tuple(kv.copy() for kv in self.kvs),
                self.coeffs.copy(),
                None,
                premultiplied=True,
                support = self._support_override)

    def coeffs_weights(self):
        """Return the non-premultiplied coefficients and weights as a pair of arrays."""
        W = self.coeffs[..., -1]
        return self.coeffs[..., :-1] / W[..., None], W.copy()

    def translate(self, offset):
        """Return a version of this geometry translated by the specified offset."""
        C, W = self.coeffs_weights()
        return NurbsFunc(self.kvs, C + offset, W, support = self._support_override)

    def scale(self, factor):
        """Scale all control points either by a scalar factor or componentwise by
        a vector, leave the weights unchanged, and return the resulting new function.
        """
        C, W = self.coeffs_weights()
        return NurbsFunc(self.kvs, C * factor, W, support=self._support_override)

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
        return NurbsFunc(self.kvs, np.squeeze(C, axis=-1), W, support=self._support_override)

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

    def as_nurbs(self):
        return self

    def as_vector(self):
        if self.is_vector():
            return self
        else:
            assert self.is_scalar()
            C = self.coeffs[..., :-1]   # keep singleton dimension, don't squeeze it
            return NurbsFunc(self.kvs, C, self.coeffs[..., -1], premultiplied=True, support=self.support)

    def __getitem__(self, I):
        C = self.coeffs[..., :-1]
        return NurbsFunc(self.kvs, C[..., I], self.coeffs[..., -1], premultiplied=True, support = self._support_override)


class UserFunction(bspline._BaseGeoFunc):
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
        self._support_override=tuple(support)
        self.jac = jac
        if dim is None:
            x0 = tuple(lo for (lo,hi) in reversed(support))
            dim = np.shape(f(*x0))
            self._output_shape = dim
            if len(dim) == 0:
                dim = 1
            elif len(dim) == 1:
                dim = dim[0]
        else:
            if np.isscalar(dim):
                self._output_shape = (dim,)
            else:
                self._output_shape = dim
        self.dim = dim
        self.sdim = len(support)

    def output_shape(self):
        return self._output_shape

    def grid_eval(self, grd):
        return utils.grid_eval(self.f, grd)

    def eval(self, *x):
        return self.f(*x)

    def pointwise_eval(self, points):
        return self.eval(*points)

    def grid_jacobian(self, grd):
        assert self.jac is not None, 'Jacobian not specified in UserFunction'
        return utils.grid_eval(self.jac, grd)

class ComposedFunction(bspline._BaseSplineFunc):
    def __init__(self, geo2, geo1):
        """Composition of two functions.

        `geo(x) = geo2(geo1(x))`
        """
        assert geo1.dim == geo2.sdim
        self.geo1 = geo1
        self.geo2 = geo2
        self.sdim = geo1.sdim
        self.dim = geo2.dim

    @property
    def support(self):
        return self.geo1.support

    @support.setter
    def support(self, new_support):
        self.geo1.support = new_support

    def grid_eval(self, grd):
        """Evaluate the function over a tensor product grid."""
        XY = self.geo1.grid_eval(grd)
        # XY is no longer a TP grid in general, need pointwise eval.
        # The last axis of XY are the coordinates, need to bring
        # them to the front.
        return self.geo2.pointwise_eval(np.rollaxis(XY, -1))

    def grid_jacobian(self, grd):
        """Evaluate the Jacobian over a tensor product grid."""
        XY = self.geo1.grid_eval(grd)
        jac1 = self.geo1.grid_jacobian(grd)
        jac2 = self.geo2.pointwise_jacobian(np.rollaxis(XY, -1))
        return np.matmul(jac2, jac1)

    def boundary(self, bdspec):
        """Return one side of the boundary as a :class:`ComposedFunction`."""
        return ComposedFunction(self.geo2, self.geo1.boundary(bdspec))

class _BoundaryFunction(bspline._BaseGeoFunc):
    """A function which represents the evaluation of the given function `f` at
    one side of its boundary, thus reducing `sdim` by one.
    """
    def __init__(self, f, bdspec, flip = None):
        self.f = f
        bdspec = bspline._parse_bdspec(bdspec, f.sdim)
        self.axis, self.sides = tuple(zip(*sorted(bdspec)))
        self.fixed_coord = {ax: f.support[ax][idx] for ax, idx in bdspec}
        self.support = list(f.support)
        for ax in np.flip(self.axis):
            del self.support[ax]
        self.support=tuple(self.support)
        self.dim = f.dim
        self.sdim = f.sdim - len(bdspec)
        
        if flip is None:
            flip = self.sdim*(False,)
        self.flip = tuple(flip)
        
        if f._support_override:
            self._support_override = self.support

    def output_shape(self):
        return self.f.output_shape()

    def eval(self, *x):
        x = list(x)
        for ax in self.axis:
            x.insert(len(x) - ax, self.fixed_coord[ax])
        return self.f(*x)

    def grid_eval(self, gridaxes):
        gridaxes = [1 - grid if flp else grid for grid, flp in zip(gridaxes, self.flip)]
        for ax in self.axis:
            gridaxes.insert(ax, np.array([self.fixed_coord[ax]]))
        vals = utils.grid_eval(self.f, gridaxes)
        return np.squeeze(vals,self.axis)

    def grid_jacobian(self, gridaxes, keep_normal=False):
        gridaxes = [1 - grid if flp else grid for grid, flp in zip(gridaxes, self.flip)]
        for ax in self.axis:
            gridaxes.insert(ax, np.array([self.fixed_coord[ax]]))
        jacs = self.f.grid_jacobian(gridaxes)
        jacs = np.squeeze(jacs, self.axis)
        if not keep_normal:
            # drop the partial derivatives corresponding to the normal
            # direction
            ax = jacs.shape[-1] - self.axis[0] - 1
            jacs = np.concatenate((jacs[..., :ax], jacs[..., ax+1:]),
                    axis=-1)
        return jacs
    
    def grid_outer_normal(self, gridaxes):
        gridaxes = [1 - np.flip(grid) if flp else grid for grid, supp, flp in zip(gridaxes, self.support, self.flip)]
        N = [len(grid) for grid in gridaxes]
        #gridaxes.insert(self.axis, np.array([self.fixed_coord]))
        jacs = self.grid_jacobian(gridaxes, keep_normal=False)
        if self.dim==2 and self.sdim==1:     # line integral
            x = jacs
            di=-1 if self.axis != self.side else 1
            x[:,0]=-x[:,0]
            x[:,[0,1]]=x[:,[1,0]]
            return di*x/np.linalg.norm(x,axis=1)[:,None]
        elif self.dim==3 and self.sdim==2:   # surface integral
            di=-1 if (self.axis[0]+self.sides[0])%2==0 else 1
            x, y = jacs[:,:,:,0], jacs[:,:,:,1]
            un=np.cross(x, y).reshape(N[0],N[1],3,1)
            return di*un/np.linalg.norm(un,axis=2)[:,:,None]
        else:
            assert False, 'do not know how to compute normal vector for Jacobian shape {}'.format(jacs.shape)

################################################################################
# Examples of 2D geometries
################################################################################

def unit_square(num_intervals=1, support = None):
    """Unit square with given number of intervals per direction.

    Returns:
        :class:`.BSplineFunc` 2D geometry
    """
    return unit_cube(dim=2, num_intervals=num_intervals, support=support)

def perturbed_square(num_intervals=5, noise=0.02, support = None):
    """Randomly perturbed unit square.

    Unit square with given number of intervals per direction;
    the control points are perturbed randomly according to the
    given noise level.

    Returns:
        :class:`.BSplineFunc` 2D geometry
    """
    return unit_square(num_intervals, support).perturb(noise)

def Quad(P,support=None):
    
    bottom = line_segment(P[:,0],P[:,1])
    top = line_segment(P[:,2],P[:,3])
    left = line_segment(P[:,0],P[:,2])
    right = line_segment(P[:,1],P[:,3])
    
    kvs, coeffs = _combine_boundary_curves(bottom,top,left,right)
    return bspline.BSplineFunc(kvs, coeffs, support=support)

def bspline_annulus(r1=1.0, r2=2.0, phi=np.pi/2, support = None):
    """A B-spline approximation of a quarter annulus in the first quadrant.

    Args:
        r1 (float): inner radius
        r2 (float): outer radius

    Returns:
        :class:`.BSplineFunc` 2D geometry
    """
    assert -np.pi/2<=phi<=np.pi/2, 'angle needs to be sharp!'
    kvx = bspline.make_knots(1, 0.0, 1.0, 1)
    kvy = bspline.make_knots(2, 0.0, 1.0, 1)

    coeffs = np.array([
            [[ r1, 0.0],
             [ r2, 0.0]],
            [[ r1,  np.tan(phi/2)*r1],
             [ r2,  np.tan(phi/2)*r2]],
            [[np.cos(phi)*r1,  np.sin(phi)*r1],
             [np.cos(phi)*r2,  np.sin(phi)*r2]],
    ])
    return BSplineFunc((kvy,kvx), coeffs, support)

def bspline_quarter_annulus(r1=1.0, r2=2.0, support = None):
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
    return BSplineFunc((kvy,kvx), coeffs, support)

def annulus(r1=1.0, r2=2.0, phi=np.pi/2, support = None):
    """A NURBS representation of a quarter annulus in the first quadrant.
    The 'bottom' and 'top' boundaries (with respect to the reference domain)
    lie on the x and y axis, respectively.

    Args:
        r1 (float): inner radius
        r2 (float): outer radius

    Returns:
        :class:`NurbsFunc` 2D geometry
    """
    assert -np.pi/2<=phi<=np.pi/2, 'angle needs to be sharp!'
    kvx = bspline.make_knots(1, 0.0, 1.0, 1)
    kvy = bspline.make_knots(2, 0.0, 1.0, 1)

    coeffs = np.array([
            [[ r1, 0.0, 1.0],
             [ r2, 0.0, 1.0]],
            [[ r1,  np.tan(phi/2)*r1, 1.0 / np.sqrt(2.0)],
             [ r2,  np.tan(phi/2)*r2, 1.0 / np.sqrt(2.0)]],
            [[np.cos(phi)*r1,  np.sin(phi)*r1, 1.0],
             [np.cos(phi)*r2,  np.sin(phi)*r2, 1.0]],
    ])
    return NurbsFunc((kvy,kvx), coeffs, weights=None, support = support)

def quarter_annulus(r1=1.0, r2=2.0, support = None):
    """A NURBS representation of a quarter annulus in the first quadrant.
    The 'bottom' and 'top' boundaries (with respect to the reference domain)
    lie on the x and y axis, respectively.

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
    return NurbsFunc((kvy,kvx), coeffs, weights=None, support = support)

def _combine_boundary_curves(bottom, top, left, right):
    kvs = (left.kvs[0], bottom.kvs[0])
    coeffs = np.full((kvs[0].numdofs, kvs[1].numdofs, left.coeffs.shape[1]), np.nan)
    coeffs[:,  0] = left.coeffs
    coeffs[:, -1] = right.coeffs
    coeffs[ 0, :] = bottom.coeffs
    coeffs[-1, :] = top.coeffs
    return kvs, coeffs

def disk(r=1.0, support = None):
    """A NURBS representation of a circular disk.

    The parametrization has four boundary singularities where the determinant
    of the Jacobian becomes 0, at the bottom, top, left and right edges.

    Args:
        r (float): radius

    Returns:
        :class:`NurbsFunc` 2D geometry
    """
    gR = circular_arc(np.pi / 2)        # upper right arc

    gL = gR.copy()
    gL.coeffs = np.flipud(gL.coeffs)    # reverse order
    gL = gL.scale(-1)                   # flip into bottom left arc

    gB = gR.rotate_2d(-np.pi / 2)       # bottom right arc
    gT = gL.rotate_2d(-np.pi / 2)       # upper left arc

    kvs, coeffs = _combine_boundary_curves(gB, gT, gL, gR)
    coeffs[1, 1] = (0.0, 0.0, 0.5)  # weight 1/2 makes weight matrix rank 1
    if r != 1.0:
        coeffs[:, :, :2] *= r
    return NurbsFunc(kvs, coeffs, None, premultiplied=True, support = support)

################################################################################
# Examples of 3D geometries
################################################################################

def unit_cube(dim=3, num_intervals=1, support = None):
    """The `dim`-dimensional unit cube with `num_intervals` intervals
    per coordinate direction.

    Returns:
        :class:`.BSplineFunc` geometry
    """
    if support:
        assert len(support)==dim, "Wrong dimension of support!"
        return functools.reduce(tensor_product, tuple(line_segment(0.0, 1.0, intervals=num_intervals, support = S) for S in support))
    else:
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

def line_segment(x0, x1, intervals=1,support=None):
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
    return BSplineFunc(bspline.make_knots(1, 0.0, 1.0, intervals,), coeffs, support = support)

def circular_arc(alpha, r=1.0):
    """Construct a circular arc with angle `alpha` and radius `r`.

    The arc is centered at the origin, starts on the positive `x` axis and
    travels in counterclockwise direction.
    """
    if 0.0 < alpha < np.pi:
        return circular_arc_3pt(alpha, r)
    elif np.pi <= alpha <= 2 * np.pi:
        return circular_arc_7pt(alpha, r)
    else:
        raise ValueError('invalid angle {}'.format(alpha))

def circular_arc_3pt(alpha, r=1.0):
    """Construct a circular arc with angle `alpha` and radius `r` using 3 control points.

    The angle `alpha` must be between 0 and `pi`.
    """
    # formulas adapted from
    # https://www.cs.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/NURBS/RB-circles.html
    assert 0.0 < alpha < np.pi, 'Invalid angle'

    kv = bspline.make_knots(2, 0.0, 1.0, 1)
    coeffs = np.array([(np.cos(a), np.sin(a)) for a in np.linspace(0, alpha, 3)])
    W = [1.0, np.cos(alpha / 2), 1.0]
    return NurbsFunc(kv, r * coeffs, weights=W, premultiplied=True)

def circular_arc_5pt(alpha, r=1.0):
    """Construct a circular arc with angle `alpha` and radius `r` using 5 control points."""
    kv = bspline.make_knots(2, 0.0, 1.0, 2, mult=2)
    coeffs = np.array([(np.cos(a), np.sin(a)) for a in np.linspace(0, alpha, 5)])
    w = np.cos(alpha / 4)
    W = [1.0, w, 1.0, w, 1.0]
    return NurbsFunc(kv, r * coeffs, weights=W, premultiplied=True)

def circular_arc_7pt(alpha, r=1.0):
    """Construct a circular arc with angle `alpha` and radius `r` using 7 control points."""
    kv = bspline.make_knots(2, 0.0, 1.0, 3, mult=2)
    coeffs = np.array([(np.cos(a), np.sin(a)) for a in np.linspace(0, alpha, 7)])
    w = np.cos(alpha / 6)
    W = np.array([1, w, 1, w, 1, w, 1])
    return NurbsFunc(kv, r * coeffs, weights=W, premultiplied=True)

def semicircle(r=1.0):
    """Construct a semicircle in the upper half-plane with radius `r` using NURBS."""
    return circular_arc_5pt(np.pi, r)

def circle(r=1.0):
    """Construct a circle with radius `r` using NURBS."""
    return circular_arc_7pt(2 * np.pi, r)

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
    if Gs != ():
        return tensor_product(G1, tensor_product(G2, *Gs))
    if G1.is_scalar():
        G1 = G1.as_vector()
    if G2.is_scalar():
        G2 = G2.as_vector()
    assert G1.is_vector() and G2.is_vector(), 'only implemented for scalar- or vector-valued functions'

    Gs = (G1, G2)
    nurbs = any(isinstance(G, NurbsFunc) for G in Gs)
    override = any(G._support_override for G in Gs)

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
        
    if override:
        if G1.sdim ==1: supp1 = (G1.support,) 
        else: supp1 =G1.support
        if G2.sdim ==1: supp2 = (G2.support,) 
        else: supp2 = G2.support
        support = supp1 + supp2
    else:
        support = None

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
        return NurbsFunc(G1.kvs + G2.kvs, C, W, support = support)
    else:
        return BSplineFunc(G1.kvs + G2.kvs, C, support = support)

