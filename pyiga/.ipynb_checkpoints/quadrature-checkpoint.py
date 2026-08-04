import numpy as np
from pyiga import solvers

def gauss_rule(deg, a, b):
    """Return nodes and weights for Gauss-Legendre rule of given degree in (a,b).

    a and b are arrays containing the start and end points of the intervals."""
    m = 0.5*(a + b)     # array of interval midpoints
    h = 0.5*(b - a)     # array of halved interval lengths
    x,w = np.polynomial.legendre.leggauss(deg)
    nodes   = (np.outer(h,x) + m[:, np.newaxis])
    weights = np.outer(h,w)
    return (nodes.ravel(), weights.ravel())

def greville_rule(kv):
    """Return nodes and weights for Greville rule of a given Knot Vector kv."""
    nodes = kv.greville()
    rhs = (kv.kv[kv.p+1:]-kv.kv[:-kv.p-1])/(kv.p+1)
    weights = solvers.make_solver(bspline.collocation(kv,nodes))@rhs
    return nodes, weights

def make_iterated_quadrature(intervals, nqp):
    return gauss_rule(nqp, intervals[:-1], intervals[1:])

def make_tensor_quadrature(meshes, nqp):
    gauss = tuple(make_iterated_quadrature(mesh, nqp) for mesh in meshes)
    grid    = tuple(g[0] for g in gauss)
    weights = tuple(g[1] for g in gauss)
    return grid, weights

def make_boundary_quadrature(meshes, nqp, bdspec):
    """Compute an iterated Gauss quadrature rule restricted to the given boundary."""
    bdax, bdside = bdspec[0], bdspec[1]
    bdcoord = meshes[bdax][0 if bdside==0 else -1]
    gauss = [make_iterated_quadrature(mesh, nqp) for mesh in meshes]
    gauss[bdax] = (np.array([bdcoord]), np.ones((1,)))
    grid    = tuple(g[0] for g in gauss)
    weights = tuple(g[1] for g in gauss)
    return grid, weights
