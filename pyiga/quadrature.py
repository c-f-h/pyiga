import numpy as np

def gauss_rule(deg, a, b):
    """Return nodes and weights for Gauss-Legendre rule of given degree in (a,b).
    
    a and b are arrays containing the start and end points of the intervals."""
    m = 0.5*(a + b)     # array of interval midpoints
    h = 0.5*(b - a)     # array of halved interval lengths
    x,w = np.polynomial.legendre.leggauss(deg)
    nodes   = (np.outer(h,x) + m[:, np.newaxis])
    weights = np.outer(h,w)
    return (nodes.ravel(), weights.ravel())

def make_iterated_quadrature(intervals, nqp):
    return gauss_rule(nqp, intervals[:-1], intervals[1:])

