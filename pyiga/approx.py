# -*- coding: utf-8 -*-
"""Methods for approximating functions in tensor product spline spaces."""
from . import bspline
from . import assemble
from . import tensor
from . import operators
from . import utils

def interpolate(kvs, f, nodes=None):
    """Compute the coefficients for the interpolant of the function `f` in
    the tensor product B-spline basis `kvs`.

    `nodes` should be a tensor grid. If not specified, the Gréville abscissae
    are used.
    """
    if nodes is None:
        nodes = [kv.greville() for kv in kvs]
    Cinvs = [operators.make_solver(bspline.collocation(kvs[i], nodes[i]))
                for i in range(len(kvs))]
    rhs = utils.grid_eval(f, nodes)
    return tensor.apply_tprod(Cinvs, rhs)

def project_L2(kvs, f):
    """Compute the coefficients for the :math:`L_2`-projection of the function `f` into
    the tensor product B-spline basis `kvs`.
    """
    Minvs = [operators.make_solver(assemble.mass(kv), spd=True) for kv in kvs]
    rhs = assemble.inner_products(kvs, f)
    return tensor.apply_tprod(Minvs, rhs)
