from .assemble_tools_cy import *

from . import bspline
import numpy as np

# returned array has axes (basis function, grid point, derivative)
def compute_values_derivs(kv, grid, derivs):
    colloc = bspline.collocation_derivs(kv, grid, derivs=derivs)
    colloc = tuple(X.T.A for X in colloc)
    # The assemblers expect the resulting array to be in C-order.  Depending on
    # numpy version, stack() does not guarantee that, so enforce contiguity.
    return np.ascontiguousarray(np.stack(colloc, axis=-1))
