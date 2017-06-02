import numpy as np
import scipy.sparse
import scipy.sparse.linalg

def grid_eval(f, grid):
    if hasattr(f, 'grid_eval'):
        return f.grid_eval(grid)
    else:
        mesh = np.meshgrid(*grid, sparse=True, indexing='ij')
        mesh.reverse() # convert order ZYX into XYZ
        values = f(*mesh)

        # If f is a function which uses only some of its arguments,
        # values may have the wrong shape. In that case, broadcast
        # it to the full mesh extent.
        num_dims = len(grid)
        # if the function is not scalar, we need include the extra dimensions
        target_shape = tuple(len(g) for g in grid) + values.shape[num_dims:]
        if values.shape != target_shape:
            values = np.broadcast_to(values, target_shape)

        return values


def read_sparse_matrix(fname):
    I,J,vals = np.loadtxt(fname, skiprows=1, unpack=True)
    I -= 1
    J -= 1
    return scipy.sparse.coo_matrix((vals, (I,J))).tocsr()
