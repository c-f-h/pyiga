import numpy as np
import scipy.sparse
import scipy.sparse.linalg

def _broadcast_to_grid(X, grid_shape):
    num_dims = len(grid_shape)
    # input might be a single scalar; make sure it's an array
    X = np.asanyarray(X)
    # if the field X is not scalar, we need include the extra dimensions
    target_shape = grid_shape + X.shape[num_dims:]
    if X.shape != target_shape:
        X = np.broadcast_to(X, target_shape)
    return X

def grid_eval(f, grid):
    if hasattr(f, 'grid_eval'):
        return f.grid_eval(grid)
    else:
        mesh = np.meshgrid(*grid, sparse=True, indexing='ij')
        mesh.reverse() # convert order ZYX into XYZ
        values = f(*mesh)

        grid_shape = tuple(len(g) for g in grid)

        # if the function returned a tuple, interpret as vector-valued function
        if isinstance(values, tuple):
            values = np.stack(tuple(_broadcast_to_grid(v, grid_shape) for v in values),
                    axis=-1)

        # If f is a function which uses only some of its arguments,
        # values may have the wrong shape. In that case, broadcast
        # it to the full mesh extent.
        values = _broadcast_to_grid(values, grid_shape)

        return values

def grid_eval_transformed(f, grid, geo):
    """Transform the tensor grid `grid` by the geometry transform `geo` and
    evaluate `f` on the resulting grid.
    """
    trf_grid = grid_eval(geo, grid) # array of size shape(grid) x dim
    # extract coordinate components
    X = tuple(trf_grid[..., i] for i in range(trf_grid.shape[-1]))
    # evaluate the function
    vals = f(*X)
    if isinstance(vals, tuple):
        vals = np.stack(vals, axis=-1)
    return vals

def read_sparse_matrix(fname):
    I,J,vals = np.loadtxt(fname, skiprows=1, unpack=True)
    I -= 1
    J -= 1
    return scipy.sparse.coo_matrix((vals, (I,J))).tocsr()
