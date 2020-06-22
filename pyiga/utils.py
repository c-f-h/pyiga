import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import itertools

def _broadcast_to_grid(X, grid_shape):
    num_dims = len(grid_shape)
    # input might be a single scalar; make sure it's an array
    X = np.asanyarray(X)
    # if the field X is not scalar, we need include the extra dimensions
    target_shape = grid_shape + X.shape[num_dims:]
    if X.shape != target_shape:
        X = np.broadcast_to(X, target_shape)
    return X

def _ensure_grid_shape(values, grid):
    """Convert tuples of arrays into higher-dimensional arrays and make sure
    the array conforms to the grid size."""
    grid_shape = tuple(len(g) for g in grid)

    # if values is a tuple, interpret as vector-valued function
    if isinstance(values, tuple):
        values = np.stack(tuple(_broadcast_to_grid(v, grid_shape) for v in values),
                axis=-1)

    # If values came from a function which uses only some of its arguments,
    # values may have the wrong shape. In that case, broadcast it to the full
    # mesh extent.
    return _broadcast_to_grid(values, grid_shape)

def grid_eval(f, grid):
    """Evaluate function `f` over the tensor grid `grid`."""
    if hasattr(f, 'grid_eval'):
        return f.grid_eval(grid)
    else:
        mesh = np.meshgrid(*grid, sparse=True, indexing='ij')
        mesh.reverse() # convert order ZYX into XYZ
        values = f(*mesh)
        return _ensure_grid_shape(values, grid)

def grid_eval_transformed(f, grid, geo):
    """Transform the tensor grid `grid` by the geometry transform `geo` and
    evaluate `f` on the resulting grid.
    """
    trf_grid = grid_eval(geo, grid) # array of size shape(grid) x dim
    # extract coordinate components
    X = tuple(trf_grid[..., i] for i in range(trf_grid.shape[-1]))
    # evaluate the function
    vals = f(*X)
    return _ensure_grid_shape(vals, grid)

def read_sparse_matrix(fname):
    I,J,vals = np.loadtxt(fname, skiprows=1, unpack=True)
    I = I.astype(int)
    J = J.astype(int)
    I -= 1
    J -= 1
    return scipy.sparse.coo_matrix((vals, (I,J))).tocsr()

def multi_kron_sparse(As, format='csr'):
    """Compute the (sparse) Kronecker product of a sequence of sparse matrices."""
    if len(As) == 1:
        return As[0].asformat(format, copy=True)
    else:
        return scipy.sparse.kron(As[0], multi_kron_sparse(As[1:], format=format), format=format)


class LazyArray:
    """An interface for lazily evaluating functions over a tensor product grid
    with array slicing notation.
    """
    def __init__(self, f, grid, mode='eval'):
        self.f = f
        self.grid = grid
        self.mode = mode

    def __getitem__(self, I):
        assert len(I) == len(self.grid), "Wrong number of indices"
        localgrid = tuple(g[i] for (g,i) in zip(self.grid, I))
        if self.mode == 'eval':
            return grid_eval(self.f, localgrid)
        elif self.mode == 'jac':
            return self.f.grid_jacobian(localgrid)
        else:
            raise ValueError('invalid mode: ' + str(self.mode))

class LazyCachingArray:
    """An interface for lazily evaluating functions over a tensor product grid
    with array slicing notation. Already computed values are cached tile-wise.

    .. warning::

        Only works correctly if the output is requested in full consecutive tiles!
    """
    def __init__(self, f, outshape, grid, tilesize, mode='eval'):
        self.f = f
        self.outshape = outshape
        self.grid = grid
        self.mode = mode
        self.ts = tilesize
        self.tiles = {}

    def get_tile(self, I):
        """I is a tile index as a d-tuple."""
        T = self.tiles.get(I)
        if T is None:
            ts = self.ts
            localgrid = tuple(g[i*ts:(i+1)*ts] for (g,i) in zip(self.grid, I))
            if self.mode == 'eval':
                T = grid_eval(self.f, localgrid)
            elif self.mode == 'jac':
                T = self.f.grid_jacobian(localgrid)
            else:
                raise ValueError('invalid mode: ' + str(self.mode))
            self.tiles[I] = T
        return T

    def __getitem__(self, I):
        assert len(I) == len(self.grid), "Wrong number of indices"
        idx = tuple(tuple(range(sl.start, sl.stop)) for sl in I)
        N = tuple(len(gi) for gi in idx)  # size of output
        output = np.empty(N + self.outshape)
        ts = self.ts
        tiles = tuple(range(gi[0]//ts, (gi[-1] + ts - 1) // ts) for gi in idx)
        J0 = tuple(gi[0] // ts for gi in idx)   # index of first tile
        for J in itertools.product(*tiles):
            dest = tuple(slice((j-j0)*ts, (j-j0+1)*ts) for (j,j0) in zip(J,J0))
            output[dest] = self.get_tile(J)
        return output

class BijectiveIndex:
    """Maps a list of values to consecutive indices in the range `0, ..., len(values) - 1`
    and allows reverse lookup of the index.
    """
    def __init__(self, values):
        self.values = values
        self._index = dict()
        for (i, v) in enumerate(self.values):
            self._index[v] = i

    def __len__(self):
        return len(self.values)

    def __getitem__(self, i):
        return self.values[i]

    def index(self, v):
        return self._index[v]
