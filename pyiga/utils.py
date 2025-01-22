import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import itertools
import functools
import operator

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
        mesh = list(np.meshgrid(*grid, sparse=True, indexing='ij'))
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

def kron_partial(As, rows, restrict=False, format='csr'):
    """Compute a partial Kronecker product between the sparse matrices
    `As = (A_1, ..., A_k)`, filling only the given `rows` in the output matrix.

    If `restrict=True`, a smaller matrix is returned which contains only the
    given rows of the original matrix. Otherwise, a matrix with the same shape
    as the full Kronecker product is returned.
    """
    from .mlmatrix import MLStructure
    # determine the I,J indices of the nonzeros in the given rows
    S = MLStructure.from_kronecker(As)
    out_shape = (len(rows), S.shape[1]) if restrict else S.shape

    if restrict:
        I, J, I_idx = S.nonzeros_for_rows(rows, renumber_rows=True)
    else:
        I, J = S.nonzeros_for_rows(rows)
    if len(I) == 0:     # no nonzeros? return zero matrix
        return scipy.sparse.csr_matrix(out_shape)

    # block sizes for unraveling the row and column indices
    bs_I = tuple(S.bs[k][0] for k in range(S.L))
    bs_J = tuple(S.bs[k][1] for k in range(S.L))
    # unravel the indices to refer to the individual blocks A_k
    I_ix = np.unravel_index(I, bs_I)
    J_ix = np.unravel_index(J, bs_J)
    # compute the values of the individual factor matrices
    values = tuple(As[k][I_ix[k], J_ix[k]].A1 for k in range(S.L))
    # compute the Kronecker product as the product of the factors
    entries = functools.reduce(operator.mul, values)
    if restrict:
        I = I_idx
    return scipy.sparse.coo_matrix((entries, (I,J)), shape=out_shape).asformat(format)

def cartesian_product(arrays):
    """Compute the Cartesian product of any number of input arrays."""
    L = len(arrays)
    shp = tuple(a.shape[0] for a in arrays)
    arr = np.empty(shp + (L,), dtype=arrays[0].dtype)
    for i in range(L):
        # broadcast the i-th array along all but the i-th axis
        ix = L * [np.newaxis]
        ix[i] = slice(shp[i])
        arr[..., i] = arrays[i][tuple(ix)]
    arr.shape = (-1, L)
    return arr

class CSRRowSlice:
    """A simple class which allows quickly applying a row slice of a CSR matrix
    to dense matrices.

    This is required since creating submatrices of scipy sparse matrices is a
    very heavyweight operation.
    """
    def __init__(self, A, row_bounds):
        assert isinstance(A, scipy.sparse.csr_matrix)
        self.A = A
        assert 0 <= row_bounds[0] <= row_bounds[1] <= A.shape[0], 'invalid row bounds'
        self.shape = (row_bounds[1] - row_bounds[0], A.shape[1])
        self.indptr = self.A.indptr[row_bounds[0]:row_bounds[1]]
        self.dtype = A.dtype

    def _matmat(self, other):
        # adapted from _mul_multivector in scipy's compressed.py
        M, N = self.shape

        y = other if other.ndim == 2 else other.reshape((-1, 1))
        n_vecs = y.shape[1]  # number of column vectors
        result = np.zeros((M, n_vecs), dtype=self.dtype)

        scipy.sparse._sparsetools.csr_matvecs(M, N, n_vecs,
                self.indptr, self.A.indices, self.A.data,
                   y.ravel(), result.ravel())

        if other.ndim == 1:
            result.shape = (result.shape[0],)
        return result

    __mul__ = _matmat
    dot = _matmat

class CSRRowSubset:
    """A simple class which allows quickly applying a subset of the rows of a
    CSR matrix to a vector.

    This is required since creating submatrices of scipy sparse matrices is a
    very heavyweight operation.
    """
    def __init__(self, A, rows):
        assert isinstance(A, scipy.sparse.csr_matrix)
        self.A = A
        self.rows = rows
        self.shape = (len(rows), A.shape[1])
        self.dtype = A.dtype

    def _matvec(self, other):
        M, N = self.shape
        assert other.shape == (N,)
        result = np.zeros(M, dtype=self.dtype)

        indptr = self.A.indptr
        y = other.ravel()
        out = result.ravel()
        for i, r in enumerate(self.rows):
            scipy.sparse._sparsetools.csr_matvec(1, N,
                    indptr[r:r+1], self.A.indices, self.A.data,
                    y, out[i:i+1])
        return result

    __mul__ = _matvec
    dot = _matvec


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


def _noop(self, *args, **kwargs): pass
class _DummyPbar:
    """No-op stand-in for tqdm."""
    def __init__(self, *args, **kwags):
        if len(args) > 0:
            self.r = args[0]
    def __iter__(self):
        return iter(self.r)
    def __enter__(self):
        return self
    __exit__ = _noop
    update   = _noop
    close    = _noop
    set_postfix = _noop

def progress_bar(enable=True):
    if enable:
        import sys
        if 'tqdm' not in sys.modules:
            # tqdm shows an ugly warning when we slightly overstep the end time in a
            # time integration routine. To avoid that we disable its warnings.
            import tqdm
            import warnings
            warnings.simplefilter('ignore', tqdm.TqdmWarning)
        else:
            import tqdm
        return tqdm.tqdm
    else:
        return _DummyPbar
