#
# Utility functions for multi-level block matrices and
# multilevel banded matrices.
#

import numpy as np
import scipy.sparse.linalg

from . import lowrank, utils

################################################################################
# Multi-level banded matrix class
################################################################################

class MLStructure:
    """Class representing the structure of a multi-level block-structured
    sparse matrix.

    This means that it represents the sparsity structure of the Kronecker
    product of `L` sparse matrices, where `L` is the number of levels.
    The k-th Kronecker factor has size `m_k x n_k` and `nnz_k` nonzeros.

    Args:
        bs: the tuple of the block sizes ((m_1, n_1), ..., (m_L, n_L))
        bidx: for each level, contains an nnz_k x 2 array with the (i,j)
            indices of the nonzero locations of the k-th factor matrix

    .. note::
        The most convenient way to create instances of this class is through
        the static methods defined below for various types of matrices.
    """
    def __init__(self, bs, bidx):
        self.bs = tuple(bs)                 # layout: ((m_1, n_1), ..., (m_L, n_L))
        self._bs_arr = np.array(self.bs)    # shape: L x 2
        assert self._bs_arr.shape[1] == 2, 'invalid block sizes'
        self.bidx = tuple(bidx)             # for each level k, contains an nnz_k x 2 array with (i,j) indices
        assert len(self.bs) == len(self.bidx)
        self.L = len(self.bs)
        M = np.prod(tuple(b[0] for b in self.bs))
        N = np.prod(tuple(b[1] for b in self.bs))
        self.shape = (M,N)

    @staticmethod
    def multi_banded(bs, bw):
        """Create the structure of a multi-level banded matrix with square blocks
        of the sizes `bs` and bandwidths `bw`.
        """
        bs = tuple((n,n) for n in bs)
        bidx = tuple(compute_banded_sparsity_ij(n[0], p)
                for (n,p) in zip(bs, bw))
        return MLStructure(bs, bidx)

    @staticmethod
    def dense(shape):
        """Create the structure of a one-level dense matrix with the given shape."""
        return MLStructure((shape,), (compute_dense_ij(shape[0], shape[1]),))

    @staticmethod
    def from_kvs(kvs0, kvs1):
        """Create the appropriate multi-level matrix structure for a stiffness
        matrix defined over the two given tensor product bases.
        """
        bs = tuple((kv1.numdofs, kv0.numdofs) for (kv0,kv1) in zip(kvs0,kvs1))
        bidx = tuple(compute_sparsity_ij(kv0, kv1) for (kv0,kv1) in zip(kvs0,kvs1))
        return MLStructure(bs, bidx)

    @staticmethod
    def from_matrix(A):
        """Create a one-level matrix structure which has the same sparsity
        pattern as `A`.
        """
        bs = (tuple(A.shape),)
        I, J = A.nonzero()
        bidx = (np.column_stack((I, J)).astype(np.uint32),)
        return MLStructure(bs, bidx)

    @staticmethod
    def from_kronecker(As):
        """Create a matrix structure which represents the sparsity pattern of
        the Kronecker product of the tuple of matrices `As`.
        """
        S = MLStructure.from_matrix(As[0])
        for A in As[1:]:
            S = S.join(MLStructure.from_matrix(A))
        return S

    def join(self, other):
        """Append the given other structure and return the result."""
        return MLStructure(self.bs + other.bs, self.bidx + other.bidx)

    def reorder(self, axes):
        """Permute the levels of the matrix according to `axes`."""
        assert len(axes) == self.L
        return MLStructure(
                bs=tuple(self.bs[j] for j in axes),
                bidx=tuple(self.bidx[j] for j in axes))

    def slice(self, start, end=None):
        """Get structure for a single dimension or several consecutive dimensions."""
        assert 0 <= start < self.L, 'invalid slice index'
        if end is None: end = start + 1
        sl = slice(start, end)
        return MLStructure(self.bs[sl], self.bidx[sl])

    def make_mlmatrix(self, data=None, matrix=None):
        """Create a multi-level matrix with the structure given by this object.

        Returns:
            :class:`MLMatrix`: the resulting multi-level matrix
        """
        return MLMatrix(structure=self, data=data, matrix=matrix)

    def nonzero(self, lower_tri=False):
        """
        Return a tuple of arrays `(row,col)` containing the indices of
        the non-zero elements of the matrix.

        If `lower_tri` is ``True``, return only the indices for the
        lower triangular part.
        """
        if self.L == 1:
            assert not lower_tri, 'Lower triangular part not implemented in 1D'
            IJ = self.bidx[0].T.copy()
        elif self.L == 2:
            IJ = ml_nonzero_2d(self.bidx, self._bs_arr, lower_tri=lower_tri)
        elif self.L == 3:
            IJ = ml_nonzero_3d(self.bidx, self._bs_arr, lower_tri=lower_tri)
        else:
            IJ = ml_nonzero_nd(self.bidx, self._bs_arr, lower_tri=lower_tri)
        return IJ[0,:], IJ[1,:]

    def transpose(self):
        """Return the structure for the transpose of this one."""
        bs = tuple((b[1], b[0]) for b in self.bs)
        ix = np.array([1,0])    # indices for swapping i and j
        bidx = tuple(np.ascontiguousarray(bx[:, ix]) for bx in self.bidx)
        return MLStructure(bs, bidx)

    def _level_rowwise_interactions(self, k):
        # return a list which contains, for each row index, a list of the
        # column indices that row interacts with on matrix level k
        num_rows = self.bs[k][0]
        bx = self.bidx[k]
        nnz = bx.shape[0]
        result = [[] for i in range(num_rows)]
        for s in range(nnz):
            result[bx[s, 0]].append(bx[s, 1])
        return [np.array(r, dtype=int) for r in result]

    def nonzeros_for_rows(self, row_indices, renumber_rows=False):
        """Compute a pair of index arrays `(I,J)` specifying the locations of
        nonzeros (just like :func:`nonzero`), but containing only those
        nonzeros which lie in the given rows.
        """
        if len(row_indices) == 0:
            if renumber_rows:
                return np.empty(0, dtype=int), np.empty(0, dtype=int), np.empty(0, dtype=int)
            else:
                return np.empty(0, dtype=int), np.empty(0, dtype=int)
        L = self.L
        lvia = tuple(self._level_rowwise_interactions(k) for k in range(L))
        bs_I = tuple(self.bs[k][0] for k in range(L))
        bs_J = tuple(self.bs[k][1] for k in range(L))

        bs_J_arr = np.array(bs_J, dtype=np.int_)       # for passing to cython function

        # convert to multi-indices: ix[i,k] = component index k of row_indices[i]
        ix = np.column_stack(np.unravel_index(row_indices, bs_I)).astype(np.int_, copy=False)

        # compute the raveled Cartesian products for each row_index
        # Js is a list of 1D integer arrays
        Js = pyx_rowwise_cartesian_product(lvia, ix, bs_J_arr)

        counts = tuple(J_i.shape[0] for J_i in Js)
        Is = np.repeat(row_indices, counts)

        if len(Js) > 0:
            Js = np.concatenate(Js)
        else:
            Js = np.empty(0, dtype=int)

        if renumber_rows:
            return Is, Js, np.repeat(np.arange(len(row_indices)), counts)
        else:
            return Is, Js

    def nonzeros_for_columns(self, col_indices):
        """Compute a pair of index arrays `(I,J)` specifying the locations of
        nonzeros (just like :func:`nonzero`), but containing only those
        nonzeros which lie in the given columns.
        """
        J, I = self.transpose().nonzeros_for_rows(col_indices)
        return I, J     # swap I and J because of transpose

    def sequential_bidx(self):
        # returns a version of bidx with ravelled indices
        return [ self.bs[j][0] * self.bidx[j][:,0] + self.bidx[j][:,1]
                 for j in range(self.L) ]


class MLMatrix(scipy.sparse.linalg.LinearOperator):
    """Compact representation of a multi-level structured sparse matrix.

    Many IgA matrices arising from tensor product bases have multi-level
    banded structure, meaning that they are block-structured, each block
    is banded, and the block pattern itself is banded. This allows
    compact storage of all coefficients in a dense matrix or tensor.
    See (Hofreither 2017) for details.

    This class allows even more general block-structured matrix representations
    where each level has an arbitrary sparsity pattern.

    Args:
        structure (:class:`MLStructure`): the multi-level structure of the matrix to create
        matrix: a dense or sparse matrix with the proper multi-level
            banded structure used as initializer
        data (ndarray): alternatively, the compact data array for the matrix
            can be specified directly
    """
    def __init__(self, structure, data=None, matrix=None):
        self.structure = structure
        self.L = self.structure.L
        self.shape = self.structure.shape

        self.datashape = tuple(len(bi) for bi in self.structure.bidx)

        # initialize data (ndarray of shape mu_1 x ... x mu_L)
        if (data is not None) and (matrix is not None):
            assert False, 'Can only specify one of `data` and `matrix`'
        if data is not None:
            assert data.shape == self.datashape, 'Wrong shape of data tensor'
            self._data = np.asarray(data, order='C')
            dtype = self._data.dtype
        elif matrix is not None:
            assert matrix.shape == self.shape, 'Matrix has wrong shape'
            data = np.asarray(matrix[self.nonzero()]).reshape(self.datashape)
            self._data = np.asarray(data, order='C')
            dtype = self._data.dtype
        else:
            self._data = None
            dtype = np.float64

        scipy.sparse.linalg.LinearOperator.__init__(self, shape=self.shape, dtype=dtype)

    @property
    def nnz(self):
        """Return the number of nonzeros in a sparse matrix representation."""
        return np.prod(self.datashape)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, X):
        assert X.shape == self.datashape
        self._data = np.asarray(X, order='C')

    def asmatrix(self, format='csr'):
        """Return a sparse matrix representation in the given format."""
        assert self._data is not None, 'matrix has no data'
        if self.L == 1:
            return scipy.sparse.coo_matrix(
                    (self._data, (self.structure.bidx[0][:,0], self.structure.bidx[0][:,1])),
                    shape=self.shape).asformat(format)
        else:
            IJ = self.nonzero()
            A = scipy.sparse.csr_matrix((self._data.ravel('C'), IJ), shape=self.shape)
            return A.asformat(format)

    def _matvec(self, x):
        """Compute the matrix-vector product with vector `x`."""
        assert self._data is not None, 'matrix has no data'
        assert len(x) == self.shape[1], 'Invalid input size'
        if self.L == 2:
            y = np.zeros(len(x))
            ml_matvec_2d(self._data, self.structure.bidx, self.structure._bs_arr, x, y)
            return y
        elif self.L == 3:
            y = np.zeros(len(x))
            ml_matvec_3d(self._data, self.structure.bidx, self.structure._bs_arr, x, y)
            return y
        else:
            return self.asmatrix().dot(x)

    def nonzero(self, lower_tri=False):
        """
        Return a tuple of arrays `(row,col)` containing the indices of
        the non-zero elements of the matrix.

        If `lower_tri` is ``True``, return only the indices for the
        lower triangular part.
        """
        return self.structure.nonzero(lower_tri=lower_tri)

    def reorder(self, axes):
        """Permute the levels of the matrix according to `axes`."""
        assert len(axes) == self.L
        if self.data is not None:
            newdata = np.transpose(self.data, axes)
        else:
            newdata = None
        return MLMatrix(
                structure=self.structure.reorder(axes),
                data=newdata)


################################################################################
# Reordering and reindexing
################################################################################

def reorder(X, m1, n1):
    """Input X has m1 x n1 blocks of size m2 x n2, i.e.,
    total size m1*m2 x n1*n2.

    Output has m1*n1 rows of length m2*n2, where each row
    is one vectorized block of X.

    This implements the matrix reordering described in
    [Van Loan, Pitsianis 1993] for dense matrices."""
    (M,N) = X.shape
    m2 = M // m1
    n2 = N // n1
    assert M == m1*m2 and N == n1*n2, "Invalid block size"
    Y = np.empty((m1*n1, m2*n2))
    for i in range(m1):
        for j in range(n1):
            B = X[i*m2 : (i+1)*m2, j*n2 : (j+1)*n2]
            Y[i*n1 + j, :] = B.ravel('C')
    return Y

def reindex_from_reordered(i,j, m1,n1,m2,n2):
    """Convert (i,j) from an index into reorder(X, m1, n1) into the
    corresponding index into X (reordered to original).

    Arguments:
        i = row = block index           (0...m1*n1)
        j = column = index within block (0...m2*n2)

    Returns:
        a pair of indices with ranges `(0...m1*m2, 0...n1*n2)`
    """
    bi0, bi1 = i // n1, i % n1      # range: m1, n1
    ii0, ii1 = j // n2, j % n2      # range: m2, n2
    return (bi0*m2 + ii0, bi1*n2 + ii1)

def from_seq(i, dims):
    """Convert sequential (lexicographic) index into multiindex.

    Same as np.unravel_index(i, dims) except for returning a list.
    """
    L = len(dims)
    I = L * [0]
    for k in reversed(range(L)):
        mk = dims[k]
        I[k] = i % mk
        i //= mk
    return I

def to_seq(I, dims):
    """Convert multiindex into sequential (lexicographic) index.

    Same as np.ravel_multi_index(I, dims).
    """
    i = 0
    for k in range(len(dims)):
        i *= dims[k]
        i += I[k]
    return i

def reindex_to_multilevel(i, j, bs):
    """Convert sequential indices (i,j) of a multilevel matrix
    with L levels and block sizes bs into a multiindex with length L.
    """
    bs = np.array(bs, copy=False)   # bs has shape L x 2
    I, J = from_seq(i, bs[:,0]), from_seq(j, bs[:,1])  # multilevel indices
    return tuple(to_seq((I[k],J[k]), bs[k,:]) for k in range(bs.shape[0]))

def reindex_from_multilevel(M, bs):
    """Convert a multiindex M with length L into sequential indices (i,j)
    of a multilevel matrix with L levels and block sizes bs.

    This is the multilevel version of reindex_from_reordered, which does the
    same for the two-level case.

    Arguments:
        M: the multiindex to convert
        bs: the block sizes; ndarray of shape Lx2. Each row gives the sizes
            (rows and columns) of the blocks on the corresponding matrix level.
    """
    bs = np.array(bs, copy=False)  # bs has shape L x 2
    IJ = np.stack((from_seq(M[k], bs[k,:]) for k in range(len(M))), axis=0)
    return tuple(to_seq(IJ[:,m], bs[:,m]) for m in range(2))

def compute_banded_sparsity(n, bw):
    """Returns list of ravelled indices which are nonzero in a square,
    banded matrix of size n and bandwidth bw.

    This is identical to np.flatnonzero(X) of such a banded matrix X.
    """
    I = []
    for i in range(n):
        for j in range(max(0, i-bw), min(n, i+bw+1)):
            I.append(i*n + j)
    return np.array(I, dtype=int)

def compute_banded_sparsity_ij(n, bw):
    """Returns an `N x 2` array of those indices (i,j) which are nonzero in a
    square, banded matrix of size n and bandwidth bw.

    This is similar to :func:`compute_banded_sparsity`, but retains
    the (i,j) indices rather than raveling them.
    """
    I = []
    for i in range(n):
        for j in range(max(0, i-bw), min(n, i+bw+1)):
            I.append((i, j))
    return np.array(I, dtype=np.uint32)

def compute_sparsity_ij(kv1, kv2):
    """Returns an `N x 2` array of those basis function indices (i,j) where the
    two given knot vectors have joint support.

    This describes the sparsity pattern of a stiffness matrix assembled for these
    two knot vectors.
    """
    def do_intersect(intva, intvb):
        c0, c1 = (max(intva[0], intvb[0]), min(intva[1], intvb[1]))
        return c1 > c0

    meshsupp1 = kv1.mesh_support_idx_all()
    meshsupp2 = kv2.mesh_support_idx_all()

    IJ = []
    for i in range(meshsupp2.shape[0]):
        j = np.searchsorted(meshsupp1[:,1], meshsupp2[i,0], side='right')
        while j < meshsupp1.shape[0] and do_intersect(meshsupp2[i], meshsupp1[j]):
            IJ.append((i,j))
            j += 1
    return np.array(IJ, dtype=np.uint32)

def compute_dense_ij(m, n):
    """Returns an `N x 2` array of all indices (i,j) in a dense `m x n` matrix.  """
    return np.array([ (i,j) for i in range(m) for j in range(n)], dtype=np.uint32)


################################################################################
# Elementwise generators for ML-reordered sparse matrices
################################################################################

def ReorderedMatrixGenerator(multiasm, structure):
    assert structure.L == 2
    n1, m1 = structure.bs[0]
    n2, m2 = structure.bs[1]
    sparsidx = structure.sequential_bidx()

    def multientryfunc(indices):
        return multiasm(
            [reindex_from_reordered(sparsidx[0][i], sparsidx[1][j], n1, m1, n2, m2)
                for (i,j) in indices])
    shp = tuple(len(si) for si in sparsidx)
    return lowrank.MatrixGenerator(shp[0], shp[1], multientryfunc=multientryfunc)

def ReorderedTensorGenerator(multiasm, structure):
    L = structure.L
    bs = structure._bs_arr
    sparsidx = structure.sequential_bidx()

    Ms = L * [None]
    def multientryfunc(indices):
        indices = list(indices)
        for n in range(len(indices)):
            for k in range(L):
                Ms[k] = sparsidx[k][indices[n][k]]
            indices[n] = reindex_from_multilevel(Ms, bs)
        return multiasm(indices)
    shp = tuple(len(si) for si in sparsidx)
    return lowrank.TensorGenerator(shp, multientryfunc=multientryfunc)



# import optimized versions as well as some additional functions
from .mlmatrix_cy import *


