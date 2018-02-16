#
# Utility functions for multi-level block matrices and
# multilevel banded matrices.
#

import numpy as np
import scipy.sparse.linalg

from . import lowrank


################################################################################
# Multi-level banded matrix class
################################################################################

class MLBandedMatrix(scipy.sparse.linalg.LinearOperator):
    """Compact representation of a multi-level banded matrix.

    Many IgA matrices arising from tensor product bases have multi-level
    banded structure, meaning that they are block-structured, each block
    is banded, and the block pattern itself is banded. This allows
    compact storage of all coefficients in a dense matrix or tensor.
    See (Hofreither 2017) for details.

    Args:
        bs (seq): list of block sizes, one per level (dimension)
        bw (seq): list of bandwidths, one per level (dimension)
        matrix: a dense or sparse matrix with the proper multi-level
            banded structure used as initializer
        data (ndarray): alternatively, the compact data array for the matrix
            can be specified directly
    """
    def __init__(self, bs, bw, bidx=None, data=None, matrix=None):
        self.bs = tuple((n,n) if np.isscalar(n) else n for n in bs)
        self._bs_arr = np.array(self.bs)
        self.L = len(bs)
        if bidx is not None:
            self.bidx = bidx
        else:
            self.bidx = tuple(compute_banded_sparsity_ij(n[0], p)
                    for (n,p) in zip(self.bs, bw))
        assert self.L == len(self.bidx), \
            'Inconsistent dimensions for block sizes and bandwidths/structure'
        # bidx is a tuple where each bidx_k has shape (mu_k, 2)

        M = np.prod(self._bs_arr[:,0])
        N = np.prod(self._bs_arr[:,1])

        self.datashape = tuple(len(bi) for bi in self.bidx)

        # initialize data (ndarray of shape mu_1 x ... x mu_L)
        if (data is not None) and (matrix is not None):
            assert False, 'Can only specify one of `data` and `matrix`'
        if data is not None:
            assert data.shape == self.datashape, 'Wrong shape of data tensor'
            self._data = np.asarray(data, order='C')
            dtype = self._data.dtype
        elif matrix is not None:
            assert matrix.shape == (M,N), 'Matrix has wrong shape'
            data = np.asarray(matrix[self.nonzero()]).reshape(self.datashape)
            self._data = np.asarray(data, order='C')
            dtype = self._data.dtype
        else:
            self._data = None
            dtype = np.float_

        scipy.sparse.linalg.LinearOperator.__init__(self, shape=(M,N), dtype=dtype)

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
                    (self._data, (self.bidx[0][:,0], self.bidx[0][:,1])),
                    shape=self.shape).asformat(format)
        elif self.L == 2:
            A = inflate_2d_bidx(self._data, self.bidx, self._bs_arr)
        elif self.L == 3:
            A = inflate_3d_bidx(self._data, self.bidx, self._bs_arr)
        else:
            assert False, 'dimension %d not implemented' % self.L
        return A.asformat(format)

    def _matvec(self, x):
        """Compute the matrix-vector product with vector `x`."""
        assert self._data is not None, 'matrix has no data'
        assert len(x) == self.shape[1], 'Invalid input size'
        if self.L == 2:
            y = np.zeros(len(x))
            ml_matvec_2d(self._data, self.bidx, self._bs_arr, x, y)
            return y
        elif self.L == 3:
            y = np.zeros(len(x))
            ml_matvec_3d(self._data, self.bidx, self._bs_arr, x, y)
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
        if self.L == 1:
            assert not lower_tri, 'Lower triangular part not implemented in 1D'
            IJ = self.bidx[0].T.copy()
        elif self.L == 2:
            IJ = ml_nonzero_2d(self.bidx, self._bs_arr, lower_tri=lower_tri)
        elif self.L == 3:
            IJ = ml_nonzero_3d(self.bidx, self._bs_arr, lower_tri=lower_tri)
        else:
            assert False, 'dimension %d not implemented' % self.L
        return IJ[0,:], IJ[1,:]

    def reorder(self, axes):
        """Permute the levels of the matrix according to `axes`."""
        assert len(axes) == self.L
        if self.data is not None:
            newdata = np.transpose(self.data, axes)
        else:
            newdata = None
        return MLBandedMatrix(
                bs=tuple(self.bs[j] for j in axes),
                bw=None,
                bidx=tuple(self.bidx[j] for j in axes),
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


################################################################################
# Elementwise generators for ML-reordered sparse matrices
################################################################################

def ReorderedMatrixGenerator(multiasm, sparsidx, n1, n2):
    def multientryfunc(indices):
        return multiasm(
            [reindex_from_reordered(sparsidx[0][i], sparsidx[1][j], n1, n1, n2, n2)
                for (i,j) in indices])
    shp = tuple(len(si) for si in sparsidx)
    return lowrank.MatrixGenerator(shp[0], shp[1], multientryfunc=multientryfunc)

def ReorderedTensorGenerator(multiasm, sparsidx, bs):
    block_sizes = np.array([(b,b) for b in bs])
    L = len(sparsidx)
    assert L == block_sizes.shape[0]
    Ms = L * [None]
    def multientryfunc(indices):
        indices = list(indices)
        for n in range(len(indices)):
            for k in range(L):
                Ms[k] = sparsidx[k][indices[n][k]]
            indices[n] = reindex_from_multilevel(Ms, block_sizes)
        return multiasm(indices)
    shp = tuple(len(si) for si in sparsidx)
    return lowrank.TensorGenerator(shp, multientryfunc=multientryfunc)



# import optimized versions as well as some additional functions
from .mlmatrix_cy import *


