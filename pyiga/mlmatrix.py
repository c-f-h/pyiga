#
# Utility functions for multi-level block matrices and
# multilevel banded matrices.
#

import numpy as np
from . import lowrank


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
            Y[i + j*m1, :] = B.ravel('F')
    return Y

def reindex_from_reordered(i,j, m1,n1,m2,n2):
    """Convert (i,j) from an index into reorder(X, m1, n1) into the
    corresponding index into X (reordered to original).

      i = row = block index           (0...m1*n1)
      j = column = index within block (0...m2*n2)
    """
    bi0, bi1 = i % m1, i // m1
    ii0, ii1 = j % m2, j // m2
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

    Same as np.ravel_multiindex(I, dims).
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
    assert bw % 2 == 1, 'Bandwidth must be an odd number'
    k = bw // 2
    for j in range(n):
        for i in range(max(0, j-k), min(n, j+k+1)):
            I.append(i + j*n)
    return np.array(I, dtype=int)



################################################################################
# Elementwise generators for ML-reordered sparse matrices
################################################################################

def ReorderedMatrixGenerator(asm, sparsidx, n1, n2):
    block_sizes = (n1,n1, n2,n2)
    def entryfunc(i, j):
        (ii,jj) = reindex_from_reordered(sparsidx[0][i], sparsidx[1][j], *block_sizes)
        return asm(ii, jj)
    shp = tuple(len(si) for si in sparsidx)
    return lowrank.MatrixGenerator(shp[0], shp[1], entryfunc)

def ReorderedTensorGenerator(asm, sparsidx, bs):
    block_sizes = np.array([(b,b) for b in bs])
    L = len(sparsidx)
    assert L == block_sizes.shape[0]
    Ms = L * [None]
    def entryfunc(M):
        for k in range(L):
            Ms[k] = sparsidx[k][M[k]]
        i,j = reindex_from_multilevel(Ms, block_sizes)
        return asm(i, j)
    shp = tuple(len(si) for si in sparsidx)
    return lowrank.TensorGenerator(shp, entryfunc)



# import optimized versions as well as some additional functions
from .mlmatrix_cy import *


