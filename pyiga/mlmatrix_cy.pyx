# cython: profile=False
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

cimport cython
from cython.parallel import prange

import numpy as np
cimport numpy as np

import scipy.sparse

################################################################################
# Reindexing
################################################################################

@cython.cdivision(True)
def reindex_from_reordered(size_t i, size_t j, size_t m1, size_t n1, size_t m2, size_t n2):
    """Convert (i,j) from an index into reorder(X, m1, n1) into the
    corresponding index into X (reordered to original).

    Arguments:
        i = row = block index           (0...m1*n1)
        j = column = index within block (0...m2*n2)

    Returns:
        a pair of indices with ranges `(0...m1*m2, 0...n1*n2)`
    """
    cdef size_t bi0, bi1, ii0, ii1
    bi0, bi1 = i // n1, i % n1      # range: m1, n1
    ii0, ii1 = j // n2, j % n2      # range: m2, n2
    return (bi0*m2 + ii0, bi1*n2 + ii1)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void from_seq2(np.int64_t i, np.int64_t[:] dims, size_t[2] out) noexcept:
    out[0] = i // dims[1]
    out[1] = i % dims[1]

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef object from_seq(np.int64_t i, np.int64_t[:] dims):
    """Convert sequential (lexicographic) index into multiindex.

    Same as np.unravel_index(i, dims) except for returning a list.
    """
    cdef np.int64_t L, k
    L = len(dims)
    I = L * [0]
    for k in reversed(range(L)):
        mk = dims[k]
        I[k] = i % mk
        i //= mk
    return I

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.int64_t to_seq(np.int64_t* I, np.int64_t* dims, int n) noexcept nogil:
    """Convert multiindex into sequential (lexicographic) index.

    Same as np.ravel_multiindex(I, dims).
    """
    cdef np.int64_t i, k
    i = 0
    for k in range(n):
        i *= dims[k]
        i += I[k]
    return i

# this doesn't work currently
#def reindex_to_multilevel(i, j, bs):
#    bs = np.array(bs, copy=False)   # bs has shape L x 2
#    I, J = from_seq(i, bs[:,0]), from_seq(j, bs[:,1])  # multilevel indices
#    return tuple(to_seq((I[k],J[k]), bs[k,:]) for k in range(bs.shape[0]))   # <<- to_seq() expects buffers

def reindex_from_multilevel(M, np.int64_t[:,:] bs):
    """Convert a multiindex M with length L into sequential indices (i,j)
    of a multilevel matrix with L levels and block sizes bs.

    This is the multilevel version of reindex_from_reordered, which does the
    same for the two-level case.

    Arguments:
        M: the multiindex to convert
        bs: the block sizes; ndarray of shape Lx2. Each row gives the sizes
            (rows and columns) of the blocks on the corresponding matrix level.
    """
    cdef size_t L = len(M), k
    cdef size_t[2] ij
    cdef size_t ii=0, jj=0

    for k in range(L):
        from_seq2(M[k], bs[k,:], ij)
        ii *= bs[k,0]
        ii += ij[0]
        jj *= bs[k,1]
        jj += ij[1]
    return (ii, jj)


def get_transpose_idx_for_bidx(bidx):
    cdef dict jidict
    jidict = {}
    for (k,(i,j)) in enumerate(bidx):
        jidict[(j,i)] = k

    transpose_bidx = np.empty(len(bidx), dtype=np.uintp)

    for k in range(len(bidx)):
        transpose_bidx[k] = jidict[ tuple(bidx[k]) ]
    return transpose_bidx


# computes the Cartesian product of indices and ravels them according to the size `dims`
@cython.boundscheck(False)
@cython.wraparound(False)
cdef pyx_raveled_cartesian_product(arrays, np.int64_t[::1] dims):
    cdef int L = len(arrays)
    cdef np.int64_t[8] I      # iteration index
    cdef np.int64_t[8] K      # corresponding indices K[k] = arrays[k][I[k]]
    cdef np.int64_t[8] shp    # size of Cartesian product
    cdef int i, k, N

    cdef (np.int64_t*)[8] arr_ptrs
    cdef np.int64_t[:] arr

    N = 1
    for k in range(L):
        # initialize shape
        shp[k] = arrays[k].shape[0]
        if shp[k] == 0:
            return np.zeros(0, dtype=int)
        N *= shp[k]
        # initialize pointer
        arr = arrays[k]
        arr_ptrs[k] = &arr[0]
        # initialize I and K
        I[k] = 0
        K[k] = arr_ptrs[k][I[k]]

    out_buf = np.empty(N, dtype=int)
    cdef np.int64_t[::1] out = out_buf

    with nogil:
        for i in range(N):
            # compute raveled index at current multi-index K
            out[i] = to_seq(&K[0], &dims[0], L)

            # increment multi-index I and update K
            for k in reversed(range(L)):
                I[k] += 1
                if I[k] < shp[k]:
                    K[k] = arr_ptrs[k][I[k]]
                    break
                else:
                    I[k] = 0
                    K[k] = arr_ptrs[k][I[k]]
                    # go on to increment previous one
    return out_buf

# helper function for nonzeros_for_rows
@cython.boundscheck(False)
@cython.wraparound(False)
def pyx_rowwise_cartesian_product(lvia, np.int64_t[:, ::1] ix, np.int64_t[::1] block_sizes):
    cdef int N = ix.shape[0]
    cdef int L = ix.shape[1]
    assert L <= 8, 'pyx_raveled_cartesian_product only implemented for L <= 8'
    cdef int i

    Js = N * [None]
    for i in range(N):      # loop over all row_indices
        # obtain the levelwise interactions for each index i_k = ix[i,k]
        ia_k = tuple(lvia[k][ix[i,k]] for k in range(L))
        # compute global interactions by taking the Cartesian product
        Js[i] = pyx_raveled_cartesian_product(ia_k, block_sizes)
    return Js

################################################################################
# Inflation and matvec
################################################################################

## 2D

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef object ml_nonzero_2d(bidx, block_sizes, bint lower_tri=False):
    cdef unsigned[:,::1] bidx1, bidx2
    cdef int m2,n2
    cdef size_t i, j, Ni, Nj, N, idx, I, J
    cdef unsigned xi0, xi1, yi0, yi1

    bidx1, bidx2 = bidx
    m2, n2 = block_sizes[1]
    Ni, Nj = len(bidx1), len(bidx2)
    N = Ni * Nj
    results = np.empty((2, N), dtype=np.uintp)
    cdef size_t[:, ::1] IJ = results

    with nogil:
        idx = 0
        for i in range(Ni):
            xi0, xi1 = bidx1[i,0], bidx1[i,1]          # range: m1, n1
            for j in range(Nj):
                yi0, yi1 = bidx2[j,0], bidx2[j,1]      # range: m2, n2

                I = xi0 * m2 + yi0      # range: m1*m2*m3
                J = xi1 * n2 + yi1      # range: n1*n2*n3
                if not lower_tri or J <= I:
                    IJ[0,idx] = I
                    IJ[1,idx] = J
                    idx += 1
    if idx < N: # TODO: what's the closed formula for N in the triangular case?
        return results[:, :idx]
    else:
        return results


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void ml_matvec_2d(double[:,::1] X, bidx, block_sizes, double[::1] x, double[::1] y) noexcept:
    cdef unsigned[:,::1] bidx1, bidx2
    bidx1,bidx2 = bidx

    cdef int m2, n2
    m2,n2 = block_sizes[1]

    cdef size_t i, j, M, N, I, J
    M, N = X.shape[0], X.shape[1]

    cdef unsigned bi0, bi1, ii0, ii1

    assert len(bidx1) == M
    assert len(bidx2) == N

    #for i in prange(M, schedule='static', nogil=True):
    # the += update is not thread safe! need atomics
    for i in range(M):
        bi0, bi1 = bidx1[i,0], bidx1[i,1]        # range: m1, n1
        for j in range(N):
            ii0, ii1 = bidx2[j,0], bidx2[j,1]    # range: m2, n2

            I = bi0*m2 + ii0     # range: m1*m2
            J = bi1*n2 + ii1     # range: n1*n2

            y[I] += X[i,j] * x[J]


## 3D

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef object ml_nonzero_3d(bidx, block_sizes, bint lower_tri=False):
    cdef unsigned[:,::1] bidx1, bidx2, bidx3
    cdef int m2,n2, m3,n3
    cdef size_t i, j, k, Ni, Nj, Nk, N, idx, I, J
    cdef unsigned xi0, xi1, yi0, yi1, zi0, zi1

    bidx1, bidx2, bidx3 = bidx
    m2, n2 = block_sizes[1]
    m3, n3 = block_sizes[2]
    Ni, Nj, Nk = len(bidx1), len(bidx2), len(bidx3)
    N = Ni * Nj * Nk
    results = np.empty((2, N), dtype=np.uintp)
    cdef size_t[:, ::1] IJ = results

    with nogil:
        idx = 0
        for i in range(Ni):
            xi0, xi1 = bidx1[i,0], bidx1[i,1]          # range: m1, n1
            for j in range(Nj):
                yi0, yi1 = bidx2[j,0], bidx2[j,1]      # range: m2, n2
                for k in range(Nk):
                    zi0, zi1 = bidx3[k,0], bidx3[k,1]  # range: m3, n3

                    I = (xi0 * m2 + yi0) * m3 + zi0    # range: m1*m2*m3
                    J = (xi1 * n2 + yi1) * n3 + zi1    # range: n1*n2*n3
                    if not lower_tri or J <= I:
                        IJ[0,idx] = I
                        IJ[1,idx] = J
                        idx += 1
    if idx < N: # TODO: what's the closed formula for N in the triangular case?
        return results[:, :idx]
    else:
        return results


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void ml_matvec_3d(double[:,:,::1] X, bidx, block_sizes, double[::1] x, double[::1] y) noexcept:
    cdef unsigned[:,::1] bidx1, bidx2, bidx3
    bidx1,bidx2,bidx3 = bidx

    cdef int m2,n2, m3,n3
    m2,n2 = block_sizes[1]
    m3,n3 = block_sizes[2]

    cdef size_t i, j, k, Ni, Nj, Nk, I, J
    cdef unsigned xi0, xi1, yi0, yi1, zi0, zi1

    Ni,Nj,Nk = X.shape[:3]
    assert len(bidx1) == Ni
    assert len(bidx2) == Nj
    assert len(bidx3) == Nk

    #for i in prange(Ni, schedule='static', nogil=True):
    # the += update is not thread safe! need atomics
    for i in range(Ni):
        xi0, xi1 = bidx1[i,0], bidx1[i,1]          # range: m1, n1

        for j in range(Nj):
            yi0, yi1 = bidx2[j,0], bidx2[j,1]      # range: m2, n2

            for k in range(Nk):
                zi0, zi1 = bidx3[k,0], bidx3[k,1]  # range: m3, n3

                I = (xi0 * m2 + yi0) * m3 + zi0    # range: m1*m2*m3
                J = (xi1 * n2 + yi1) * n3 + zi1    # range: n1*n2*n3

                y[I] += X[i,j,k] * x[J]


# generic dimension

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef object ml_nonzero_nd(bidx, block_sizes, bint lower_tri=False):
    cdef unsigned L = len(bidx)
    assert L <= 8, 'ml_nonzero_nd only implemented for L <= 8'

    cdef (unsigned*)[8] bidx_ptr
    cdef np.int64_t NN[8]             # number of nonzeros per dimension
    cdef np.int64_t block_rows[8]     # number of rows in dimension k
    cdef np.int64_t block_cols[8]     # number of columns in dimension k
    cdef np.int64_t cur_idx[8]        # current nonzero idx (0..NN[k])
    cdef np.int64_t block_i[8]        # row of current nonzero corresponding to cur_idx[k]
    cdef np.int64_t block_j[8]        # column of current nonzero corresponding to cur_idx[k]

    cdef unsigned[:,::1] bidx_temp
    cdef size_t k, idx, N, I, J

    N = 1
    for i in range(L):
        bidx_temp = bidx[i]
        bidx_ptr[i] = &bidx_temp[0,0]
        NN[i] = len(bidx_temp)
        N *= NN[i]
        block_rows[i], block_cols[i] = block_sizes[i]

        # intialize cur_idx to 0 and set the corresponding block_i/block_j values
        cur_idx[i] = 0
        block_i[i], block_j[i] = bidx_ptr[i][0], bidx_ptr[0][1]

    results = np.empty((2, N), dtype=np.uintp)
    cdef size_t[:, ::1] IJ = results

    cdef bint done = (N == 0)

    with nogil:
        idx = 0

        while not done:
            # compute sequential index for current (I,J) multi-indices
            I = to_seq(&block_i[0], &block_rows[0], L)
            J = to_seq(&block_j[0], &block_cols[0], L)

            if not lower_tri or J <= I:
                IJ[0,idx] = I
                IJ[1,idx] = J
                idx += 1

            # increment multi-index cur_idx and update block_i, block_j
            for k in reversed(range(L)):
                cur_idx[k] += 1
                if cur_idx[k] < NN[k]:
                    block_i[k], block_j[k] = bidx_ptr[k][2*cur_idx[k]+0], bidx_ptr[k][2*cur_idx[k]+1]
                    break
                else:
                    if k == 0:      # reached the end of the first dimension?
                        done = True
                        break
                    cur_idx[k] = 0
                    block_i[k], block_j[k] = bidx_ptr[k][2*cur_idx[k]+0], bidx_ptr[k][2*cur_idx[k]+1]
                    # go on to increment previous one

    if idx < N: # TODO: what's the closed formula for N in the triangular case?
        return results[:, :idx]
    else:
        return results

