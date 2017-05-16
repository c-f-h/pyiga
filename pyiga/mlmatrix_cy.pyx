# cython: profile=False

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
cdef inline void from_seq2(np.int_t i, np.int_t[:] dims, size_t[2] out):
    out[0] = i // dims[1]
    out[1] = i % dims[1]

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef object from_seq(np.int_t i, np.int_t[:] dims):
    """Convert sequential (lexicographic) index into multiindex.

    Same as np.unravel_index(i, dims) except for returning a list.
    """
    cdef np.int_t L, k
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
cdef np.int_t to_seq(np.int_t[:] I, np.int_t[:] dims):
    """Convert multiindex into sequential (lexicographic) index.

    Same as np.ravel_multiindex(I, dims).
    """
    cdef np.int_t i, k
    i = 0
    for k in range(dims.shape[0]):
        i *= dims[k]
        i += I[k]
    return i

# this doesn't work currently
#def reindex_to_multilevel(i, j, bs):
#    bs = np.array(bs, copy=False)   # bs has shape L x 2
#    I, J = from_seq(i, bs[:,0]), from_seq(j, bs[:,1])  # multilevel indices
#    return tuple(to_seq((I[k],J[k]), bs[k,:]) for k in range(bs.shape[0]))   # <<- to_seq() expects buffers

def reindex_from_multilevel(M, np.int_t[:,:] bs):
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


################################################################################
# Inflation and matvec
################################################################################

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def make_block_indices(sparsidx, block_sizes):
    cdef np.int_t[:] sidx
    cdef unsigned[:, ::1] bidx
    cdef list bidx_list = []
    cdef size_t i, s, N, n

    assert len(sparsidx) == len(block_sizes)
    for k in range(len(sparsidx)):
        sidx = sparsidx[k]
        n = block_sizes[k][1]
        N = len(sidx)
        bidx = np.empty((N,2), dtype=np.uint32)

        for i in range(N):
            s = sidx[i]
            bidx[i,0], bidx[i,1] = s // n, s % n

        bidx_list.append(bidx)

    return tuple(bidx_list)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef object ml_nonzero_2d(bidx, block_sizes):
    cdef unsigned[:,::1] bidx1, bidx2
    cdef int m2,n2
    cdef size_t i, j, Ni, Nj, idx
    cdef unsigned xi0, xi1, yi0, yi1

    bidx1, bidx2 = bidx
    m2, n2 = block_sizes[1]
    Ni, Nj = len(bidx1), len(bidx2)
    results = np.empty((2, Ni * Nj), dtype=np.uintp)
    cdef size_t[:, ::1] IJ = results

    with nogil:
        idx = 0
        for i in range(Ni):
            xi0, xi1 = bidx1[i,0], bidx1[i,1]          # range: m1, n1
            for j in range(Nj):
                yi0, yi1 = bidx2[j,0], bidx2[j,1]      # range: m2, n2

                IJ[0,idx] = xi0 * m2 + yi0      # range: m1*m2*m3
                IJ[1,idx] = xi1 * n2 + yi1      # range: n1*n2*n3
                idx += 1
    return results


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef object inflate_2d(object X, np.int_t[:] sparsidx1, np.int_t[:] sparsidx2,
        int m1, int n1, int m2, int n2):
    """Convert the dense ndarray X from reordered, compressed two-level
    banded form back to a standard sparse matrix format.
    """
    cdef long[::1] entries_i, entries_j
    cdef size_t i, j, si, sj, M, N, k=0
    M, N = X.shape[0], X.shape[1]

    cdef size_t bi0, bi1, ii0, ii1

    assert len(sparsidx1) == M
    assert len(sparsidx2) == N

    entries_i = np.empty(M*N, dtype=int)
    entries_j = np.empty(M*N, dtype=int)

    for i in range(M):
        si = sparsidx1[i]                   # range: m1*n1
        bi0, bi1 = si // n1, si % n1        # range: m1, n1
        for j in range(N):
            sj = sparsidx2[j]               # range: m2*n2
            ii0, ii1 = sj // n2, sj % n2    # range: m2, n2

            entries_i[k] = bi0*m2 + ii0     # range: m1*m2
            entries_j[k] = bi1*n2 + ii1     # range: n1*n2
            k += 1

    return scipy.sparse.csr_matrix((X.ravel('C'), (entries_i, entries_j)),
            shape=(m1*m2, n1*n2))


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void ml_matvec_2d(double[:,::1] X,
        unsigned[:,::1] bidx1, unsigned[:,::1] bidx2,
        int m1, int n1, int m2, int n2,
        double[::1] x, double[::1] y):
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


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef object ml_nonzero_3d(bidx, block_sizes):
    cdef unsigned[:,::1] bidx1, bidx2, bidx3
    cdef int m2,n2, m3,n3
    cdef size_t i, j, k, Ni, Nj, Nk, idx
    cdef unsigned xi0, xi1, yi0, yi1, zi0, zi1

    bidx1, bidx2, bidx3 = bidx
    m2, n2 = block_sizes[1]
    m3, n3 = block_sizes[2]
    Ni, Nj, Nk = len(bidx1), len(bidx2), len(bidx3)
    results = np.empty((2, Ni * Nj * Nk), dtype=np.uintp)
    cdef size_t[:, ::1] IJ = results

    with nogil:
        idx = 0
        for i in range(Ni):
            xi0, xi1 = bidx1[i,0], bidx1[i,1]          # range: m1, n1
            for j in range(Nj):
                yi0, yi1 = bidx2[j,0], bidx2[j,1]      # range: m2, n2
                for k in range(Nk):
                    zi0, zi1 = bidx3[k,0], bidx3[k,1]  # range: m3, n3

                    IJ[0,idx] = (xi0 * m2 + yi0) * m3 + zi0    # range: m1*m2*m3
                    IJ[1,idx] = (xi1 * n2 + yi1) * n3 + zi1    # range: n1*n2*n3
                    idx += 1
    return results


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef object inflate_3d(object X, sparsidx, block_sizes):
    """Convert the dense ndarray X from reordered, compressed three-level
    banded form back to a standard sparse matrix format.

    Returns:
        sparse matrix of size `m1*m2*m3 x n1*n2*n3`
    """
    cdef np.int_t[:] sparsidx1, sparsidx2, sparsidx3
    sparsidx1,sparsidx2,sparsidx3 = sparsidx

    cdef int m1,n1, m2,n2, m3,n3
    m1,n1 = block_sizes[0]
    m2,n2 = block_sizes[1]
    m3,n3 = block_sizes[2]

    cdef long[::1] entries_i, entries_j
    cdef size_t i, j, k, si, sj, sk, Ni, Nj, Nk
    cdef size_t idx=0
    Ni,Nj,Nk = X.shape

    assert len(sparsidx1) == Ni
    assert len(sparsidx2) == Nj
    assert len(sparsidx3) == Nk

    cdef size_t xi0, xi1, yi0, yi1, zi0, zi1

    entries_i = np.empty(Ni*Nj*Nk, dtype=int)
    entries_j = np.empty(Ni*Nj*Nk, dtype=int)

    for i in range(Ni):
        si = sparsidx1[i]                       # range: m1*n1
        xi0, xi1 = si // n1, si % n1            # range: m1, n1

        for j in range(Nj):
            sj = sparsidx2[j]                   # range: m2*n2
            yi0, yi1 = sj // n2, sj % n2        # range: m2, n2

            for k in range(Nk):
                sk = sparsidx3[k]               # range: m3*n3
                zi0, zi1 = sk // n3, sk % n3    # range: m3, n3

                entries_i[idx] = (xi0 * m2 + yi0) * m3 + zi0    # range: m1*m2*m3
                entries_j[idx] = (xi1 * n2 + yi1) * n3 + zi1    # range: n1*n2*n3
                idx += 1

    return scipy.sparse.csr_matrix((X.ravel('C'), (entries_i, entries_j)),
            shape=(m1*m2*m3, n1*n2*n3))


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void ml_matvec_3d(double[:,:,::1] X, bidx, block_sizes, double[::1] x, double[::1] y):
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

