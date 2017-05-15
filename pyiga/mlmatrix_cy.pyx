# cython: profile=False

cimport cython

import numpy as np
cimport numpy as np

import scipy.sparse

################################################################################
# Structured matrix reordering
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


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef object inflate_3d(object X, sparsidx, block_sizes):
    """Convert the dense ndarray X from reordered, compressed three-level
    banded form back to a standard sparse matrix format.
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
        si = sparsidx1[i]
        xi0, xi1 = si % m1, si // m1

        for j in range(Nj):
            sj = sparsidx2[j]
            yi0, yi1 = sj % m2, sj // m2

            for k in range(Nk):
                sk = sparsidx3[k]
                zi0, zi1 = sk % m3, sk // m3

                entries_i[idx] = (xi0 * m2 + yi0) * m3 + zi0
                entries_j[idx] = (xi1 * n2 + yi1) * n3 + zi1
                idx += 1

    return scipy.sparse.csr_matrix((X.ravel('C'), (entries_i, entries_j)),
            shape=(m1*m2*m3, n1*n2*n3))

