# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void rank_1_update(double[:,::1] X, double alpha, double[::1] u, double[::1] v) noexcept:
    """Perform the update `X += alpha * u * v^T`.

    This does the same thing as the BLAS function `dger`, but OpenBLAS
    tries to parallelize it, which hurts more than it helps. Instead of
    forcing OMP_NUM_THREADS=1, which slows down many other things,
    we write our own.
    """
    cdef double au
    cdef size_t i, j
    for i in range(X.shape[0]):
        au = alpha * u[i]
        for j in range(X.shape[1]):
            X[i,j] += au * v[j]


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void aca3d_update(double[:,:,::1] X, double alpha, double[::1] u, double[:,::1] V) noexcept:
    cdef double au
    cdef size_t i, j, k
    for i in range(X.shape[0]):
        au = alpha * u[i]
        for j in range(X.shape[1]):
            for k in range(X.shape[2]):
                X[i,j,k] += au * V[j,k]

