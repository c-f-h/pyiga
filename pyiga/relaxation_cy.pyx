# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gauss_seidel(const int[:] row_ptr, const int[:] col_indices, const double[:] data,
        double[:] x, const double[:] b,
        int row_start, int row_stop, int row_step):

    cdef int i, j, jj, start, end
    cdef double rsum, diag

    i = row_start
    while i != row_stop:
        start = row_ptr[i]
        end   = row_ptr[i+1]
        rsum = 0.0
        diag = 0.0

        for jj in range(start, end):
            j = col_indices[jj]
            if i == j:
                diag = data[jj]
            else:
                rsum += data[jj] * x[j]

        if diag != 0.0:
            x[i] = (b[i] - rsum) / diag

        i += row_step

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gauss_seidel_indexed(int[:] row_ptr, int[:] col_indices, double[:] data,
        double[:] x, double[:] b,
        int[:] indices, bint reverse):

    cdef int idx, I0, I1, Is, i, j, jj, start, end
    cdef double rsum, diag

    if reverse:
        I0,I1,Is = indices.shape[0] - 1, -1, -1
    else:
        I0,I1,Is = 0, indices.shape[0], 1

    idx = I0
    while idx != I1:
        i = indices[idx]
        start = row_ptr[i]
        end   = row_ptr[i+1]
        rsum = 0.0
        diag = 0.0

        for jj in range(start, end):
            j = col_indices[jj]
            if i == j:
                diag = data[jj]
            else:
                rsum += data[jj] * x[j]

        if diag != 0.0:
            x[i] = (b[i] - rsum) / diag

        idx += Is
