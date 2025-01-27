
ctypedef double (*MatrixEntryFn)(size_t i, size_t j, void * data) noexcept

cdef object fast_assemble_2d_wrapper(MatrixEntryFn entry_func, void * data, kvs,
        double tol, int maxiter, int skipcount, int tolcount, int verbose)

cdef object fast_assemble_3d_wrapper(MatrixEntryFn entry_func, void * data, kvs,
        double tol, int maxiter, int skipcount, int tolcount, int verbose)
