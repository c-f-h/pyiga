# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

cimport cython
from libcpp.vector cimport vector
from cython cimport view    # avoid compiler crash
from cpython cimport pycapsule

import numpy as np
cimport numpy as np

import scipy.sparse

#
# Imports from fastasm.cc:
#
cdef extern void set_log_func(void (*logfunc)(const char * str, size_t))

cdef extern void fast_assemble_2d_cimpl "fast_assemble_2d"(
        MatrixEntryFn entryfunc, void * data,
        size_t n0, int bw0,
        size_t n1, int bw1,
        double tol, int maxiter, int skipcount, int tolcount,
        int verbose,
        vector[size_t]& entries_i, vector[size_t]& entries_j, vector[double]& entries) noexcept

cdef extern void fast_assemble_3d_cimpl "fast_assemble_3d"(
        MatrixEntryFn entryfunc, void * data,
        size_t n0, int bw0,
        size_t n1, int bw1,
        size_t n2, int bw2,
        double tol, int maxiter, int skipcount, int tolcount,
        int verbose,
        vector[size_t]& entries_i, vector[size_t]& entries_j, vector[double]& entries) noexcept
#
# Imports end
#
# this is so that IPython notebooks can capture the output
cdef void _stdout_log_func(const char * s, size_t nbytes) noexcept:
    import sys
    sys.stdout.write(s[:nbytes].decode('ascii'))

# slightly higher-level wrapper for the C++ implementation
cdef object fast_assemble_2d_wrapper(MatrixEntryFn entry_func, void * data, kvs,
        double tol, int maxiter, int skipcount, int tolcount, int verbose):
    cdef vector[size_t] entries_i
    cdef vector[size_t] entries_j
    cdef vector[double] entries

    set_log_func(_stdout_log_func)

    fast_assemble_2d_cimpl(entry_func, data,
            kvs[0].numdofs, kvs[0].p,
            kvs[1].numdofs, kvs[1].p,
            tol, maxiter, skipcount, tolcount,
            verbose,
            entries_i, entries_j, entries)

    cdef size_t ne = entries.size()
    cdef size_t N = kvs[0].numdofs * kvs[1].numdofs

    #cdef double[:] edata =
    <double[:ne]>(entries.data())

    return scipy.sparse.coo_matrix(
            (<double[:ne]> entries.data(),
                (<size_t[:ne]> entries_i.data(),
                 <size_t[:ne]> entries_j.data())),
            shape=(N,N)).tocsr()

# slightly higher-level wrapper for the C++ implementation
cdef object fast_assemble_3d_wrapper(MatrixEntryFn entry_func, void * data, kvs,
        double tol, int maxiter, int skipcount, int tolcount, int verbose):
    cdef vector[size_t] entries_i
    cdef vector[size_t] entries_j
    cdef vector[double] entries

    set_log_func(_stdout_log_func)

    fast_assemble_3d_cimpl(entry_func, data,
            kvs[0].numdofs, kvs[0].p,
            kvs[1].numdofs, kvs[1].p,
            kvs[2].numdofs, kvs[2].p,
            tol, maxiter, skipcount, tolcount,
            verbose,
            entries_i, entries_j, entries)

    cdef size_t ne = entries.size()
    cdef size_t N = kvs[0].numdofs * kvs[1].numdofs * kvs[2].numdofs

    return scipy.sparse.coo_matrix(
            (<double[:ne]> entries.data(),
                (<size_t[:ne]> entries_i.data(),
                 <size_t[:ne]> entries_j.data())),
            shape=(N,N)).tocsr()


################################################################################
# Generic wrapper for arbitrary dimension
################################################################################

def fast_assemble(asm, kvs, tol=1e-10, maxiter=100, skipcount=3, tolcount=3, verbose=2):
    capsule = asm.entry_func_ptr()
    cdef MatrixEntryFn entryfunc = <MatrixEntryFn>(pycapsule.PyCapsule_GetPointer(capsule, "entryfunc"))

    dim = len(kvs)
    if dim == 2:
        return fast_assemble_2d_wrapper(entryfunc, <void*>asm, kvs,
                tol, maxiter, skipcount, tolcount, verbose)
    elif dim == 3:
        return fast_assemble_3d_wrapper(entryfunc, <void*>asm, kvs,
                tol, maxiter, skipcount, tolcount, verbose)
    else:
        raise NotImplementedError('fast assemblers only implemented for 2D and 3D')
