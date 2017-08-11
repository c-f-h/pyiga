cimport cython
from libcpp.vector cimport vector
from cython cimport view    # avoid compiler crash

import numpy as np
cimport numpy as np

import scipy.sparse

#
# Imports from fastasm.cc:
#
cdef extern void set_log_func(void (*logfunc)(char * str, size_t))

cdef extern void fast_assemble_2d_cimpl "fast_assemble_2d"(
        MatrixEntryFn entryfunc, void * data,
        size_t n0, int bw0,
        size_t n1, int bw1,
        double tol, int maxiter, int skipcount, int tolcount,
        int verbose,
        vector[size_t]& entries_i, vector[size_t]& entries_j, vector[double]& entries)

cdef extern void fast_assemble_3d_cimpl "fast_assemble_3d"(
        MatrixEntryFn entryfunc, void * data,
        size_t n0, int bw0,
        size_t n1, int bw1,
        size_t n2, int bw2,
        double tol, int maxiter, int skipcount, int tolcount,
        int verbose,
        vector[size_t]& entries_i, vector[size_t]& entries_j, vector[double]& entries)
#
# Imports end
#


# this is so that IPython notebooks can capture the output
cdef void _stdout_log_func(char * s, size_t nbytes):
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
# Wrappers for particular assemblers
################################################################################

from .assemble_tools_cy cimport _entry_func_2d, _entry_func_3d
from .assemblers cimport (
    MassAssembler2D, MassAssembler3D,
    StiffnessAssembler2D, StiffnessAssembler3D,
)

def fast_mass_2d(kvs, geo, tol=1e-10, maxiter=100, skipcount=3, tolcount=3, verbose=2):
    cdef MassAssembler2D asm = MassAssembler2D(kvs, geo)
    return fast_assemble_2d_wrapper(_entry_func_2d, <void*>asm, kvs,
            tol, maxiter, skipcount, tolcount, verbose)

def fast_stiffness_2d(kvs, geo, tol=1e-10, maxiter=100, skipcount=3, tolcount=3, verbose=2):
    cdef StiffnessAssembler2D asm = StiffnessAssembler2D(kvs, geo)
    return fast_assemble_2d_wrapper(_entry_func_2d, <void*>asm, kvs,
            tol, maxiter, skipcount, tolcount, verbose)


def fast_mass_3d(kvs, geo, tol=1e-10, maxiter=100, skipcount=3, tolcount=3, verbose=2):
    cdef MassAssembler3D asm = MassAssembler3D(kvs, geo)
    return fast_assemble_3d_wrapper(_entry_func_3d, <void*>asm, kvs,
            tol, maxiter, skipcount, tolcount, verbose)

def fast_stiffness_3d(kvs, geo, tol=1e-10, maxiter=100, skipcount=3, tolcount=3, verbose=2):
    cdef StiffnessAssembler3D asm = StiffnessAssembler3D(kvs, geo)
    return fast_assemble_3d_wrapper(_entry_func_3d, <void*>asm, kvs,
            tol, maxiter, skipcount, tolcount, verbose)

