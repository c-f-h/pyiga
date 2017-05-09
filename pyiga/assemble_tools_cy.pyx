# cython: language_level=3
# cython: profile=False
# cython: linetrace=False
# cython: binding=False

from builtins import range as range_it   # Python 2 compatibility

cimport cython
from cython.parallel import prange
from libcpp.vector cimport vector

import numpy as np
cimport numpy as np

import scipy.sparse

import pyiga
from . import bspline
from .quadrature import make_iterated_quadrature
from . cimport fast_assemble_cy

from concurrent.futures import ThreadPoolExecutor
import multiprocessing

import itertools

################################################################################
# Public utility functions
################################################################################

cpdef void rank_1_update(double[:,::1] X, double alpha, double[::1] u, double[::1] v):
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

################################################################################
# Internal helper functions
################################################################################

cdef struct IntInterval:
    int a
    int b

cdef IntInterval make_intv(int a, int b) nogil:
    cdef IntInterval intv
    intv.a = a
    intv.b = b
    return intv

cdef IntInterval intersect_intervals(IntInterval intva, IntInterval intvb) nogil:
    return make_intv(max(intva.a, intvb.a), min(intva.b, intvb.b))


cdef int next_lexicographic2(size_t[2] cur, size_t start[2], size_t end[2]) nogil:
    cdef size_t i
    for i in range(2):
        cur[i] += 1
        if cur[i] == end[i]:
            if i == (2-1):
                return 0
            else:
                cur[i] = start[i]
        else:
            return 1

cdef int next_lexicographic3(size_t[3] cur, size_t start[3], size_t end[3]) nogil:
    cdef size_t i
    for i in range(3):
        cur[i] += 1
        if cur[i] == end[i]:
            if i == (3-1):
                return 0
            else:
                cur[i] = start[i]
        else:
            return 1

@cython.boundscheck(False)
@cython.wraparound(False)
cdef IntInterval find_joint_support_functions(ssize_t[:,::1] meshsupp, long i) nogil:
    cdef long j, n, minj, maxj
    minj = j = i
    while j >= 0 and meshsupp[j,1] > meshsupp[i,0]:
        minj = j
        j -= 1

    maxj = i
    j = i + 1
    n = meshsupp.shape[0]
    while j < n and meshsupp[j,0] < meshsupp[i,1]:
        maxj = j
        j += 1
    return make_intv(minj, maxj+1)
    #return IntInterval(minj, maxj+1)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void outer_prod(double[::1] x1, double[::1] x2, double[:,:] out) nogil:
    cdef size_t n1 = x1.shape[0], n2 = x2.shape[0]
    cdef size_t i, j

    for i in range(n1):
        for j in range(n2):
            out[i,j] = x1[i] * x2[j]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void outer_prod3(double[::1] x1, double[::1] x2, double[::1] x3, double[:,:,:] out) nogil:
    cdef size_t n1 = x1.shape[0], n2 = x2.shape[0], n3 = x3.shape[0]
    cdef size_t i, j, k

    for i in range(n1):
        for j in range(n2):
            for k in range(n3):
                out[i,j,k] = x1[i] * x2[j] * x3[k]


#### determinants and inverses

def det_and_inv(X):
    """Return (np.linalg.det(X), np.linalg.inv(X)), but much
    faster for 2x2- and 3x3-matrices."""
    d = X.shape[-1]
    if d == 2:
        det = np.empty(X.shape[:-2])
        inv = det_and_inv_2x2(X, det)
        return det, inv
    elif d == 3:
        det = np.empty(X.shape[:-2])
        inv = det_and_inv_3x3(X, det)
        return det, inv
    else:
        return np.linalg.det(X), np.linalg.inv(X)

def determinants(X):
    """Compute the determinants of an ndarray of square matrices.

    This behaves mostly identically to np.linalg.det(), but is faster for 2x2 matrices."""
    shape = X.shape
    d = shape[-1]
    assert shape[-2] == d, "Input matrices need to be square"
    if d == 2:
        # optimization for 2x2 matrices
        assert len(shape) == 4, "Only implemented for n x m x 2 x 2 arrays"
        return X[:,:,0,0] * X[:,:,1,1] - X[:,:,0,1] * X[:,:,1,0]
    elif d == 3:
        return determinants_3x3(X)
    else:
        return np.linalg.det(X)

def inverses(X):
    if X.shape[-2:] == (2,2):
        return inverses_2x2(X)
    elif X.shape[-2:] == (3,3):
        return inverses_3x3(X)
    else:
        return np.linalg.inv(X)

#### 2D determinants and inverses

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double[:,:,:,::1] det_and_inv_2x2(double[:,:,:,::1] X, double[:,::1] det_out):
    cdef long m,n, i,j
    cdef double det, a,b,c,d
    m,n = X.shape[0], X.shape[1]

    cdef double[:,:,:,::1] Y = np.empty_like(X)
    for i in prange(m, nogil=True, schedule='static'):
        for j in range(n):
            a,b,c,d = X[i,j, 0,0], X[i,j, 0,1], X[i,j, 1,0], X[i,j, 1,1]
            det = a*d - b*c
            det_out[i,j] = det
            Y[i,j, 0,0] =  d / det
            Y[i,j, 0,1] = -b / det
            Y[i,j, 1,0] = -c / det
            Y[i,j, 1,1] =  a / det
    return Y

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double[:,:,:,::1] inverses_2x2(double[:,:,:,::1] X):
    cdef size_t m,n, i,j
    cdef double det, a,b,c,d
    m,n = X.shape[0], X.shape[1]

    cdef double[:,:,:,::1] Y = np.empty_like(X)
    for i in range(m):
        for j in range(n):
            a,b,c,d = X[i,j, 0,0], X[i,j, 0,1], X[i,j, 1,0], X[i,j, 1,1]
            det = a*d - b*c
            Y[i,j, 0,0] =  d / det
            Y[i,j, 0,1] = -b / det
            Y[i,j, 1,0] = -c / det
            Y[i,j, 1,1] =  a / det
    return Y

#### 3D determinants and inverses

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double[:,:,:,:,::1] det_and_inv_3x3(double[:,:,:,:,::1] X, double[:,:,::1] det_out):
    cdef long n0, n1, n2, i0, i1, i2
    cdef double det, invdet
    n0,n1,n2 = X.shape[0], X.shape[1], X.shape[2]
    cdef double x00,x01,x02,x10,x11,x12,x20,x21,x22

    cdef double[:,:,:,:,::1] Y = np.empty_like(X)

    for i0 in prange(n0, nogil=True, schedule='static'):
        for i1 in range(n1):
            for i2 in range(n2):
                x00,x01,x02 = X[i0, i1, i2, 0, 0], X[i0, i1, i2, 0, 1], X[i0, i1, i2, 0, 2]
                x10,x11,x12 = X[i0, i1, i2, 1, 0], X[i0, i1, i2, 1, 1], X[i0, i1, i2, 1, 2]
                x20,x21,x22 = X[i0, i1, i2, 2, 0], X[i0, i1, i2, 2, 1], X[i0, i1, i2, 2, 2]

                det = x00 * (x11 * x22 - x21 * x12) - \
                      x01 * (x10 * x22 - x12 * x20) + \
                      x02 * (x10 * x21 - x11 * x20)

                det_out[i0, i1, i2] = det

                invdet = 1.0 / det

                Y[i0, i1, i2, 0, 0] = (x11 * x22 - x21 * x12) * invdet
                Y[i0, i1, i2, 0, 1] = (x02 * x21 - x01 * x22) * invdet
                Y[i0, i1, i2, 0, 2] = (x01 * x12 - x02 * x11) * invdet
                Y[i0, i1, i2, 1, 0] = (x12 * x20 - x10 * x22) * invdet
                Y[i0, i1, i2, 1, 1] = (x00 * x22 - x02 * x20) * invdet
                Y[i0, i1, i2, 1, 2] = (x10 * x02 - x00 * x12) * invdet
                Y[i0, i1, i2, 2, 0] = (x10 * x21 - x20 * x11) * invdet
                Y[i0, i1, i2, 2, 1] = (x20 * x01 - x00 * x21) * invdet
                Y[i0, i1, i2, 2, 2] = (x00 * x11 - x10 * x01) * invdet

    return Y

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double[:,:,::1] determinants_3x3(double[:,:,:,:,::1] X):
    cdef size_t n0, n1, n2, i0, i1, i2
    n0,n1,n2 = X.shape[0], X.shape[1], X.shape[2]

    cdef double[:,:,::1] Y = np.empty((n0,n1,n2))
    cdef double[:,::1] x

    for i0 in range(n0):
        for i1 in range(n1):
            for i2 in range(n2):
                x = X[i0, i1, i2, :, :]

                Y[i0,i1,i2] = x[0, 0] * (x[1, 1] * x[2, 2] - x[2, 1] * x[1, 2]) - \
                              x[0, 1] * (x[1, 0] * x[2, 2] - x[1, 2] * x[2, 0]) + \
                              x[0, 2] * (x[1, 0] * x[2, 1] - x[1, 1] * x[2, 0])
    return Y

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double[:,:,:,:,::1] inverses_3x3(double[:,:,:,:,::1] X):
    cdef size_t n0, n1, n2, i0, i1, i2
    cdef double det, invdet
    n0,n1,n2 = X.shape[0], X.shape[1], X.shape[2]

    cdef double[:,:,:,:,::1] Y = np.empty_like(X)
    cdef double[:,::1] x, y

    for i0 in range(n0):
        for i1 in range(n1):
            for i2 in range(n2):
                x = X[i0, i1, i2, :, :]
                y = Y[i0, i1, i2, :, :]

                det = x[0, 0] * (x[1, 1] * x[2, 2] - x[2, 1] * x[1, 2]) - \
                      x[0, 1] * (x[1, 0] * x[2, 2] - x[1, 2] * x[2, 0]) + \
                      x[0, 2] * (x[1, 0] * x[2, 1] - x[1, 1] * x[2, 0])

                invdet = 1.0 / det

                y[0, 0] = (x[1, 1] * x[2, 2] - x[2, 1] * x[1, 2]) * invdet
                y[0, 1] = (x[0, 2] * x[2, 1] - x[0, 1] * x[2, 2]) * invdet
                y[0, 2] = (x[0, 1] * x[1, 2] - x[0, 2] * x[1, 1]) * invdet
                y[1, 0] = (x[1, 2] * x[2, 0] - x[1, 0] * x[2, 2]) * invdet
                y[1, 1] = (x[0, 0] * x[2, 2] - x[0, 2] * x[2, 0]) * invdet
                y[1, 2] = (x[1, 0] * x[0, 2] - x[0, 0] * x[1, 2]) * invdet
                y[2, 0] = (x[1, 0] * x[2, 1] - x[2, 0] * x[1, 1]) * invdet
                y[2, 1] = (x[2, 0] * x[0, 1] - x[0, 0] * x[2, 1]) * invdet
                y[2, 2] = (x[0, 0] * x[1, 1] - x[1, 0] * x[0, 1]) * invdet

    return Y

#### Parallelization

def chunk_tasks(tasks, num_chunks):
    """Generator that splits the list `tasks` into roughly `num_chunks` equally-sized parts."""
    n = len(tasks) // num_chunks + 1
    for i in range(0, len(tasks), n):
        yield tasks[i:i+n]

cdef object _threadpool = None

cdef object get_thread_pool():
    global _threadpool
    if _threadpool is None:
        _threadpool = ThreadPoolExecutor(pyiga.get_max_threads())
    return _threadpool

################################################################################
# Assembler kernels
################################################################################

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef double combine_mass_2d(
        double[:,::1] J,
        double* Vu0, double* Vu1,
        double* Vv0, double* Vv1
    ) nogil:
    """Compute the sum of J*u*v over a 2D grid."""
    cdef size_t i0, i1
    cdef double result = 0.0
    cdef double vu0, vu1, vv0, vv1

    cdef size_t n0 = J.shape[0]
    cdef size_t n1 = J.shape[1]

    for i0 in range(n0):
        vu0 = Vu0[i0]
        vv0 = Vv0[i0]
        for i1 in range(n1):
            vu1 = Vu1[i1]
            vv1 = Vv1[i1]
            result += vu0*vu1 * vv0*vv1 * J[i0,i1]
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef double combine_stiff_2d(
        double[:,:,:,::1] B,
        double[:,::1] J,
        double* Vu0, double* Du0,
        double* Vu1, double* Du1,
        double* Vv0, double* Dv0,
        double* Vv1, double* Dv1
    ) nogil:
    """Compute the sum of J*(B^T grad(u), B^T grad(v)) over a 2D grid."""
    cdef size_t n0 = B.shape[0]
    cdef size_t n1 = B.shape[1]
    #cdef size_t m0 = B.shape[2]    # == 2
    #cdef size_t m1 = B.shape[3]    # == 2
    cdef size_t i0, i1, k
    cdef double result = 0.0, x, y, z, b0, b1
    cdef double vu0, vu1, du0, du1
    cdef double vv0, vv1, dv0, dv1

    for i0 in range(n0):
        vu0 = Vu0[i0]
        du0 = Du0[i0]
        vv0 = Vv0[i0]
        dv0 = Dv0[i0]
        for i1 in range(n1):
            vu1 = Vu1[i1]
            du1 = Du1[i1]
            vv1 = Vv1[i1]
            dv1 = Dv1[i1]
            z = 0.0
            for k in range(2):
                b0 = B[i0,i1,0,k]
                b1 = B[i0,i1,1,k]

                x = b0 * vu0*du1 + b1 * du0*vu1
                y = b0 * vv0*dv1 + b1 * dv0*vv1
                z += x*y
            result += J[i0,i1] * z
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef double combine_mass_3d(
        double[:,:,::1] J,
        double* Vu0, double* Vu1, double* Vu2,
        double* Vv0, double* Vv1, double* Vv2
    ) nogil:
    """Compute the sum of J*u*v over a 2D grid."""
    cdef size_t i0, i1, i2
    cdef double result = 0.0
    cdef double vu0, vu1, vu2, vv0, vv1, vv2

    cdef size_t n0 = J.shape[0]
    cdef size_t n1 = J.shape[1]
    cdef size_t n2 = J.shape[2]

    for i0 in range(n0):
        vu0 = Vu0[i0]
        vv0 = Vv0[i0]

        for i1 in range(n1):
            vu1 = Vu1[i1]
            vv1 = Vv1[i1]

            for i2 in range(n2):
                vu2 = Vu2[i2]
                vv2 = Vv2[i2]

                result += vu0*vu1*vu2 * vv0*vv1*vv2 * J[i0,i1,i2]

    return result

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef double combine_stiff_3d(double[:,:,:,:,::1] B,
                             double[:,:,::1] J,
                             double[:,:,:,::1] u,
                             double[:,:,:,::1] v) nogil:
    """Compute the sum of J*(B^T u, B^T v) over a 3D grid"""
    cdef size_t n0 = B.shape[0]
    cdef size_t n1 = B.shape[1]
    cdef size_t n2 = B.shape[2]
    #cdef size_t m0 = B.shape[3]    # == 3
    #cdef size_t m1 = B.shape[4]    # == 3
    cdef size_t i0,i1,i2,k,l
    cdef double result = 0.0, x, y, z, b
    cdef double * u_I
    cdef double * v_I
    cdef double * B_I

    for i0 in range(n0):
        for i1 in range(n1):
            for i2 in range(n2):
                z = 0.0

                u_I = &u[i0,i1,i2, 0]
                v_I = &v[i0,i1,i2, 0]
                B_I = &B[i0,i1,i2, 0,0] # assume B is contiguous in last two axes

                for k in range(3):
                    x = y = 0.0
                    for l in range(3):
                        #b = B[i0,i1,i2, l,k]
                        b = B_I[3*l + k]
                        x += b * u_I[l]
                        y += b * v_I[l]
                    z += x*y
                result += J[i0,i1,i2] * z
    return result



################################################################################
# 2D Assemblers
################################################################################

cdef class BaseAssembler2D:
    cdef int nqp
    cdef size_t[2] ndofs
    cdef vector[ssize_t[:,::1]] meshsupp
    cdef list _asm_pool     # list of shared clones for multithreading

    cdef void base_init(self, kvs):
        assert len(kvs) == 2, "Assembler requires two knot vectors"
        self.nqp = max([kv.p for kv in kvs]) + 1
        self.ndofs[:] = [kv.numdofs for kv in kvs]
        self.meshsupp = [kvs[k].mesh_support_idx_all() for k in range(2)]
        self._asm_pool = []

    cdef _share_base(self, BaseAssembler2D asm):
        asm.nqp = self.nqp
        asm.ndofs[:] = self.ndofs[:]
        asm.meshsupp = self.meshsupp

    cdef BaseAssembler2D shared_clone(self):
        return None     # not implemented

    cdef inline size_t to_seq(self, size_t[2] ii) nogil:
        # by convention, the order of indices is (y,x)
        return ii[0] * self.ndofs[1] + ii[1]

    @cython.cdivision(True)
    cdef inline void from_seq(self, size_t i, size_t[2] out) nogil:
        out[0] = i / self.ndofs[1]
        out[1] = i % self.ndofs[1]

    cdef double assemble_impl(self, size_t[2] i, size_t[2] j) nogil:
        return -9999.99  # Not implemented

    cpdef double assemble(self, size_t i, size_t j):
        cdef size_t[2] I, J
        with nogil:
            self.from_seq(i, I)
            self.from_seq(j, J)
            return self.assemble_impl(I, J)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void multi_assemble_chunk(self, size_t[:,::1] idx_arr, double[::1] out) nogil:
        cdef size_t[2] I, J
        cdef size_t k

        for k in range(idx_arr.shape[0]):
            self.from_seq(idx_arr[k,0], I)
            self.from_seq(idx_arr[k,1], J)
            out[k] = self.assemble_impl(I, J)

    def multi_assemble(self, indices):
        cdef size_t[:,::1] idx_arr = np.array(list(indices), dtype=np.uintp)
        cdef double[::1] result = np.empty(idx_arr.shape[0])

        num_threads = pyiga.get_max_threads()
        if num_threads <= 1:
            self.multi_assemble_chunk(idx_arr, result)
        else:
            thread_pool = get_thread_pool()
            if not self._asm_pool:
                self._asm_pool = [self] + [self.shared_clone()
                        for i in range(1, thread_pool._max_workers)]

            results = thread_pool.map(_asm_chunk_2d,
                        self._asm_pool,
                        chunk_tasks(idx_arr, num_threads),
                        chunk_tasks(result, num_threads))
            list(results)   # wait for threads to finish
        return result

cpdef void _asm_chunk_2d(BaseAssembler2D asm, size_t[:,::1] idxchunk, double[::1] out):
    with nogil:
        asm.multi_assemble_chunk(idxchunk, out)


cdef class MassAssembler2D(BaseAssembler2D):
    # shared data
    cdef vector[double[::1,:]] C
    cdef double[:,::1] geo_weights

    def __init__(self, kvs, geo):
        assert geo.dim == 2, "Geometry has wrong dimension"
        self.base_init(kvs)

        gauss = [make_iterated_quadrature(np.unique(kv.kv), self.nqp) for kv in kvs]
        gaussgrid = [g[0] for g in gauss]
        gaussweights = [g[1] for g in gauss]
        self.C  = [bspline.collocation(kvs[k], gaussgrid[k])
                   .toarray(order='F') for k in range(2)]

        geo_jac    = geo.grid_jacobian(gaussgrid)
        geo_det    = np.abs(determinants(geo_jac))
        self.geo_weights = gaussweights[0][:,None] * gaussweights[1][None,:] * geo_det

    cdef MassAssembler2D shared_clone(self):
        return self     # no shared data; class is thread-safe

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double assemble_impl(self, size_t[2] i, size_t[2] j) nogil:
        cdef int k
        cdef IntInterval intv
        cdef size_t g_sta[2]
        cdef size_t g_end[2]

        cdef (double*) values_i[2]
        cdef (double*) values_j[2]

        for k in range(2):
            intv = intersect_intervals(make_intv(self.meshsupp[k][i[k],0], self.meshsupp[k][i[k],1]),
                                       make_intv(self.meshsupp[k][j[k],0], self.meshsupp[k][j[k],1]))
            if intv.a >= intv.b:
                return 0.0      # no intersection of support
            g_sta[k] = self.nqp * intv.a    # start of Gauss nodes
            g_end[k] = self.nqp * intv.b    # end of Gauss nodes

            values_i[k] = &self.C[k][ g_sta[k], i[k] ]
            values_j[k] = &self.C[k][ g_sta[k], j[k] ]

        return combine_mass_2d(
            self.geo_weights[ g_sta[0]:g_end[0], g_sta[1]:g_end[1] ],
            values_i[0], values_i[1],
            values_j[0], values_j[1]
        )


cdef class StiffnessAssembler2D(BaseAssembler2D):
    # shared data
    cdef vector[double[::1,:]] C, Cd
    cdef double[:,::1] geo_weights
    cdef double[:,:,:,::1] geo_jacinv

    def __init__(self, kvs, geo):
        assert geo.dim == 2, "Geometry has wrong dimension"
        self.base_init(kvs)

        gauss = [make_iterated_quadrature(np.unique(kv.kv), self.nqp) for kv in kvs]
        gaussgrid = [g[0] for g in gauss]
        gaussweights = [g[1] for g in gauss]
        colloc = [bspline.collocation_derivs(kvs[k], gaussgrid[k], derivs=1) for k in range(2)]
        self.C  = [X.toarray(order='F') for (X,Y) in colloc]
        self.Cd = [Y.toarray(order='F') for (X,Y) in colloc]

        geo_jac    = geo.grid_jacobian(gaussgrid)
        geo_det, self.geo_jacinv = det_and_inv(geo_jac)

        self.geo_weights = gaussweights[0][:,None] * gaussweights[1][None,:] * np.abs(geo_det)

    cdef StiffnessAssembler2D shared_clone(self):
        return self     # no shared data; class is thread-safe

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double assemble_impl(self, size_t[2] i, size_t[2] j) nogil:
        cdef int k
        cdef IntInterval intv
        cdef size_t g_sta[2]
        cdef size_t g_end[2]
        cdef (double*) values_i[2]
        cdef (double*) values_j[2]
        cdef (double*) derivs_i[2]
        cdef (double*) derivs_j[2]

        for k in range(2):
            intv = intersect_intervals(make_intv(self.meshsupp[k][i[k],0], self.meshsupp[k][i[k],1]),
                                       make_intv(self.meshsupp[k][j[k],0], self.meshsupp[k][j[k],1]))
            if intv.a >= intv.b:
                return 0.0      # no intersection of support
            g_sta[k] = self.nqp * intv.a    # start of Gauss nodes
            g_end[k] = self.nqp * intv.b    # end of Gauss nodes

            values_i[k] = &self.C[k][ g_sta[k], i[k] ]
            values_j[k] = &self.C[k][ g_sta[k], j[k] ]

            derivs_i[k] = &self.Cd[k][ g_sta[k], i[k] ]
            derivs_j[k] = &self.Cd[k][ g_sta[k], j[k] ]

        return combine_stiff_2d(
                self.geo_jacinv [ g_sta[0]:g_end[0], g_sta[1]:g_end[1] ],
                self.geo_weights[ g_sta[0]:g_end[0], g_sta[1]:g_end[1] ],
                values_i[0], derivs_i[0], values_i[1], derivs_i[1],
                values_j[0], derivs_j[0], values_j[1], derivs_j[1])



################################################################################
# 3D Assemblers
################################################################################

cdef class BaseAssembler3D:
    cdef int nqp
    cdef size_t[3] ndofs
    cdef vector[ssize_t[:,::1]] meshsupp
    cdef list _asm_pool     # list of shared clones for multithreading

    cdef base_init(self, kvs):
        assert len(kvs) == 3, "Assembler requires three knot vectors"
        self.nqp = max([kv.p for kv in kvs]) + 1
        self.ndofs[:] = [kv.numdofs for kv in kvs]
        self.meshsupp = [kvs[k].mesh_support_idx_all() for k in range(3)]
        self._asm_pool = []

    cdef _share_base(self, BaseAssembler3D asm):
        asm.nqp = self.nqp
        asm.ndofs[:] = self.ndofs[:]
        asm.meshsupp = self.meshsupp

    cdef BaseAssembler3D shared_clone(self):
        return None     # not implemented

    cdef inline size_t to_seq(self, size_t[3] ii) nogil:
        # by convention, the order of indices is (z,y,x)
        return (ii[0] * self.ndofs[1] + ii[1]) * self.ndofs[2] + ii[2]

    @cython.cdivision(True)
    cdef inline void from_seq(self, size_t i, size_t[3] out) nogil:
        out[2] = i % self.ndofs[2]
        i /= self.ndofs[2]
        out[1] = i % self.ndofs[1]
        i /= self.ndofs[1]
        out[0] = i

    cdef double assemble_impl(self, size_t[3] i, size_t[3] j) nogil:
        return -9999.99  # Not implemented

    cpdef double assemble(self, size_t i, size_t j):
        cdef size_t[3] I, J
        with nogil:
            self.from_seq(i, I)
            self.from_seq(j, J)
            return self.assemble_impl(I, J)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void multi_assemble_chunk(self, size_t[:,::1] idx_arr, double[::1] out) nogil:
        cdef size_t[3] I, J
        cdef size_t k

        for k in range(idx_arr.shape[0]):
            self.from_seq(idx_arr[k,0], I)
            self.from_seq(idx_arr[k,1], J)
            out[k] = self.assemble_impl(I, J)

    def multi_assemble(self, indices):
        cdef size_t[:,::1] idx_arr = np.array(list(indices), dtype=np.uintp)
        cdef double[::1] result = np.empty(idx_arr.shape[0])

        num_threads = pyiga.get_max_threads()
        if num_threads <= 1:
            self.multi_assemble_chunk(idx_arr, result)
        else:
            thread_pool = get_thread_pool()
            if not self._asm_pool:
                self._asm_pool = [self] + [self.shared_clone()
                        for i in range(1, thread_pool._max_workers)]

            results = thread_pool.map(_asm_chunk_3d,
                        self._asm_pool,
                        chunk_tasks(idx_arr, num_threads),
                        chunk_tasks(result, num_threads))
            list(results)   # wait for threads to finish
        return result

cpdef void _asm_chunk_3d(BaseAssembler3D asm, size_t[:,::1] idxchunk, double[::1] out):
    with nogil:
        asm.multi_assemble_chunk(idxchunk, out)


cdef class MassAssembler3D(BaseAssembler3D):
    # shared data
    cdef vector[double[::1,:]] C
    cdef double[:,:,::1] geo_weights

    def __init__(self, kvs, geo):
        assert geo.dim == 3, "Geometry has wrong dimension"
        self.base_init(kvs)

        gauss = [make_iterated_quadrature(np.unique(kv.kv), self.nqp) for kv in kvs]
        gaussgrid = [g[0] for g in gauss]
        gaussweights = [g[1] for g in gauss]
        self.C  = [bspline.collocation(kvs[k], gaussgrid[k])
                   .toarray(order='F') for k in range(3)]

        geo_jac    = geo.grid_jacobian(gaussgrid)
        geo_det    = np.abs(determinants(geo_jac))
        self.geo_weights = gaussweights[0][:,None,None] * gaussweights[1][None,:,None] * gaussweights[2][None,None,:] * geo_det

    cdef MassAssembler3D shared_clone(self):
        return self     # no shared data; class is thread-safe

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double assemble_impl(self, size_t[3] i, size_t[3] j) nogil:
        cdef int k
        cdef IntInterval intv
        cdef size_t g_sta[3]
        cdef size_t g_end[3]
        cdef (double*) values_i[3]
        cdef (double*) values_j[3]

        for k in range(3):
            intv = intersect_intervals(make_intv(self.meshsupp[k][i[k],0], self.meshsupp[k][i[k],1]),
                                       make_intv(self.meshsupp[k][j[k],0], self.meshsupp[k][j[k],1]))
            if intv.a >= intv.b:
                return 0.0      # no intersection of support
            g_sta[k] = self.nqp * intv.a    # start of Gauss nodes
            g_end[k] = self.nqp * intv.b    # end of Gauss nodes

            values_i[k] = &self.C[k][ g_sta[k], i[k] ]
            values_j[k] = &self.C[k][ g_sta[k], j[k] ]

        return combine_mass_3d(
            self.geo_weights[ g_sta[0]:g_end[0], g_sta[1]:g_end[1], g_sta[2]:g_end[2] ],
            values_i[0], values_i[1], values_i[2],
            values_j[0], values_j[1], values_j[2]
        )



cdef class StiffnessAssembler3D(BaseAssembler3D):
    # shared data
    cdef vector[double[::1,:]] C, Cd
    cdef double[:,:,::1] geo_weights
    cdef double[:,:,:,:,::1] geo_jacinv
    # local data
    cdef double[:,:,:,:,::1] grad_buffer
    cdef vector[double[::1]] values_i, values_j, derivs_i, derivs_j

    def __init__(self, kvs, geo):
        assert geo.dim == 3, "Geometry has wrong dimension"
        self.base_init(kvs)

        gauss = [make_iterated_quadrature(np.unique(kv.kv), self.nqp) for kv in kvs]
        gaussgrid = [g[0] for g in gauss]
        gaussweights = [g[1] for g in gauss]
        colloc = [bspline.collocation_derivs(kvs[k], gaussgrid[k], derivs=1) for k in range(3)]
        self.C  = [X.toarray(order='F') for (X,Y) in colloc]
        self.Cd = [Y.toarray(order='F') for (X,Y) in colloc]

        geo_jac    = geo.grid_jacobian(gaussgrid)
        geo_det, self.geo_jacinv = det_and_inv(geo_jac)
        self.geo_weights = gaussweights[0][:,None,None] * gaussweights[1][None,:,None] * gaussweights[2][None,None,:] * np.abs(geo_det)

        # initialize local storage
        self.values_i.resize(3)
        self.values_j.resize(3)
        self.derivs_i.resize(3)
        self.derivs_j.resize(3)
        # indices: (basis function i or j), coord0, coord1, coord2, (x,y,z)
        self.grad_buffer = np.empty((2,
            self.nqp * (2*kvs[0].p + 1),
            self.nqp * (2*kvs[1].p + 1),
            self.nqp * (2*kvs[2].p + 1),
            3), np.double)

    cdef StiffnessAssembler3D shared_clone(self):
        cdef StiffnessAssembler3D asm = StiffnessAssembler3D.__new__(StiffnessAssembler3D)

        # copy references to shared data
        self._share_base(asm)
        asm.C = self.C
        asm.Cd = self.Cd
        asm.geo_weights = self.geo_weights
        asm.geo_jacinv = self.geo_jacinv

        # initialize local data
        asm.values_i.resize(3)
        asm.values_j.resize(3)
        asm.derivs_i.resize(3)
        asm.derivs_j.resize(3)
        asm.grad_buffer = np.empty_like(self.grad_buffer)

        return asm

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef double assemble_impl(self, size_t[3] i, size_t[3] j) nogil:
        cdef int k
        cdef IntInterval intv
        cdef size_t m[3]
        cdef size_t g_sta[3]
        cdef size_t g_end[3]
        cdef double[:,:,:,::1] grad_i, grad_j

        for k in range(3):
            intv = intersect_intervals(make_intv(self.meshsupp[k][i[k],0], self.meshsupp[k][i[k],1]),
                                       make_intv(self.meshsupp[k][j[k],0], self.meshsupp[k][j[k],1]))
            if intv.a >= intv.b:
                return 0.0      # no intersection of support
            g_sta[k] = self.nqp * intv.a    # start of Gauss nodes
            g_end[k] = self.nqp * intv.b    # end of Gauss nodes
            m[k] = self.nqp * (intv.b - intv.a)  # length

            self.values_i[k] = self.C[k][ g_sta[k]:g_end[k], i[k] ]
            self.values_j[k] = self.C[k][ g_sta[k]:g_end[k], j[k] ]

            self.derivs_i[k] = self.Cd[k][ g_sta[k]:g_end[k], i[k] ]
            self.derivs_j[k] = self.Cd[k][ g_sta[k]:g_end[k], j[k] ]

        grad_i = self.grad_buffer[0, :m[0], :m[1], :m[2], :]
        grad_j = self.grad_buffer[1, :m[0], :m[1], :m[2], :]

        outer_prod3(self.values_i[0], self.values_i[1], self.derivs_i[2], out=grad_i[:,:,:,0])
        outer_prod3(self.values_i[0], self.derivs_i[1], self.values_i[2], out=grad_i[:,:,:,1])
        outer_prod3(self.derivs_i[0], self.values_i[1], self.values_i[2], out=grad_i[:,:,:,2])

        outer_prod3(self.values_j[0], self.values_j[1], self.derivs_j[2], out=grad_j[:,:,:,0])
        outer_prod3(self.values_j[0], self.derivs_j[1], self.values_j[2], out=grad_j[:,:,:,1])
        outer_prod3(self.derivs_j[0], self.values_j[1], self.values_j[2], out=grad_j[:,:,:,2])

        return combine_stiff_3d(
                self.geo_jacinv [ g_sta[0]:g_end[0], g_sta[1]:g_end[1], g_sta[2]:g_end[2] ],
                self.geo_weights[ g_sta[0]:g_end[0], g_sta[1]:g_end[1], g_sta[2]:g_end[2] ],
                grad_i, grad_j)



################################################################################
# Driver routines for 2D assemblers
################################################################################

@cython.boundscheck(False)
@cython.wraparound(False)
cdef object generic_assemble_2d(BaseAssembler2D asm, long chunk_start=-1, long chunk_end=-1):
    cdef size_t[2] i, j
    cdef size_t k, ii, jj
    cdef IntInterval intv

    cdef size_t[2] dof_start, dof_end, neigh_j_start, neigh_j_end
    cdef double entry
    cdef vector[double] entries
    cdef vector[size_t] entries_i, entries_j

    dof_start[:] = (0,0)
    dof_end[:] = asm.ndofs[:]

    if chunk_start >= 0:
        dof_start[0] = chunk_start
    if chunk_end >= 0:
        dof_end[0] = chunk_end

    i[:] = dof_start[:]
    with nogil:
        while True:         # loop over all i
            ii = asm.to_seq(i)

            for k in range(2):
                intv = find_joint_support_functions(asm.meshsupp[k], i[k])
                neigh_j_start[k] = intv.a
                neigh_j_end[k] = intv.b
            j[0] = neigh_j_start[0]
            j[1] = neigh_j_start[1]

            while True:     # loop j over all neighbors of i
                jj = asm.to_seq(j)
                if jj >= ii:
                    entry = asm.assemble_impl(i, j)

                    entries.push_back(entry)
                    entries_i.push_back(ii)
                    entries_j.push_back(jj)

                    if ii != jj:
                        entries.push_back(entry)
                        entries_i.push_back(jj)
                        entries_j.push_back(ii)

                if not next_lexicographic2(j, neigh_j_start, neigh_j_end):
                    break
            if not next_lexicographic2(i, dof_start, dof_end):
                break

    cdef size_t ne = entries.size()
    cdef size_t N = asm.ndofs[0] * asm.ndofs[1]
    return scipy.sparse.coo_matrix(
            (<double[:ne]> entries.data(),
                (<size_t[:ne]> entries_i.data(),
                 <size_t[:ne]> entries_j.data())),
            shape=(N,N)).tocsr()


cdef generic_assemble_2d_parallel(BaseAssembler2D asm):
    num_threads = pyiga.get_max_threads()
    if num_threads <= 1:
        return generic_assemble_2d(asm)
    def asm_chunk(rg):
        cdef BaseAssembler2D asm_clone = asm.shared_clone()
        return generic_assemble_2d(asm_clone, rg.start, rg.stop)
    results = get_thread_pool().map(asm_chunk, chunk_tasks(range_it(asm.ndofs[0]), 4*num_threads))
    return sum(results)


def mass_2d(kvs, geo):
    return generic_assemble_2d_parallel(MassAssembler2D(kvs, geo))

def stiffness_2d(kvs, geo):
    return generic_assemble_2d_parallel(StiffnessAssembler2D(kvs, geo))



################################################################################
# Driver routines for 3D assemblers
################################################################################

@cython.boundscheck(False)
@cython.wraparound(False)
cdef object generic_assemble_3d(BaseAssembler3D asm, long chunk_start=-1, long chunk_end=-1):
    cdef size_t[3] i, j
    cdef size_t k, ii, jj
    cdef IntInterval intv

    cdef size_t[3] dof_start, dof_end, neigh_j_start, neigh_j_end
    cdef double entry
    cdef vector[double] entries
    cdef vector[size_t] entries_i, entries_j

    dof_start[:] = (0,0,0)
    dof_end[:] = asm.ndofs[:]

    if chunk_start >= 0:
        dof_start[0] = chunk_start
    if chunk_end >= 0:
        dof_end[0] = chunk_end

    i[:] = dof_start[:]
    with nogil:
        while True:         # loop over all i
            ii = asm.to_seq(i)

            for k in range(3):
                intv = find_joint_support_functions(asm.meshsupp[k], i[k])
                neigh_j_start[k] = intv.a
                neigh_j_end[k] = intv.b
            j[0] = neigh_j_start[0]
            j[1] = neigh_j_start[1]
            j[2] = neigh_j_start[2]

            while True:     # loop j over all neighbors of i
                jj = asm.to_seq(j)
                if jj >= ii:
                    entry = asm.assemble_impl(i, j)

                    entries.push_back(entry)
                    entries_i.push_back(ii)
                    entries_j.push_back(jj)

                    if ii != jj:
                        entries.push_back(entry)
                        entries_i.push_back(jj)
                        entries_j.push_back(ii)

                if not next_lexicographic3(j, neigh_j_start, neigh_j_end):
                    break
            if not next_lexicographic3(i, dof_start, dof_end):
                break

    cdef size_t ne = entries.size()
    cdef size_t N = asm.ndofs[0] * asm.ndofs[1] * asm.ndofs[2]
    return scipy.sparse.coo_matrix(
            (<double[:ne]> entries.data(),
                (<size_t[:ne]> entries_i.data(),
                 <size_t[:ne]> entries_j.data())),
            shape=(N,N)).tocsr()


cdef generic_assemble_3d_parallel(BaseAssembler3D asm):
    num_threads = pyiga.get_max_threads()
    if num_threads <= 1:
        return generic_assemble_3d(asm)
    def asm_chunk(rg):
        cdef BaseAssembler3D asm_clone = asm.shared_clone()
        return generic_assemble_3d(asm_clone, rg.start, rg.stop)
    results = get_thread_pool().map(asm_chunk, chunk_tasks(range_it(asm.ndofs[0]), 4*num_threads))
    return sum(results)


def mass_3d(kvs, geo):
    return generic_assemble_3d_parallel(MassAssembler3D(kvs, geo))

def stiffness_3d(kvs, geo):
    return generic_assemble_3d_parallel(StiffnessAssembler3D(kvs, geo))



################################################################################
# Bindings for the C++ low-rank assembler (fastasm.cc)
################################################################################

cdef double _entry_func_2d(size_t i, size_t j, void * data):
    return (<BaseAssembler2D>data).assemble(i, j)

cdef double _entry_func_3d(size_t i, size_t j, void * data):
    return (<BaseAssembler3D>data).assemble(i, j)


def fast_mass_2d(kvs, geo, tol=1e-10, maxiter=100, skipcount=3, tolcount=3, verbose=2):
    cdef MassAssembler2D asm = MassAssembler2D(kvs, geo)
    return fast_assemble_cy.fast_assemble_2d_wrapper(_entry_func_2d, <void*>asm, kvs,
            tol, maxiter, skipcount, tolcount, verbose)

def fast_stiffness_2d(kvs, geo, tol=1e-10, maxiter=100, skipcount=3, tolcount=3, verbose=2):
    cdef StiffnessAssembler2D asm = StiffnessAssembler2D(kvs, geo)
    return fast_assemble_cy.fast_assemble_2d_wrapper(_entry_func_2d, <void*>asm, kvs,
            tol, maxiter, skipcount, tolcount, verbose)


def fast_mass_3d(kvs, geo, tol=1e-10, maxiter=100, skipcount=3, tolcount=3, verbose=2):
    cdef MassAssembler3D asm = MassAssembler3D(kvs, geo)
    return fast_assemble_cy.fast_assemble_3d_wrapper(_entry_func_3d, <void*>asm, kvs,
            tol, maxiter, skipcount, tolcount, verbose)

def fast_stiffness_3d(kvs, geo, tol=1e-10, maxiter=100, skipcount=3, tolcount=3, verbose=2):
    cdef StiffnessAssembler3D asm = StiffnessAssembler3D(kvs, geo)
    return fast_assemble_cy.fast_assemble_3d_wrapper(_entry_func_3d, <void*>asm, kvs,
            tol, maxiter, skipcount, tolcount, verbose)

