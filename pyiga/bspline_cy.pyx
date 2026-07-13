# cython: profile=False
# cython: linetrace=False
# cython: binding=False
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# distutils: language = c++
# cython: language_level=3, boundscheck=False, cdivision=True, wraparound=False, initializedcheck=False, nonecheck=False

cimport cython

import numpy as np
cimport numpy as np

from libc.stdlib cimport malloc, free
from libc.math cimport fabs, fmin, fmax
#from libcpp.string cimport string
#from libc.stdio cimport snprintf

###Helper functions
# cdef inline int count_multiplicity(double[:] kv, int idx, int n, double tol):
#     cdef int m = 1
#     while (idx + m) < n and fabs(kv[idx + m] - kv[idx]) < tol:
#         m += 1
#     return m

# cdef inline double clamp(double x, double a, double b):
#     if x < a:
#         return a
#     elif x > b:
#         return b
#     else:
#         return x

# cdef class KnotVector:
#     cdef int _p
#     cdef double[:] _knots  # typed memoryview
#     cdef int size
#     cdef double[:] _mesh
#     cdef int[:] _m
#     cdef int _meshsize
#     cdef int _numdofs

#     def __cinit__(self, int p, np.ndarray[np.float64_t, ndim=1] knots):
#         self._p = p
#         if not knots.flags['C_CONTIGUOUS']:
#             raise ValueError("knots array must be contiguous")
#         self._knots = knots
#         self.size = len(knots)
#         self._numdofs = self.size - self._p-1

#         if not self.sanity_check(): raise AssertionError("not a p-open knot vector")

#         cdef int i = 0, count = 0
#         cdef double[:] _knots = knots
#         while i < self.size:
#             count += 1
#             i += count_multiplicity(_knots, i, self.size, 1e-12)

#         self._meshsize = count
#         cdef np.ndarray[np.float64_t, ndim=1] mesh = np.empty(count, dtype=np.float64)
#         cdef np.ndarray[np.int32_t, ndim=1] m = np.empty(count, dtype=np.int32)
#         self._mesh = mesh
#         self._m = m
#         cdef double[:] _mesh = self._mesh
#         cdef int[:] _m = self._m

#         cdef int k = 0
#         i = 0
#         while i < self.size:
#             _mesh[k] = _knots[i]
#             _m[k] = count_multiplicity(_knots, i, self.size, 1e-12)
#             i += _m[k]
#             k += 1
            
#     def __richcmp__(self, other, int op):
#         if not isinstance(other, KnotVector):
#             return NotImplemented
    
#         if op == 0:  # <
#             return self.leq(other) and not self.eq(other)
#         elif op == 1:  # <=
#             return self.leq(other)
#         elif op == 2:  # ==
#             return self.eq(other)
#         elif op == 3:  # !=
#             return not self.eq(other)
#         elif op == 4:  # >
#             return not self.leq(other)
#         elif op == 5:  # >=
#             return self.eq(other) or not self.leq(other)
#         else:
#             return NotImplemented

#     cdef bint eq(self, KnotVector other):
#         cdef double[:] mesh1 = self._mesh
#         cdef double[:] mesh2 = other._mesh
#         cdef int[:] m1 = self._m
#         cdef int[:] m2 = other._m
#         cdef int i, n = self._meshsize
#         cdef double tol = 1e-12
    
#         if self._p != other._p or n!=other._meshsize:
#             return False
#         for i in range(n):
#             if fabs(mesh1[i] - mesh2[i]) > tol or m1[i] != m2[i]:
#                 return False
#         return True

#     cdef bint leq(self, KnotVector other): ###TODO: since mesh is now anyway precomputed, use mesh and not knots
#         cdef double[:] knots1 = self._knots
#         cdef double[:] knots2 = other._knots
#         cdef int[:] m1 = self._m
#         cdef int[:] m2 = other._m

#         cdef double a1 = knots1[0], b1 = knots1[self.size-1], a2 = knots2[0], b2 = knots2[other.size-1] 
        
#         cdef int i1=0, i2=0
#         cdef int delta_p=other._p - self._p
#         cdef int n1=self.size, n2=other.size
#         cdef double tol=1e-12
#         if delta_p<0: return False
#         if a1 > a2 + tol or b2 > b1 + tol: return False
    
#         while knots1[i1] < a2 - tol: 
#             i1 += 1
    
#         while i1 < n1 and i2 < n2:  
#             while i2 < n2 and knots2[i2] < knots1[i1] - tol:
#                 i2 += 1
#             if i2 == n2:
#                 break
    
#             while i1 < n1 and knots1[i1] < knots2[i2] - tol:
#                 i1 += 1
#             if i1 == n1:
#                 break
    
#             if m2[i2] < m1[i1] + delta_p:
#                 return False
    
#             i1 += m1[i1]
#             i2 += m2[i2]
#         return True

#     def __str__(self):
#         cdef double[:] knots = self._knots
#         cdef int n = self.size
#         cdef int bufsize = 100 + 14 * min(n, 50)  # estimate smaller buffer since truncation
#         cdef char* buffer = <char*>malloc(bufsize)
#         cdef int i, pos = 0, written
    
#         if not buffer:
#             raise MemoryError("Could not allocate buffer")
    
#         written = snprintf(buffer, bufsize, b"Knot vector (degree %d): [", self._p)
#         pos += written
    
#         if n <= 50:
#             # print all knots
#             for i in range(n):
#                 written = snprintf(buffer + pos, bufsize - pos, b"%.3g", knots[i])
#                 pos += written
#                 if i < n - 1:
#                     buffer[pos] = ord(',')
#                     buffer[pos + 1] = ord(' ')
#                     pos += 2
#         else:
#             # print first 3 knots
#             for i in range(3):
#                 written = snprintf(buffer + pos, bufsize - pos, b"%.3g", knots[i])
#                 pos += written
#                 buffer[pos] = ord(',')
#                 buffer[pos + 1] = ord(' ')
#                 pos += 2
    
#             # print ellipsis "..."
#             written = snprintf(buffer + pos, bufsize - pos, b"...")
#             pos += written
#             buffer[pos] = ord(',')
#             buffer[pos + 1] = ord(' ')
#             pos += 2
    
#             # print last 3 knots
#             for i in range(n - 3, n):
#                 written = snprintf(buffer + pos, bufsize - pos, b"%.3g", knots[i])
#                 pos += written
#                 if i < n - 1:
#                     buffer[pos] = ord(',')
#                     buffer[pos + 1] = ord(' ')
#                     pos += 2
    
#         buffer[pos] = ord(']')
#         pos += 1
#         buffer[pos] = 0
    
#         cdef bytes s_bytes = (<char *>buffer)[:pos]
#         free(buffer)
#         return s_bytes.decode('ascii')

#     @property
#     def p(self):
#         return self._p

#     @property
#     def kv(self):
#         return self._knots.base

#     @property
#     def numknots(self):
#         return self.size

#     @property
#     def numdofs(self):
#         """Number of basis functions in a B-spline basis defined over this knot vector"""
#         return self._numdofs

#     @property
#     def numspans(self):
#         """Number of nontrivial intervals in the knot vector"""
#         return self._meshsize - 1

#     @property
#     def mesh(self):
#         return self._mesh.base

#     cpdef (double, double) support(self, int j=-1):
#         if j<0: 
#             return (self._knots[0], self._knots[self.size-1])
#         elif j >= self.size - self._p -1:
#             raise IndexError(f"Basis function index j={j} out of range (max={self.size - self._p - 2})")
#         else:
#             return (self._knots[j], self._knots[j+self._p+1])

#     cpdef (int, int) support_idx(self, int j):
#         """Knot indices of support of j-th B-spline"""
#         return (j, j+self._p+1)

#     # cpdef (int, int) mesh_support_idx(self, j):
#     #     """Return the first and last mesh index of the support of i"""
#     #     return (self._knots_to_mesh[j], self._knots_to_mesh[j+self._p+1])

#     # cpdef (int, int) mesh_support_idx_all(self):
#     #     """Compute an integer array of size `N × 2`, where N = self.numdofs, which
#     #     contains for each B-spline the result of :func:`mesh_support_idx`.
#     #     """
#     #     n = self.numdofs
#     #     startend = np.stack((np.arange(0,n), np.arange(self.p+1, n+self.p+1)), axis=1)
#     #     return self._knots_to_mesh[startend]

#     # cpdef mesh_span_indices(self):
#     #     """Return an array of indices i such that kv[i] != kv[i+1], i.e., the indices
#     #     of the nonempty spans. Return value has length self.numspans.
#     #     """
#     #     self._ensure_mesh()
#     #     k2m = self._knots_to_mesh
#     #     return np.where(k2m[1:] != k2m[:-1])[0]

#     cpdef int findspan(self, double u):
#         """Returns an index i such that
#          kv[i] <= u < kv[i+1]     (except for the boundary, where u <= kv[m-p] is allowed)
#          and p <= i < len(kv) - 1 - p"""

#         cdef int n = self.size
    
#         if u >= self._knots[n - self._p - 1]:
#             return n - self._p - 2  # last interval
    
#         cdef int a = 0, b = n - 1, c
    
#         while b - a > 1:
#             c = a + (b - a) / 2
#             if self._knots[c] > u:
#                 b = c
#             else:
#                 a = c
#         return a

#     cpdef np.ndarray[double, ndim=1] greville(self):
#         cdef double[:] knots = self._knots
#         cdef int n = self.size - self._p - 1
#         cdef np.ndarray[double, ndim=1] grev_pts = np.empty(n, dtype=np.float64)
#         cdef int i, j
#         cdef double s
#         cdef double minv = knots[0], maxv = knots[self.size-1]

#         if self._p == 0:
#             for i in range(n):
#                 grev_pts[i] = (knots[i] + knots[i+1])/2
#             return grev_pts

#         for i in range(n):
#             s = 0.0
#             for j in range(1, self._p + 1):  # sum from kv[i+1] to kv[i+p]
#                 s += knots[i + j]
#             grev_pts[i] = clamp(s / self._p, minv, maxv)
#         return grev_pts

#     cdef bint sanity_check(self):
#         cdef double[:] knots = self._knots
#         cdef int i, n = self.size
#         cdef double first_val, last_val
    
#         # Quick check on knot vector length
#         if n < 2 * self._p + 2:
#             return False
    
#         first_val = knots[0]
#         last_val = knots[n - 1]
    
#         # Check first p+1 knots equal
#         for i in range(1, self._p + 1):
#             if knots[i] != first_val:
#                 return False
    
#         # Check last p+1 knots equal
#         for i in range(n - self._p - 1, n):
#             if knots[i] != last_val:
#                 return False
    
#         # Check nondecreasing sequence
#         for i in range(n - 1):
#             if knots[i] > knots[i + 1]:
#                 return False

#         return True

# cdef KnotVector make_knots(int p, double a, double b, int n, int mult):
#     cdef int i,j, size = 2*(p+1)+mult*(n-1)
#     cdef np.ndarray[np.float64_t, ndim=1] knots = np.empty(size, dtype=np.float64)
#     cdef double[:]_knots = knots
#     cdef double step = (b-a)/n
    
#     for i in range(p+1):
#         _knots[i]=a
#         _knots[size-i-1]=b
#     for i in range(n-1):
#         for j in range(mult):
#             _knots[p + 1 + i * mult + j] = a + (i + 1) * step
#     return KnotVector(p, knots)
##################################################################################################
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef bint pyx_checkSWcondition(double[::1] kv, int p, int size, double[::1] nodes, int m, bint with_edges):
    """
    Check Schönberg-Whitney condition:
    Given B-spline knot vector `kv` of degree `p` and `m` collocation `nodes`,
    verify that each node can be assigned uniquely to a basis function
    whose support contains it (with tolerance adjustments at boundaries).
    If with_edges is true it is also checked if we can assign basis functions 
    that are nonzero for each interval between the nodes.
    """
    cdef int n = size - p - 1  # number of basis functions
    if n < m + (m-1 if with_edges else 0):
        return False

    cdef int i, j, k = 0
    cdef double tol = 1e-14, s0, s1

    # Check all nodes are strictly increasing and lie in the domain defined by kv[p] and kv[size-p-1]
    for i in range(m):
        if i < m-1 and nodes[i] > nodes[i+1]: return False
        if nodes[i] < kv[p] - tol or nodes[i] > kv[size-p-1] + tol: return False

    for i in range(m):
        #check to find a basis function for the node itself
        for j in range(k, n):
            s0 = kv[j] - tol*(j==0)
            s1 = kv[j+p+1] + tol*(j==n-1)

            if s0 < nodes[i] < s1:
                k = j + 1
                break
            elif nodes[i] < s0:
                return False
        else:
            # No valid basis function support found for nodes[i]
            return False

        if with_edges and i<m-1:
            #check the next interval between the nodes if we can find a basis function with intersecting support
            for j in range(k,n):
                s0 = kv[j] - tol if j <= p else kv[j] + tol
                s1 = kv[j+p+1] + tol if j >= n-p else kv[j+p+1] - tol

                if not (nodes[i] > s1 or nodes[i+1] < s0):
                    k = j + 1
                    break
                elif nodes[i+1] < kv[j] - tol:
                    return False
            else:
                # no valid basis function support found for [nodes[i],nodes[i+1]]
                return False
    return True

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple pyx_unique_knots(double[::1] knots, int size, double atol=1e-14, double rtol=1e-14):
    """
    Identifies unique knot values and their multiplicities from a non-decreasing knot vector.

    Parameters:
        knots : 1D array of knot values (assumed sorted non-decreasing)
        size  : number of knot entries
        atol  : absolute tolerance for float comparison
        rtol  : relative tolerance for float comparison

    Returns:
        mesh_to_knots : np.ndarray[int] mapping each unique knot index to the corresponding index in `knots`
        knots_to_mesh : np.ndarray[int] mapping each knot index to its unique-knot index
        m             : np.ndarray[int] giving the multiplicity of each unique knot
    """
    if size == 0:
        return (
            np.empty(0, dtype=np.intp),
            np.empty(0, dtype=np.intp),
            np.empty(0, dtype=np.intp)
        )

    cdef np.ndarray[np.intp_t, ndim=1] mesh_to_knots = np.empty(size, dtype=np.intp)
    cdef np.ndarray[np.intp_t, ndim=1] knots_to_mesh = np.empty(size, dtype=np.intp)
    cdef np.ndarray[np.intp_t, ndim=1] m             = np.empty(size, dtype=np.intp)

    cdef int i, k = 0, count = 1
    cdef double prev = knots[0]

    mesh_to_knots[0] = 0
    knots_to_mesh[0] = 0

    for i in range(1, size):
        if fabs(prev - knots[i]) < atol + rtol * fabs(knots[i]):
            count += 1
        else:
            m[k] = count
            k += 1
            mesh_to_knots[k] = i
            count = 1
            prev = knots[i]
        knots_to_mesh[i] = k

    m[k] = count  # Final group

    return mesh_to_knots[:k+1], knots_to_mesh, m[:k+1]
    
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef bint pyx_knots_leq(double[:] kv1, int n1, int p1, double a1, double b1,
                         double[:] kv2, int n2, int p2, double a2, double b2):
    cdef int i1 = 0, i2 = 0
    cdef int m1, m2, delta_p = p2 - p1
    cdef double tol = 1e-14
    cdef double x

    if delta_p < 0:
        return 0

    # support containment: [a2,b2] ⊆ [a1,b1]
    if a1 > a2 + tol or b2 > b1 + tol:
        return 0

    # skip knots of kv1 before a2
    while i1 < n1 and kv1[i1] < a2 - tol:
        i1 += 1

    # main walk
    while i1 < n1 and kv1[i1] <= b2 + tol:
        x = kv1[i1]

        # multiplicity of x in kv1
        m1 = 1
        while i1 + m1 < n1 and fabs(kv1[i1 + m1] - x) < tol:
            m1 += 1

        # find x in kv2
        while i2 < n2 and kv2[i2] < x - tol:
            i2 += 1
        if i2 == n2 or fabs(kv2[i2] - x) > tol:
            return 0

        # multiplicity of x in kv2
        m2 = 1
        while i2 + m2 < n2 and fabs(kv2[i2 + m2] - x) < tol:
            m2 += 1

        if m2 < m1 + delta_p:
            return 0

        i1 += m1
        i2 += m2

    return 1

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int pyx_findspan(double[::1] kv, int p, double u) noexcept nogil:
    cdef int n = kv.shape[0]

    if u >= kv[n - p - 1]:
        return n - p - 2  # last interval

    cdef int a = 0, b = n - 1, c

    while b - a > 1:
        c = a + (b - a) // 2
        if kv[c] > u:
            b = c
        else:
            a = c
    return a

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef object pyx_findspans(double[::1] kv, int p, double[::1] u):
    out = np.empty(u.shape[0], dtype=np.int64)
    cdef long[::1] result = out
    cdef int i
    for i in range(u.shape[0]):
        result[i] = pyx_findspan(kv, p, u[i])
    return out

# @cython.boundscheck(False)
# @cython.wraparound(False)
# cpdef np.ndarray[np.int64_t, ndim=1] pyx_findspans(double[::1] kv, int p, double[::1] u, int m):
#     cdef np.ndarray[np.int64_t, ndim=1] out = np.empty(m, dtype=np.int64)
#     cdef long[::1] result = out
#     cdef int i
#     with nogil:
#         for i in range(m):
#             result[i] = pyx_findspan(kv, p, u[i])
#     return out

# @cython.cdivision(True)
# @cython.boundscheck(False)
# @cython.wraparound(False)
# cpdef np.ndarray[np.float64_t, ndim=1] pyx_knot_insertion(double[:] kv, int size, int p, double[:] u, int m):
#     cdef int n = size - p - 1
#     cdef np.ndarray[np.float64_t, ndim=1] kv_new = np.empty(size + m, dtype=np.float64)
#     cdef double[:] kv_new_ = kv_new
#     cdef int i1=0,i2=0,k=0
#     while i1 < size or i2 < m:
#         if i2==m: kv_new_[k] = kv[i1]; i1+=1
#         elif i1==size: kv_new_[k] = u[i2]; i2+=1
#         elif kv[i1] <= u[i2]: kv_new_[k] = kv[i1]; i1+=1
#         else: kv_new_[k] = u[i2]; i2+=1
#         k+=1
#     cdef int n_new = k - p - 1
#     #return kv_new

#     cdef int max_nnz = (n + m) * (p + 1)
#     cdef np.ndarray[np.float64_t, ndim=1] data = np.empty(max_nnz, dtype=np.float64)
#     cdef np.ndarray[np.int32_t, ndim=1] ii = np.empty(max_nnz, dtype=np.int32)
#     cdef np.ndarray[np.int32_t, ndim=1] jj = np.empty(max_nnz, dtype=np.int32)
#     cdef double[:] data_ = data
#     cdef int[:] ii_ = ii
#     cdef int[:] jj_ = jj
#     cdef double* alpha = <double *>malloc((m + p + 1) * sizeof(double))

#     cdef int i,j, idx=0
#     k=0
#     for j in range(n):
#         k = pyx_findspan(kv_new_, p, kv[j])
#         r = 0
#         while r + k + p + 1 < size + m and kv_new_[r + k + p + 1] < kv[j + p + 1]:
#             r+=1
#         pyx_compute_alpha(kv, kv_new_, p, k, r, j, alpha)
#         for i in range(r + p + 1):
#             data_[idx] = alpha[i]
#             jj_[idx] = j
#             ii_[idx] = i + k
#             idx+=1
#     free(alpha)

# @cython.cdivision(True)
# @cython.boundscheck(False)
# @cython.wraparound(False)
# cdef void pyx_compute_alpha(double[:] kv, double[:] kv_new, int p, int start_index, int r, int j, double* alpha) nogil:
#     """
#     """
#     #cdef int i, r, s  # number of inserted knots
#     #cdef double alpha, beta
#     cdef int i, k
#     cdef double beta
#     # Step 1: initialize
#     for i in range(p + 1):
#         alpha[i] = 0.0
#     alpha[0] = 1.0
    
#     for i in range(r):
#         # loop backward to update coefficients
#         for k in range(p + i, 0, -1):
#             if kv[j + k + p] != kv[j + k]:
#                 beta = #TODO
#             else:
#                 beta = 0.0
#             alpha[k] = beta * alpha[k-1] + (1.0 - beta) * alpha[k]
    
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double[:,:] bspline_active_deriv_single(object knotvec, double u, int numderiv, double[:,:] result=None) noexcept:
    """Evaluate all active B-spline basis functions and their derivatives
    up to `numderiv` at a single point `u`"""
    cdef double[::1] kv
    cdef int p, j, r, k, span, rk, pk, fac, j1, j2
    cdef double[:,::1] NDU
    cdef double saved, temp, d
    cdef double[64] left, right, a1buf, a2buf
    cdef double* a1
    cdef double* a2

    kv, p = knotvec.kv, knotvec.p
    assert p < 64, "Spline degree too high"  # need to change constant array sizes above (p+1)

    NDU = np.empty((p+1, p+1), order='C')
    if result is None:
        result = np.empty((numderiv+1, p+1))
    else:
        assert result.shape[0] is numderiv+1 and result.shape[1] is p+1

    span = pyx_findspan(kv, p, u)

    NDU[0,0] = 1.0

    for j in range(1, p+1):
        # Compute knot splits
        left[j-1]  = u - kv[span+1-j]
        right[j-1] = kv[span+j] - u
        saved = 0.0

        for r in range(j):     # For all but the last basis functions of degree j (ndu row)
            # Strictly lower triangular part: Knot differences of distance j
            NDU[j, r] = right[r] + left[j-r-1]
            temp = NDU[r, j-1] / NDU[j, r]
            # Upper triangular part: Basis functions of degree j
            NDU[r, j] = saved + right[r] * temp  # r-th function value of degree j
            saved = left[j-r-1] * temp

        # Diagonal: j-th (last) function value of degree j
        NDU[j, j] = saved

    # copy function values into result array
    for j in range(p+1):
        result[0, j] = NDU[j, p]

    (a1,a2) = a1buf, a2buf

    for r in range(p+1):    # loop over basis functions
        a1[0] = 1.0

        fac = p        # fac = fac(p) / fac(p-k)

        # Compute the k-th derivative of the r-th basis function
        for k in range(1, numderiv+1):
            rk = r - k
            pk = p - k
            d = 0.0

            if r >= k:
                a2[0] = a1[0] / NDU[pk+1, rk]
                d = a2[0] * NDU[rk, pk]

            j1 = 1 if rk >= -1  else -rk
            j2 = k-1 if r-1 <= pk else p - r

            for j in range(j1, j2+1):
                a2[j] = (a1[j] - a1[j-1]) / NDU[pk+1, rk+j]
                d += a2[j] * NDU[rk+j, pk]

            if r <= pk:
                a2[k] = -a1[k-1] / NDU[pk+1, r]
                d += a2[k] * NDU[r, pk]

            result[k, r] = d * fac
            fac *= pk          # update fac = fac(p) / fac(p-k) for next k

            # swap rows a1 and a2
            (a1,a2) = (a2,a1)

    return result

# @cython.cdivision(True)
# @cython.boundscheck(False)
# @cython.wraparound(False)
# cdef double[:,:] bspline_active_deriv_single(object knotvec, double u, int numderiv, double[:,:] result=None) noexcept:
#     """Evaluate all active B-spline basis functions and their derivatives
#     up to `numderiv` at a single point `u`"""
#     cdef double[::1] kv
#     cdef int p, j, r, k, span, rk, pk, fac, j1, j2
#     cdef double[:,::1] NDU
#     cdef double saved, temp, d
#     cdef double[64] left, right, a1buf, a2buf
#     cdef double* a1
#     cdef double* a2

#     kv, p = knotvec.kv, knotvec.p
#     assert p < 64, "Spline degree too high"  # need to change constant array sizes above (p+1)

#     NDU = np.empty((p+1, p+1), order='C')
#     if result is None:
#         result = np.empty((numderiv+1, p+1))
#     else:
#         assert result.shape[0] is numderiv+1 and result.shape[1] is p+1

#     span = pyx_findspan(kv, p, u)

#     NDU[0,0] = 1.0

#     for j in range(1, p+1):
#         # Compute knot splits
#         left[j-1]  = u - kv[span+1-j]
#         right[j-1] = kv[span+j] - u
#         saved = 0.0

#         for r in range(j):     # For all but the last basis functions of degree j (ndu row)
#             # Strictly lower triangular part: Knot differences of distance j
#             NDU[j, r] = right[r] + left[j-r-1]
#             temp = NDU[r, j-1] / NDU[j, r]
#             # Upper triangular part: Basis functions of degree j
#             NDU[r, j] = saved + right[r] * temp  # r-th function value of degree j
#             saved = left[j-r-1] * temp

#         # Diagonal: j-th (last) function value of degree j
#         NDU[j, j] = saved

#     # copy function values into result array
#     for j in range(p+1):
#         result[0, j] = NDU[j, p]

#     (a1,a2) = a1buf, a2buf

#     for r in range(p+1):    # loop over basis functions
#         a1[0] = 1.0

#         fac = p        # fac = fac(p) / fac(p-k)

#         # Compute the k-th derivative of the r-th basis function
#         for k in range(1, numderiv+1):
#             rk = r - k
#             pk = p - k
#             d = 0.0

#             if r >= k:
#                 a2[0] = a1[0] / NDU[pk+1, rk]
#                 d = a2[0] * NDU[rk, pk]

#             j1 = 1 if rk >= -1  else -rk
#             j2 = k-1 if r-1 <= pk else p - r

#             for j in range(j1, j2+1):
#                 a2[j] = (a1[j] - a1[j-1]) / NDU[pk+1, rk+j]
#                 d += a2[j] * NDU[rk+j, pk]

#             if r <= pk:
#                 a2[k] = -a1[k-1] / NDU[pk+1, r]
#                 d += a2[k] * NDU[r, pk]

#             result[k, r] = d * fac
#             fac *= pk          # update fac = fac(p) / fac(p-k) for next k

#             # swap rows a1 and a2
#             (a1,a2) = (a2,a1)

#     return result

@cython.boundscheck(False)
@cython.wraparound(False)
def active_deriv(object knotvec, u, int numderiv):
    """Evaluate all active B-spline basis functions and their derivatives
    up to `numderiv` at the points `u`.

    Returns an array with shape (numderiv+1, p+1) if `u` is scalar or
    an array with shape (numderiv+1, p+1, len(u)) otherwise.
    """
    cdef double[:,:,:] result
    cdef double[:] u_arr
    cdef int i, n

    if np.isscalar(u):
        return bspline_active_deriv_single(knotvec, u, numderiv)
    else:
        u_arr = u
        n = u.shape[0]
        result = np.empty((numderiv+1, knotvec.p+1, n))
        for i in range(n):
            bspline_active_deriv_single(knotvec, u_arr[i], numderiv, result=result[:,:,i])
        return result

