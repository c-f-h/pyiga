# cython: profile=False
# cython: linetrace=False
# cython: binding=False

cimport cython

import numpy as np
cimport numpy as np

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double[:,:] bspline_active_deriv_single(object knotvec, double u, int numderiv, double[:,:] result=None):
    """Evaluate all active B-spline basis functions and their derivatives
    up to `numderiv` at a single point `u`"""
    cdef double[:] kv
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

    span = knotvec.findspan(u)

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

