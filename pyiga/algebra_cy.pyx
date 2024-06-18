cimport cython

import numpy as np
cimport numpy as np

#cpdef pyx_eval_charPolynomial():

@cython.cdivision(False)
@cython.boundscheck(True)
@cython.wraparound(False)
cpdef tuple pyx_eval_charPolynomial(double[::1] delta, double[::1] gamma, double lambda_):
    cdef int n = delta.shape[0]
    cdef double[::1] v = np.empty(n+1, dtype=float) #actually just need vector with 3 entries
    cdef double[::1] d = np.empty(n+1, dtype=float)
    v[0] = 1.
    v[1] = delta[0]-lambda_
    d[0] = 0.
    d[1] = -1.
    for i in range(2,n+1):
        v[i] = (delta[i-1]-lambda_) * v[i-1] - gamma[i-2] * gamma[i-2] * v[i-2]
        d[i] = (delta[i-1]-lambda_) * d[i-1] - v[i-1] - gamma[i-2] * gamma[i-2] * d[i-2]
    return v[n],d[n]