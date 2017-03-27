cimport cython

import numpy as np
cimport numpy as np

cdef class MatrixGenerator:
    cdef readonly tuple shape
    cdef public object entryfunc

    cdef size_t m, n

    def __init__(self, size_t m, size_t n, entryfunc):
        self.m = m
        self.n = n
        self.shape = (m,n)
        self.entryfunc = entryfunc

    cpdef double entry(self, size_t i, size_t j):
        """Generate the entry at row i and column j"""
        return self.entryfunc(i, j)

    def row(self, size_t i):
        """Generate the i-th row"""
        cdef double[:] result = np.empty(self.n)
        for j in range(self.n):
            result[j] = self.entry(i, j)
        return result

    def column(self, size_t j):
        """Generate the j-th column"""
        cdef double[:] result = np.empty(self.m)
        for i in range(self.m):
            result[i] = self.entry(i, j)
        return result

    def full(self):
        """Generate the entire matrix as an np.ndarray"""
        cdef double[:,:] X = np.empty(self.shape)
        for i in range(self.m):
            for j in range(self.n):
                X[i,j] = self.entry(i, j)
        return X
