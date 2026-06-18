cimport cython

import numpy as np
cimport numpy as np

from libc.math cimport fabs
from libc.stdlib cimport rand, srand

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.int32_t, ndim=1] pyx_doerfler_mark(np.ndarray[np.float64_t ,ndim=1] x_a, int n, double theta, double TOL):
    cdef np.ndarray[np.int32_t, ndim=1] idx_a = np.argsort(-x_a).astype(np.int32)
    cdef int[:] idx = idx_a
    cdef double[:] x = x_a
    cdef int i, k
    cdef double total=0, S=0

    for i in range(n):
        total += x[i]**2

    for i in range(n):
        S += x[idx[i]]**2
        if S > theta * total:
            k = i
            while k<n and (fabs((x[idx[i]]-x[idx[k]])/(x[idx[i]]+1e-16)) < TOL):       
                k+=1
            break
    else: return idx.base
            
    return idx_a[:k]

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.int32_t, ndim=1] pyx_quick_mark(double[:] x, double theta, int[:] idx, int l, int u, double v):
    cdef int i, p
    cdef double sigma

    while l < u:
        p = (u + l) / 2  # integer division, choosing the slot of the median (rounded to the lower slot of the array)
    
        quickselect(x, idx, l, u, p)
    
        sigma = 0.0
        for i in range(l,p):
            sigma += x[idx[i]] * x[idx[i]]
        if sigma > v:                                                                         
            u=p-1                      
        elif sigma + x[idx[p]] * x[idx[p]] > v:              
            return idx.base[:p+1]                                                    
        else:     
            v = v - sigma - x[idx[p]] * x[idx[p]]
            l=p+1
            #return pyx_quick_mark(x, theta, idx, p+1, u, v-sigma-x[idx[p]]**2)
    return idx.base[:u+1]

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void swap(int[:] idx, int i, int j):
    cdef int temp = idx[i]
    idx[i] = idx[j]
    idx[j] = temp
    return

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef int partition(double[:] x, int[:] idx, int l, int u, int p):
    cdef double pivot_value = x[idx[p]]
    if p!=u:
        swap(idx, p, u)
    cdef int store_index = l
    cdef int i
    for i in range(l, u):
        if x[idx[i]] > pivot_value:
            swap(idx, store_index, i)
            store_index += 1
    swap(idx, store_index, u)
    return store_index

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void quickselect(double[:] x, int[:] idx, int l, int u, int q):
    cdef int p, k
    while l < u:
        p = l + rand() % (u - l + 1)
        #p =  (u + l)/2
        k = partition(x, idx, l, u, p)
        if k == q:
            return
        elif q < k:
            u = k - 1
        else:
            l = k + 1
    return

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void quicksort(double[:] x, int[:] idx, int l, int u):
    cdef int k, p
    if u-l>128:
        p = l + rand() % (u - l + 1)
        k = partition(x, idx, l, u, p)
        quicksort(x, idx, l, k-1)
        quicksort(x, idx, k+1, u)
    else:
        insertionsort(x, idx, l, u)
    return

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void insertionsort(double[:] x, int[:] idx, int l, int u):
    cdef int i, j
    cdef double key

    for i in range(l+1,u+1):
        key_idx = idx[i]
        key = x[key_idx]
        j = i - 1
        while j >= l and x[idx[j]] < key:
            idx[j + 1] = idx[j]
            j -= 1
        idx[j + 1] = key_idx