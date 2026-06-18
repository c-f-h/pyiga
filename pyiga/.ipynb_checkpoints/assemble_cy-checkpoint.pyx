cimport cython

import numpy as np
cimport numpy as np

import scipy

from libc.math cimport fabs

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef object pyx_right_inverse_C0_Basis(int[:] indptr, int[:] indices, double[:] data, int n, int m):
    cdef int j,r,k=0

    cdef int[:] ii = np.empty(m, dtype=np.int32)        
    cdef int[:] jj = np.empty(m, dtype=np.int32)        
    cdef double[:] vv = np.empty(m, dtype=np.float64) 
    
    for j in range(m):
        jj[j]=j
        vv[k]=1.
        for r in range(indptr[j],indptr[j+1]):
            if fabs(data[r]-1.)<1e-12:
                ii[k]=indices[r]
                k+=1
                break
    return scipy.sparse.csr_matrix((vv.base,(jj.base,ii.base)),(m,n))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef tuple pyx_find_global_indices(int[:] indptr, int[:] indices, double[:] data, int[:] local_dir_dofs, double[:] local_dir_vals):
    cdef int n = len(local_dir_dofs), i, r, k=0
    
    cdef int[:] g_dofs = np.empty(n, dtype=np.int32) 
    cdef double[:] g_vals = np.empty(n, dtype=np.float64)

    for i in range(n):
        for r in range(indptr[local_dir_dofs[i]],indptr[local_dir_dofs[i]+1]):
            if fabs(data[r]-1.)<1e-14: 
                g_dofs[k]=indices[r]
                g_vals[k]=local_dir_vals[i]
                k+=1
                break;
    return g_dofs[:k].base, g_vals[:k].base
                
        

#cpdef object pyx_eval_nodal()
                
                
                
            
            