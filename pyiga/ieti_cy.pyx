cimport cython

import numpy as np
cimport numpy as np
#from libcpp.vector cimport vector
#from libc.stdlib cimport malloc, free

import time 
import scipy
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix

@cython.cdivision(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple pyx_compute_decoupled_coarse_basis(object global_Basis, int[:] N_ofs, int[:,:] p_intfs): 
    #cdef vector[int] dofs[np]
    cdef int K = len(N_ofs)-1
    cdef int n = global_Basis.shape[0]
    cdef int[:] dofs = np.empty(n, dtype=np.int32)
    cdef int[:] N = np.zeros(K ,dtype=np.int32)
    
    cdef int[:] indptr = global_Basis.indptr
    cdef int[:] indices = global_Basis.indices
    #cdef double[:] data = global_Basis.data
    #cdef int[:] N_ofs_ = N_ofs
    
    cdef int i, j, ind, p, last_p
    
    for j in range(global_Basis.shape[1]):
        last_p=-1
        for ind in range(indptr[j],indptr[j+1]):
            for p in range(max(last_p,0),K):
                if indices[ind] < N_ofs[p+1]:
                    break
            if p!=last_p:
                dofs[N_ofs[p]+N[p]] = j 
                N[p]+=1
                last_p=p
    
    cdef list Basisk = [global_Basis[N_ofs[p]:N_ofs[p+1],:][:,dofs.base[N_ofs[p]:(N_ofs[p]+N[p])]] for p in range(K)]
    cdef int[:] N_ofs_ = np.r_[0,np.cumsum(N.base, dtype=np.int32)]
    
    cdef int k, p1, p2, m, s
    cdef int[:] dofs_intfs, idx1, idx2
    
    cdef int l = 0
    cdef int[:] jj = np.empty(n, dtype=np.int32)            ###TODO: generate list of patch jump matrices immediately, also generate them in csr-format (data, indices, inptr)
    cdef int[:] ii = np.empty(n, dtype=np.int32)
    cdef int[:] Bdata = np.empty(n, dtype=np.int32)
    
    for k in range(p_intfs.shape[1]):
        p1 = p_intfs[0,k]
        p2 = p_intfs[1,k]
        dofs_intfs, idx1, idx2, m = intersect(dofs[N_ofs[p1]:N_ofs[p1]+N[p1]],dofs[N_ofs[p2]:N_ofs[p2]+N[p2]])
        for s in range(m):
            jj[l] = idx1[s] + N_ofs_[p1]
            jj[l+1] = idx2[s] + N_ofs_[p2]
            ii[l] = l//2
            ii[l+1] = l//2
            Bdata[l]= 1
            Bdata[l+1]= -1
            l+=2
        
    B = scipy.sparse.coo_matrix((Bdata.base[:l],(ii.base[:l],jj.base[:l])),(l//2,N_ofs_[len(N_ofs_)-1])).tocsr()
        #N_ofs_[p1]+idx1
    
    return Basisk, N_ofs_.base, N.base, B
    #return (1,2)

@cython.cdivision(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef int max(int a, int b):
    if a > b: 
        return a
    else: 
        return b
    
@cython.cdivision(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef int min(int a, int b):
    if a < b: 
        return a
    else: 
        return b
    
@cython.cdivision(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef tuple intersect(int[:] arr1, int[:] arr2):
    cdef int i = 0
    cdef int j = 0
    cdef int k = 0
    
    cdef int n = min(len(arr1),len(arr2))
    cdef int[:] idx1 = np.empty(n, dtype=np.int32)
    cdef int[:] idx2 = np.empty(n, dtype=np.int32)
    cdef int[:] result = np.empty(n, dtype=np.int32)
    
    while (i < len(arr1)) & (j < len(arr2)):
        if arr1[i] == arr2[j]:
            result[k] = arr1[i]
            idx1[k] = i
            idx2[k] = j
            #print(result[k])
            k+=1
            i+=1
            j+=1
        elif arr1[i] < arr2[j]:
            i+=1
        else:
            j+=1
            
    return result.base[:k], idx1.base[:k], idx2.base[:k], k
    
    
    
    
    
    
    
    
        #object Basis, int[:] N, object B