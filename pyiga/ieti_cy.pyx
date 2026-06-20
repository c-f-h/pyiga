# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
cimport cython

#from algebra_cy cimport pyx_compute_basis

import numpy as np
cimport numpy as np
#from libcpp.vector cimport vector
from libc.stdlib cimport malloc, free

from libc.math cimport fabs

import time 
import scipy
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef object identify_T_coefficients_from_corner_basis(int[:] CBasis_indptr, int[:] CBasis_indices, double[:] CBasis_data, int n,   int m,
                                                       int[:] B_indptr,      int[:] B_indices,      double[:] B_data,      int lam, int deg): 
    cdef int j, r, ind, ind2, nnz=0
    cdef bint found = False, found2= False
    cdef int[:] ii = np.empty((deg+1)*m, dtype=np.int32)
    cdef int[:] jj = np.empty((deg+1)*m, dtype=np.int32)
    cdef double[:] vv = np.empty((deg+1)*m, dtype=np.float64)

    for j in range(m):
        found=False
        found2=False
        for ind in range(CBasis_indptr[j],CBasis_indptr[j+1]):
            if found: break
            for r in range(lam):
                for ind2 in range(B_indptr[r],B_indptr[r+1]):
                    if B_indices[ind2]==CBasis_indices[ind] and fabs(B_data[ind2]-1)<1e-12:
                        found = True
                        for ind2 in range(B_indptr[r], B_indptr[r+1]):
                            if fabs(fabs(B_data[ind2])-1)>1e-12:
                                found2 = True
                                ii[nnz] = B_indices[ind2]
                                jj[nnz] = j
                                vv[nnz] = fabs(B_data[ind2])
                                nnz+=1
                    
    return scipy.sparse.csc_matrix((vv.base[:nnz],(ii.base[:nnz],jj.base[:nnz])),(n,m))

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.uint8_t, ndim=1] eliminate_corner_constraints(int[:] indptr, int[:] indices, double[:] data, int m, int n, int[:] corners, int nC):
    cdef np.ndarray[np.uint8_t, ndim=1] out = np.zeros(m, dtype=np.uint8)
    cdef int i, ind
    for i in range(nC):
        for ind in range(indptr[corners[i]],indptr[corners[i]+1]):
            if fabs(1-data[ind])<1e-12:
                out[indices[ind]]=1
    return out

@cython.cdivision(True)
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
    
    cdef int i, j, ind, p, last_p, row

    # Loop over columns
    for j in range(global_Basis.shape[1]):
        p = 0

        for ind in range(indptr[j], indptr[j+1]):
            row = indices[ind]

            # advance p monotonically
            while p < K-1 and row >= N_ofs[p+1]:
                p += 1

            # assign DOF once per block
            if N[p] == 0 or dofs[N_ofs[p] + N[p] - 1] != j:
                dofs[N_ofs[p] + N[p]] = j
                N[p] += 1
    
    cdef list Basisk = [global_Basis[N_ofs[p]:N_ofs[p+1],dofs.base[N_ofs[p]:(N_ofs[p]+N[p])]]] for p in range(K)]
    cdef int[:] N_ofs_ = np.r_[0,np.cumsum(N.base, dtype=np.int32)]
    
    cdef int k, p1, p2, m, s
    cdef int[:] dofs_intfs, idx1, idx2
    
    cdef int l = 0
    cdef int[:] jj = np.empty(n, dtype=np.int32)            ###TODO: generate list of patch jump matrices immediately, also generate them in csr-format (data, indices, inptr)
    cdef int[:] ii = np.empty(n, dtype=np.int32)
    cdef double[:] Bdata = np.empty(n, dtype=np.float64)
    
    for k in range(p_intfs.shape[1]):
        p1 = p_intfs[0,k]
        p2 = p_intfs[1,k]
        dofs_intfs, idx1, idx2, m = intersect(dofs[N_ofs[p1]:N_ofs[p1]+N[p1]],dofs[N_ofs[p2]:N_ofs[p2]+N[p2]])
        for s in range(m):
            jj[l] = idx1[s] + N_ofs_[p1]
            jj[l+1] = idx2[s] + N_ofs_[p2]
            ii[l] = l//2
            ii[l+1] = l//2
            Bdata[l]= 1.0
            Bdata[l+1]= -1.0
            l+=2
        
    B = scipy.sparse.coo_matrix((Bdata.base[:l],(ii.base[:l],jj.base[:l])),(l//2,N_ofs_[len(N_ofs_)-1])).tocsr()
        #N_ofs_[p1]+idx1
    
    return Basisk, N_ofs_.base, N.base, B
    #return (1,2)

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef int max(int a, int b):
    if a > b: 
        return a
    else: 
        return b
    
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef int min(int a, int b):
    if a < b: 
        return a
    else: 
        return b
    
@cython.cdivision(True)
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

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void pyx_parametersort(int[:] indptr, int[:] indices, double[:] data, int m, int n, double[:] a):
    cdef int pos,neg,i,ind
    for i in range(m):
        pos=-1
        neg=-1
        for ind in range(indptr[i],indptr[i+1]):
            if fabs(1.0-data[ind])<1e-12:
                pos = ind
            if fabs(1.0+data[ind])<1e-12:
                neg = ind
        if (pos >= 0) and (neg >= 0):
            if (a[indices[pos]] > a[indices[neg]]):
                data[pos]*=-1
                data[neg]*=-1

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=1] pyx_constraint_scaling(int[:] indptr, int[:] indices, double[:] data, int m, int n):

    cdef double[:] d = np.zeros(n, dtype=np.float64)
    cdef int i, j, ind

    for i in range(m):
        for ind in range(indptr[i],indptr[i+1]):
            j=indices[ind]
            d[j]+=1

    for j in range(n):
        d[j]=1.0/(1.0+d[j])
    return d.base

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=1] pyx_weight_scaling(int[:] indptr, int[:] indices, double[:] data, int m, int n):

    cdef double[:] d = np.zeros(n, dtype=np.float64)
    cdef int i, j, ind

    for i in range(m):
        for ind in range(indptr[i],indptr[i+1]):
            d[indices[ind]]+=data[ind]

    for j in range(n):
        d[j]=1.0/(1.0+d[j])
    return d.base

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=1] pyx_absolute_weight_scaling(int[:] indptr, int[:] indices, double[:] data, int m, int n):

    cdef double[:] d = np.zeros(n, dtype=np.float64)
    cdef int i, j, ind

    for i in range(m):
        for ind in range(indptr[i],indptr[i+1]):
            d[indices[ind]]+=fabs(data[ind])

    for j in range(n):
        d[j]=1.0/(1.0+d[j])
    return d.base
                
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.int32_t, ndim=1] pyx_selection_scaling(int[:] indptr, int[:] indices, double[:] data, int m, int n):

    cdef np.ndarray[np.int32_t, ndim=1] d_ = np.zeros(n, dtype=np.int32)
    cdef int[:] d = d_
    cdef char* valid = <char*> malloc(n * sizeof(char))
    cdef int i, j, ind
    
    for i in range(n):
        valid[i] = 1

    for i in range(m):
        for ind in range(indptr[i],indptr[i+1]):
            j=indices[ind]
            if fabs(1.0-data[ind])<1e-14:
                d[j]+=1
            else:
                valid[j]=0

    for j in range(n):
        if d[j] != 1 or not valid[j]:
            d[j]=0
    free(valid)
    return d_


                






    