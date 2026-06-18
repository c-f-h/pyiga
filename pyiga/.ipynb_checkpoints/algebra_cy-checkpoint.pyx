# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: linetrace=True
cimport cython

import numpy as np
import time 
import scipy
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
#import math

cimport numpy as np
cimport libc.math as math
from libc.math cimport INFINITY, fabs
from libcpp.map cimport map
from libc.stdlib cimport malloc, free
from libc.string cimport memset
from cython.operator cimport dereference as deref, postincrement as inc
#from cython.cimports.cpython.mem import PyMem_Malloc, PyMem_Realloc, PyMem_Free

# @cython.cdivision(True)
# @cython.boundscheck(False)
# @cython.wraparound(False)
# cpdef object pyx_compute_basis_full_rank(int m, int n, object Constr): 
#     cdef bint *pivot = <bint *>malloc(n * sizeof(bint))
#     cdef int *pivot_row = <int *>malloc(m * sizeof(int))
#     memset(pivot, 0, n * sizeof(bint))
#     cdef int i,j=0, k=0

#     print(1)
#     pyx_find_pivot_full_rank(Constr.indptr, Constr.indices, Constr.data, pivot, pivot_row, m)
#     print(2)
#     Basis = pyx_update_basis_full_rank(Constr.indptr, Constr.indices, Constr.data, pivot, pivot_row, m, n)

#     print(3)
#     cdef int[:] free_dofs = np.empty(n-m, dtype=np.int32)
#     for i in range(n):
#         if not pivot[i]:
#             free_dofs[j]=i
#             j+=1
#     free(pivot)
#     free(pivot_row)

#     return Basis

# @cython.cdivision(True)
# @cython.boundscheck(False)
# @cython.wraparound(False)
# cdef void pyx_find_pivot_full_rank(int[:] Cindptr, int[:] Cindices, double[:] Cdata, bint* pivot, int* pivot_row, int m):
#     cdef int elim_dof, ind, c, i
#     cdef double v, elim_val

#     for i in range(m):
#         elim_dof = -1
#         elim_val = -INFINITY
#         for ind in range(Cindptr[i], Cindptr[i+1]):
#             c = Cindices[ind]
#             v = Cdata[ind]
#             if v > elim_val and pivot[c]==0:
#                 elim_dof = c
#                 elim_val = v
#         pivot[elim_dof] = 1
#         pivot_row[i] = elim_dof
#         #pivot_col[elim_dof] = i
#     return 

# @cython.cdivision(True)
# @cython.boundscheck(False)
# @cython.wraparound(False)
# cdef object pyx_update_basis_full_rank(int[:] Cindptr, int[:] Cindices, double[:] Cdata, bint* pivot, int* pivot_row, int m, int n):   
#     cdef int i=0, nnz=0, r, c, ind, k=0
#     cdef double v, v_pivot
    
#     for i in range(m):
#         nnz+=Cindptr[i+1]-Cindptr[i]

#     cdef int nnz_basis = nnz - 2*m + n
        
#     cdef int[:] ii = np.empty(nnz_basis, dtype=np.int32)        
#     cdef int[:] jj = np.empty(nnz_basis, dtype=np.int32)        
#     cdef double[:] data = np.empty(nnz_basis, dtype=np.float64) 

#     for i in range(m):
#         j = pivot_row[i]
#         for ind in range(Cindptr[i], Cindptr[i+1]):
#             if j == Cindices[ind]:
#                 v_pivot = Cdata[ind]
#                 break;
#         for ind in range(Cindptr[i], Cindptr[i+1]):
#             c = Cindices[ind]
#             v = Cdata[ind]
#             if j != c:
#                 ii[k] = j
#                 jj[k] = c
#                 data[k] = - v / v_pivot
#                 k+=1

#     for j in range(n):
#         if not pivot[j]:
#             ii[k] = j
#             jj[k] = j
#             data[k] = 1.0
#             k+=1
                    
#     cdef object Basis = scipy.sparse.coo_matrix((data.base,(ii.base,jj.base)),(n,n)).tocsc()
#     return Basis

##############################################################################################################################################

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef object pyx_compute_basis(int m, int n, object Constr, int maxiter, int switch): 
    cdef int *active = <int *>malloc(m * sizeof(int))
    cdef int i, j=0, it, num_active=0
    cdef map[int,int] dDofs, pivot
    
    cdef object Basis=scipy.sparse.identity(n, format="csr")
    num_active = pyx_compute_active_constr(m, n, Constr.indptr, Constr.data, active, False)

    it=0
    while num_active!=0:
        if it>maxiter:
            print("maxiter reached.")
            break
        pivot = pyx_find_pivot(Constr.indptr, Constr.indices, Constr.data, active, num_active, take_max=False)
        #assert not pivot.empty(), 'Unable to derive further dofs.'
        Basis = pyx_update_basis(Constr.indptr, Constr.indices, Constr.data, pivot, dDofs, Basis, n)
        Constr = Constr @ Basis   
        num_active = pyx_compute_active_constr(m, n, Constr.indptr, Constr.data, active, False)
        #print(0, num_active)
        it+=1

    if switch>=1:
        num_active = pyx_compute_active_constr(m, n, Constr.indptr, Constr.data, active, True)
    
        it=0
        while num_active!=0:
            if it>maxiter:
                print("maxiter reached.")
                break
            pivot = pyx_find_pivot(Constr.indptr, Constr.indices, Constr.data, active, num_active, take_max=True)
            #assert not pivot.empty(), 'Unable to derive further dofs.'
            Basis = pyx_update_basis(Constr.indptr, Constr.indices, Constr.data, pivot, dDofs, Basis, n)
            Constr = Constr @ Basis   
            num_active = pyx_compute_active_constr(m, n, Constr.indptr, Constr.data, active, True)
            #print(0, num_active)
            it+=1
            
    free(active)
    cdef int[:] ndDofs = np.empty(n-dDofs.size(), dtype=np.int32)
    for i in range(n):
        if dDofs.count(i)==0:
            ndDofs[j]=i
            j+=1
    #print(Basis)
    return Basis[:,ndDofs.base], Constr

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef map[int,int] pyx_find_pivot(int[:] Cindptr, int[:] Cindices, double[:] Cdata, int* active, int num_active, bint take_max=False):
    cdef map[int,int] pivot
    cdef int r, elim_dof, ind, c
    cdef double v, elim_val
    cdef bint feasible

    for i in range(num_active):
        r=active[i]
        elim_dof = -1
        elim_val = 0.
        feasible = True
        for ind in range(Cindptr[r], Cindptr[r+1]):
            c = Cindices[ind]
            v = Cdata[ind]
            if take_max:
                if fabs(fabs(v)-elim_val)<1e-14:
                    if v>1e-14:
                        # if elim_dof >= 0:
                        #     feasible = False
                        # else:
                        elim_dof = c
                        elim_val = fabs(v)
                if fabs(v) > elim_val+1e-14: 
                    elim_dof = c
                    elim_val = fabs(v)
            else:
                if fabs(v-1.0)<1e-14:
                    # if elim_dof >= 0:
                    #     feasible = False
                    # else:
                    elim_dof = c
                    elim_val = fabs(v)
        for ind in range(Cindptr[r], Cindptr[r+1]):
            c = Cindices[ind]
            v = Cdata[ind]
            if fabs(v) > 1e-14 and pivot.count(c)>0:
                feasible=False
        if elim_dof == -1: # Empty row (TODO: check)
            feasible = False
        if feasible:
            #print(r,elim_dof)
            pivot[elim_dof] = r
    return pivot
        
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef int pyx_compute_active_constr(int m, int n, int[:] Cindptr, double[:] Cdata, int* active, int switch):
    cdef int r, a, b, ind, num_active= 0
    cdef int n1=0, n2=0, n3=0
    
    for r in range(m):
        a=0
        b=0
        for ind in range(Cindptr[r], Cindptr[r+1]):
            if Cdata[ind] > 1e-14:
                a += 1
            if Cdata[ind] < -1e-14:
                b += 1
        if (a==1 and b>0):
            n1+=1
            active[num_active]=r
            num_active+=1
        elif (b==1 and a>0):
            n1+=1
            active[num_active]=r
            num_active+=1
            for ind in range(Cindptr[r], Cindptr[r+1]):
                Cdata[ind]=-Cdata[ind]
        if (a==0 and b>0):
            n2+=1
            if switch:
                active[num_active]=r
                num_active+=1   
                for ind in range(Cindptr[r], Cindptr[r+1]):
                    Cdata[ind]=-Cdata[ind]
        if (b==0 and a>0):
            n2+=1
            if switch:
                active[num_active]=r
                num_active+=1     
        if (a>1 and b>1):
            n3+=1
            if switch:
                active[num_active]=r
                num_active+=1                
    #print(n1,n2,n3)
    return num_active
    
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef object pyx_update_basis(int[:] Cindptr, int[:] Cindices, double[:] Cdata, map[int,int]& pivot, map[int,int]& dDofs, object Basis, int n):   
    #assert isinstance(Constr, csr_matrix), "Constraint matrix is not CSR."
    #assert isinstance(Basis, csc_matrix), "Basis matrix is not CSC."
    cdef int i=0, nnz=0, r, c, ind, n_dd = pivot.size(), k=0
    cdef double v, v0
    cdef map[int, int].iterator it = pivot.begin()
    cdef int *ddofs = <int *>malloc(n_dd * sizeof(int)) 
    
    while it!=pivot.end():
        ddofs[i]= deref(it).first
        dDofs[deref(it).first]=deref(it).second
        i+=1
        nnz+=Cindptr[deref(it).second+1]-Cindptr[deref(it).second]
        inc(it)

    cdef int num_elem = nnz - 2*n_dd + n
        
    cdef int[:] ii = np.empty(num_elem, dtype=np.int32)        
    cdef int[:] jj = np.empty(num_elem, dtype=np.int32)        
    cdef double[:] data = np.empty(num_elem, dtype=np.float64) 
    
    for i in range(n): #lBasis is assembled here as a COO matrix. Is it possible also with CSC?
        if pivot.count(i)==0:
            ii[k] = i
            jj[k] = i
            data[k] = 1.0
            k+=1
        else:
            r  = pivot[i]
            for ind in range(Cindptr[r], Cindptr[r+1]):
                if i == Cindices[ind]:
                    v0 = Cdata[ind]
                    break;
            for ind in range(Cindptr[r], Cindptr[r+1]):
                c = Cindices[ind]
                v = Cdata[ind]
                if i != c:
                    ii[k] = i
                    jj[k] = c
                    data[k] = - v / v0
                    k+=1
                    
    cdef object lBasis = scipy.sparse.coo_matrix((data.base,(ii.base,jj.base)),(n,n)).tocsc()@Basis

    k=0
    while pyx_check_col(lBasis.indptr, ddofs, n_dd):
        lBasis = lBasis @ lBasis
        k+=1
        # if k>10: 
        #     print("maxiter reached for sequence reduction. maybe encountered a cycle?")
        #     break;

    free(ddofs)
    return lBasis

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef bint pyx_check_col(int[:] Bindptr, int* ddofs, int n_dd):
    cdef int i, dof
    cdef bint check = False
    
    for i in range(n_dd):
        dof = ddofs[i]
        if Bindptr[dof+1]-Bindptr[dof] != 0:  #check if there are entries in columns that correspond to derived dofs.
            check=True
            break;
    return check
        
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple pyx_eval_charPolynomial(double[:] delta, double[:] gamma, double lambda_):
    cdef int i
    cdef int n = delta.shape[0]
    cdef double[:] v = np.empty(n+1, dtype=float) #actually just need vector with 3 entries
    cdef double[:] d = np.empty(n+1, dtype=float)
    cdef double[:] d2 = np.empty(n+1, dtype=float)
    v[0] = 1.
    v[1] = delta[0]-lambda_
    d[0] = 0.
    d[1] = -1.
    d2[0] = 0.
    d2[0] = 0.
    for i in range(2,n+1):
        v[i] = (delta[i-1]-lambda_) * v[i-1] - gamma[i-2] * gamma[i-2] * v[i-2]
        d[i] = (delta[i-1]-lambda_) * d[i-1] - v[i-1] - gamma[i-2] * gamma[i-2] * d[i-2]
        d2[i] = (delta[i-1]-lambda_) * d2[i-1] - 2*d[i-1] - gamma[i-2] * gamma[i-2] * d2[i-2]
    return v[n],d[n],d2[n]

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=2] HilbertMatrix(int n): 
    cdef double[:,:] out = np.empty((n,n), dtype=np.float64)
    cdef int i, j
    for i in range(n):
        for j in range(n):
            out[i,j]=1./(i+j+1)
    return out.base

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=2] HilbertMatrixInv(int n): 
    assert n<=10 , "Dimension of matrix must not exceed 10."
    cdef double[:,:] out = np.empty((n,n), dtype=np.float64)
    #cdef double[:] temp = np.empty(n, dtype=np.float64)
    cdef double *temp = <double *>malloc(n * sizeof(double))
    cdef int i, j
    for i in range(1,n+1):
        for j in range(1,n+1):
            if i == 1:
                temp[j-1]=(-1.)**j*factorial(n+j-1)/factorial(n-j)/(factorial(j-1)**2)
            out[i-1,j-1]=temp[i-1]*temp[j-1]/(i+j-1)
    free(temp)
    return out.base

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=2] CauchyMatrix(int n): 
    cdef double[:,:] out = np.empty((n,n), dtype=np.float64)
    #cdef long[:] temp = np.empty(n, dtype=np.int64) 
    cdef long *temp = <long *>malloc(n * sizeof(long)) 
    cdef int i, j
    for i in range(1,n+1):
        for j in range(1,n+1):
            if i==1:
                temp[j-1]=factorial(n-j)
            out[i-1,j-1]=1./temp[i-1]/temp[j-1]/(2*n+1-i-j)
    free(temp)
    return out.base

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=2] CauchyMatrixInv(int n): 
    assert n<=10 , "Dimension of matrix must not exceed 10."
    cdef double[:,:] out = np.empty((n,n), dtype=np.float64)
    #cdef double[:] temp = np.empty(n, dtype=np.float64) 
    cdef double *temp = <double *>malloc(n * sizeof(double))
    cdef int i, j, r
    cdef long prod1, prod2
    for i in range(1,n+1):
        for j in range(1,n+1):
            if i==1:
                prod1=1
                prod2=1
                for r in range(1,n+1):
                    prod1 *= 2*n-j-r+1
                    if r!=j:
                        prod2 *= r-j
                temp[j-1]=factorial(n-j)*prod1/prod2
            out[i-1,j-1]=temp[i-1]*temp[j-1]/(2*n+1-i-j)
    free(temp)
    return out.base

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef long factorial(int n) noexcept:
    cdef int i
    cdef long r = 1
    for i in range(1,n):
        r *= (i+1)
    return r

# @cython.cdivision(True)
# @cython.boundscheck(False)
# @cython.wraparound(False)
# cpdef object csr_multiply(
#     double[:] data_a,
#     int[:] indices_a,
#     int[:] indptr_a,
#     int rows_a, int cols_a,
#     double[:] data_b,
#     int[:] indices_b,
#     int[:] indptr_b,
#     int rows_b, int cols_b):
#     """
#     Multiplies two CSR matrices.

#     Parameters:
#         data_a, indices_a, indptr_a: CSR representation of matrix A.
#         rows_a, cols_a: Dimensions of matrix A.

#         data_b, indices_b, indptr_b: CSR representation of matrix B.
#         rows_b, cols_b: Dimensions of matrix B.

#     Returns:
#         data_c, indices_c, indptr_c: CSR representation of the result matrix.
#     """
#     if cols_a != rows_b:
#         raise ValueError("Matrix dimensions do not allow multiplication.")

#     cdef:
#         int i, j, k, nnz_c
#         double value
#         int[:] indptr_c = np.zeros(rows_a + 1, dtype=np.int32)
#         double[:] data_c
#         int[:] indices_c
#         map[int, double] row_accumulator
#         int pos
#         int col_a, col_b
#         double val_a, val_b

#     # First pass: calculate row pointers and nnz
#     nnz_c = 0
#     for i in range(rows_a):
#         row_accumulator.clear()
#         for j in range(indptr_a[i], indptr_a[i + 1]):
#             col_a = indices_a[j]
#             val_a = data_a[j]
#             for k in range(indptr_b[col_a], indptr_b[col_a + 1]):
#                 col_b = indices_b[k]
#                 val_b = data_b[k]
#                 if col_b not in row_accumulator:
#                     row_accumulator[col_b] = 0.0
#                 row_accumulator[col_b] += val_a * val_b
#         nnz_c += len(row_accumulator)
#         indptr_c[i + 1] = nnz_c

#     # Preallocate arrays for data and indices
#     data_c = np.zeros(nnz_c, dtype=np.float64)
#     indices_c = np.zeros(nnz_c, dtype=np.int32)

#     # Second pass: fill data and indices arrays
#     pos = 0
#     for i in range(rows_a):
#         row_accumulator.clear()
#         for j in range(indptr_a[i], indptr_a[i + 1]):
#             col_a = indices_a[j]
#             val_a = data_a[j]
#             for k in range(indptr_b[col_a], indptr_b[col_a + 1]):
#                 col_b = indices_b[k]
#                 val_b = data_b[k]
#                 if col_b not in row_accumulator:
#                     row_accumulator[col_b] = 0.0
#                 row_accumulator[col_b] += val_a * val_b
#         for key, value in row_accumulator.items():
#             indices_c[pos] = key
#             data_c[pos] = value
#             pos += 1

#     return data_c, indices_c, indptr_c
