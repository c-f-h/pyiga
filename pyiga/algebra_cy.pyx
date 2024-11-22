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
from libcpp.map cimport map
from libc.stdlib cimport malloc, free
from cython.operator cimport dereference as deref, postincrement as inc
#from cython.cimports.cpython.mem import PyMem_Malloc, PyMem_Realloc, PyMem_Free

@cython.cdivision(False)
@cython.boundscheck(True)
@cython.wraparound(True)
cpdef object pyx_compute_basis(int m, int n, object Constr, int maxiter): 
    cdef int *active = <int *>malloc(m * sizeof(int))
    cdef int i, j=0, it=1, num_active=m
    cdef map[int,int] alldDofs, dDofs
    
    cdef object Basis=scipy.sparse.identity(n, format="csr")
    
    for i in range(m):
        active[i]=i
        
    while num_active!=0:
        if it>maxiter:
            print("maxiter reached.")
            break
        dDofs = pyx_find_ddofs(Constr.indptr, Constr.indices, Constr.data, active, num_active)
        assert not dDofs.empty(), 'Unable to derive further dofs.'
        Basis = pyx_update_basis(Constr.indptr, Constr.indices, Constr.data, dDofs, alldDofs, Basis, n)
        Constr = Constr @ Basis   
        num_active = pyx_compute_active_constr(m, n, Constr.indptr, Constr.data, active)
        it+=1
        
    free(active)
    cdef int[:] ndDofs = np.empty(n-alldDofs.size(), dtype=np.int32)
    for i in range(n):
        if alldDofs.count(i)==0:
            ndDofs[j]=i
            j+=1
    return Basis[:,ndDofs.base].tocsr()#, Constr

@cython.cdivision(False)
@cython.boundscheck(True)
@cython.wraparound(True)
cdef map[int,int] pyx_find_ddofs(int[:] Cindptr, int[:] Cindices, double[:] Cdata, int* active, int num_active):
    cdef map[int,int] ddofs
    cdef int r, elim_dof, ind
    cdef bint feasible

    for i in range(num_active):
        r=active[i]
        elim_dof = -1
        feasible = True
        for ind in range(Cindptr[r], Cindptr[r+1]):
            c = Cindices[ind]
            v = Cdata[ind]
            if v > 1e-12: # We know that there is only one (see assertion above!)
                if elim_dof >= 0:
                    feasible = False
                else:
                    elim_dof = c
        if elim_dof == -1: # Empty row (TODO: check)
            feasible = False
        for ind in range(Cindptr[r], Cindptr[r+1]):
            c = Cindices[ind]
            v = Cdata[ind]
            if abs(v) > 1e-12 and ddofs.count(c)>0:
                #print("{} cannot be eliminated (constraint #{}) because it refers to eliminated dof {}.".format(dofToBeEliminated,r,c))
                feasible = False
        if feasible:
            ddofs[elim_dof] = r
    return ddofs
        
@cython.cdivision(False)
@cython.boundscheck(True)
@cython.wraparound(True)
cdef int pyx_compute_active_constr(int m, int n, int[:] Cindptr, double[:] Cdata, int* active):
    cdef int r, a, b, ind, num_active= 0
    
    for r in range(m):
        a=0
        b=0
        for ind in range(Cindptr[r], Cindptr[r+1]):
            if Cdata[ind] > 1e-12:
                a += 1
            if Cdata[ind] < -1e-12:
                b += 1
        if (a==1 and b>0):
            active[num_active]=r
            num_active+=1
        if (b==1 and a>0):
            active[num_active]=r
            num_active+=1
            for ind in range(Cindptr[r], Cindptr[r+1]):
                Cdata[ind]=-Cdata[ind]
    return num_active
    
@cython.cdivision(False)
@cython.boundscheck(True)
@cython.wraparound(True)
cdef object pyx_update_basis(int[:] Cindptr, int[:] Cindices, double[:] Cdata, map[int,int]& Ddofs, map[int,int]& alldDofs, object Basis, int n):   
    #assert isinstance(Constr, csr_matrix), "Constraint matrix is not CSR."
    #assert isinstance(Basis, csc_matrix), "Basis matrix is not CSC."
    cdef int i=0, nnz=0, r, c, ind, n_dd = Ddofs.size(), k=0
    cdef double v, v0
    cdef map[int, int].iterator it = Ddofs.begin()
    cdef int *ddofs = <int *>malloc(n_dd * sizeof(int)) 
    
    while it!=Ddofs.end():
        ddofs[i]= deref(it).first
        alldDofs[deref(it).first]=deref(it).second
        i+=1
        nnz+=Cindptr[deref(it).second+1]-Cindptr[deref(it).second]
        inc(it)

    cdef int num_elem = nnz - 2*n_dd + n
        
    cdef int[:] ii = np.empty(num_elem, dtype=np.int32)        
    cdef int[:] jj = np.empty(num_elem, dtype=np.int32)        
    cdef double[:] data = np.empty(num_elem, dtype=np.float64) 
    
    for i in range(n): #lBasis is assembled here as a COO matrix. Is it possible also with CSC?
        if Ddofs.count(i)==0:
            ii[k] = i
            jj[k] = i
            data[k] = 1.0
            k+=1
        else:
            r  = Ddofs[i]
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
    while pyx_check_col(lBasis.indptr, ddofs, n_dd):
        lBasis = lBasis @ lBasis

    free(ddofs)
    return lBasis

@cython.cdivision(False)
@cython.boundscheck(True)
@cython.wraparound(True)
cdef bint pyx_check_col(int[:] Bindptr, int* ddofs, int n_dd):
    cdef int i, dof
    cdef bint check = False
    
    for i in range(n_dd):
        dof = ddofs[i]
        if Bindptr[dof+1]-Bindptr[dof] != 0:  #check if there are entries in columns that correspond to derived dofs.
            check=True
            break;
    return check
        
@cython.cdivision(False)
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