cimport cython

import numpy as np
import time 
import scipy
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
#import math

cimport numpy as np
cimport libc.math as math

@cython.cdivision(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple pyx_compute_basis(object Constr, int maxiter):  #in general this function could probably be optimized a bit more
    cdef int n = Constr.shape[1]
    cdef int m = Constr.shape[0]
    cdef np.ndarray[np.int64_t] active = np.arange(m, dtype=int)
    Basis=scipy.sparse.csc_matrix(scipy.sparse.identity(n))
    alldDofs = {}
    cdef int it = 1
    
    while len(active)!=0:
        if it>maxiter:
            print("maxiter reached.")
            break
        dDofs = pyx_find_ddofs(Constr, active)
        alldDofs.update(dDofs)                               #not yet sure if ddofs should stay a dictionary
        assert dDofs, 'Unable to derive further dofs.'
        Basis = pyx_update_basis(Constr, dDofs, Basis)
        Constr = Constr @ Basis   
        active = pyx_compute_active_constr(Constr)
        it+=1
    
    nonderivedDofs = np.setdiff1d(np.arange(n),np.array(list(alldDofs.keys()))) #not yet completely optimal
    return Basis[:,nonderivedDofs].tocsr(), Constr

@cython.cdivision(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef object pyx_find_ddofs(object Constr, long[:] active):
    cdef long n = len(active)
    cpdef object ddofs={}
    
    cdef int[:] Cindptr = Constr.indptr
    cdef int[:] Cindices = Constr.indices
    cdef double[:] Cdata = Constr.data
    
    cdef int r, elim_dof
    cdef bint feasible
    
    for i in range(n):
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
            if abs(v) > 1e-12 and c in ddofs:
                #print("{} cannot be eliminated (constraint #{}) because it refers to eliminated dof {}.".format(dofToBeEliminated,r,c))
                feasible = False
        if feasible:
            ddofs[elim_dof] = r
    return ddofs
        
@cython.cdivision(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef object pyx_compute_active_constr(object Constr):
    cdef int n = Constr.shape[0]
    cdef long[:] active= np.empty(n, dtype=int)
    cdef int k = 0
    cdef int r, a, b
    
    cdef int[:] indptr = Constr.indptr
    cdef double[:] data = Constr.data

    for r in range(Constr.shape[0]):
        a=0
        b=0
        for ind in range(indptr[r], indptr[r+1]):
            if data[ind] > 1e-12:
                a += 1
            if data[ind] < -1e-12:
                b += 1
        if (a==1 and b>0):
            active[k]=r
            k+=1
        if (b==1 and a>0):
            active[k]=r
            k+=1
            for ind in range(indptr[r], indptr[r+1]):
                data[ind]=-data[ind]

    return active.base[:k]
    
@cython.cdivision(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef object pyx_update_basis(object Constr, object Ddofs, object Basis):   
    #assert isinstance(Constr, csr_matrix), "Constraint matrix is not CSR."
    #assert isinstance(Basis, csc_matrix), "Basis matrix is not CSC."
    cdef long[:] ddofs = np.array(list(Ddofs.keys()))
    cdef long[:] dconstr = np.array(list(Ddofs.values()))
    cdef int n_dd = len(Ddofs)
    cdef int n = Constr.shape[1]

    cdef long num_elem = Constr[np.asarray(dconstr),:].nnz - 2*n_dd + n 
    
    cdef int[:] Cindptr = Constr.indptr
    cdef int[:] Cindices = Constr.indices
    cdef double[:] Cdata = Constr.data
    
    cdef int i, r, c, ind
    cdef double v, v0
    
    cdef long[:] ii = np.empty(num_elem, dtype=int)       ###
    cdef long[:] jj = np.empty(num_elem, dtype=int)       ### Is it an advantage to use memoryview here???
    cdef double[:] data = np.empty(num_elem, dtype=float) ###

    cdef int k = 0
    
    for i in range(n):                              #lBasis is assembled here as a COO matrix. Is it possible also with CSC?
        if not i in Ddofs:
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
                    
    lBasis = scipy.sparse.coo_matrix((np.asarray(data),(np.asarray(ii),np.asarray(jj))),(n,n)).tocsc()@Basis
    while pyx_check_col(lBasis, ddofs, n_dd):
        lBasis = lBasis @ lBasis
    
    return lBasis

@cython.cdivision(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef bint pyx_check_col(object lBasis, long[:] ddofs, int n_dd):
    cdef int[:] indptr = lBasis.indptr
    cdef bint check = False
    
    for i in range(n_dd):
        if indptr[ddofs[i]+1]-indptr[ddofs[i]] != 0:  #check if there are entries in columns that correspond to derived dofs.
            check=True
            break;
    return check
        
@cython.cdivision(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple pyx_eval_charPolynomial(double[::1] delta, double[::1] gamma, double lambda_):
    cdef int n = delta.shape[0]
    cdef double[:] v = np.empty(n+1, dtype=float) #actually just need vector with 3 entries
    cdef double[:] d = np.empty(n+1, dtype=float)
    v[0] = 1.
    v[1] = delta[0]-lambda_
    d[0] = 0.
    d[1] = -1.
    for i in range(2,n+1):
        v[i] = (delta[i-1]-lambda_) * v[i-1] - gamma[i-2] * gamma[i-2] * v[i-2]
        d[i] = (delta[i-1]-lambda_) * d[i-1] - v[i-1] - gamma[i-2] * gamma[i-2] * d[i-2]
    return v[n],d[n]

@cython.cdivision(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef object pyx_HilbertMatrix(int n): 
    cdef double[:,:] out = np.empty((n,n), dtype=float)
    cdef int i
    cdef int j
    for i in range(n):
        for j in range(n):
            out[i,j]=1/(i+j+1)
    return out.base

@cython.cdivision(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef object pyx_HilbertMatrixInv(int n): 
    cdef double[:,:] out = np.empty((n,n), dtype=float)
    cdef double[::1] temp = np.empty(n, dtype=float)
    cdef int i
    cdef int j
    cpdef object m
    for i in range(1,n+1):
        for j in range(1,n+1):
            if i == 1:
                m = factorial(j-1)
                temp[j]=factorial(n+j-1)/m/m/factorial(n-j)
            out[i-1,j-1]=(-1)**(i+j)*(i+j-1)/(i+j-1)**2*temp[i]*temp[j]
    return out.base

@cython.cdivision(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef object pyx_CauchyMatrix(int n): 
    cdef double[:,:] out = np.empty((n,n), dtype=float)
    cdef long long[::1] temp = np.empty(n, dtype=int) 
    cdef int i
    cdef int j
    for i in range(1,n+1):
        for j in range(1,n+1):
            if i==1:
                temp[j]=factorial(n-j)
            out[i-1,j-1]=1/temp[i]/temp[j]/(2*n+1-i-j)
    return out.base

@cython.cdivision(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef object pyx_CauchyMatrixInv(int n): 
    cdef double[:,:] out = np.empty((n,n), dtype=float)
    cdef double[::1] temp = np.empty(n, dtype=float) 
    cdef int i
    cdef int j
    cdef int r
    cdef int prod1
    cdef int prod2
    for i in range(1,n+1):
        for j in range(1,n+1):
            if i==1:
                prod1=1
                prod2=1
                for r in range(1,n+1):
                    prod1 *= 2*n-j-r+1
                    if r!=j:
                        prod2 *= r-j
                temp[j]=factorial(n-j)*prod1/prod2
            out[i-1,j-1]=temp[i]*temp[j]/(2*n+1-i-j)
    return out.base

@cython.cdivision(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int factorial(int n):
    cdef int i
    cdef int r = 1
    for i in range(1,n):
        r *= i
    return r