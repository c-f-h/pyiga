# -*- coding: utf-8 -*-
"""Functions and classes for algebraic operations.

"""

import numpy as np
import scipy.sparse
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
import scipy.linalg
from scipy.sparse.linalg import LinearOperator, onenormest, splu
from . import solvers
from . import algebra_cy
import time
import numba as nb

def condest(A, spd=False):
    luA = splu(A)
    iA = LinearOperator(luA.shape, matvec = lambda x : luA.solve(x), rmatvec = lambda x : luA.solve(x))
    return onenormest(iA)*onenormest(A)

def find_ddofs(Constr, activeConstraints):
    derivedDofs={}
    for r in activeConstraints:
        dofToBeEliminated = -1
        feasible = True
        #mx=0
        for ind in range(Constr.indptr[r], Constr.indptr[r+1]):
            c = Constr.indices[ind]
            v = Constr.data[ind]
            
            if v > 1e-12: # We know that there is only one (see assertion above!)
                if dofToBeEliminated >= 0:
                    feasible = False 
                else:
                    dofToBeEliminated = c
                    #mx=v
        if dofToBeEliminated == -1: # Empty row (TODO: check)
            feasible = False
        for ind in range(Constr.indptr[r], Constr.indptr[r+1]):
            c = Constr.indices[ind]
            v = Constr.data[ind]
            if abs(v) > 1e-12 and c in derivedDofs:
                #print("{} cannot be eliminated (constraint #{}) because it refers to eliminated dof {}.".format(dofToBeEliminated,r,c))
                feasible = False
        if feasible:
            derivedDofs[dofToBeEliminated] = r
    return derivedDofs

def update_basis(Constr, derivedDofs, Basis):
    n=Constr.shape[1]
    #print(Basis)
    
    #variant 1
    #     lBasis = scipy.sparse.lil_matrix((n,n)) #coo_matrix ?
    #     for i in range(n):
    #         if not i in derivedDofs:
    #             lBasis[i,i] = 1

    #     for i, r in derivedDofs.items():
    #         for ind in range(Constr.indptr[r], Constr.indptr[r+1]):
    #             c = Constr.indices[ind]
    #             v = Constr.data[ind]    
    #             if v < -1e-12:
    #                 lBasis[i,c] = - v / Constr[r,i]
    
    #fast variant
    ddofs=np.array(list(derivedDofs.keys()))   #derived dofs
    n2 = len(ddofs)
    r = np.array(list(derivedDofs.values()))  #which constraints do the dofs derive from
    nddofs=np.setdiff1d(np.arange(n),ddofs)   #still free dofs (TODO: make this faster)
    n1=len(nddofs)

    lBasis=scipy.sparse.csr_matrix((n,n))
    c = 1/(Constr[r,ddofs].A.ravel())
    B1 = scipy.sparse.coo_matrix((np.ones(n1),(np.arange(n1),nddofs)),(n1,n)).tocsr()
    t=time.time()
    B2 = - Constr[r].multiply(Constr[r]<0)  
    B2 = scipy.sparse.spdiags(c,0,n2,n2)@B2  #row scaling
    
    Q1 = scipy.sparse.coo_matrix((np.ones(n1),(nddofs,np.arange(n1))),(n,n1)).tocsr()
    Q2 = scipy.sparse.coo_matrix((np.ones(n2),(ddofs,np.arange(n2))),(n,n2)).tocsr()
    lBasis = Q1@B1 + Q2@B2
    
    lBasis = lBasis @ Basis
    
    testVec = scipy.sparse.coo_array((np.ones(n2),(ddofs,np.zeros(n2))),shape=(n,1))
        
    while True:
        found = 0
        tmp = lBasis @ testVec
        found = sum(abs(tmp.data)>1e-12)
        
        if found > 0:
            lBasis = lBasis @ lBasis
        else:
            break
    return lBasis

def compute_active_constr(Constr):
    #fast variant
#     A = (Constr>1e-12)
#     B = (Constr<-1e-12)
#     a=A.indptr[1:]-A.indptr[:-1]
#     b=B.indptr[1:]-B.indptr[:-1]
#     active=np.where(((a==1) & (b>0)) | ((b==1) & (a>0)))[0]
#     sign = np.ones(len(a), dtype=int)
#     sign[np.where((b==1) & (a>0))[0]]=-1
    
#     #print(signs)
#     S=scipy.sparse.spdiags(sign,0,len(a),len(a))
    
#     Constr = S@Constr ### not by reference
    
    # assert np.all(((a[activeConstraints]==1) | (b[activeConstraints]==1))), 'error in constraint matrix.'

    #variant 1
    active=[]
    for r in range(Constr.shape[0]):
        a = 0
        b = 0
        for ind in range(Constr.indptr[r], Constr.indptr[r+1]):
            if Constr.data[ind] > 1e-12:
                a += 1
            if Constr.data[ind] < -1e-12:
                b += 1
        if ((a==1 and b>0) or (b==1 and a>0)):
            active.append(r)
            #print("{}: {}, {}".format(r,a,b))
            #if not (a==1 or b==1): 
                #print(a,b)
                #print(Constr[r,:])
            #assert (a==1 or b==1), 'error in constraint matrix.'
            if b==1 and a>0:
                #print( "Re-sign" )
                Constr[r,:] *= -1  ###bottleneck
                
    return active

def compute_basis(Constr, maxiter):
    n=Constr.shape[1]
    nonderivedDofs = allLocalDofs=np.arange(n)
    allderivedDofs={}
    activeConstraints=np.arange(Constr.shape[0])
    Basis=scipy.sparse.csc_matrix(scipy.sparse.identity(n))
    time_find_active = 0
    time_find_ddofs = 0
    time_update = 0
    time_main1 = 0
    time_main2 = 0
    time_main3 = 0
    i=1
    while len(activeConstraints)!=0:
        #print(Constr[activeConstraints])
        if i>maxiter:
            print("maxiter reached.")
            break
            
        t = time.time()
        derivedDofs = find_ddofs(Constr, activeConstraints)  
        time_find_ddofs += time.time()-t
        #assert derivedDofs, 'Unable to derive any further dofs.'
        assert derivedDofs, 'Unable to derive further dofs.'
        
        #update which dofs were already derived
        t = time.time()
        allderivedDofs.update(derivedDofs)
        time_main1 += time.time()-t
        t=time.time()
        Basis = update_basis(Constr, derivedDofs, Basis)
        #assert isinstance(Basis, csc_matrix)
        time_update += time.time()-t
        #eliminate used constraints from constraint matrix
        t = time.time()
        Constr = Constr @ Basis    
        time_main2 += time.time()-t
        
        t=time.time()
        activeConstraints = compute_active_constr(Constr)
        time_find_active += time.time()-t
        i+=1
        
    #print(np.array(list(allderivedDofs.keys())))
    #nonderivedDofs=np.setdiff1d(allLocalDofs, np.array(list(allderivedDofs.keys())))
    #print(nonderivedDofs)
    #Basis = scipy.sparse.csc_matrix(Basis)
    
    t=time.time()
    nonderivedDofs=np.setdiff1d(allLocalDofs, np.array(list(allderivedDofs.keys())))
    Basis = Basis[:,nonderivedDofs].tocsr()
    time_main3+= time.time()-t

    
    # print('finding active constraints took '+str(time_find_active)+' seconds.')
    # print('finding derived dofs took '+str(time_find_ddofs)+' seconds.')
    # print('updating basis and constraints took '+str(time_update)+' seconds.')
    # print('main1 took '+str(time_main1)+' seconds.')
    # print('main2 took '+str(time_main2)+' seconds.')
    # print('main3 took '+str(time_main3)+' seconds.')
    
    return Basis, Constr  #,Constr,activeConstraints

def rref(A, tol=1e-8):
    B=A.astype(float).copy()
    m,n=B.shape
    rows=np.arange(m)
    i=0
    j=0
    piv=[]  #indices of pivot columns, rank = length(piv)
    while (i<m and j<n):
        k = np.argmax(abs(B[i:m,j]))
        k=k+i
        if abs(B[k,j])<=tol:
            B[i:m,j]=0
            j+=1
        else:
            piv.append(j)
            if k!=i:
                rows[[i,k]]=rows[[k,i]]
                B[[i,k],j:n]=B[[k,i],j:n]
            B[i,j:n]=B[i,j:n]/B[i,j]
            for l in np.r_[np.arange(i),np.arange(i+1,m)]:
                B[l,j:n]=B[l,j:n]-B[l,j]*B[i,j:n]
            i+=1
            j+=1
    return B, np.array(piv), rows

class LanczosMatrix():
    def __init__(self, delta, gamma):
        assert len(delta)==len(gamma)+1, "size mismatch."
        self.gamma = gamma
        self.delta = delta
        self.n = len(delta)
    
    @property
    def mat(self):
        return scipy.sparse.spdiags(np.c_[np.r_[self.gamma,0],self.delta,np.r_[0,self.gamma]].T,[-1,0,1])
        
    @property
    def A(self):
        return self.mat.A
        
    def maxEigenvalue(self):
        if self.n==1: return self.delta[0]
    
        x0 = abs(self.mat).sum(axis=1).T.A[0].max()
        return self.newton(x0)
        
    def minEigenvalue(self): 
        if self.n==1: return self.delta[0]
        return self.newton(x0 = 0.)
        
    def eval_charPolynomial(self,lambda_):
        return algebra_cy.pyx_eval_charPolynomial(self.delta, self.gamma, lambda_)
        
    def newton(self, x0, maxiter=50, tol=1e-6):
        i = 0
        res = 1
        x_old = x0
        x_new = x0
        while(i < maxiter and res > tol):
            v, d = self.eval_charPolynomial(x_old)
            x_new = x_old - v/d
            res = np.abs(x_old - x_new)
            x_old = x_new
            i+=1
        return x_new
        #return algebra_cy.pyx_newton
        
def HilbertMatrix(n, return_inv = False):
    if return_inv:
        assert n < 11, "dimension of matrix too large to compute inverse exactly."
        return algebra_cy.pyx_HilbertMatrix(n), algebra_cy.pyx_HilbertMatrixInv(n)
    else:
        return algebra_cy.pyx_HilbertMatrix(n)

def CauchyMatrix(n, return_inv = False):
    assert n < 170, "dimension of matrix too large."
    if return_inv:
        assert n < 8, "dimension of matrix too large to compute inverse exactly."
        return algebra_cy.pyx_CauchyMatrix(n), algebra_cy.pyx_CauchyMatrixInv(n)
    else:
        return algebra_cy.pyx_CauchyMatrix(n)