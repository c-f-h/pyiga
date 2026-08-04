# -*- coding: utf-8 -*-
"""Functions and classes for algebraic operations.

"""

import numpy as np
import scipy.sparse
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
import scipy.linalg
from scipy.sparse.linalg import LinearOperator, onenormest, splu
from .operators import make_solver
from . import solvers
from . import algebra_cy
import time
#import numba as nb

def condest(A, spd=False, symm=False):
    iA = make_solver(A, spd=spd, symmetric=symm)
    return onenormest(iA)*onenormest(A)

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
        return self.newton(x0 = x0)
        
    def minEigenvalue(self): 
        if self.n==1: return self.delta[0]
        return self.newton(x0 = 1e-15)
        
    def eval_charPolynomial(self,lambda_):
        return algebra_cy.pyx_eval_charPolynomial(self.delta, self.gamma, lambda_)
        
    def newton(self, x0, maxiter=20, tol=1e-6):
        x_old = x0
        x_new = x0
        res=1
        i = 0

        while (i < maxiter) and (res > tol):
            v, d, _ = self.eval_charPolynomial(x_old)
            x_new = x_old - v/d
            res = abs(x_new - x_old)
            x_old = x_new
            i+=1
            
        # for i in range(maxiter):
        #     x_new = x_old - v/d
        #     if abs(x_new - x_old)<rtol:
        #         break
        #     if i<maxiter-1:
        #         v, d, _ = self.eval_charPolynomial(x_old)
        # if i==maxiter-1: 
        #     print("Eigenvalue computation did not converge!")
        return x_new

    def halley(self, x0, maxiter=20, tol=1e-6):
        x_old = x0
        v, d, d2 = self.eval_charPolynomial(x_old)

        for i in range(maxiter):
            x_new = x_old - (2*v*d)/(2*d**2-v*d2)
            if abs(x_new - x_old)<tol:
                break
            if i<maxiter-1:
                v, d, d2 = self.eval_charPolynomial(x_old)
        if i==maxiter-1: 
            print("Eigenvalue computation did not converge!")
        return x_new

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