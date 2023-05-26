import numpy as np
import scipy.sparse

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
    testVec = np.zeros(n)
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
    nddofs=np.setdiff1d(np.arange(n),ddofs)   #still free dofs
    n1=len(nddofs)

    lBasis=scipy.sparse.csr_matrix((n,n))
    c = 1/(Constr[r,ddofs].A.ravel())
    B1 = scipy.sparse.csr_matrix(scipy.sparse.coo_matrix((np.ones(n1),(np.arange(n1),nddofs)),(n1,n)))
    B2 = - Constr[r].multiply(Constr[r]<0)  
    B2 = scipy.sparse.spdiags(c,0,n2,n2)@B2  #row scaling
    
    Q1 = scipy.sparse.coo_matrix((np.ones(n1),(nddofs,np.arange(n1))),(n,n1))
    Q2 = scipy.sparse.coo_matrix((np.ones(n2),(ddofs,np.arange(n2))),(n,n2))
    lBasis = Q1@B1 + Q2@B2
    
    lBasis = lBasis @ Basis
    
    lastFound = n+1
    #print(len(ddofs))
    testVec[ddofs]=1 
        
    while True:
        found = 0
        tmp = lBasis @ testVec
        # for i in range(n):
        #     if abs(tmp[i])>1e-12:
        #         #print('{}'.format(i))
        #         found += 1
        found = sum(abs(tmp)>1e-12)
        
        # NEW
        # for i in ddofs:
        #     for j in ddofs:
        #         if abs(lBasis[i,j]) > 1e-12:
        #             found = 1
        # ENDNEW
        
        if found > 0:
            #print(found, lastFound)
            #assert(found < lastFound)
            lastFound = found
            lBasis = lBasis @ lBasis
            #print("multiply & repeat")
        else:
            #print("done")
            break
    return lBasis

def compute_active_constr(Constr):
    #fast variant
    
    # a=(Constr>1e-12).sum(axis=1).A.ravel(); b=(Constr<-1e-12).sum(axis=1).A.ravel()
    # activeConstraints=np.where(a+b>0)[0]
    # sign = np.where(b==1 & a>0)
    
#     signs=1.*((a<=1) | (a+b==0))-1.*((a>1)&(a+b>0))
#     #print(signs)
#     S=scipy.sparse.spdiags(signs,0,len(a),len(a))
    
#     Constr = S@Constr
    
    # assert np.all(((a[activeConstraints]==1) | (b[activeConstraints]==1))), 'error in constraint matrix.'

    #variant 1
    activeConstraints=[]
    for r in range(Constr.shape[0]):
        a = 0
        b = 0
        for ind in range(Constr.indptr[r], Constr.indptr[r+1]):
            if Constr.data[ind] > 1e-12:
                a += 1
            if Constr.data[ind] < -1e-12:
                b += 1
        if a+b>0:
            activeConstraints.append(r)
            #print("{}: {}, {}".format(r,a,b))
            #if not (a==1 or b==1): 
                #print(a,b)
                #print(Constr[r,:])
            #assert (a==1 or b==1), 'error in constraint matrix.'
            if b==1 and a>0:
                #print( "Re-sign" )
                Constr[r,:] *= -1
                
    return np.array(activeConstraints)

def compute_basis(Constr, maxiter):
    n=Constr.shape[1]
    nonderivedDofs = allLocalDofs=np.arange(n)
    allderivedDofs={}
    activeConstraints=np.arange(Constr.shape[0])
    Basis=scipy.sparse.csr_matrix(scipy.sparse.identity(n))
    i=1
    while len(activeConstraints)!=0:
        #print(Constr[activeConstraints])
        if i>maxiter:
            print("maxiter reached.")
            break
            
        derivedDofs = find_ddofs(Constr, activeConstraints)  
        #assert derivedDofs, 'Unable to derive any further dofs.'
        assert derivedDofs, 'Unable to derive further dofs.'
        #if not derivedDofs: 
        #    print(derivedDofs)
        #    break
        
        #update which dofs were already derived
        allderivedDofs.update(derivedDofs)
        nonderivedDofs=np.setdiff1d(allLocalDofs, np.array(list(allderivedDofs.keys())))
        Basis = update_basis(Constr, derivedDofs, Basis)
        
        #eliminate used constraints from constraint matrix
        Constr = Constr @ Basis    
        activeConstraints = compute_active_constr(Constr)
            
        i+=1
        
    #print(np.array(list(allderivedDofs.keys())))
    #nonderivedDofs=np.setdiff1d(allLocalDofs, np.array(list(allderivedDofs.keys())))
    #print(nonderivedDofs)
    return Basis[:,nonderivedDofs]  #,Constr,activeConstraints

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