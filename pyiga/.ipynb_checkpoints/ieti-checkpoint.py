import numpy as np
import time
import scipy
from scipy.sparse import coo_matrix, block_diag, identity, hstack, csr_matrix, csc_matrix, vstack

from pyiga import assemble, bspline, vform, geometry, vis, solvers, utils, topology, ieti, algebra, operators, adaptive
from pyiga import ieti_cy, algebra_cy
from scipy.sparse.linalg import aslinearoperator as LinOp

class IetiMapper(assemble.MultiBasis):
    def __init__(self, M, dir_data, neu_data=None, elim=False, **kwargs):
        super().__init__(M, dir_data, **kwargs)
        self.elim=bool(elim)

        if not self.subspace=='C0':
            self.B = self.set_constraints('C0')
        self.global_fixed_idx,_ = self.set_fixed_boundary(dir_data)

        self.free={}
        for p in range(self.nPatches):
            if p in self.fixed_idx:
                self.free[p] = np.setdiff1d(np.arange(self.N[p]),self.fixed_idx[p],assume_unique=True)
            else:
                self.free[p] = np.arange(self.N[p])
            
        self.global_free = np.setdiff1d(np.arange(self.N_ofs[-1]),self.global_fixed_idx, assume_unique=True)

        self.corners = np.concatenate([assemble.boundary_dofs(kvs,m=0,ravel=True)+self.N_ofs[p] for p, kvs in enumerate(self.mesh.kvs)])

        self.Bk = [self.B[:,self.N_ofs[p]:self.N_ofs[p+1]] for p in range(self.nPatches)]
        nnz_per_col = self.B.getnnz(axis=0)
        # self.intfs = np.setdiff1d(np.where(nnz_per_col > 0)[0], self.global_fixed_idx)
        self.skeleton = np.setdiff1d(np.where(nnz_per_col > 0)[0], self.global_fixed_idx)
        self.interior = np.setdiff1d(np.where(nnz_per_col == 0)[0], self.global_fixed_idx)
        
        self.R_interior = self.nPatches*[None] ###TODO: without loops
        self.R_skeleton = self.nPatches*[None]
        self.R_interfaces = {}
        
        for p in range(self.nPatches):
            Id = scipy.sparse.eye(self.N[p], format='csr')
            mask_skeleton = np.zeros(self.N[p], dtype=bool)
            intfs = np.where(self.Bk[p].getnnz(0) > 0)[0]
            mask_interior = np.ones(self.N[p], dtype=bool)
            mask_interior[intfs]=False
            if p in self.fixed_idx:
                mask_interior[self.fixed_idx[p]]=False
            self.R_interior[p]=Id[mask_interior,:][:,self.free[p]]
            for b in range(4):
                if not any([(p,b) in self.mesh.outer_boundaries[key] for key in self.mesh.outer_boundaries]):
                    mask_intf = np.zeros(self.N[p], dtype=bool)
                    interface_dofs = assemble.boundary_dofs(self.mesh.kvs[p],bdspec=b,ravel=True)
                    mask_intf[interface_dofs[1:-1]] = True
                    mask_skeleton[interface_dofs] = True
                    if p in self.fixed_idx:
                        mask_intf[self.fixed_idx[p]]=False
                        mask_skeleton[self.fixed_idx[p]]=False

                    self.R_interfaces[(p,b)] = Id[mask_intf,:][:,self.free[p]]
            self.R_skeleton[p] = Id[mask_skeleton,:][:,self.free[p]]
            
    def assemble(self, a, f, M):
        if M is None:
            M = {k:(0.0,0.0) for k in self.mesh.domains}
        if self.elim:
            A = [self.Basis.T @ assemble.assemble('a * inner(grad(u), grad(v)) * dx', kvs, a=a[self.mesh.patch_domains[k]], 
                                                  bfuns=[('u',1), ('v',1)], geo=geo) @ self.Basis for k, ((kvs, geo),_) in enumerate(self.mesh.patches)]
            RHS = [self.Basis.T @ assemble.assemble('f * v * dx', kvs, bfuns=[('v',1)], geo=geo, f=f[self.mesh.patch_domains[k]]).ravel() 
                   for k, ((kvs, geo),_) in enumerate(self.mesh.patches)]
        else:
            A = [assemble.assemble('a * inner(grad(u), grad(v)) * dx', kvs, a=a[self.mesh.patch_domains[k]], 
                                   bfuns=[('u',1), ('v',1)], geo=geo) for k, ((kvs, geo),_) in enumerate(self.mesh.patches)]
            RHS = [assemble.assemble('(f * v - inner(Ma_T,grad(v))) * dx', kvs, bfuns=[('v',1)], geo=geo, 
                                    f=f[self.mesh.patch_domains[k]], Ma_T = M[self.mesh.patch_domains[k]]).ravel() for k, ((kvs, geo),_) in enumerate(self.mesh.patches)]
        
        self.BCRestr = {p:assemble.RestrictedLinearSystem(A[p], RHS[p], (self.fixed_idx[p],self.fixed_vals[p])) for p in self.fixed_idx}
        RHS = [rhs if p not in self.fixed_idx else self.BCRestr[p].b for p, rhs in enumerate(RHS)]
        A = [a if p not in self.fixed_idx else self.BCRestr[p].A for p, a in enumerate(A)]

        return A, RHS

    def ConstraintMatrices(self, redundant = False):
        eliminated_constraints = np.repeat(False, self.B.shape[0])
        B = self.B.tocsc()
        if not redundant:
            eliminated_constraints = ieti_cy.eliminate_corner_constraints(B.indptr.astype(np.int32), B.indices.astype(np.int32), B.data, *B.shape, 
                                                                          self.corners.astype(np.int32), len(self.corners)).astype(bool)

        B = self.B[:,self.global_free]

        eliminated_constraints = eliminated_constraints | (B.getnnz(1)==0)
        B = B[~eliminated_constraints,:]
        ofs = np.cumsum([0]+[len(self.free[p]) for p in range(self.nPatches)])
        return [B[:,ofs[p]:ofs[p+1]] for p in range(self.nPatches)], eliminated_constraints

    def parametersort(self, a):
        D = np.array([a[key] for key in self.mesh.patch_domains.values()], dtype=float)
        ieti_cy.pyx_parametersort(self.B.indptr, self.B.indices, self.B.data, *self.B.shape, np.repeat(D,self.N))

    def nodes_as_primals(self, dir_boundary=False):  ###TODO: cythonize(?)
        """Get global vertices of the multipatch object as well as local nodal degrees of freedom corresponding to the vertices. 
        In case of T-junctions also obtain the $p$ global degrees of freedom and $p$ local degrees of freedom on the coarse patch.
        Additionally may include nodes on the Dirichlet boundary if desired."""
        deg = self.mesh.patches[0][0][0][0].p
        n = self.N_ofs[-1]
        loc_c = self.corners
        if not dir_boundary: loc_c = np.setdiff1d(loc_c, self.global_fixed_idx, assume_unique=True) 
  
        idx = (self.B[:,loc_c].getnnz(1)>0) & (self.B.getnnz(1)==2)
        B = self.B[idx,:]
        #B = B[B.getnnz(1)==2,:]
        loc_c = np.unique(B.indices)
        if not dir_boundary: loc_c = np.setdiff1d(loc_c, self.global_fixed_idx, assume_unique=True) 

        q = np.where(B.getnnz(0)>0)[0]
        R = scipy.sparse.coo_matrix((np.ones(len(q)),(np.arange(len(q)),q)),shape=(len(q),n)).tocsr()
        c_B = B@R.T
        c_B.eliminate_zeros()

        Basis , _ = algebra_cy.pyx_compute_basis(c_B.shape[0], c_B.shape[1], c_B, maxiter=100, switch=0)
        nodal_coeff = R.T@Basis
        #print(nodal_coeff)

        nodal_coeff += ieti_cy.identify_T_coefficients_from_corner_basis(nodal_coeff.indptr, nodal_coeff.indices, nodal_coeff.data, *nodal_coeff.shape,
                                                                         self.B.indptr,      self.B.indices,      self.B.data,       self.B.shape[0], deg)
        return nodal_coeff

    def interface_averages_as_primals(self, level=None):
        # def stepfunction(a,b):
        #     def f(x):
        #         if a<x and x<b: return 1.0
        #         else: return 0.0
        #     return np.vectorize(f)
        vv = []
        ii=[]
        jj=[]
        k=0
        #B = self.B.tocsc()
        for (p1,b1) in self.mesh.L_intfs:
            #supp1 = self.mesh.boundaries(p1)[1][b1]
            kv1 = assemble.boundary_kv(self.mesh.kvs[p1],b1)[0]
            left1, right1 = kv1.support()
            dofs1 = assemble.boundary_dofs(self.mesh.kvs[p1],b1,ravel=True)
            #P = B[:,dofs1+self.N_ofs[p1]].tocsr()
            for (p2, b2) in self.mesh.L_intfs[(p1,b1)]:
                kv2 = assemble.boundary_kv(self.mesh.kvs[p2],b2)[0]
                left2, right2 = kv2.support()
                print((right1-left1)/(right2-left2))
                if level is None or (level is not None and (right1-left1)/(right2-left2) < 2**level+1e-6):
                    dofs2 = assemble.boundary_dofs(self.mesh.kvs[p2],b2,ravel=True)
    
                    P = bspline.prolongation(kv1,kv2)
                    #a, b = self.Constr[(p1,p2)]
    
                    moments2 = assemble.assemble("v * ds", arity=1, kvs = self.mesh.kvs[p2], geo = self.mesh.geos[p2], boundary=b2).ravel()
                    #moments1 = assemble.assemble("v * ds", arity=1, kvs = self.mesh.kvs[p1], geo = self.mesh.geos[p1], boundary=b1).ravel()
    
                    vv.append(np.r_[P.T@moments2,moments2]/sum(moments2))
                    ii.append(np.r_[dofs1 + self.N_ofs[p1], dofs2+ self.N_ofs[p2]])
                    jj.append(np.repeat(k, len(dofs1)+len(dofs2)))
                    k+=1

        vv = np.concatenate(vv)
        ii = np.concatenate(ii)
        jj = np.concatenate(jj)

        Prim = csc_matrix((vv,(ii,jj)),(self.nLocDofs,k))
        Prim.eliminate_zeros()
        return Prim

    def completeDirichlet(self, U):
        return [self.BCRestr[p].complete(u) if p in self.BCRestr else u for p,u in enumerate(U)]

class PrimalSystem():
    def __init__(self, Prim, eliminate_constraints=False):
        self.Prim = Prim
        self.nPrim = self.Prim.shape[1]

        self.Psi = scipy.sparse.csc_matrix((0,self.nPrim))
        self.A_prim = scipy.sparse.csr_matrix(2*(self.nPrim,))
        self.RHS_prim = np.zeros(self.nPrim)
        self.R = []
        self.eliminate_constraints = eliminate_constraints ###not implemented yet

    def incorporate_PrimalConstraints(self, A, B, RHS, IMap):
        self.nLagrangeMultipliers = B[0].shape[0]
        
        #if self.nPrim == 0:
            #return A, B, RHS
        K = len(A)
        self.C=[]
        
        for p in range(K):
            c = (self.Prim[IMap.N_ofs[p]:IMap.N_ofs[p+1],:].T).tocsr()
            c.eliminate_zeros()
            jj = np.where((c.indptr[1:]-c.indptr[:-1])>0)[0]
            c = c[:,IMap.free[p]]
            c = c[jj,:]
            #diff = np.linalg.matrix_rank(c.toarray())==c.shape[0]
            self.C.append(c)
            self.R.append(scipy.sparse.coo_matrix((np.ones(c.shape[0]),(np.arange(c.shape[0]),jj)),(c.shape[0],self.nPrim)))
            assert np.linalg.matrix_rank(c.toarray())==c.shape[0], "Local saddle point system not full row rank in patch "+str(p)+": number of rows: "+str(c.shape[0])+", row rank: "+str(np.linalg.matrix_rank(c.toarray()))+" ."
        #assert np.all(np.array([np.linalg.matrix_rank(c.toarray())==c.shape[0] for c in self.C if c.shape[0]!=0])), "Local saddle point system not full rank."
            
        self.nPrimConstr = [c.shape[0] for c in self.C]

        #if self.eliminate_constraints:
            #return A, B, RHS, self.C

        mod_A = [scipy.sparse.bmat([[A[p],self.C[p].T],[self.C[p], None]]) for p in range(K)]
        mod_RHS = [np.concatenate([RHS[p],np.zeros(self.nPrimConstr[p])]) for p in range(K)]

        mod_B = [scipy.sparse.hstack([B[p],scipy.sparse.csr_matrix((self.nLagrangeMultipliers, self.nPrimConstr[p]))]) for p in range(K)]
        return mod_A, mod_B, mod_RHS, self.C

    def compute_PrimalBasis(self, mod_A, mod_B, mod_RHS, C):
        K = len(mod_A)
        self.Psi = []
        Delta = []
    
        A_prim = []
        B_prim = []
        loc_solvers = K*[None]
    
        for p in range(K):    
            
            loc_solvers[p] = solvers.make_solver(mod_A[p],spd=False, symm=True)

            n_constr = self.nPrimConstr[p]
            n_total = loc_solvers[p].shape[0]
            n_free = n_total - n_constr

            if n_constr == 0:
                self.Psi.append(csr_matrix((0,self.Prim.shape[1])))
                Delta.append(None)
    
            # Build RHS: stacked [0; I]
            RHS = np.zeros((n_total, n_constr))
            RHS[n_free:, :] = np.eye(n_constr)
    
            # Solve with dense RHS (assumed requirement of solver)
            sol = loc_solvers[p] @ RHS
            psi = sol[:n_free, :]
            delta = sol[n_free:, :]
    
            self.Psi.append(csr_matrix(psi @ self.R[p]))
            Delta.append(csr_matrix(delta))

            #A_prim.append(self.R[p].T @ Delta[-1] @ self.R[p])
            A_prim.append(self.Psi[-1].T@(csr_matrix(mod_A[p])[:,:-self.nPrimConstr[p]][:-self.nPrimConstr[p],:])@self.Psi[-1])
            B_prim.append(mod_B[p][:, :n_free] @ self.Psi[-1])
            self.RHS_prim += self.Psi[-1].T@mod_RHS[p][:-self.nPrimConstr[p]]
    
        # One-time sparse sum
        #self.A_prim = -1*sum(A_prim)
        self.A_prim = sum(A_prim)
        self.B_prim = sum(B_prim)
        #self.RHS_prim += sum(RHS_prim_contribs)
    
        return loc_solvers

    def distributePrimalSolution(self, u):
        u_prim = u[-1]
        return [u[p]+self.Psi[p]@u_prim for p in range(len(u)-1)]

    def PrimalSolution(self, u_prim):
        return [psi@u_prim for psi in self.Psi]

class IetiSystem():
    def __init__(self, A, B ,RHS, N, loc_solver = None, spd=False, symm=False):
        self.A = A
        self.B = B
        self.RHS = RHS
        self.N = N

        self.sanity_check()

        if loc_solver:
            assert len(loc_solver)==len(A), 'amount of local solvers does not match amount of local system matrices!'
            self.loc_solver=loc_solver
        else:
            self.loc_solver = [solvers.make_solver(a, spd=spd, symm=symm) for a in self.A]

        self.K = len(A)
        self.nLagrangeMultipliers = self.B[0].shape[0]

    def SaddlePointSystem(self, format='csr'):
        B = scipy.sparse.hstack(self.B)
        return scipy.sparse.bmat([[scipy.sparse.block_diag(self.A),B.T],[B,None]], format=format)

    def RHSforSaddlePointSystem(self):
        return np.concatenate(self.RHS+[np.zeros(self.nLagrangeMultipliers)])

    def SchurComplement(self, as_matrix=False):
        F = operators.SumOperator([LinOp(self.B[p])@self.loc_solver[p]@LinOp(self.B[p].T) for p in range(self.K)])
        if as_matrix:
            return F@np.identity(F.shape[1])
        return F

    def RHSforSchurComplement(self):
        #print("any NaN in RHS?", np.any([np.isnan(self.RHS[p]).any() for p in range(self.K)]))
        return np.sum([self.B[p]@(self.loc_solver[p](self.RHS[p])) for p in range(self.K)], axis=0)

    def constructSolutionFromLagrangeMultipliers(self, lam):
        return [(self.loc_solver[p]@(self.RHS[p]-self.B[p].T@lam))[:self.N[p]] for p in range(self.K)]
        
    def sanity_check(self):
        assert len(self.A)==len(self.B)==len(self.RHS), 'Length of input data incompatible!'
        K = len(self.A)
        assert np.all([self.B[0].shape[0]==self.B[p].shape[0] for p in range(K)]), 'Constraint matrices have incompatible number of constraints!'
        assert np.all([self.A[p].shape[0]==self.A[p].shape[1] for p in range(K)]), 'Local system matrices are not square!'
        assert np.all([self.B[p].shape[1] == self.A[p].shape[1] for p in range(K)]), 'Constraint matrices have incompatible dimension!'
        assert np.all([self.A[p].shape[0]==len(self.RHS[p]) for p in range(K)]), 'Local rhs vectors have incompatible dimension!'

class ScaledDirichletPreconditioner():
    def __init__(self, A ,B, IMap):
        self.A = A
        self.IMap = IMap
        self.K   = len(B)

        self.B = [B[p]@IMap.R_skeleton[p].T for p in range(self.K)]
        self.B_full   = scipy.sparse.hstack(self.B, format='csr')
        self.BN  = np.array([IMap.R_skeleton[p].shape[0] for p in range(self.K)]) 
        self.BN_ofs = np.cumsum(np.r_[0,self.BN])

        self.D = []
        self.S = [LinOp(IMap.R_skeleton[p]@A[p]@IMap.R_skeleton[p].T) - 
                  LinOp(IMap.R_skeleton[p]@A[p]@IMap.R_interior[p].T)
                  @solvers.make_solver(IMap.R_interior[p]@A[p]@IMap.R_interior[p].T, spd=True)
                  @LinOp(IMap.R_interior[p]@A[p].T@IMap.R_skeleton[p].T) for p in range(self.K)]

    def SchurMatrices(self): #for large number of degrees of freedom not feasible
        return [s@np.eye(s.shape[0]) for s in self.S]

    def setupMultiplicityScaling(self):
        d = ieti_cy.pyx_multiplicity_scaling(self.B_full.indptr, self.B_full.indices, self.B_full.data, *self.B_full.shape)
        self.D = [scipy.sparse.diags(d[self.BN_ofs[p]:self.BN_ofs[p+1]], format='csr') for p in range(self.K)] 
        
    # def setupConstraintScaling(self):
    #     d = ieti_cy.pyx_constraint_scaling(self.B_full.indptr, self.B_full.indices, self.B_full.data, *self.B_full.shape)
    #     self.D = [scipy.sparse.diags(d[self.BN_ofs[p]:self.BN_ofs[p+1]], format='csr') for p in range(self.K)] 

    def setupWeightScaling(self):
        d = ieti_cy.pyx_weight_scaling(self.B_full.indptr, self.B_full.indices, self.B_full.data, *self.B_full.shape)
        self.D = [scipy.sparse.diags(d[self.BN_ofs[p]:self.BN_ofs[p+1]], format='csr') for p in range(self.K)] 
        
    def setupCoefficientScaling(self, a):  ###TODO: cythonize
        self.D = [scipy.sparse.csr_matrix((self.B[p].shape[1],self.B[p].shape[1])) for p in range(self.K)]
        for (p1,b1) in self.IMap.mesh.L_intfs:
            R1 = self.IMap.R_interfaces[(p1,b1)]@self.IMap.R_skeleton[p1].T
            for (p2,b2) in self.IMap.mesh.L_intfs[(p1,b1)]:
                R2 = self.IMap.R_interfaces[(p2,b2)]@self.IMap.R_skeleton[p2].T
                a1, a2 = a[self.IMap.mesh.patch_domains[p1]], a[self.IMap.mesh.patch_domains[p2]]
                self.D[p1] += (a2)/(a1+a2)*R1.T@scipy.sparse.identity(R1.shape[0])@R1
                self.D[p2] += (a1)/(a1+a2)*R2.T@scipy.sparse.identity(R2.shape[0])@R2
        
    def setupSelectionScaling(self):
        assert isinstance(self.B_full, scipy.sparse.csr_matrix)
        d = ieti_cy.pyx_selection_scaling(self.B_full.indptr, self.B_full.indices, self.B_full.data, *self.B_full.shape)
        self.D = [scipy.sparse.diags(d[self.BN_ofs[p]:self.BN_ofs[p+1]], format='csr') for p in range(self.K)] 

    def setupDeluxeScaling(self):  ###TODO: cythonize and/or write it as solution of sparse systems (no computation of Schur matrices)
        self.D = [np.zeros((self.B[p].shape[1],self.B[p].shape[1])) for p in range(self.K)]
        S = self.SchurMatrices()
        for (p1,b1) in self.IMap.L_intfs:
            R1 = self.IMap.R_interfaces[(p1,b1)]@self.IMap.R_skeleton[p1].T
            S1 = R1@S[p1]@R1.T
            for (p2,b2) in self.IMap.L_intfs[(p1,b1)]:
                R2 = self.IMap.R_interfaces[(p2,b2)]@self.IMap.R_skeleton[p2].T
                S2 = R2@S[p2]@R2.T
                Inv = np.linalg.inv(S1+S2)
                self.D[p1]+=R1.T@(Inv@S2)@R1
                self.D[p2]+=R2.T@(Inv@S1)@R2
        
    def prec(self):
        assert len(self.D)==self.K, 'Not all scaling matrices given! Call a setup routine first!'
        self.BgD = [LinOp(self.B[p]@self.D[p]) for p in range(self.K)]
        B = scipy.sparse.hstack(self.B)
        print("Convergence condition: {:.3}".format(scipy.sparse.linalg.norm(B@scipy.sparse.block_diag(self.D).T@B.T@B-B))) #check Algebraic condition
        return operators.SumOperator([self.BgD[p]@self.S[p]@self.BgD[p].T for p in range(self.K)])     

def EdgePreconditionerFull(IMap, S, B, C):
    S_ = [np.zeros(s.shape) for s in S]
    B_ = len(B)*[None]
    C_ = len(C)*[None]
    for p in range(IMap.nPatches):
        C_[p]=C[p]@IMap.R_skeleton[p].T
        B_[p] = scipy.sparse.hstack([B[p],scipy.sparse.csr_matrix((B[p].shape[0],C_[p].shape[0]))])
        for b in range(4):
            if (p,b) in IMap.R_interfaces:
                R = IMap.R_interfaces[(p,b)]@IMap.R_skeleton[p].T
                S_[p] += R.T@R@S[p]@R.T@R
        #print(S_[p].shape, C_[p].shape)
    return np.linalg.inv(np.array([B_[p]@np.linalg.inv(scipy.sparse.block_array([[S_[p],C_[p].T],[C_[p],np.zeros(2*(C_[p].shape[0],))]]).toarray())@B_[p].T for p in range(len(S))]).sum(axis=0))
    
def EdgePreconditioner(IMap, S, B, C):
    m = B[0].shape[0]
    MsD = np.zeros((m,m))
    for (p1,b1) in IMap.mesh.L_intfs:
        R1 = IMap.R_interfaces[(p1,b1)]@IMap.R_skeleton[p1].T
        c = C[p1]@IMap.R_interfaces[(p1,b1)].T
        c = c[c.getnnz(1)>0,:].toarray()
        if c.shape[0]>0:
            S1_inv = np.linalg.inv(np.block([[R1@S[p1]@R1.T,c.T],[c, np.zeros((c.shape[0], c.shape[0]))]]))[:c.shape[1],:c.shape[1]]
        else:
            S1_inv = np.linalg.inv(R1@S[p1]@R1.T)
        S2_inv = []
        P = []
        B2 = []
        for (p2,b2) in IMap.mesh.L_intfs[(p1,b1)]:
            R2 = IMap.R_interfaces[(p2,b2)]@IMap.R_skeleton[p2].T
            S2_inv.append(np.linalg.inv(R2@S[p2]@R2.T))
            X = (B[p2]@R2.T)
            constr = np.where(X.getnnz(1)>0)[0]
            B2.append(B[p2]@R2.T)
            P.append(B[p1][constr,:]@R1.T)
        S2_inv = scipy.sparse.block_diag(S2_inv).toarray()
        P = scipy.sparse.vstack(P)
        B2 = scipy.sparse.hstack(B2)
        
        MsD+=B2@np.linalg.inv(S2_inv + P@S1_inv@P.T)@B2.T
    return MsD

def EdgePreconditioner2(IMap, S, B, C):
    m = B[0].shape[0]
    MsD = np.zeros((m,m))
    for (p1,b1) in IMap.L_intfs:
        R1 = IMap.R_interfaces[(p1,b1)]@IMap.R_skeleton[p1].T
        c = C[p1]@IMap.R_interfaces[(p1,b1)].T
        c = c[c.getnnz(1)>0,:].toarray()
        if c.shape[0]>0:
            S1 = np.block([[R1@S[p1]@R1.T,c.T],[c, np.zeros((c.shape[0], c.shape[0]))]])
        else:
            S1 = R1@S[p1]@R1.T
        S2 = []
        P = []
        B2 = []
        for (p2,b2) in IMap.L_intfs[(p1,b1)]:
            R2 = IMap.R_interfaces[(p2,b2)]@IMap.R_skeleton[p2].T
            S2.append(R2@S[p2]@R2.T)
            X = (B[p2]@R2.T)
            constr = np.where(X.getnnz(1)>0)[0]
            B2.append(B[p2]@R2.T)
            P.append(B[p1][constr,:]@R1.T)
        S2 = scipy.sparse.block_diag(S2).toarray()
        P = scipy.sparse.vstack(P)
        B2 = scipy.sparse.hstack(B2)
        MsD+=B2@(S2 + P@S1@P.T)@B2.T
    return MsD
    
def EdgePreconditionerOP(IMap, A, B, C):
    M = len(IMap.L_intfs)*[None]

    for i,(p1, b1) in enumerate(IMap.L_intfs):
        R1 = scipy.sparse.vstack([
            IMap.R_interior[p1],
            IMap.R_interfaces[(p1, b1)]
        ], format='csr')
        c = C[p1] @ R1.T
        c = c[c.getnnz(axis=1) > 0, :]

        A_blocks = [scipy.sparse.bmat(
            [[R1 @ A[p1] @ R1.T, c.T],
             [c, None]], format='csc')]
        B_blocks = [scipy.sparse.hstack([
            B[p1] @ R1.T,
            scipy.sparse.csc_matrix((B[p1].shape[0], c.shape[0]))
        ], format='csc')]
        P_blocks = []

        for (p2, b2) in IMap.L_intfs[(p1, b1)]:
            R2 = scipy.sparse.vstack([
                IMap.R_interior[p2],
                IMap.R_interfaces[(p2, b2)]
            ], format='csr')

            A_blocks.append(R2 @ A[p2] @ R2.T)
            B_blocks.append(B[p2] @ R2.T)
            P_blocks.append(B[p2] @ IMap.R_interfaces[(p2, b2)].T)

        A_ = scipy.sparse.block_diag(A_blocks, format='csc')
        B_ = scipy.sparse.hstack(B_blocks, format='csc')
        P = scipy.sparse.hstack(P_blocks, format='csc') 

        B_ = B_.tocsr()
        r_idx = B_.getnnz(axis=1) > 0
        B_=B_[r_idx,:]
        n_c = r_idx.sum()

        Mat = -scipy.sparse.bmat([
            [A_, B_.T],
            [B_, None]
        ], format='csc')

        B_coarse=B_.tocsc()
        B_coarse = B_coarse[:, (R1.shape[0] + c.shape[0]):]
        c_idx = B_coarse.getnnz(axis=0) > 0

        n_total = Mat.shape[0]
        X = LinOp(
            scipy.sparse.hstack([
                scipy.sparse.csc_matrix((P.shape[0], n_total - n_c)),
                P @ B_coarse[:, c_idx].T
            ])
        )
        solver = solvers.make_solver(Mat, symm=True, spd=False)
        M[i] = X @ solver @ X.T

    return operators.SumOperator(M)
    
# def EdgePreconditionerFull(IMap, S, B, C):
#     S_ = [np.zeros(s.shape) for s in S]
#     B_ = len(B)*[None]
#     C_ = len(C)*[None]
#     for p in range(IMap.numpatches):
#         C_[p]=C[p]@IMap.R_skeleton[p].T
#         B_[p] = scipy.sparse.hstack([B[p],scipy.sparse.csr_matrix((B[p].shape[0],C_[p].shape[0]))])
#         for b in range(4):
#             if (p,b) in IMap.R_interfaces:
#                 R = IMap.R_interfaces[(p,b)]@IMap.R_skeleton[p].T
#                 S_[p] += R.T@R@S[p]@R.T@R
#         #print(S_[p].shape, C_[p].shape)
#     return np.linalg.inv(np.array([B_[p]@np.linalg.inv(scipy.sparse.block_array([[S_[p],C_[p].T],[C_[p],np.zeros(2*(C_[p].shape[0],))]]).toarray())@B_[p].T for p in range(len(S))]).sum(axis=0))
    
# def EdgePreconditioner(IMap, S, B, C):
#     m = B[0].shape[0]
#     MsD = np.zeros((m,m))
#     for (p1,b1) in IMap.L_intfs:
#         R1 = IMap.R_interfaces[(p1,b1)]@IMap.R_skeleton[p1].T
#         c = C[p1]@IMap.R_interfaces[(p1,b1)].T
#         c = c[c.getnnz(1)>0,:].toarray()
#         if c.shape[0]>0:
#             S1_inv = np.linalg.inv(np.block([[R1@S[p1]@R1.T,c.T],[c, np.zeros((c.shape[0], c.shape[0]))]]))
#         else:
#             S1_inv = np.linalg.inv(R1@S[p1]@R1.T)
#         S2_inv = []
#         P = []
#         B2 = []
#         for (p2,b2) in IMap.L_intfs[(p1,b1)]:
#             R2 = IMap.R_interfaces[(p2,b2)]@IMap.R_skeleton[p2].T
#             S2_inv.append(np.linalg.inv(R2@S[p2]@R2.T))
#             X = (B[p2]@R2.T)
#             constr = np.where(X.getnnz(1)>0)[0]
#             B2.append(B[p2]@R2.T)
#             P.append(B[p1][constr,:]@R1.T)
#         S2_inv = scipy.sparse.block_diag(S2_inv).toarray()
#         P = scipy.sparse.vstack(P)
#         B2 = scipy.sparse.hstack(B2)
#         MsD+=B2@np.linalg.inv(S2_inv + P@S1_inv@P.T)@B2.T
#     return MsD
    
# def EdgePreconditionerOP(IMap, A, B, C):
#     M = len(IMap.L_intfs)*[None]

#     for i,(p1, b1) in enumerate(IMap.L_intfs):
#         R1 = scipy.sparse.vstack([
#             IMap.R_interior[p1],
#             IMap.R_interfaces[(p1, b1)]
#         ], format='csr')
#         c = C[p1] @ R1.T
#         c = c[c.getnnz(axis=1) > 0, :]

#         A_blocks = [scipy.sparse.bmat(
#             [[R1 @ A[p1] @ R1.T, c.T],
#              [c, None]], format='csc')]
#         B_blocks = [scipy.sparse.hstack([
#             B[p1] @ R1.T,
#             scipy.sparse.csc_matrix((B[p1].shape[0], c.shape[0]))
#         ], format='csc')]
#         P_blocks = []

#         for (p2, b2) in IMap.L_intfs[(p1, b1)]:
#             R2 = scipy.sparse.vstack([
#                 IMap.R_interior[p2],
#                 IMap.R_interfaces[(p2, b2)]
#             ], format='csr')

#             A_blocks.append(R2 @ A[p2] @ R2.T)
#             B_blocks.append(B[p2] @ R2.T)
#             P_blocks.append(B[p2] @ IMap.R_interfaces[(p2, b2)].T)

#         A_ = scipy.sparse.block_diag(A_blocks, format='csc')
#         B_ = scipy.sparse.hstack(B_blocks, format='csc')
#         P = scipy.sparse.hstack(P_blocks, format='csc') 

#         B_ = B_.tocsr()
#         r_idx = B_.getnnz(axis=1) > 0
#         B_=B_[r_idx,:]
#         n_c = r_idx.sum()

#         Mat = -scipy.sparse.bmat([
#             [A_, B_.T],
#             [B_, None]
#         ], format='csc')

#         B_coarse=B_.tocsc()
#         B_coarse = B_coarse[:, (R1.shape[0] + c.shape[0]):]
#         c_idx = B_coarse.getnnz(axis=0) > 0

#         n_total = Mat.shape[0]
#         X = LinOp(
#             scipy.sparse.hstack([
#                 scipy.sparse.csc_matrix((P.shape[0], n_total - n_c)),
#                 P @ B_coarse[:, c_idx].T
#             ])
#         )
#         solver = solvers.make_solver(Mat, symm=True, spd=False)
#         M[i] = X @ solver @ X.T

#     return operators.SumOperator(M)