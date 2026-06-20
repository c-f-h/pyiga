import numpy as np
import time
import scipy
from scipy.sparse import coo_matrix, block_diag, identity, hstack, csr_matrix, csc_matrix, vstack

from pyiga import assemble, bspline, vform, geometry, vis, solvers, utils, topology, ieti, algebra, operators, adaptive
from pyiga import ieti_cy, algebra_cy
from scipy.sparse.linalg import aslinearoperator as LinOp

class IetiMapper(assemble.MultiBasis):
    def __init__(self, M, dir_data, neu_data=None, elim=False, **kwargs):
        super().__init__(M, dir_data=dir_data, **kwargs)
        self.elim=bool(elim)

        if not self.subspace=='C0':
            if self.elim:
                self.set_subspace(subspace='C0')
            else:
                self.B = self.set_constraints('C0')

        self.global_fixed_idx,_ = self.set_fixed_boundary(dir_data, local=True) 

        self.N_elim = self.N
        self.N_ofs_elim = self.N_ofs
        if self.elim: 
            self.X=[]
            p_intfs = np.array([[p1,p2] for (p1,_),(p2,_),_ in self.intfs], dtype=np.int32).T
            self.Basis_loc, self.N_ofs_elim, self.N_elim, self.B = ieti_cy.pyx_compute_decoupled_coarse_basis(self.Basis.tocsc(), self.N_ofs.astype(np.int32), p_intfs)
            self.Basis_loc_global = scipy.sparse.block_diag(self.Basis_loc)
            
            for p in range(self.nPatches):
                X = self.Basis_loc[p].T.tocoo()

                mask = np.isclose(X.data, 1)

                col_to_row = np.full(X.shape[1], -1, dtype=np.int64)
                col_to_row[X.col[mask]] = X.row[mask]
                
                self.X.append(col_to_row)

                idx = self.X[p][self.fixed_idx[p]]

                if np.any(idx < 0):
                    raise ValueError("Some fixed indices are not present in the selection matrix.")
                lookup = np.argsort(idx)
                self.fixed_idx[p] = self.fixed_idx[p][lookup]
                self.fixed_vals[p] = self.fixed_vals[p][lookup] 

            self.global_fixed_idx = np.concatenate([self.fixed_idx[p] + self.N_ofs_elim[p] for p in self.fixed_idx])
  
        self.global_free = np.setdiff1d(np.arange(self.N_ofs_elim[-1]),self.global_fixed_idx, assume_unique=True)

        self.N_free = [self.N_elim[p] - len(self.fixed_idx.get(p, ())) for p in range(self.nPatches)]
        self.N_ofs_free = np.cumsum([0]+self.N_free)

        self.free={}
        for p in range(self.nPatches):
            if p in self.fixed_idx:
                self.free[p] = np.setdiff1d(np.arange(self.N_elim[p]),self.fixed_idx[p],assume_unique=True)
            else:
                self.free[p] = np.arange(self.N_elim[p])

        if self.elim:
            self.corners = np.concatenate([self.X[p][assemble.boundary_dofs(kvs,m=0,ravel=True)] + self.N_ofs_elim[p] for p, kvs in enumerate(self.mesh.kvs)])
        else:
            self.corners = np.concatenate([assemble.boundary_dofs(kvs,m=0,ravel=True) + self.N_ofs_elim[p] for p, kvs in enumerate(self.mesh.kvs)])

        self.Bk = [self.B[:,self.N_ofs_elim[p]:self.N_ofs_elim[p+1]] for p in range(self.nPatches)]
        nnz_per_col = self.B.getnnz(axis=0)

        self.skeleton = np.setdiff1d(np.where(nnz_per_col > 0)[0], self.global_fixed_idx)
        self.interior = np.setdiff1d(np.where(nnz_per_col == 0)[0], self.global_fixed_idx)

        self.DOFS_interior = self.nPatches*[None]
        self.DOFS_skeleton = self.nPatches*[None]
        self.DOFS_interfaces  = {}
        self.DOFS_corners = self.nPatches*[None]

        def to_free(idx_full, fixed):
            if fixed is None:
                return idx_full
            return idx_full - np.searchsorted(fixed, idx_full)
        
        for p in range(self.nPatches):
            fixed = self.fixed_idx.get(p)
            mask_skeleton = np.zeros(self.N_elim[p], dtype=bool)
            mask_corners = np.zeros(self.N_elim[p], dtype=bool)
            intfs = np.where(self.Bk[p].getnnz(0) > 0)[0]
            mask_interior = np.ones(self.N_elim[p], dtype=bool)
            mask_interior[intfs]=False

            cdofs = assemble.boundary_dofs(self.mesh.kvs[p],m=0,ravel=True)
            
            if self.elim:
                cdofs = self.X[p][cdofs]

            mask_corners[cdofs]=True
            if fixed is not None:
                mask_interior[fixed]=False
                mask_corners[fixed]=False
            self.DOFS_interior[p] = to_free(np.flatnonzero(mask_interior), fixed)
            self.DOFS_corners[p] = to_free(np.flatnonzero(mask_corners), fixed)

            for b in range(4):
                if not any([(p,b) in self.mesh.outer_boundaries[key] for key in self.mesh.outer_boundaries]):
                    mask_intf = np.zeros(self.N_elim[p], dtype=bool)
                    interface_dofs = assemble.boundary_dofs(self.mesh.kvs[p],bdspec=b,ravel=True)[1:-1]
                    if self.elim:
                        interface_dofs = self.X[p][interface_dofs]
                    mask_intf[interface_dofs] = True
                    mask_skeleton[interface_dofs] = True
                    if fixed is not None:
                        mask_intf[fixed]=False
                        mask_skeleton[fixed]=False

                    self.DOFS_interfaces[(p,b)] = to_free(np.flatnonzero(mask_intf), fixed)

            self.DOFS_skeleton[p] = to_free(np.flatnonzero(mask_skeleton), fixed)

    def assemble(self, a, f, M):
        if M is None:
            M = {k:(0.0,0.0) for k in self.mesh.domains}
        if a is None:
            a = {k:1 for k in self.mesh.domains}
        if isinstance(a, (int, float)):
            a = {k:a for k in self.mesh.domains}
        if isinstance(f, (int, float)):
            f = {k:f for k in self.mesh.domains}
        if self.elim:
            A = [self.Basis_loc[k].T @ assemble.assemble('a * inner(grad(u), grad(v)) * dx', kvs, a=a[self.mesh.patch_domains[k]], 
                                                  bfuns=[('u',1), ('v',1)], geo=geo) @ self.Basis_loc[k] for k, ((kvs, geo),_) in enumerate(self.mesh.patches)]
            RHS = [self.Basis_loc[k].T @ assemble.assemble('f * v * dx', kvs, bfuns=[('v',1)], geo=geo, f=f[self.mesh.patch_domains[k]]).ravel() 
                   for k, ((kvs, geo),_) in enumerate(self.mesh.patches)]
        else:
            A = [assemble.assemble('a * inner(grad(u), grad(v)) * dx', kvs, a=a[self.mesh.patch_domains[k]], 
                                   bfuns=[('u',1), ('v',1)], geo=geo) for k, ((kvs, geo),_) in enumerate(self.mesh.patches)]
            RHS = [assemble.assemble('f * v  * dx', kvs, bfuns=[('v',1)], geo=geo, 
                                    f=f[self.mesh.patch_domains[k]], Ma_T = M[self.mesh.patch_domains[k]]).ravel() for k, ((kvs, geo),_) in enumerate(self.mesh.patches)]
        
        self.BCRestr = {p:assemble.RestrictedLinearSystem(A[p], RHS[p], (self.fixed_idx[p],self.fixed_vals[p])) for p in self.fixed_idx}
        RHS = [rhs if p not in self.fixed_idx else self.BCRestr[p].b for p, rhs in enumerate(RHS)]
        A = [a if p not in self.fixed_idx else self.BCRestr[p].A for p, a in enumerate(A)]

        return A, RHS

    def ConstraintMatrices(self, eliminate = None):
        if self.elim:
            return [self.Bk[p][:,self.free[p]] for p in range(self.nPatches)], np.repeat(False,self.nConstr)
        if eliminate is not None:
            eliminated_constraints = eliminate
        else:
            eliminated_constraints = np.repeat(False,self.nConstr)
        B = self.B.tocsc()
        # if not corners:
        #     eliminated_constraints = ieti_cy.eliminate_corner_constraints(B.indptr.astype(np.int32), B.indices.astype(np.int32), B.data, *B.shape, 
        #                                                                   self.corners.astype(np.int32), len(self.corners)).astype(bool)

        B = self.B[:,self.global_free]
        
        eliminated_constraints = eliminated_constraints | (B.getnnz(1)==0)
        B = B[~eliminated_constraints,:]
        return [B[:,self.N_ofs_free[p] : self.N_ofs_free[p+1]] for p in range(self.nPatches)], eliminated_constraints

    def parametersort(self, a):
        D = np.array([a[key] for key in self.mesh.patch_domains.values()], dtype=float)
        ieti_cy.pyx_parametersort(self.B.indptr, self.B.indices, self.B.data, *self.B.shape, np.repeat(D,self.N_elim))

    def nodes_as_primals(self, dir_boundary=False):  ###TODO: cythonize(?)
        """Get global vertices of the multipatch object as well as local nodal degrees of freedom corresponding to the vertices. 
        In case of T-junctions also obtain the $p$ global degrees of freedom and $p$ local degrees of freedom on the coarse patch.
        Additionally may include nodes on the Dirichlet boundary if desired."""
        if self.elim:
            to_be_eliminated = np.zeros(self.B.shape[0],dtype=bool)
            n = self.N_ofs_elim[-1]
            q = np.where(self.B.getnnz(0)>1)[0]
            #q = np.setdiff1d(q,self.global_fixed_idx)
            #B = self.B[:,q]
            R = scipy.sparse.coo_matrix((np.ones(len(q)),(np.arange(len(q)),q)),shape=(len(q),n)).tocsr()
            c_B = self.B@R.T
            c_B.eliminate_zeros()
            to_be_eliminated[(c_B.getnnz(1)>0)] = True

            Basis , _ = algebra_cy.pyx_compute_basis(c_B.shape[0], c_B.shape[1], c_B, maxiter=100, switch=0)
            nodal_coeff = R.T@Basis
            return nodal_coeff, to_be_eliminated
            
        deg = self.mesh.patches[0][0][0][0].p
        n = self.N_ofs[-1]
        loc_c = self.corners
        #if not dir_boundary: loc_c = np.setdiff1d(loc_c, self.global_fixed_idx, assume_unique=True) 
  
        #idx = (self.B[:,loc_c].getnnz(1)>0) & (self.B.getnnz(1)==2)
        idx = (self.CornerConstr) & (self.B.getnnz(1)==2)
        to_be_eliminated = self.CornerConstr
        B = self.B[idx,:]
        #B = B[B.getnnz(1)==2,:]
        loc_c = np.unique(B.indices)
        #if not dir_boundary: loc_c = np.setdiff1d(loc_c, self.global_fixed_idx, assume_unique=True) 

        q = np.where(B.getnnz(0)>0)[0]
        R = scipy.sparse.coo_matrix((np.ones(len(q)),(np.arange(len(q)),q)),shape=(len(q),n)).tocsr()
        c_B = B@R.T
        c_B.eliminate_zeros()

        Basis , _ = algebra_cy.pyx_compute_basis(c_B.shape[0], c_B.shape[1], c_B, maxiter=100, switch=0)
        nodal_coeff = R.T@Basis

        nodal_coeff += ieti_cy.identify_T_coefficients_from_corner_basis(nodal_coeff.indptr, nodal_coeff.indices, nodal_coeff.data, *nodal_coeff.shape,
                                                                         self.B.indptr,      self.B.indices,      self.B.data,       self.B.shape[0], deg)

        if not dir_boundary:
            cols = np.flatnonzero(nodal_coeff[self.global_fixed_idx, :].getnnz(axis=0))
            keep = np.ones(nodal_coeff.shape[1], dtype=bool)
            keep[cols] = False
            nodal_coeff=nodal_coeff[:,keep]

            #TODO:Do not remove respective constraints if the primals are on the dirichlet boundary and finally dropped from the primal system.
            
        return nodal_coeff, to_be_eliminated

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
            for (p2, b2,_) in self.mesh.L_intfs[(p1,b1)]:
                kv2 = assemble.boundary_kv(self.mesh.kvs[p2],b2)[0]
                left2, right2 = kv2.support()
                
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

    def function(self, u):
        if self.subspace and len(u)==self.nDofs:
            u_loc=self.Basis@u
        elif len(u)==self.N_ofs_elim[-1] and self.elim:
            u_loc = self.Basis_loc_global@u
        elif len(u)==self.N_ofs[-1]:
            u_loc = u
        else:
            raise ValueError('dimension mismatch')
        return [geometry.BSplineFunc(kvs,u_loc[self.N_ofs[p]:self.N_ofs[p+1]]) for p, kvs in enumerate(self.mesh.kvs)]

class PrimalSystem():
    def __init__(self, Prim):
        self.Prim = Prim
        self.nPrim = self.Prim.shape[1]

        self.Psi = scipy.sparse.csc_matrix((0,self.nPrim))
        self.A_prim = scipy.sparse.csr_matrix(2*(self.nPrim,))
        self.RHS_prim = np.zeros(self.nPrim)
        self.R = []

    def incorporate_PrimalConstraints(self, A, B, RHS, IMap):
        Prim_free = self.Prim[IMap.global_free, :]
        self.nLagrangeMultipliers = B[0].shape[0]
        
        #if self.nPrim == 0:
            #return A, B, RHS
        K = len(A)
        self.C=[]
        
        for p in range(K):
            c = (Prim_free[IMap.N_ofs_free[p]:IMap.N_ofs_free[p+1],:].T).tocsr()
            c.eliminate_zeros()
            jj = np.where((c.indptr[1:]-c.indptr[:-1])>0)[0]
            c = c[jj,:]
            #diff = np.linalg.matrix_rank(c.toarray())==c.shape[0]
            self.C.append(c)
            self.R.append(scipy.sparse.coo_matrix((np.ones(c.shape[0]),(np.arange(c.shape[0]),jj)),(c.shape[0],self.nPrim)))
            #assert np.linalg.matrix_rank(c.toarray())==c.shape[0], "Local saddle point system not full row rank in patch "+str(p)+": number of rows: "+str(c.shape[0])+", row rank: "+str(np.linalg.matrix_rank(c.toarray()))+" ."
        #assert np.all(np.array([np.linalg.matrix_rank(c.toarray())==c.shape[0] for c in self.C if c.shape[0]!=0])), "Local saddle point system not full rank."
            
        self.nPrimConstr = [c.shape[0] for c in self.C]

        #if self.eliminate_constraints:
            #return A, B, RHS, self.C

        mod_A = [scipy.sparse.block_array([[A[p],self.C[p].T],[self.C[p], None]], format='csr') for p in range(K)]
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

            # if n_constr == 0:
            #     self.Psi.append(csr_matrix((0,self.Prim.shape[1])))
            #     Delta.append(None)
    
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
            
            A_prim.append(self.Psi[-1].T@(csr_matrix(mod_A[p])[:,:n_free][:n_free,:])@self.Psi[-1])
            B_prim.append(mod_B[p][:, :n_free] @ self.Psi[-1])
            self.RHS_prim += self.Psi[-1].T@mod_RHS[p][:n_free]
    
        # One-time sparse sum
        #self.A_prim = -1*sum(A_prim)
        self.A_prim = sum(A_prim)
        self.B_prim = sum(B_prim)
        #self.RHS_prim += sum(RHS_prim_contribs)
    
        return loc_solvers

    def distributePrimalSolution(self, u):
        u_prim = u[-1]
        #print(self.Psi[].shape,u_prim.shape)
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
        return scipy.sparse.block_array([[scipy.sparse.block_diag(self.A),B.T],[B,None]], format=format)

    def RHSforSaddlePointSystem(self):
        return np.concatenate(self.RHS+[np.zeros(self.nLagrangeMultipliers)])

    def SchurComplement(self, as_matrix=False):
        F = IETIOperator(self.loc_solver,self.B)
        #F = operators.SumOperator([LinOp(self.B[p])@self.loc_solver[p]@LinOp(self.B[p].T) for p in range(self.K)])
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

class IETIOperator(scipy.sparse.linalg.LinearOperator):
    def __init__(self, OP, B):
        assert len(B)==len(OP), 'Amount of operators and transition matrices does not match.'
        K = len(B)
        assert all(B[p].shape[0] == B[0].shape[0] for p in range(1, K)), 'Transition matrices do not have matching image dimension!'
        assert all(OP[p].shape[0] == OP[p].shape[1] for p in range(K)), 'Operator dimension are not square!'
        assert all(B[p].shape[1] == OP[p].shape[0] for p in range(K)), 'Dimensions of operators and transition matrices do not agree!'

        self.OP = OP
        self.B = B
        self.BT = [B.T for B in B]
        
        m = B[0].shape[0]
        super().__init__(dtype=np.float64,shape=(m, m))

    def _matvec(self, x):
        y = np.zeros(self.shape[0], dtype=self.dtype)
        for B, BT, OP in zip(self.B, self.BT, self.OP):
            y += B @ (OP @ (BT @ x))
        return y
    
    def _matmat(self, X):
        Y = np.zeros((self.shape[0], X.shape[1]), dtype=self.dtype)
        for B, BT, OP in zip(self.B, self.BT, self.OP):
            Y += B @ (OP @ (BT @ X))
        return Y

    def _adjoint(self):
        return self
            
class SchurOperator(scipy.sparse.linalg.LinearOperator):
    def __init__(self,A,B,C,D,symmetric=False):
        assert A.shape[0]==A.shape[1]
        assert D.shape[0]==D.shape[1]
        assert A.shape[0]==B.shape[0]
        assert A.shape[1]==C.shape[1]
        assert C.shape[0]==D.shape[0]
        assert B.shape[1]==D.shape[1]

        self.A=A
        self.B=B
        self.C=C
        self.D=D

        self.symmetric=symmetric

        super().__init__(dtype=A.dtype,shape=A.shape)

    def _matvec(self, x):
        return self.A @ x - self.B @ (self.D @ (self.C @ x))

    def _matmat(self, X):
        return self.A @ X - self.B @ (self.D @ (self.C @ X))

    def _adjoint(self):
        if self.symmetric:
            return self

class ScaledDirichletPreconditioner():
    def __init__(self, A ,B, IMap):
        self.A = A
        self.IMap = IMap
        self.K   = len(B)
        self.B = [B[p][:,IMap.DOFS_skeleton[p]] for p in range(self.K)]
        self.B_full   = scipy.sparse.hstack(self.B, format='csr')
        self.BN  = np.array([len(IMap.DOFS_skeleton[p]) for p in range(self.K)]) 
        self.BN_ofs = np.cumsum(np.r_[0,self.BN])
        self.S=[]
        for p in range(self.K):
            Gamma = IMap.DOFS_skeleton[p]
            Delta = IMap.DOFS_interior[p]
    
            Ass = A[p][Gamma][:,Gamma]
            Asi = A[p][Gamma][:,Delta]
            Ais = A[p][Delta][:,Gamma]
            Aii = A[p][Delta][:,Delta]
            solver = solvers.make_solver(Aii, spd=True)

            self.S.append(SchurOperator(Ass,Asi,Ais,solver,symmetric=True))

        self.D = []
        self.D_Is_Diagonal = False
        self.Schur_Is_Matrix = False

    def SchurAsMatrices(self): #for large number of degrees of freedom not feasible
        self.S_dense = [s@np.eye(s.shape[0]) for s in self.S]
        self.Schur_Is_Matrix = True
        #return [s@np.eye(s.shape[0]) for s in self.S]

    def setupConstraintScaling(self):
        d = ieti_cy.pyx_constraint_scaling(self.B_full.indptr, self.B_full.indices, self.B_full.data, *self.B_full.shape)
        self.D = [scipy.sparse.diags(d[self.BN_ofs[p]:self.BN_ofs[p+1]], format='csr') for p in range(self.K)] 
        
    def setupWeightScaling(self):
        d = ieti_cy.pyx_weight_scaling(self.B_full.indptr, self.B_full.indices, self.B_full.data, *self.B_full.shape)
        self.D = [d[self.BN_ofs[p]:self.BN_ofs[p+1]] for p in range(self.K)] 
        self.D_Is_Diagonal = True

    def setupAbsoluteWeightScaling(self):
        d = ieti_cy.pyx_absolute_weight_scaling(self.B_full.indptr, self.B_full.indices, self.B_full.data, *self.B_full.shape)
        self.D = [d[self.BN_ofs[p]:self.BN_ofs[p+1]] for p in range(self.K)] 
        self.D_Is_Diagonal = True
        
    def setupCoefficientScaling(self, a):  ###TODO: cythonize and return 1d array instead of sparse diag
        self.D = [scipy.sparse.csr_matrix((self.B[p].shape[1],self.B[p].shape[1])) for p in range(self.K)]
        for (p1,b1) in self.IMap.mesh.L_intfs:
            R1 = self.IMap.R_interfaces[(p1,b1)]@self.IMap.R_skeleton[p1].T
            for (p2,b2) in self.IMap.mesh.L_intfs[(p1,b1)]:
                R2 = self.IMap.R_interfaces[(p2,b2)]@self.IMap.R_skeleton[p2].T
                a1, a2 = a[self.IMap.mesh.patch_domains[p1]], a[self.IMap.mesh.patch_domains[p2]]
                self.D[p1] += (a2)/(a1+a2)*R1.T@scipy.sparse.identity(R1.shape[0])@R1
                self.D[p2] += (a1)/(a1+a2)*R2.T@scipy.sparse.identity(R2.shape[0])@R2
        self.D_Is_Diagonal = False
        
    def setupSelectionScaling(self):
        assert isinstance(self.B_full, scipy.sparse.csr_matrix)
        d = ieti_cy.pyx_selection_scaling(self.B_full.indptr, self.B_full.indices, self.B_full.data, *self.B_full.shape)
        self.D = [d[self.BN_ofs[p]:self.BN_ofs[p+1]] for p in range(self.K)]
        self.D_Is_Diagonal = True
        #self.D = [scipy.sparse.diags(d[self.BN_ofs[p]:self.BN_ofs[p+1]], format='csr') for p in range(self.K)] 

    def setupDeluxeScaling(self):  ###TODO: cythonize and/or write it as solution of sparse systems (no computation of Schur matrices)
        self.D = [np.zeros((self.B[p].shape[1],self.B[p].shape[1])) for p in range(self.K)]
        for (p1,b1) in self.IMap.mesh.L_intfs:
            R1 = self.IMap.R_interfaces[(p1,b1)]@self.IMap.R_skeleton[p1].T
            S1 = R1@self.S_dense[p1]@R1.T
            for (p2,b2) in self.IMap.mesh.L_intfs[(p1,b1)]:
                R2 = self.IMap.R_interfaces[(p2,b2)]@self.IMap.R_skeleton[p2].T
                S2 = R2@self.S_dense[p2]@R2.T
                X = (self.B[p2]@R2.T)
                constr = np.where(X.getnnz(1)>0)[0]
                P = self.B[p1][constr,:] @ R1.T
                Inv = np.linalg.inv(P@S1@P.T+S2)
                self.D[p1]+=R1.T@ P.T @ (Inv@S2) @ P @R1
                self.D[p2]+=R2.T@(Inv@P@S1@P.T)@R2
        self.D_Is_Diagonal = False
        
    def AsOperator(self):
        assert len(self.D)==self.K, 'Not all scaling matrices given! Call a setup routine first!'
        D = self.D
        B = self.B
        S = self.S
        if self.Schur_Is_Matrix: 
            S_dense = self.S_dense
            m = self.B_full.shape[0]
            M = np.zeros((m,m))
            if self.D_Is_Diagonal:
                for p in range(self.K):
                    BgD = B[p].multiply(D[p])
                    M += BgD @S_dense[p] @ BgD.T
            else:
                for p in range(self.K):
                    BgD = B[p] @ D[p]
                    M += BgD @S_dense[p] @ BgD.T
            return M
            
            #self.BgD = [self.B[p] @ self.D[p] for p in range(self.K)]
            
        #B = scipy.sparse.hstack(self.B)
        #print("Convergence condition: {:.3}".format(scipy.sparse.linalg.norm(B@scipy.sparse.block_diag(self.D).T@B.T@B-B))) #check Algebraic condition
        #return operators.SumOperator([self.BgD[p]@self.S[p]@self.BgD[p].T for p in range(self.K)])  
        if self.D_Is_Diagonal:
            BgD = [b.multiply(d) for b,d in zip(B,D)]
        else:
            BgD = [b @ d for b,d in zip(B,D)]
        return IETIOperator(self.S,BgD)

def EdgePreconditionerFull(IMap, S, B, C):
    S_ = [np.zeros(s.shape) for s in S]
    X = len(B)*[None]
    C_ = len(C)*[None]
    for p in range(IMap.nPatches):
        Gamma = IMap.DOFS_skeleton[p]
        m = len(Gamma)
        assert m == S_[p].shape[0]
        C_[p]=C[p][:,Gamma]
        X[p] = scipy.sparse.hstack([scipy.sparse.identity(B[p].shape[1],format='csc'),scipy.sparse.csr_matrix((B[p].shape[1],C_[p].shape[0]))])
        for b in range(4):
            if (p,b) in IMap.DOFS_interfaces:
                I = IMap.DOFS_interfaces[(p,b)]
                I_in_Gamma = np.searchsorted(Gamma,I)
                Q = scipy.sparse.csr_matrix((np.ones(len(I_in_Gamma)),(I_in_Gamma, I_in_Gamma)),shape=(m, m))
                S_[p] += S[p]
        #print(B_[p].shape,S_[p].shape,C_[p].shape)
    #return np.linalg.inv(np.array([B_[p]@np.linalg.inv(scipy.sparse.block_array([[S_[p],C_[p].T],[C_[p],np.zeros(2*(C_[p].shape[0],))]]).toarray())@B_[p].T for p in range(len(S_))]).sum(axis=0))
    return np.linalg.inv(np.array([B[p]@X[p]@np.linalg.inv(scipy.sparse.block_array([[S_[p],C_[p].T],[C_[p],np.zeros(2*(C_[p].shape[0],))]]).toarray())@X[p].T@B[p].T for p in range(len(S_))]).sum(axis=0))

def EdgePreconditionerDense(IMap, S, B, C):
    m = B[0].shape[0]
    M = np.zeros((m,m))

    for (p1,b1) in IMap.mesh.L_intfs:
        Gamma = IMap.DOFS_skeleton[p1]
        I     = IMap.DOFS_interfaces[(p1,b1)]
        I_in_Gamma = np.searchsorted(Gamma,I)
        c = C[p1][:, I]
        c = c[c.getnnz(1)>0,:].toarray()
        S1 = S[p1][I_in_Gamma,:][:,I_in_Gamma]
        if c.shape[0]>0:
            #S1 = np.block([[S1,c.T],[c, np.zeros((c.shape[0], c.shape[0]))]])
            S1_inv=np.linalg.inv(np.block([[S1,c.T],[c, np.zeros((c.shape[0], c.shape[0]))]]))
        else:
            S1_inv = np.linalg.inv(S1)
            
        P = []
        B2 = []
        S2_inv=[]
        for (p2,b2,_) in IMap.mesh.L_intfs[(p1,b1)]:
            Gamma2 = IMap.DOFS_skeleton[p2]
            I2 = IMap.DOFS_interfaces[(p2,b2)]
            I2_in_Gamma2 = np.searchsorted(Gamma2,I2)

            S2 = S[p2][I2_in_Gamma2,:][:,I2_in_Gamma2]
            S2_inv.append(np.linalg.inv(S2))

            B_ = B[p2][:,I2_in_Gamma2]
            B2.append(B_)
            constr = np.where(B_.getnnz(1)>0)[0]
            P.append(B[p1][constr,:][:,I_in_Gamma])

        S2_inv = scipy.sparse.block_diag(S2_inv)
        P = scipy.sparse.vstack(P)
        B2 = scipy.sparse.hstack(B2)

        if c.shape[0]>0:
            M+=B2@np.linalg.inv(S2_inv + P@S1_inv[:-(c.shape[0]),:-(c.shape[0])]@P.T)@B2.T
        else:
            M+=B2@np.linalg.inv(S2_inv + P@S1_inv@P.T)@B2.T
    return M

def EdgePreconditioner(IMap, S, B, C):
    m = B[0].shape[0]
    solver = []
    X = []
    M = []
    for (p1,b1) in IMap.mesh.L_intfs:
        Gamma = IMap.DOFS_skeleton[p1]
        I     = IMap.DOFS_interfaces[(p1,b1)]
        I_in_Gamma = np.searchsorted(Gamma,I)
        c = C[p1][:, I]
        c = c[c.getnnz(1)>0,:].toarray()
        S1 = S[p1][I_in_Gamma,:][:,I_in_Gamma]
        if c.shape[0]>0:
            S1 = np.block([[S1,c.T],[c, np.zeros((c.shape[0], c.shape[0]))]])

        P = []
        B2 = []
        S2=[]
        for (p2,b2,_) in IMap.mesh.L_intfs[(p1,b1)]:
            Gamma2 = IMap.DOFS_skeleton[p2]
            I2 = IMap.DOFS_interfaces[(p2,b2)]
            I2_in_Gamma2 = np.searchsorted(Gamma2,I2)

            c2 = C[p2][:, I2]
            c2 = c2[c2.getnnz(1)>0,:].toarray()

            S2_ = S[p2][I2_in_Gamma2][:,I2_in_Gamma2]
            if c2.shape[0]>0:
                S2.append(np.block([[S2_,c2.T],[c2, np.zeros((c2.shape[0], c2.shape[0]))]]))
                B_ = scipy.sparse.hstack([B[p2][:,I2_in_Gamma2],scipy.sparse.csc_matrix((B[p2].shape[0],c2.shape[0]))])
            else:
                S2.append(np.block([[S2_,c2.T],[c2, np.zeros((c2.shape[0], c2.shape[0]))]]))
                B_ = B[p2][:,I2_in_Gamma2]
            #S2_inv.append(solvers.make_solver(S_, symm=True, spd=True))
            
            #B_ = B[p2][:,I2_in_Gamma2]
            B2.append(B_)
            constr = np.where(B_.getnnz(1)>0)[0]
            if c2.shape[0]>0:
                P.append(scipy.sparse.vstack([B[p1][constr,:][:,I_in_Gamma],scipy.sparse.csc_matrix((c2.shape[0],len(I_in_Gamma)))]))
            else:
                P.append(B[p1][constr,:][:,I_in_Gamma])

        S2 = scipy.sparse.block_diag(S2)
        P = scipy.sparse.vstack(P)
        if c.shape[0]>0:
            P = scipy.sparse.hstack([P,scipy.sparse.csc_matrix((P.shape[0],c.shape[0]))])
        #print(np.linalg.eigvals(S1 + P.T @ S2 @ P))
        M.append(SchurOperator(S2,S2 @ P,P.T @ S2, operators.make_solver(S1 + P.T @ S2 @ P, symm=True, spd=False))) ###inverse of SchurOperator(S2_inv,P,P.T,-S1_inv)
        X.append(scipy.sparse.hstack(B2))

    return IETIOperator(M,X)
    
def EdgePreconditionerOP(IMap, A, B, C):
    M = len(IMap.mesh.L_intfs)*[None]
    X = len(IMap.mesh.L_intfs)*[None]

    for i,(p1, b1) in enumerate(IMap.mesh.L_intfs):
        Gamma = IMap.DOFS_interfaces[(p1,b1)]
        Q = scipy.sparse.csr_matrix((np.ones(len(Gamma)),(Gamma, Gamma)),shape=(C[p1].shape[1], C[p1].shape[1]))
        c = C[p1]@Q
        c.eliminate_zeros()
        mask = c.getnnz(axis=1) > 0
        c = c[mask, :]
        #print(c)

        n_constr = 0

        b = B[p1]@Q
        b.eliminate_zeros()

        A_blocks = [scipy.sparse.block_array(
            [[A[p1], c.T],
             [c, None]], format='csc')]
        B_blocks = [scipy.sparse.hstack([b,scipy.sparse.csc_matrix((b.shape[0], c.shape[0]))], format='csc')]

        B2_blocks = []
        for (p2, b2,_) in IMap.mesh.L_intfs[(p1, b1)]:
            Gamma2 = IMap.DOFS_interfaces[(p2,b2)]
            n_constr += len(Gamma2)
            #Q = scipy.sparse.csr_matrix((np.ones(len(Gamma2)),(Gamma2, Gamma2)),shape=(C[p1].shape[1], C[p1].shape[1]))
            b = B[p2]@scipy.sparse.csr_matrix((np.ones(len(Gamma2)),(Gamma2, Gamma2)),shape=(B[p2].shape[1], B[p2].shape[1]))
            A_blocks.append(A[p2])
            B_blocks.append(b)
            B2_blocks.append(B[p2][:,Gamma2])

        A_ = scipy.sparse.block_diag(A_blocks, format='csc')
        B_ = scipy.sparse.hstack(B_blocks, format='csc')
        B2 = scipy.sparse.hstack(B2_blocks, format='csc') 
        #print(B_.toarray())

        B_ = B_.tocsr()
        r_idx = B_.getnnz(axis=1) > 0 #boolean array
        assert n_constr == r_idx.sum()
        B_=B_[r_idx,:]
        #print(B_.toarray())
        #n_constr = r_idx.sum()

        Mat = scipy.sparse.block_array([
            [A_, B_.T],
            [B_, None]
        ], format='csc')

        # B_fine=B_.tocsc()
        # B_fine = B_fine[:, (A_blocks[0].shape[0]):]
        # c_idx = B_fine.getnnz(axis=0) > 0
        # print(B_fine[:,c_idx].toarray())

        n_total = Mat.shape[0]

        solver = solvers.make_solver(-Mat, symm=True, spd=False)
        M[i] = solver 
        X[i] = B2 @ scipy.sparse.hstack([scipy.sparse.csc_matrix((n_constr, n_total - n_constr)),scipy.sparse.identity(n_constr)])
        # X[i] = scipy.sparse.hstack([
        #         scipy.sparse.csc_matrix((B2.shape[0], n_total - n_constr)),
        #         B2
        # ])

    return IETIOperator(M, X)

def EdgePreconditioner2(IMap, S, B, C):
    m = B[0].shape[0]
    MsD = np.zeros((m,m))
    for (p1,b1) in IMap.mesh.L_intfs:
        R1 = IMap.R_interfaces[(p1,b1)]@IMap.R_skeleton[p1].T
        c = C[p1]@IMap.R_interfaces[(p1,b1)].T
        c = c[c.getnnz(1)>0,:].toarray()
        #c = C[p1][:, IMap.DOFS_interfaces[(p1,b1)]]
        if c.shape[0]>0:
            S1 = np.block([[R1@S[p1]@R1.T,c.T],[c, np.zeros((c.shape[0], c.shape[0]))]])
        else:
            S1 = R1@S[p1]@R1.T
        S2 = []
        P = []
        B2 = []
        for (p2,b2,_) in IMap.mesh.L_intfs[(p1,b1)]:
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
    
