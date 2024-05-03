import numpy as np
import time
import scipy
from scipy.sparse import coo_matrix

from pyiga import bspline, vis, assemble, solvers, algebra

class IetiDP:
    def __init__(self, MP, dir_data, neu_data=None):
        self.space = MP
        
        self.dir_idx=dict()
        self.dir_vals=dict()
        kvs = self.space.mesh.kvs
        geos = self.space.mesh.geos
        for key in dir_data:
            for p,b in self.space.mesh.outer_boundaries[key]:
                idx_, vals_ = assemble.compute_dirichlet_bc(kvs[p], geos[p], [(b//2,b%2)], dir_data[key])
                if p in self.dir_idx:
                    self.dir_idx[p].append(idx_)
                    self.dir_vals[p].append(vals_)
                else:
                    self.dir_idx[p]=[idx_]
                    self.dir_vals[p]=[vals_]
                
        for p in self.dir_idx:
            self.dir_idx[p], lookup = np.unique(self.dir_idx[p], return_index = True)
            self.dir_vals[p] = np.concatenate(self.dir_vals[p])[lookup]
            
        self.global_dir_idx = np.concatenate([self.dir_idx[p] + self.space.N_ofs[p] for p in self.dir_idx])
        self.free_dofs = np.setdiff1d(np.arange(self.space.N_ofs[-1]),self.global_dir_idx)
            
        self.B = self.space.Constr[:,self.free_dofs]
        
        nnz_per_col = MP.Constr.getnnz(axis=0)
        self.intf_dofs = np.where(nnz_per_col > 0)[0]
        self.intfs = np.setdiff1d(self.intf_dofs, self.global_dir_idx)

        #self.dir_ofs = np.cumsum(np.array([len(np.unique(idx[p])) for p in range(MP.numpatches)]))
        
        #self.dir_idx, lookup = np.unique(idx, return_index=True)
        #self.dir_vals = np.concatenate(vals)[lookup]
        
    def assemble(self, f):
        Ak = [assemble.assemble('inner(grad(u),grad(v)) * dx', kvs, bfuns=[('u',1), ('v',1)], geo=geo) for (kvs, geo),_ in self.space.mesh.patches]
        A = scipy.sparse.block_diag(Ak, format='csr')
        rhsk = [assemble.assemble('f * v * dx', kvs, bfuns=[('v',1)], geo=geo, f=f).ravel() for (kvs, geo),_ in self.space.mesh.patches]
        
        Id = scipy.sparse.eye(A.shape[1], format='csr')
        mask = np.zeros(A.shape[1], dtype=bool)
        mask[list(self.intfs)] = True
        self.Rbb = Id[mask]
        mask = np.ones(A.shape[1], dtype=bool)
        
        kvs = self.space.mesh.kvs

        for p in range(self.space.numpatches):
            bnd_dofs = np.concatenate([indices for indices in assemble.boundary_dofs(kvs[p])])
            mask[bnd_dofs+self.space.N_ofs[p]] = False

        Rii = Id[mask]

        self.Abb = self.Rbb.dot(A).dot(self.Rbb.T)
        self.Aii = Rii.dot(A).dot(Rii.T)
        self.Abi = self.Rbb.dot(A).dot(Rii.T)
        self.Aib = self.Abi.T
        
        BCRestr = {p:assemble.RestrictedLinearSystem(Ak[p], rhsk[p], (self.dir_idx[p],self.dir_vals[p])) for p in self.dir_idx}
        self.rhsk = [rhsk[p] if p not in self.dir_idx else BCRestr[p].b for p in range(self.space.numpatches)]
        self.Ak = [Ak[p] if p not in self.dir_idx else BCRestr[p].A for p in range(self.space.numpatches)]
        self.A = scipy.sparse.block_diag(self.Ak)
        # self.A = BCRestr.A
        # self.rhs = BCRestr.b
        
    def construct_primal_constraints(self):
        self.Ck = []
        kvs = self.space.mesh.kvs
        geos = self.space.mesh.geos
        self.eliminate_constraints = np.array([], dtype=int)
        
        for p in range(len(self.space.mesh.patches)):
            #bndindices = (bcs[0] < MP.N_ofs[p+1]) & (bcs[0] >= MP.N_ofs[p])
            if p in self.dir_idx:
                to_eliminate = self.dir_idx[p]
            else:
                to_eliminate = np.array([])
                
            free = np.setdiff1d(np.arange(self.space.N[p]),to_eliminate)
            primal = assemble.boundary_dofs(kvs[p], m=0, ravel=1)
            primal_free = np.setdiff1d(primal, to_eliminate)
            B = self.space.Constr[:,primal_free+self.space.N_ofs[p]].tocoo()
            result = B.row[np.isclose(abs(B.data),1)]
            
            # nnz_per_row = np.isclose(self.space.Constr[:,primal_free+self.space.N_ofs[p]].getnnz(axis=1)
            # result = np.where(nnz_per_row > 0)[0]
            
            self.eliminate_constraints = np.union1d(result, self.eliminate_constraints)

            #loc, _ = cpp[p][0]

            data = np.ones(len(primal_free))
            rows = np.arange(len(primal_free))
            cols = primal_free
            ck = coo_matrix((data, (rows, cols)),(len(primal_free),self.space.N[p])).tocsc()
            ck = ck[:,free]
            self.Ck.append(ck)
            print(ck.A)
            
#         eliminate_constraints = np.array([], dtype=int)
#         Ck = []
#         loc_dirichlet = []
#         cpp = self.space.get_crosspoints()
#         for p in range(len(self.space.mesh.patches)):
#             # bndindices = (bcs[0] < self.space.N_ofs[p+1]) & (bcs[0] >= MP.N_ofs[p])
#             # to_eliminate = bcs[0][bndindices]-MP.N_ofs[p]
#             to_eliminate = self.dir_idx[p]

#             if p in cpp:
#                 loc, _ = cpp[p][0]
#                 nnz_per_row = self.space.Constr[:,loc+self.space.N_ofs[p]].getnnz(axis=1)
#                 result = np.where(nnz_per_row > 0)[0]
#             else:
#                 result=np.array([])
#             eliminate_constraints = np.union1d(eliminate_constraints, result)

#             ck = self.space.Constr[result,self.space.N_ofs[p]:self.space.N_ofs[p+1]]
#             ck = np.delete(ck.A, 
#                            to_eliminate, 
#                            axis=1)

#             q,r = np.linalg.qr(ck.T)
#             print(ck)
#             ck = scipy.sparse.csr_matrix(ck[np.abs(np.diag(r))>=1e-10])

#             #if p == 3:
#             #    print("Ck", result)

#             Ck.append(ck)
            #loc_dirichlet.append(to_eliminate)
            
        #self.eliminate_constraints = eliminate_constraints
        self.B = self.space.Constr[:,self.free_dofs]
        self.B = self.B[np.setdiff1d(np.arange(self.B.shape[0]),self.eliminate_constraints),:]
        
        self.C = scipy.sparse.block_diag(self.Ck)
        
    def construct_primal_basis(self):
        Psik = []
        # for p in range(len(self.space.mesh.patches)):
        #     a = self.Ak[p]
        #     c = self.Ck[p]
        #     AC = scipy.sparse.bmat(
        #     [[a, c.T],
        #      [c,  None   ]], format='csr')
        #     RHS = np.vstack([np.zeros((a.shape[0],c.shape[0])), np.identity(c.shape[0])])
        #     psi = scipy.sparse.linalg.spsolve(AC, RHS)
        #     psi, delta = psi[:a.shape[0],], psi[a.shape[0]:,]
        #     Psik.append(psi)
            
        Psik = [scipy.sparse.linalg.spsolve(scipy.sparse.bmat([[self.Ak[p], self.Ck[p].T],[self.Ck[p],  None   ]], format='csr'),np.vstack([np.zeros((self.Ak[p].shape[0],self.Ck[p].shape[0])), np.identity(self.Ck[p].shape[0])]))[:self.Ak[p].shape[0],] for p in range(self.space.numpatches)]
        Psik = [psi[None] if len(psi.shape)==1 else psi.T for psi in Psik]
        
        self.Psik = Psik
        self.Psi = scipy.sparse.block_diag(Psik)
        
    def compute_F(self):
        PTAP = self.Psi@self.A@self.Psi.T
        PTBT = self.Psi@self.B.T
        BP   = self.B@self.Psi.T

        BL = scipy.sparse.bmat([[self.B,    np.zeros((self.B.shape[0],self.C.shape[0])), BP]], format='csr')
        BR = scipy.sparse.bmat([[self.B.T],    
                                [np.zeros((self.C.shape[0],self.B.shape[0]))], 
                                [PTBT]], format='csr')
        A0 = scipy.sparse.bmat(
            [[self.A,    self.C.T,  None],
             [self.C,    None,      None],
             [None,      None,      PTAP]], format='csr')

            # print("Rank ", np.linalg.matrix_rank(PTAP.A), " vs. shape ", PTAP.shape)
        
        rhs = np.concatenate(self.rhsk)
        b = np.hstack((rhs, np.zeros(self.C.shape[0],), self.Psi.dot(rhs), np.zeros(self.B.shape[0],)))

        BR = scipy.sparse.linalg.aslinearoperator(BR)
        BL = scipy.sparse.linalg.aslinearoperator(BL)
        A0inv = solvers.make_solver(A0.A, spd=False)

        F = BL@A0inv.dot(BR)

        TR = np.hstack((rhs, np.zeros((self.C.shape[0],)), self.Psi.dot(rhs)))
        b = BL@(A0inv.dot(TR))
        
        return F, b
    
    def MsD(self):
        B_gamma = self.space.Constr
        B_gamma = B_gamma[np.setdiff1d(np.arange(B_gamma.shape[0]),self.eliminate_constraints),:]
        self.B_gamma = B_gamma@self.Rbb.T

        #print(np.linalg.matrix_rank(Rbb.A, 1e-8))
        #print(B_gamma.shape)

        #Aib = scipy.sparse.linalg.aslinearoperator(Aib)
        AiiinvB = scipy.sparse.linalg.spsolve(self.Aii, self.Aib.A)
        print(AiiinvB)
        self.S = self.Abb - self.Abi@AiiinvB
        D = np.abs(self.B_gamma).sum(axis=0)
        D = [1/(1+D[0,v]) for v in range(D.shape[1])]
        D = scipy.sparse.diags(D, format='csr')

        BgD = self.B_gamma@D
        MsD = BgD@self.S@BgD.T  
        
        return MsD
            
    