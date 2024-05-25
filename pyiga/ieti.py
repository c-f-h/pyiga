import numpy as np
import time
import scipy
from scipy.sparse import coo_matrix

from pyiga import bspline, vis, assemble, solvers, algebra

class IetiDP:
    def __init__(self, MP, dir_data, neu_data=None, elim=True):
        self.space = MP
        self.elim=elim
        
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
            
        # Idx = np.where(self.space.Constr.getnnz(axis=1)>2)[0]
        # Bas, self.B = algebra.compute_basis(self.space.Constr, maxiter=5, Idx = Idx)
        # self.Basisk = []
        
        if elim:
            dofs=dict()
            Basis=MP.Basis.tocsc()
            constr=[]
            for p in range(MP.numpatches):
                idx_per_col = [Basis.indices[Basis.indptr[c]:Basis.indptr[c+1]] for c in range(Basis.shape[1])]
                dofs[p] = np.where([np.any((i<MP.N_ofs[p+1]) & (i>=MP.N_ofs[p])) for i in idx_per_col])[0]
                #print(dofs)
            N = [len(dofs_) for dofs_ in dofs.values()]
            N_ofs = np.cumsum([0]+N)

            self.Basisk=[Basis[MP.N_ofs[p]:MP.N_ofs[p+1],:][:,dofs[p]] for p in range(MP.numpatches)]

            J1=[]
            J2=[]
            for (p1,b1,_),(p2,b2,_),_ in MP.intfs:
                #print(np.intersect1d(dofs[p1],dofs[p2]))
                J1.append([np.where(dofs[p1]==g)[0][0]+N_ofs[p1] for g in np.intersect1d(dofs[p1],dofs[p2])])
                J2.append([np.where(dofs[p2]==g)[0][0]+N_ofs[p2] for g in np.intersect1d(dofs[p1],dofs[p2])])

            J1=np.concatenate(J1)
            J2=np.concatenate(J2)
            data=np.r_[np.ones(len(J1)),-np.ones(len(J2))]
            I=np.r_[np.arange(len(J1)),np.arange(len(J2))]
            J = np.r_[J1,J2]
            self.B = scipy.sparse.coo_matrix((data,(I,J)),(len(J1),sum(N))).tocsr()
        else:
            self.Basisk = [scipy.sparse.identity(MP.N[p]) for p in range(MP.numpatches)]
            self.B = MP.Constr
            
        self.P2G =[]
        
        for p in range(self.space.numpatches):
            X = self.Basisk[p].tocoo()
            idx = np.where(np.isclose(X.data,1))
            X.data, X.row, X.col = X.data[idx], X.row[idx], X.col[idx]
            D = (X.T@self.Basisk[p]).sum(axis=1).A.ravel()
            #assert all(abs(D)>1e-12), 'D has zeros.'
            #S = scipy.sparse.spdiags(1/D,[0],len(D),len(D))
            self.P2G.append(X.T)
            I = np.zeros(self.Basisk[p].shape[0])
            if p in self.dir_idx:
                I[self.dir_idx[p]] = 1
                self.dir_idx[p] = np.where(np.isclose(self.P2G[p]@I,1))[0]
        
        self.N = [Ba.shape[1] for Ba in self.Basisk]
        self.N_ofs = np.cumsum([0]+self.N)
        self.global_dir_idx = np.concatenate([self.dir_idx[p] + self.N_ofs[p] for p in self.dir_idx])
        self.free_dofs = np.setdiff1d(np.arange(self.N_ofs[-1]),self.global_dir_idx)
        
        #self.B = self.B @ scipy.sparse.block_diag(self.Basisk)
        
        nnz_per_col = self.B.getnnz(axis=0)
        self.intf_dofs = np.where(nnz_per_col > 0)[0]
        self.intfs = np.setdiff1d(self.intf_dofs, self.global_dir_idx)
        
        #self.B = self.B[:,self.free_dofs]

        #self.dir_ofs = np.cumsum(np.array([len(np.unique(idx[p])) for p in range(MP.numpatches)]))
        
        #self.dir_idx, lookup = np.unique(idx, return_index=True)
        #self.dir_vals = np.concatenate(vals)[lookup]
        
    def assemble(self, f):
        Ak = [Ba.T @ assemble.assemble('(inner(grad(u),grad(v)))* dx', kvs, bfuns=[('u',1), ('v',1)], geo=geo)@Ba for Ba, ((kvs, geo),_) in zip(self.Basisk, self.space.mesh.patches)]
        A = scipy.sparse.block_diag(Ak, format='csr')
        rhsk = [Ba.T @ assemble.assemble('f * v * dx', kvs, bfuns=[('v',1)], geo=geo, f=f).ravel() for Ba, ((kvs, geo),_) in zip(self.Basisk,self.space.mesh.patches)]
        
        Id = scipy.sparse.eye(A.shape[1], format='csr')
        mask = np.zeros(A.shape[1], dtype=bool)
        mask[list(self.intfs)] = True
        mask[list(self.global_dir_idx)]=False
        self.Rbb = Id[mask]
        
        mask = np.ones(A.shape[1], dtype=bool)
        
        kvs=self.space.mesh.kvs
        
        for p in range(self.space.numpatches):
            bnd_dofs = np.concatenate([indices for indices in assemble.boundary_dofs(kvs[p])])
            I = np.zeros(self.Basisk[p].shape[0])
            I[bnd_dofs] = 1
            bnd_dofs = np.where(np.isclose(self.P2G[p]@I,1))[0]
            mask[bnd_dofs+self.N_ofs[p]] = False
            #mask[]

        Rii = Id[mask]

        # for p in range(self.space.numpatches):
        #     bnd_dofs = np.concatenate([indices for indices in assemble.boundary_dofs(kvs[p])])
        #     mask[bnd_dofs+self.space.N_ofs[p]] = False

        self.Rii = Id[mask]

        self.Abb = self.Rbb.dot(A).dot(self.Rbb.T)
        self.Aii = self.Rii.dot(A).dot(self.Rii.T)
        self.Abi = self.Rbb.dot(A).dot(self.Rii.T)
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
        Nodes = self.space.get_nodes()
        loc_prim = np.concatenate(list(Nodes.values()))
        cpp = {p : loc_prim[(loc_prim >= self.space.N_ofs[p]) & (loc_prim < self.space.N_ofs[p+1])] for p in range(self.space.numpatches)}
        
        for p in range(len(self.space.mesh.patches)):
            #bndindices = (bcs[0] < MP.N_ofs[p+1]) & (bcs[0] >= MP.N_ofs[p])
            if p in self.dir_idx:
                to_eliminate = self.dir_idx[p]
            else:
                to_eliminate = np.array([])
                
            free = np.setdiff1d(np.arange(self.N[p]),to_eliminate)
            primal_free = cpp[p] - self.space.N_ofs[p]
            
            if self.elim:
                I = np.zeros(self.Basisk[p].shape[0])
                I[primal_free] = 1
                primal_free = np.where(np.isclose(self.P2G[p]@I,1))[0]
            
            #print(primal_free)
            # B = self.B[:,primal_free+self.N_ofs[p]].tocoo()
            # result = B.row[np.isclose(abs(B.data),1)]
            
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
            #print(ck.A)
            
        self.B = self.B[np.setdiff1d(np.arange(self.B.shape[0]),self.eliminate_constraints),:]
        
        self.C = scipy.sparse.block_diag(self.Ck)
        
    def construct_primal_basis(self):
        PsiK=[]
        Nodes = self.space.get_nodes()
        #prim = np.array(list(Nodes.keys()))
        loc_prim = np.concatenate(list(Nodes.values()))
        cpp = {p : loc_prim[(loc_prim >= self.space.N_ofs[p]) & (loc_prim < self.space.N_ofs[p+1])] for p in range(self.space.numpatches)}

        data = np.ones(len(loc_prim))
        ii = loc_prim
        V =[len(Nodes[key]) for key in Nodes] 
        jj= np.arange(len(Nodes)).repeat(V)
        R = scipy.sparse.coo_matrix((data,(ii,jj)),(self.N_ofs[-1],len(Nodes))).tocsr()
        
        for p in range(len(self.space.mesh.patches)):
            a = self.Ak[p]
            c = self.Ck[p]
            AC = scipy.sparse.bmat(
            [[a, c.T],
             [c,  None   ]], format='csr')
            RHS = np.vstack([np.zeros((a.shape[0],c.shape[0])), np.identity(c.shape[0])])
            psi = scipy.sparse.linalg.spsolve(AC, RHS)
            psi, delta = psi[:a.shape[0],], psi[a.shape[0]:,]
            if psi.ndim==1: psi=psi[:,None]

#             jj=[len() for g in Nodes]
#             r = scipy.sparse.coo_matrix((np.ones(psi.shape[1]),(cpp[p]-self.space.N_ofs[p],)),(psi.shape[1],len(Nodes)))
            PsiK.append(psi@(R[cpp[p],:]))

        self.Psi=np.vstack(PsiK)
            
#         Psik = [scipy.sparse.linalg.spsolve(scipy.sparse.bmat([[self.Ak[p], self.Ck[p].T],[self.Ck[p],  None   ]], format='csr'),np.vstack([np.zeros((self.Ak[p].shape[0],self.Ck[p].shape[0])), np.identity(self.Ck[p].shape[0])]))[:self.Ak[p].shape[0],] for p in range(self.space.numpatches)]
#         Psik = [psi[None] if len(psi.shape)==1 else psi.T for psi in Psik]
        
#         self.Psik = Psik
#         self.Psi = scipy.sparse.block_diag(Psik)
        
    def compute_F(self):
        B = self.B[:,self.free_dofs]
        B = B[np.where(B.getnnz(axis=1)>0)[0]]
        PTAP = self.Psi.T@self.A@self.Psi
        PTBT = self.Psi.T@B.T
        BP   = B@self.Psi

        self.BL = scipy.sparse.bmat([[B,    np.zeros((B.shape[0],self.C.shape[0])), BP]], format='csr')
        self.BR = scipy.sparse.bmat([[B.T],    
                                [np.zeros((self.C.shape[0],B.shape[0]))], 
                                [PTBT]], format='csr')
        self.A0 = scipy.sparse.bmat(
            [[self.A,    self.C.T,  None],
             [self.C,    None,      None],
             [None,      None,      PTAP]], format='csr')

            # print("Rank ", np.linalg.matrix_rank(PTAP.A), " vs. shape ", PTAP.shape)
        
        rhs = np.concatenate(self.rhsk)
        #print(rhs)
        #b = np.hstack((rhs, np.zeros(self.C.shape[0],), self.Psi.dot(rhs), np.zeros(self.B[:,self.free_dofs].shape[0],)))

        BR = scipy.sparse.linalg.aslinearoperator(self.BR)
        BL = scipy.sparse.linalg.aslinearoperator(self.BL)
        A0inv = solvers.make_solver(self.A0, spd=False, symmetric=True)

        F = BL@A0inv.dot(BR)

        self.TR = np.hstack((rhs, np.zeros((self.C.shape[0],)), self.Psi.T.dot(rhs)))
        b = BL@(A0inv.dot(self.TR))
        
        return F, b
    
    def MsD(self):
        B = self.B[:,self.free_dofs]
        B = B[np.where(B.getnnz(axis=1)>0)[0]]
        #B_gamma = B_gamma[np.setdiff1d(np.arange(B_gamma.shape[0]),self.eliminate_constraints),:]
        self.B_gamma = B@self.Rbb[:,self.free_dofs].T

        #print(np.linalg.matrix_rank(Rbb.A, 1e-8))
        #print(B_gamma.shape)

        #Aib = scipy.sparse.linalg.aslinearoperator(Aib)
        AiiinvB = solvers.make_solver(self.Aii, spd=True)
        #AiiinvB = scipy.sparse.linalg.spsolve(self.Aii, self.Aib.A)
        self.S = scipy.sparse.linalg.aslinearoperator(self.Abb) - scipy.sparse.linalg.aslinearoperator(self.Abi)@AiiinvB.dot(scipy.sparse.linalg.aslinearoperator(self.Aib))
        D = np.abs(self.B_gamma).sum(axis=0)
        D = [1/(1+D[0,v]) for v in range(D.shape[1])]
        self.D = scipy.sparse.diags(D, format='csr')

        self.BgD = scipy.sparse.linalg.aslinearoperator(self.B_gamma@self.D)
        return self.BgD@self.S.dot(self.BgD.T)  
            
    