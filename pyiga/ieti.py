import numpy as np
import time
import scipy
from scipy.sparse import coo_matrix

from pyiga import bspline, vis, assemble, solvers, algebra, assemble
from pyiga import ieti_cy
from scipy.sparse.linalg import aslinearoperator as LinOp

class IetiMapper:
    def __init__(self, M, dir_data, neu_data, elim=False):
        

class IetiMapper:
    def __init__(self, MP, dir_data, neu_data=None, elim=False):
        self.space = MP
        kvs = self.space.mesh.kvs
        geos = self.space.mesh.geos

        # self.elim=elim

        # if self.elim:
        #     p_intfs = np.array([[p1,p2] for (p1,_,_),(p2,_,_),_ in self.space.intfs], dtype=np.int32).T
        #     self.Basisk, self.N_ofs, self.N, self.B = ieti_cy.pyx_compute_decoupled_coarse_basis(self.space.Basis.tocsc(), MP.N_ofs.astype(np.int32), p_intfs)
        # else:
            #self.Basisk = [scipy.sparse.identity(self.space.N[p]) for p in range(self.space.numpatches)]
        self.B = self.space.B
        self.N = self.space.N
        self.N_ofs = self.space.N_ofs

        self.Bk = [self.B[:,self.N_ofs[p]:self.N_ofs[p+1]] for p in range(self.space.numpatches)]

        nnz_per_col = self.B.getnnz(axis=0)
        self.intfs = np.setdiff1d(np.where(nnz_per_col > 0)[0], self.space.global_dir_idx)
        self.skeleton = np.union1d(self.intfs, self.space.global_dir_idx)
        
        self.Rbb = []
        self.Rii = []
        for p in range(self.space.numpatches):
            Id = scipy.sparse.eye(self.N[p], format='csr')
            mask = np.zeros(self.N[p], dtype=bool)
            nnz_per_col = self.Bk[p].getnnz(axis=0)
            if p in self.space.dir_idx:
                intfs = np.setdiff1d(np.where(nnz_per_col > 0)[0], self.space.dir_idx[p])
            else:
                intfs = np.where(nnz_per_col > 0)[0]
            mask[intfs]=True
            if p in self.space.dir_idx:
                mask[self.space.dir_idx[p]]=False
            self.Rbb.append(Id[mask].tocsc())
            mask = np.ones(self.N[p], dtype=bool)
            mask[intfs]=False
            if p in self.space.dir_idx:
                mask[self.space.dir_idx[p]]=False
            self.Rii.append(Id[mask].tocsc())
            
        #self.Nodes = self.get_nodes()

    def get_nodes(self, dir_boundary=False):      ###TODO: make it dependent on a constraint matrix not on the precomputed Basis matrix
        """Get global vertices of the multipatch object as well as local nodal degrees of freedom corresponding to the vertices. 
        In case of T-junctions also obtain the $p$ global degrees of freedom and $p$ local degrees of freedom on the coarse patch.
        Additionally may include nodes on the Dirichlet boundary if desired."""
        loc_c = np.concatenate([assemble.boundary_dofs(kvs,m=0,ravel=True) + self.N_ofs[p] for p, kvs in enumerate(self.space.mesh.kvs)])

        #loc_c = np.setdiff1d(loc_c, self.global_dir_idx)

        # idx = (self.B[:,loc_c].getnnz(axis=1)==2)
        # R = scipy.sparse.coo_matrix((np.ones(len(loc_c)),(np.arange(len(loc_c)),loc_c)),shape=(len(loc_c),self.B.shape[1])).tocsr()
        # c_B = self.B[idx,:]@R.T
        # print(c_B.shape)
        # nodal_indicator = R.T@algebra_cy.pyx_compute_basis(c_B.shape[0], c_B.shape[1], c_B, maxiter=5).tocsc()
        # nodal_indicator = nodal_indicator[:,nodal_indicator.getnnz(axis=0)>1]
        
        # return np.split(nodal_indicator.indices, nodal_indicator.indptr[1:-1])
        
        B = self.space.Basis[loc_c,:]
        C_dofs = np.unique(B[B.getnnz(axis=1)==1].indices)
        X = scipy.sparse.coo_matrix(self.space.Basis)
        idx = np.where(np.isclose(X.data,1))
        X.data, X.row, X.col = X.data[idx], X.row[idx], X.col[idx]
        X = X.tocsc()
        self.Nodes = {c:[X[:,c].indices] for c in C_dofs}
        t_idx = loc_c[np.where(B.getnnz(axis=1)>1)[0]]
        T_dofs = {}
        for i in t_idx:
            t = tuple(self.space.Basis[i,:].indices)
            coeff = tuple()
            if t not in T_dofs:
                T_dofs[t] = [(i,),set(self.space.B[self.space.B.tocsc()[:,i].indices,:].indices)-{i}]
            else:
                T_dofs[t][0] = T_dofs[t][0]+(i,)
                T_dofs[t][1] = T_dofs[t][1] & (set(self.space.B[self.space.B.tocsc()[:,i].indices,:].indices)-{i})
        T_dofs = {t:[np.sort(T_dofs[t][0]),np.sort(list(T_dofs[t][1]))] for t in T_dofs}
        self.Nodes.update(T_dofs)
        if not dir_boundary:
            self.Nodes = {key:self.Nodes[key] for key in self.Nodes if len(np.intersect1d(self.Nodes[key][0],self.space.global_dir_idx))==0}
        

    def generate_primal_info(fat):      ###TODO: more efficient
        #Nodes = self.space.get_nodes()
        self.Prim = {}
        i=0
        total_dofs=set()
        
        if self.elim:
            if fat:
                for key in self.Nodes:
                    if isinstance(key,np.int32):
                        dofs = self.Basis.tocsr()[self.Nodes[key][0],:].indices
                        self.Prim[i] = (dofs,np.ones(len(dofs)))
                        i+=1
                    else:
                        dofs = self.Basis.tocsr()[self.Nodes[key][1],:].indices
                        for dof in dofs:
                            if dof not in total_dofs:
                                total_dofs.add(dof)
                                dofs = np.unique(self.B.tocsr()[self.B.tocsc()[:,dof].indices,:].indices)
                                self.Prim[i] = (dofs,np.ones(len(dofs)))
                                i+=1
            else:
                for key in self.Nodes:
                    if isinstance(key,np.int32):
                        dofs = self.Basis.tocsr()[self.Nodes[key][0],:].indices
                        self.Prim[i] = (dofs,np.ones(len(dofs)))
                        i+=1
                    else:
                        constr = (self.space.B.tocsc()[:,self.Nodes[key][0][0]]==1).indices
                        coeffs = abs(self.space.B[constr,:][:,self.Nodes[key][1]].data)
                        dofs_coarse = self.Basis.tocsr()[self.Nodes[key][1],:].indices
                        dofs=[]
                        for dof in dofs_coarse:
                            found_dofs = np.unique(self.B.tocsr()[self.B.tocsc()[:,dof].indices,:].indices)
                            dofs.append(found_dofs)
                        self.Prim[i] = (np.concatenate(dofs),np.repeat(coeffs,len(found_dofs)))
                        i+=1
        else:
            for key in self.Nodes:
                if isinstance(key,np.int32):    #regular corner
                    dofs = self.Basis.tocsr()[self.Nodes[key][0],:].indices
                    self.Prim[i] = (dofs,np.ones(len(dofs)))
                    i+=1
                else:                        #T-junction
                    constr = (self.space.B.tocsc()[:,self.Nodes[key][0][0]]==1).indices
                    coeffs = abs(self.space.B[constr,:][:,self.Nodes[key][1]].data)
                    dofs_coarse = self.Nodes[key][1]
                    dofs=np.concatenate([self.Nodes[key][0],self.Nodes[key][1]])
                    coeffs = np.concatenate([np.ones(len(self.Nodes[key][0])),coeffs])
                    self.Prim[i] = (dofs,coeffs)
                    i+=1
        #self.n_prim = len(Prim)

class IetiSystem:
    def __init__(self, IMap, redundant=False):
        self.IMap = IMap
        self.A = []
        self.B = self.IMap.Bk
        self.C = []
        self.rhsk = []

        self.redundant = redundant
        #self.SaddlePoint = 
        
    def assemble(self, problem, f, a):
        self.A = [assemble.assemble(problem, kvs, a=a[self.space.mesh.patch_domains[k]], bfuns=[('u',1), ('v',1)], geo=geo) for k, ((kvs, geo),_) in enumerate(self.space.mesh.patches)]
        self.rhsk = [assemble.assemble('f * v * dx', kvs, bfuns=[('v',1)], geo=geo, f=f[self.space.mesh.patch_domains[k]]).ravel() for k, ((kvs, geo),_) in enumerate(self.space.mesh.patches)]
        if self.IMap.elim:
            self.A = [b.T@a@b for b,a in zip(self.IMap.Basisk,self.A)]
            self.rhsk = [b.T@r for b,r in zip(self.IMap.Basisk, self.rhsk)]
        self.BCRestr = {p:assemble.RestrictedLinearSystem(self.A[p], self.rhsk[p], (self.IMap.dir_idx[p],self.IMap.dir_vals[p])) for p in self.IMap.dir_idx}
        self.rhsk = [r if p not in self.dir_idx else self.BCRestr[p].b for p, r in enumerate(self.rhsk)]
        self.A = [A if p not in self.dir_idx else self.BCRestr[p].A for p, A in enumerate(self.A)]

    def construct_primal_constraints(self, redundant=False):
        kvs = self.space.mesh.kvs
        geos = self.space.mesh.geos
        self.eliminate_constraints = np.array([], dtype=int)

        if self.IMap.Prim:
            loc_c_prim = np.concatenate([self.Prim[key][0] for key in self.Prim])
            coeffs = np.concatenate([self.Prim[key][1] for key in self.Prim])
        else:
            loc_c_prim = np.array([])
            coeffs = np.array([])
        loc_c_prim_idx = np.repeat(np.arange(len(self.Prim)),[len(self.Prim[i][0]) for i in self.Prim])
        p_idx = [(loc_c_prim >= self.N_ofs[p]) & (loc_c_prim < self.N_ofs[p+1]) for p in range(self.space.numpatches)]
        self.Prim_pp = {p : (loc_c_prim[p_idx[p]],coeffs[p_idx[p]],loc_c_prim_idx[p_idx[p]]) for p in range(self.space.numpatches)}

        for (p1,b1) in self.space.L_intfs:
            R1 = self.R_interfaces[(p1,b1)]
            diag=np.ones(R1.shape[1])
            idx = np.where(np.isclose(self.Prim_pp[p1][1],1))[0]
            diag[self.Prim_pp[p1][0][idx]-self.N_ofs[p1]]=0
            R1 = R1@scipy.sparse.spdiags(diag,0,R1.shape[1],R1.shape[1])
            R1.eliminate_zeros()
            self.R_interfaces[(p1,b1)] = R1[R1.getnnz(1)>0,:]
            for (p2,b2) in self.space.L_intfs[(p1,b1)]:
                R2 = self.R_interfaces[(p2,b2)]
                diag=np.ones(R2.shape[1])
                idx = np.where(np.isclose(self.Prim_pp[p2][1],1))[0]
                diag[self.Prim_pp[p2][0][idx]-self.N_ofs[p2]]=0
                R2 = R2@scipy.sparse.spdiags(diag,0,R2.shape[1],R2.shape[1])
                R2.eliminate_zeros()
                self.R_interfaces[(p2,b2)] = R2[R2.getnnz(1)>0,:]
                    
        for p in range(MP.numpatches):
            c_primal_free = self.Prim_pp[p][0] - self.N_ofs[p]
            data = self.Prim_pp[p][1]
            idx = np.bincount(self.Prim_pp[p][2])
            idx = idx[idx>0]
            rows = np.repeat(np.arange(len(idx)),idx)
            cols = c_primal_free
            ck = coo_matrix((data, (rows, cols)),(len(idx),self.N[p])).tocsc()
            ck = ck[:,self.free_dofs_pp[p]]
            self.Ck.append(ck.tocsr())
            m, n = ck.shape[0], len(self.Prim)
            jj = np.unique(self.Prim_pp[p][2])
            self.Rk.append(scipy.sparse.coo_matrix((np.ones(m),(np.arange(m),jj)),(m,n)))
            
        if not redundant:
            if self.elim:
                if fat:
                    nnz_per_row = self.B[:,np.concatenate([self.Prim_pp[p][0] for p in range(self.space.numpatches)])].getnnz(axis=1)
                    self.eliminate_constraints = np.where(nnz_per_row > 0)[0]
                else:
                    dofs = [Nodes[key][0] for key in Nodes if isinstance(key,np.int32)]
                    if len(dofs)>0:
                        dofs = np.concatenate([Nodes[key][0] for key in Nodes if isinstance(key,np.int32)])
                    dofs_new = (self.Basis[dofs,:]==1).indices
                    self.eliminate_constraints = np.where(self.B[:,dofs_new].getnnz(axis=1))
            else:
                self.eliminate_constraints = (MP.Constr.tocsc()[:,np.concatenate([Nodes[key][0] for key in Nodes])]==1).indices
                    
        keep_constr = np.setdiff1d(np.arange(self.B.shape[0]),self.eliminate_constraints)
        self.B = self.B[keep_constr,:]
        self.Bk = [B[keep_constr,:] for B in self.Bk]
        self.C = scipy.sparse.block_diag(self.Ck)
            
                

class IetiSystem:
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
        
        if self.elim:
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
            
        self.N = [Ba.shape[1] for Ba in self.Basisk]
        self.N_ofs = np.cumsum([0]+self.N)
        self.Bk = [self.B[:,self.N_ofs[p]:self.N_ofs[p+1]] for p in range(self.space.numpatches)]
        self.Basis=scipy.sparse.block_diag(self.Basisk)
        self.P2Gk =[]
        
        for p in range(self.space.numpatches):
            X = self.Basisk[p].tocoo()
            idx = np.where(np.isclose(X.data,1))
            X.data, X.row, X.col = X.data[idx], X.row[idx], X.col[idx]
            D = (X.T@self.Basisk[p]).sum(axis=1).A.ravel()
            #assert all(abs(D)>1e-12), 'D has zeros.'
            #S = scipy.sparse.spdiags(1/D,[0],len(D),len(D))
            self.P2Gk.append(X.T)
            I = np.zeros(self.Basisk[p].shape[0])
            if p in self.dir_idx:
                I[self.dir_idx[p]] = 1
                self.dir_idx[p] = np.where(np.isclose(self.P2Gk[p]@I,1))[0]
        
        self.P2G = scipy.sparse.block_diag(self.P2Gk)
        self.global_dir_idx = np.concatenate([self.dir_idx[p] + self.N_ofs[p] for p in self.dir_idx])
        self.free_dofs = np.setdiff1d(np.arange(self.N_ofs[-1]),self.global_dir_idx)
        self.free_dofs_pp = [np.arange(self.N[p]) if p not in self.dir_idx else np.setdiff1d(np.arange(self.N[p]),self.dir_idx[p]) for p in range(self.space.numpatches)]
        
        #self.B = self.B @ scipy.sparse.block_diag(self.Basisk)
        
        nnz_per_col = self.B.getnnz(axis=0)
        self.intfs = np.setdiff1d(np.where(nnz_per_col > 0)[0], self.global_dir_idx)
        self.skeleton = np.union1d(self.intfs, self.global_dir_idx)
        
        self.Rbb = []
        self.Rii = []
        for p in range(self.space.numpatches):
            Id = scipy.sparse.eye(self.N[p], format='csr')
            mask = np.zeros(self.N[p], dtype=bool)
            nnz_per_col = self.Bk[p].getnnz(axis=0)
            if p in self.dir_idx:
                intfs = np.setdiff1d(np.where(nnz_per_col > 0)[0], self.dir_idx[p])
            else:
                intfs = np.where(nnz_per_col > 0)[0]
            mask[intfs]=True
            if p in self.dir_idx:
                mask[self.dir_idx[p]]=False
            self.Rbb.append(Id[mask].tocsc())
            mask = np.ones(self.N[p], dtype=bool)
            mask[intfs]=False
            if p in self.dir_idx:
                mask[self.dir_idx[p]]=False
            self.Rii.append(Id[mask].tocsc())
            
        
    def assemble(self, f):
        Ak = [Ba.T @ assemble.assemble('(inner(grad(u),grad(v)))* dx', kvs, bfuns=[('u',1), ('v',1)], geo=geo)@Ba for Ba, ((kvs, geo),_) in zip(self.Basisk, self.space.mesh.patches)]
        A = scipy.sparse.block_diag(Ak, format='csr')
        rhsk = [Ba.T @ assemble.assemble('f * v * dx', kvs, bfuns=[('v',1)], geo=geo, f=f).ravel() for Ba, ((kvs, geo),_) in zip(self.Basisk,self.space.mesh.patches)]

        # self.Abb = self.Rbb.dot(A).dot(self.Rbb.T)
        # self.Aii = self.Rii.dot(A).dot(self.Rii.T)
        # self.Abi = self.Rbb.dot(A).dot(self.Rii.T)
        # self.Aib = self.Abi.T
        
        BCRestr = {p:assemble.RestrictedLinearSystem(Ak[p], rhsk[p], (self.dir_idx[p],self.dir_vals[p])) for p in self.dir_idx}
        self.rhsk = [rhsk[p] if p not in self.dir_idx else BCRestr[p].b for p in range(self.space.numpatches)]
        self.Ak = [Ak[p] if p not in self.dir_idx else BCRestr[p].A for p in range(self.space.numpatches)]
        self.A = scipy.sparse.block_diag(self.Ak)
        # self.A = BCRestr.A
        # self.rhs = BCRestr.b
        
    def construct_primal_constraints(self):
        self.Ck = []
        self.Rk = []
        kvs = self.space.mesh.kvs
        geos = self.space.mesh.geos
        self.eliminate_constraints = np.array([], dtype=int)
        Nodes=self.space.get_nodes()
        self.Prim = {}
        
        if self.elim:
            total_dofs=set()
            i=0
            for key in Nodes:
                if isinstance(key,tuple):
                    dofs = self.Basis.tocsr()[Nodes[key][1],:].indices
                    #print(dofs)
                    for dof in dofs:
                        if dof not in total_dofs:
                            total_dofs.add(dof)
                            self.Prim[i] = np.unique(self.B.tocsr()[self.B.tocsc()[:,dof].indices,:].indices)
                            i+=1
                else:
                    self.Prim[i] = self.Basis.tocsr()[Nodes[key][0],:].indices
                    i+=1
            loc_c_prim = np.concatenate([self.Prim[key] for key in self.Prim])
            #print(loc_c_prim)
            loc_c_prim_idx = np.repeat(np.arange(len(self.Prim)),[len(self.Prim[i]) for i in self.Prim])
            self.Prim_pp = {p : (loc_c_prim[(loc_c_prim >= self.N_ofs[p]) & (loc_c_prim < self.N_ofs[p+1])],loc_c_prim_idx[(loc_c_prim >= self.N_ofs[p]) & (loc_c_prim < self.N_ofs[p+1])]) for p in range(self.space.numpatches)}
        else:
            self.Prim = {i: val for i,val in enumerate(self.space.get_nodes().values())}
            loc_c_prim = np.concatenate([Nodes[key][0] for key in Nodes])
            loc_c_prim_idx = np.repeat(np.arange(len(self.Prim)),[len(self.Prim[i][0]) for i in self.Prim])
            self.cpp = {p : (loc_c_prim[(loc_c_prim >= self.space.N_ofs[p]) & (loc_c_prim < self.space.N_ofs[p+1])],loc_c_prim_idx[(loc_c_prim >= self.space.N_ofs[p]) & (loc_c_prim < self.space.N_ofs[p+1])]) for p in range(self.space.numpatches)}
            self.tpp = {p : {key:val for key,val in self.Prim.items() if len(val)>1 and all((val[1] >= self.space.N_ofs[p]) & (val[1] < self.space.N_ofs[p+1]))} for p in range(self.space.numpatches)}
        
        for p in range(len(self.space.mesh.patches)):
            if p in self.dir_idx:
                to_eliminate = self.dir_idx[p]
            else:
                to_eliminate = np.array([])
            free = np.setdiff1d(np.arange(self.N[p]),to_eliminate)
            #c_primal_free = self.cpp[p][0] - self.space.N_ofs[p]
                
            if self.elim:
                c_primal_free = self.Prim_pp[p][0] - self.N_ofs[p]
                #print(c_primal_free)
                data = np.ones(len(c_primal_free))
                rows = np.arange(len(c_primal_free))
                cols = c_primal_free
                ck = coo_matrix((data, (rows, cols)),(len(c_primal_free),self.N[p])).tocsc()
                ck = ck[:,free]
                self.Ck.append(ck.tocsr())
                m, n = ck.shape[0], len(self.Prim)
                jj = self.Prim_pp[p][1]
                self.Rk.append(scipy.sparse.coo_matrix((np.ones(m),(np.arange(m),jj)),(m,n)))
                
                nnz_per_row = self.B[:,self.Prim_pp[p][0]].getnnz(axis=1)
                result = np.where(nnz_per_row > 0)[0]

                self.eliminate_constraints = np.union1d(result, self.eliminate_constraints)
            else:
                c_primal_free = self.cpp[p][0] - self.space.N_ofs[p]

                # if self.elim:
                #     I = np.zeros(self.Basisk[p].shape[0])
                #     I[c_primal_free] = 1
                #     c_primal_free = np.where(np.isclose(self.P2Gk[p]@I,1))[0]

                nnz_per_row = self.space.Constr[:,self.cpp[p][0]].getnnz(axis=1)
                result = np.where(nnz_per_row > 0)[0]

                self.eliminate_constraints = np.union1d(result, self.eliminate_constraints)

                data = np.ones(len(c_primal_free))
                rows = np.arange(len(c_primal_free))
                cols = c_primal_free
                ck = coo_matrix((data, (rows, cols)),(len(c_primal_free),self.space.N[p])).tocsc()

                V = []
                for t in self.tpp[p]:
                    constr = (self.space.Constr.tocsc()[:,self.tpp[p][t][0][0]]==1).indices
                    self.eliminate_constraints = np.union1d(constr, self.eliminate_constraints)
                    X = self.space.Constr[constr,:][:,self.space.N_ofs[p]:self.space.N_ofs[p+1]].tocsr()
                    V.append(X[X.getnnz(axis=1)>0,:])
                ck = (scipy.sparse.vstack([ck]+V)@self.Basisk[p]).tocsc()

                ck = ck[:,free]
                self.Ck.append(ck.tocsr())
                m, n = ck.shape[0], len(Nodes)
                jj = np.concatenate([self.cpp[p][1],np.array(list(self.tpp[p].keys()), dtype=int)])
                #print(m, jj)
                self.Rk.append(scipy.sparse.coo_matrix((np.ones(m),(np.arange(m),jj)),(m,n)))
                #print(ck.A)
            
        #self.eliminate_constraints = np.unique(self.B.tocsc()[:,loc_c_prim].indices)
        keep_constr = np.setdiff1d(np.arange(self.B.shape[0]),self.eliminate_constraints)
        self.B = self.B[keep_constr,:]
        self.Bk = [B[keep_constr,:] for B in self.Bk]
        self.C = scipy.sparse.block_diag(self.Ck)
        
    def construct_primal_basis(self):
        PsiK=[]
        
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
            PsiK.append(psi@self.Rk[p])

        self.Psi=np.vstack(PsiK)
        
    def compute_F(self):
        B = self.B[:,self.free_dofs]
        keep = np.where(B.getnnz(axis=1)>0)[0]
        B=B[keep,:]
        idx_p = [(self.free_dofs < self.N_ofs[p+1]) & (self.free_dofs >= self.N_ofs[p]) for p in range(self.space.numpatches)]
        Bk = [B[:,idx_p[p]] for p in range(self.space.numpatches)]
        #Bk = [b[np.where(b.getnnz(axis=1)>0)[0]] for b in Bk]
        Bk_ = [aslinearoperator(scipy.sparse.bmat([[b,np.zeros((b.shape[0],self.Ck[p].shape[0]))]], format='csr')) for p,b in enumerate(Bk)] 
        PTAP = self.Psi.T@self.A@self.Psi
        PTBT = self.Psi.T@B.T
        BP   = B@self.Psi
        
        rhs = np.concatenate(self.rhsk)
        rhsk_ = [np.concatenate([f,np.zeros(self.Ck[p].shape[0])]) for p,f in enumerate(self.rhsk)]
        
        loc_solver = [solvers.make_solver(scipy.sparse.bmat([[a,    c.T], [c,    None]], format='csr'), spd=False, symmetric=True) for a,c in zip(self.Ak, self.Ck)]
        F1 = aslinearoperator(BP@solvers.make_solver(PTAP, spd=True, symmetric=True).dot(BP.T)) 
        F2 = sum([b@Ak_inv.dot(b.T) for b, Ak_inv in zip(Bk_,loc_solver)])
        print(F1,F2)
        b1 = BP@solvers.make_solver(PTAP, spd=True, symmetric=True).dot(self.Psi.T@rhs)
        b2 = sum([b@Ak_inv@f for b, Ak_inv,f in zip(Bk_, loc_solver, rhsk_)])
        print(b1, b2)
        return F1+F2, b1+b2
    
    def MsD(self, pseudo=False):
        B = self.B[:,self.free_dofs]
        keep = np.where(B.getnnz(axis=1)>0)[0]
        B = B[keep]
        Bk = [self.Bk[p][keep,:][:,self.free_dofs_pp[p]] for p in range(self.space.numpatches)] 
        Rb = [self.Rbb[p][:,self.free_dofs_pp[p]] for p in range(self.space.numpatches)]
        Ri = [self.Rii[p][:,self.free_dofs_pp[p]] for p in range(self.space.numpatches)]
        #B_gamma = B_gamma[np.setdiff1d(np.arange(B_gamma.shape[0]),self.eliminate_constraints),:]
        self.B_gamma = scipy.sparse.hstack([Bk[p]@Rb[p].T for p in range(self.space.numpatches)])

        #print(np.linalg.matrix_rank(Rbb.A, 1e-8))
        #print(B_gamma.shape)

        #Aib = scipy.sparse.linalg.aslinearoperator(Aib)
        #AiiinvB = solvers.make_solver(self.Aii, spd=True)
        #AiiinvB = scipy.sparse.linalg.spsolve(self.Aii, self.Aib.A)
        #self.S = aslinearoperator(self.Abb) - aslinearoperator(self.Abi)@AiiinvB.dot(scipy.sparse.linalg.aslinearoperator(self.Aib))
        Abb = [aslinearoperator(Rb[p]@self.Ak[p]@Rb[p].T) for p in range(self.space.numpatches)]
        Aii = [Ri[p]@self.Ak[p]@Ri[p].T for p in range(self.space.numpatches)]
        Abi = [aslinearoperator(Rb[p]@self.Ak[p]@Ri[p].T) for p in range(self.space.numpatches)]
        
        self.S = [Abb - Abi@solvers.make_solver(Aii, spd=True).dot(Abi.T) for Abb,Abi,Aii in zip(Abb,Abi,Aii)]
        ofs = np.cumsum([0]+[s.shape[0] for s in self.S])
        #print(self.S)
        
        if self.elim:
            D = self.B_gamma.getnnz(axis=0)
            D = 1/(1+D)
            self.D = scipy.sparse.diags(D, format='csr')
        else:
            if pseudo:
                t = time.time()
                D = np.linalg.pinv(self.B_gamma.A)
                D[abs(D)<1e-16]=0.0
                print("computing the pseudoinverse and pruning took " + str(time.time()-t) + " seconds.")
                D=scipy.sparse.csr_matrix(D)
                self.D=D@D.T
            else:
                D = self.B_gamma.getnnz(axis=0)
                D = 1/(1+D)
                self.D = scipy.sparse.diags(D, format='csr')
        self.BgD = self.B_gamma@self.D
        # for p in range(self.space.numpatches):
        #     print(self.BgD[:,ofs[p]:ofs[p+1]].shape, self.S[p].shape, self.BgD[:,ofs[p]:ofs[p+1]].T.shape)
        #print(self.BgD.shape, ofs[-1])
        #return self.BgD@scipy.sparse.block_diag(self.S).dot(self.BgD.T)
        return sum([aslinearoperator(self.BgD[:,ofs[p]:ofs[p+1]])@self.S[p].dot(aslinearoperator(self.BgD[:,ofs[p]:ofs[p+1]].T)) for p in range(self.space.numpatches)])
            
    