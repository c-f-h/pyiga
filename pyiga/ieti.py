import numpy as np
import time

from pyiga import bspline, vis, assemble

class IetiDP:
    def __init__(self, MP, dir_data, neu_data=None):
        self.space = MP
        
        idx=[]
        vals=[]
        kvs = MP.mesh.kvs
        geos = MP.mesh.geos
        for key in dir_data:
            for p,b in MP.mesh.outer_boundaries[key]:
                idx_, vals_ = assemble.compute_dirichlet_bc(kvs[p], geos[p], [(b//2,b%2)], dir_data[key])
                idx.append(idx_ + MP.N_ofs[p])
                vals.append(vals_)
        self.dir_ofs = np.cumsum(np.array([len(np.unique(idx[p])) for p in range(MP.numpatches)]))
        
        self.dir_idx, lookup = np.unique(idx, return_index=True)
        self.dir_vals = np.concatenate(vals)[lookup]
        
    def assemble(self):
        Ak = [assemble.assemble('inner(grad(u),grad(v)) * dx', kvs, bfuns=[('u',1), ('v',1)], geo=g) for (kvs, geo),_ in self.space.patches]
        A = scipy.sparse.block_diag(Ak, format='csr')
        rhs = np.concatenate([assemble.assemble('f * v * dx', kvs, params, bfuns=[('v',1)], geo=g).ravel() for (kvs, geo),_ in self.space.patches])
        
        kvs = MP.mesh.kvs
        geos=MP.mesh.geos
        
        idx=[]
        vals=[]
        for key in dir_bcs:
            for p,b in MP.mesh.outer_boundaries[key]:
                idx_, vals_ = assemble.compute_dirichlet_bc(kvs[p], geos[p], [(b//2,b%2)], dir_bcs[key])
                idx.append(idx_+MP.N_ofs[p])
                vals.append(vals_)
        
        uidx, lookup = np.unique(idx, return_index=True)
        vals = np.concatenate(vals)
        
        BCRestr = assemble.RestrictedLinearSystem(A, rhs, bcs)
        self.Ak
        self.A = BCRestr.A
        self.rhs = BCRestr.b
        
#     def construct_primal_constraints(self):
#         Ck = []
#         loc_dirichlet = []
#         for p in range(len(MP.mesh.patches)):
#             bndindices = (bcs[0] < MP.N_ofs[p+1]) & (bcs[0] >= MP.N_ofs[p])
#             to_eliminate = bcs[0][bndindices]-MP.N_ofs[p]
#             free = np.setdiff1d(np.arange(MP.N[p]),to_eliminate)
#             primal = assemble.boundary_dofs(kvs, m=0, ravel=1)
#             primal_free = np.setdiff1d(primal, to_eliminate)

#             #loc, _ = cpp[p][0]

#             data = np.ones(len(primal_free))
#             rows = np.arange(len(primal_free))
#             cols = primal_free
#             ck = coo_matrix((data, (rows, cols)),(len(primal_free),MP.N[p])).tocsc()
#             ck = ck[:,free]
#             Ck.append(ck)
        
    def construct_primal_basis(self):
        Psik = []
        for p in range(len(patches)):
            a = self.A[p*Ck[p].shape[1]:(p+1)*Ck[p].shape[1], p*Ck[p].shape[1]:(p+1)*Ck[p].shape[1]]
            c = Ck[p]
            AC = scipy.sparse.bmat(
            [[a, c.T],
             [c,  None   ]], format='csr')
            RHS = np.vstack([np.zeros((a.shape[0],c.shape[0])), np.identity(c.shape[0])])
            psi = scipy.sparse.linalg.spsolve(AC, RHS)
            psi, delta = psi[:a.shape[0],], psi[a.shape[0]:,]
        
        Psik.append(psi)
        self.Psi = scipy.sparse.block_diag(Psik)
        
    