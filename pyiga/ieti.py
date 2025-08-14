import numpy as np
import time
import scipy
from scipy.sparse import coo_matrix

from pyiga import bspline, vis, assemble, solvers, algebra, assemble
from pyiga import ieti_cy
from scipy.sparse.linalg import aslinearoperator as LinOp

class IetiMapper(assemble.MultiBasis):
    def __init__(self, M, dir_data, neu_data=None, elim=False):
        super().__init__(M)
        self.elim=bool(elim)

        self.set_constraints('C0')
        self.global_dir_idx,_ = self.set_dirichlet_boundary(dir_data)

        self.free={}
        for p in self.dir_idx:
            self.free[p] = np.setdiff1d(np.arange(self.N[p]),self.dir_idx[p],assume_unique=True)
            
        self.global_free = np.setdiff1d(np.arange(self.N_ofs[-1]),self.global_dir_idx, assume_unique=True)

        self.corners = np.concatenate([assemble.boundary_dofs(kvs,m=0,ravel=True)+self.N_ofs[p] for p, kvs in enumerate(self.mesh.kvs)])

        self.Bk = [self.B[:,self.N_ofs[p]:self.N_ofs[p+1]] for p in range(self.nPatches)]
        nnz_per_col = self.B.getnnz(axis=0)
        # self.intfs = np.setdiff1d(np.where(nnz_per_col > 0)[0], self.global_dir_idx)
        self.skeleton = np.setdiff1d(np.where(nnz_per_col > 0)[0], self.global_dir_idx)
        self.interior = np.setdiff1d(np.where(nnz_per_col == 0)[0], self.global_dir_idx)
        
        self.R_interior = self.nPatches*[None] ###TODO: without loops
        self.R_skeleton = self.nPatches*[None]
        self.R_interfaces = {}
        
        for p in range(self.nPatches):
            Id = scipy.sparse.eye(self.N[p], format='csr')
            mask_skeleton = np.zeros(self.N[p], dtype=bool)
            intfs = np.where(self.Bk[p].getnnz(0) > 0)[0]
            mask_interior = np.ones(self.N[p], dtype=bool)
            mask_interior[intfs]=False
            if p in self.dir_idx:
                mask_interior[self.dir_idx[p]]=False
            self.R_interior[p]=Id[mask_interior,:][:,self.free[p]]
            for b in range(4):
                if not any([(p,b) in self.mesh.outer_boundaries[key] for key in self.mesh.outer_boundaries]):
                    mask_intf = np.zeros(self.N[p], dtype=bool)
                    interface_dofs = assemble.boundary_dofs(self.mesh.kvs[p],bdspec=b,ravel=True)
                    mask_intf[interface_dofs[1:-1]] = True
                    mask_skeleton[interface_dofs] = True
                    if p in self.dir_idx:
                        mask_intf[self.dir_idx[p]]=False
                        mask_skeleton[self.dir_idx[p]]=False

                    self.R_interfaces[(p,b)] = Id[mask_intf,:][:,self.free[p]]
            self.R_skeleton[p] = Id[mask_skeleton,:][:,self.free[p]]

    def ConstraintMatrices(self, redundant = False):
        eliminated_constraints = np.repeat(False, self.B.shape[0])
        B = self.B.tocsc()
        if not redundant:
            eliminated_constraints = ieti_cy.eliminate_corner_constraints(B.indptr.astype(np.int32), B.indices.astype(np.int32), B.data, *B.shape, 
                                                                          self.corners.astype(np.int32), len(self.corners)).astype(bool)

        B = self.B[:,self.global_free]

        eliminated_constraints = eliminated_constraints | (B.getnnz(1)==0)
        B = B[~eliminated_constraints,:]
        ofs = np.cumsum([0]+[len(x) for x in self.free])
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
        if not dir_boundary: loc_c = np.setdiff1d(loc_c, self.global_dir_idx, assume_unique=True) 
  
        idx = (self.B[:,loc_c].getnnz(1)>0) & (self.B.getnnz(1)==2)
        B = self.B[idx,:]
        #B = B[B.getnnz(1)==2,:]
        loc_c = np.unique(B.indices)
        if not dir_boundary: loc_c = np.setdiff1d(loc_c, self.global_dir_idx, assume_unique=True) 

        q = np.where(B.getnnz(0)>0)[0]
        R = scipy.sparse.coo_matrix((np.ones(len(q)),(np.arange(len(q)),q)),shape=(len(q),n)).tocsr()
        c_B = B@R.T
        c_B.eliminate_zeros()

        nodal_coeff = R.T@algebra_cy.pyx_compute_basis(c_B.shape[0], c_B.shape[1], c_B, maxiter=10)
        #print(nodal_coeff)

        nodal_coeff += ieti_cy.identify_T_coefficients_from_corner_basis(nodal_coeff.indptr, nodal_coeff.indices, nodal_coeff.data, *nodal_coeff.shape,
                                                                         self.B.indptr,      self.B.indices,      self.B.data,       self.B.shape[0], deg)
        return nodal_coeff

    def completeDirichlet(self, U):
        return [self.BCRestr[p].complete(u) if p in self.BCRestr else u for p,u in enumerate(U)]
    