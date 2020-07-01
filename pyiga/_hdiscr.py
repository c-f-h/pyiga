import numpy as np
import scipy.sparse
from . import assemble, compile

class HDiscretization:
    def __init__(self, hspace, vform, asm_args, truncate=False):
        self.hs = hspace
        self._I_hb = hspace.represent_fine(truncate=truncate)
        self.truncate = truncate
        self.asm_class = compile.compile_vform(vform, on_demand=True)
        self.asm_args = asm_args

    def _assemble_level(self, k):
        asm = self.asm_class(self.hs.knotvectors(k), **self.asm_args)
        return assemble.assemble(asm, symmetric=True)

    def assemble_matrix(self):
        if self.truncate:
            # levelwise assembling not implemented for THBs
            A_fine = self._assemble_level(-1)
            return (self._I_hb.T @ A_fine @ self._I_hb).tocsr()
        else:
            hs = self.hs
            # compute dofs interacting with active dofs on each level
            neighbors = hs.cell_supp_indices(remove_dirichlet=False)
            # interactions on the same level are handled separately, so remove them
            for k in range(hs.numlevels):
                neighbors[k][k] = []

            # dofs to be assembled for interlevel contributions - all level k dofs which
            # are required to represent the coarse functions which interact with level k
            to_assemble = []
            for k in range(hs.numlevels):
                to_assemble.append(set())
                for lv in range(max(0, k - hs.disparity), k):
                    to_assemble[-1] |= set(hs.hmesh.function_babys(lv, neighbors[k][lv], k))
            # convert them to raveled form
            to_assemble = hs._ravel_indices(to_assemble)

            # compute neighbors as matrix indices
            neighbors = [hs._ravel_indices(idx) for idx in neighbors]
            neighbors = hs.raveled_to_virtual_matrix_indices(neighbors)

            # new indices per level as local tensor product indices
            new_loc = hs.active_indices()
            # new indices per level as global matrix indices
            na = tuple(len(ii) for ii in hs.active_indices())
            new = [np.arange(sum(na[:k]), sum(na[:k+1])) for k in range(hs.numlevels)]

            kvs = tuple(hs.knotvectors(lv) for lv in range(hs.numlevels))
            As = [self._assemble_level(k) for k in range(hs.numlevels)]
            # I_hb[k]: maps HB-coefficients to TP coefficients on level k
            I_hb = [hs.represent_fine(lv=k) for k in range(hs.numlevels)]

            # the diagonal block consisting of interactions on the same level
            A_hb_new = [As[k][new_loc[k]][:,new_loc[k]]
                    for k in range(hs.numlevels)]
            # the off-diagonal blocks which describe interactions with coarser levels
            A_hb_interlevel = [(I_hb[k][to_assemble[k]][:, neighbors[k]].T
                                @ As[k][to_assemble[k]][:, new_loc[k]]
                                @ I_hb[k][new_loc[k]][:, new[k]])
                               for k in range(hs.numlevels)]
            #A_hb_interlevel = [(I_hb[k][:, neighbors[k]].T @ As[k] @ I_hb[k][:, new[k]])
            #        for k in range(hs.numlevels)]

            # assemble the matrix from the levelwise contributions
            A = scipy.sparse.lil_matrix((hs.numdofs, hs.numdofs))

            for k in range(hs.numlevels):
                # store the k-th diagonal block
                A[np.ix_(new[k], new[k])] = A_hb_new[k]
                A[np.ix_(neighbors[k], new[k])] = A_hb_interlevel[k]
                A[np.ix_(new[k], neighbors[k])] = A_hb_interlevel[k].T
            return A

    def assemble_rhs(self, f):
        kvs_fine = self.hs.knotvectors(-1)
        f_fine = assemble.inner_products(kvs_fine, f).ravel()
        return self._I_hb.T @ f_fine
