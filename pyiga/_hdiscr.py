import numpy as np
import scipy.sparse
from . import assemble, compile, mlmatrix

def _assemble_partial_rows(asm, row_indices):
    """Assemble a submatrix which contains only the given rows."""
    kvs0, kvs1 = asm.kvs
    S = mlmatrix.MLStructure.from_kvs(kvs0, kvs1)
    I, J = S.nonzeros_for_rows(row_indices)     # the nonzero indices in the given rows
    data = [asm.entry(i, j) for (i,j) in zip(I,J)]
    return scipy.sparse.coo_matrix((data, (I,J)), shape=S.shape).tocsr()

class HDiscretization:
    def __init__(self, hspace, vform, asm_args, truncate=False):
        self.hs = hspace
        self._I_hb = hspace.represent_fine(truncate=truncate)
        self.truncate = truncate
        self.asm_class = compile.compile_vform(vform, on_demand=False)
        self.asm_args = asm_args

    def _assemble_level(self, k, rows=None):
        asm = self.asm_class(self.hs.knotvectors(k), **self.asm_args)
        if rows is None:
            return assemble.assemble(asm, symmetric=True)
        else:
            return _assemble_partial_rows(asm, rows)

    def assemble_matrix(self):
        if self.truncate:
            # compute HB version and transform it
            # HACK - overwrite truncate flag and replace it afterwards
            try:
                self.truncate = False
                A_hb = self.assemble_matrix()
            finally:
                self.truncate = True
            T = self.hs.thb_to_hb()
            return T.T @ A_hb @ T
        else:
            hs = self.hs
            # compute dofs interacting with active dofs on each level
            neighbors = hs.cell_supp_indices(remove_dirichlet=False)
            # interactions on the same level are handled separately, so remove them
            for k in range(hs.numlevels):
                neighbors[k][k] = []

            # Determine the rows of the matrix we need to assemble:
            # 1. dofs for interlevel contributions - all level k dofs which are required
            #    to represent the coarse functions which interact with level k
            # 2. all active dofs on level k
            to_assemble, interlevel_ix = [], []
            for k in range(hs.numlevels):
                indices = set()
                for lv in range(max(0, k - hs.disparity), k):
                    indices |= set(hs.hmesh.function_babys(lv, neighbors[k][lv], k))
                interlevel_ix.append(indices)
                to_assemble.append(indices | hs.actfun[k])

            # convert them to raveled form
            to_assemble = hs._ravel_indices(to_assemble)
            interlevel_ix = hs._ravel_indices(interlevel_ix)

            # compute neighbors as matrix indices
            neighbors = [hs._ravel_indices(idx) for idx in neighbors]
            neighbors = hs.raveled_to_virtual_matrix_indices(neighbors)

            # new indices per level as local tensor product indices
            new_loc = hs.active_indices()
            # new indices per level as global matrix indices
            na = tuple(len(ii) for ii in new_loc)
            new = [np.arange(sum(na[:k]), sum(na[:k+1])) for k in range(hs.numlevels)]

            kvs = tuple(hs.knotvectors(lv) for lv in range(hs.numlevels))
            As = [self._assemble_level(k, rows=to_assemble[k]) for k in range(hs.numlevels)]
            # I_hb[k]: maps HB-coefficients to TP coefficients on level k
            I_hb = [hs.represent_fine(lv=k, rows=to_assemble[k]) for k in range(hs.numlevels)]

            # the diagonal block consisting of interactions on the same level
            A_hb_new = [As[k][new_loc[k]][:,new_loc[k]]
                    for k in range(hs.numlevels)]
            # the off-diagonal blocks which describe interactions with coarser levels
            A_hb_interlevel = [(I_hb[k][interlevel_ix[k]][:, neighbors[k]].T
                                @ As[k][interlevel_ix[k]][:, new_loc[k]]
                                @ I_hb[k][new_loc[k]][:, new[k]])
                               for k in range(hs.numlevels)]

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
