import numpy as np
import scipy.sparse
from . import assemble, compile, mlmatrix

def _assemble_partial_rows(asm, row_indices):
    """Assemble a submatrix which contains only the given rows."""
    kvs0, kvs1 = asm.kvs
    S = mlmatrix.MLStructure.from_kvs(kvs0, kvs1)
    I, J = S.nonzeros_for_rows(row_indices)     # the nonzero indices in the given rows
    data = asm.multi_entries(np.column_stack((I,J)))
    return scipy.sparse.coo_matrix((data, (I,J)), shape=S.shape).tocsr()

class HDiscretization:
    """Represents the discretization of a variational problem over a
    hierarchical spline space.

    Args:
        hspace (:class:`HSpace`): the HB- or THB-spline space in which to discretize
        vform (:class:`.VForm`): the variational form describing the problem
        asm_args (dict): a dictionary which provides named inputs for the assembler. Most
            problems will require at least a geometry map; this can be given in
            the form ``{'geo': geo}``, where ``geo`` is a geometry function
            defined using the :mod:`pyiga.geometry` module. Further inputs
            declared via the :meth:`.VForm.input` method must be included in
            this dict.

            The assemblers both for the matrix and any linear functionals will draw
            their input arguments from this dict.
    """
    def __init__(self, hspace, vform, asm_args):
        self.hs = hspace
        self.truncate = hspace.truncate
        self.vf = vform
        self.asm_args = asm_args
        self.asm_class = None

    def _assemble_level(self, k, rows=None, bbox=None, symmetric=False):
        """Assemble a subset of the rows of the full tensor product stiffness
        matrix on a given level `k`."""
        if rows is not None and len(rows) == 0:
            # work around a Cython bug with contiguity of 0-sized arrays:
            # https://github.com/cython/cython/issues/2093
            n = np.prod(self.hs.mesh(k).numdofs)
            return scipy.sparse.csr_matrix((n,n))

        # get needed assembler arguments
        asm_args = { inp.name: self.asm_args[inp.name]
                for inp in self.vf.inputs}
        asm_args['bbox'] = bbox

        if not self.asm_class:
            self.asm_class = compile.compile_vform(self.vf, on_demand=True)
        asm = self.asm_class(self.hs.knotvectors(k), **asm_args)
        if rows is None:
            return assemble.assemble(asm, symmetric=symmetric)
        else:
            return _assemble_partial_rows(asm, rows)

    def assemble_matrix(self, symmetric=False):
        """Assemble the stiffness matrix for the hierarchical discretization and return it.

        Returns:
            a sparse matrix whose size corresponds to the
            :attr:`HSpace.numdofs` attribute of `hspace`
        """
        if self.truncate:
            # compute HB version and transform it
            # HACK - overwrite truncate flag and replace it afterwards
            try:
                self.truncate = False
                A_hb = self.assemble_matrix(symmetric=symmetric)
            finally:
                self.truncate = True
            T = self.hs.thb_to_hb()
            return (T.T @ A_hb @ T).tocsr()
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
            bboxes = []
            for k in range(hs.numlevels):
                indices = set()
                for lv in range(max(0, k - hs.disparity), k):
                    indices |= set(hs.hmesh.function_grandchildren(lv, neighbors[k][lv], k))
                interlevel_ix.append(indices)
                to_assemble.append(indices | hs.actfun[k])

                # compute a bounding box for the supports of all functions to be assembled
                bboxes.append(self._bbox_for_functions(k, to_assemble[-1]))

            # convert them to raveled form
            to_assemble = hs.ravel_indices(to_assemble)
            interlevel_ix = hs.ravel_indices(interlevel_ix)

            # compute neighbors as matrix indices
            neighbors = [hs.raveled_to_virtual_canonical_indices(lv, hs.ravel_indices(idx))
                    for lv, idx in enumerate(neighbors)]

            # new indices per level as local tensor product indices
            new_loc = hs.active_indices()
            # new indices per level as global matrix indices
            na = tuple(len(ii) for ii in new_loc)
            new = [np.arange(sum(na[:k]), sum(na[:k+1])) for k in range(hs.numlevels)]

            # assemble the matrix from the levelwise contributions
            coo_I, coo_J, values = [], [], []   # blockwise COO data

            def insert_block(B, rows, columns):
                # store the block B into the given rows/columns of the output matrix
                B = B.tocsr()       # this does nothing if B is already CSR
                I, J = B.nonzero()
                coo_I.append(rows[I])
                coo_J.append(columns[J])
                values.append(B.data)

            for k in range(hs.numlevels):
                # compute the needed rows of the tensor product stiffness matrix
                A_k = self._assemble_level(k, rows=to_assemble[k], bbox=bboxes[k], symmetric=symmetric)

                # matrix which maps HB-coefficients to TP coefficients on level k
                I_hb_k = hs.represent_fine(lv=k, truncate=False, rows=to_assemble[k])

                # compute the diagonal block consisting of interactions on the same level
                A_hb_new = A_k[new_loc[k]][:,new_loc[k]]

                # store the diagonal block
                insert_block(A_hb_new, new[k], new[k])

                # compute the off-diagonal block(s) which describe interactions with coarser levels
                A_hb_interlevel = (I_hb_k[interlevel_ix[k]][:, neighbors[k]].T
                                    @ A_k[interlevel_ix[k]][:, new_loc[k]]
                                    @ I_hb_k[new_loc[k]][:, new[k]])

                if symmetric:
                    A_hb_interlevel2 = A_hb_interlevel.T
                else:
                    A_hb_interlevel2 = (I_hb_k[new_loc[k]][:, new[k]].T
                                        @ A_k[new_loc[k]][:, interlevel_ix[k]]
                                        @ I_hb_k[interlevel_ix[k]][:, neighbors[k]])

                # store the two blocks containing interactions with coarser levels
                insert_block(A_hb_interlevel,  neighbors[k], new[k])
                insert_block(A_hb_interlevel2, new[k], neighbors[k])

            # convert the blockwise COO data into a CSR matrix
            coo_I  = np.concatenate(coo_I)
            coo_J  = np.concatenate(coo_J)
            values = np.concatenate(values)
            return scipy.sparse.csr_matrix((values, (coo_I, coo_J)),
                    shape=(hs.numdofs, hs.numdofs))

    def assemble_rhs(self, vf=None):
        """Assemble the right-hand side vector for the hierarchical discretization and return it.

        By default (if `vf=None`), a standard L2 inner product `<f, v>` is used
        for computing the right-hand side, and the function `f` is taken from
        the key ``'f'`` of the ``asm_args`` dict. It is assumed to be given in
        physical coordinates.

        A different functional can be specified by passing a :class:`.VForm`
        with ``arity=1`` as the `vf` parameter.

        Returns:
            a vector whose length is equal to the :attr:`HSpace.numdofs`
            attribute of `hspace`, corresponding to the active basis functions
            in canonical order
        """
        if vf is None:
            from .vform import L2functional_vf
            vf = L2functional_vf(dim=self.hs.dim, physical=True)
        return self.assemble_functional(vf)

    def assemble_functional(self, vf):
        """Assemble a linear functional described by the :class:`.VForm` `vf` over
        the hierarchical spline space.

        Returns:
            a vector whose length is equal to the :attr:`HSpace.numdofs`
            attribute of `hspace`, corresponding to the active basis functions
            in canonical order
        """
        if not vf.arity == 1:
            raise ValueError('vf must be a linear functional (arity=1)')
        RhsAsm = compile.compile_vform(vf, on_demand=True)

        # get needed assembler arguments
        asm_args = { inp.name: self.asm_args[inp.name]
                for inp in vf.inputs}

        def asm_rhs_level(k, rows):
            if len(rows) == 0:
                return np.zeros(0)

            # determine bounding box for active functions
            bbox = self._bbox_for_functions(k, self.hs.actfun[k])

            kvs = self.hs.knotvectors(k)
            asm_args['bbox'] = bbox
            asm = RhsAsm(kvs, **asm_args)
            return asm.multi_entries(rows)

        act = self.hs.active_indices()
        na = tuple(len(ii) for ii in act)
        rhs = np.zeros(self.hs.numdofs)
        i = 0
        # collect the contributions from the active functions per level
        for k, na_k in enumerate(na):
            rhs[i:i+na_k] = asm_rhs_level(k, act[k])
            i += na_k

        # if using THBs, apply the transformation matrix
        if self.truncate:
            rhs = self.hs.thb_to_hb().T @ rhs
        return rhs

    def _bbox_for_functions(self, lv, funcs):
        """Compute a bounding box for the supports of the given functions on the given level."""
        supp_cells = np.array(sorted(self.hs.mesh(lv).support(funcs)))
        if len(supp_cells) == 0:
            return tuple((0,0) for j in range(self.hs.dim))
        else:
            return tuple(
                    (supp_cells[:,j].min(), supp_cells[:,j].max() + 1)  # upper limit is exclusive
                    for j in range(supp_cells.shape[1]))
