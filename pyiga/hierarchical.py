import numpy as np
import scipy.sparse
import itertools

from . import bspline, utils

def _make_unique(L):
    """Return a list which contains the entries of `L` with consecutive duplicates removed."""
    n = len(L)
    if n == 0:
        return []
    else:
        U = [L[0]]
        last = U[0]
        for x in L[1:]:
            if x != last:
                U.append(x)
            last = x
        return U

def _compute_supported_functions(kv, meshsupp):
    """Compute an array containing for each cell the index of the first and
    one beyond the last function supported in it.
    """
    n = kv.numspans
    sf = np.zeros((n,2), dtype=meshsupp.dtype)
    sf[:,0] = kv.numdofs
    for j in range(meshsupp.shape[0]):
        for k in range(meshsupp[j,0], meshsupp[j,1]):
            sf[k,0] = min(sf[k,0], j)
            sf[k,1] = max(sf[k,1], j)
    sf[:,1] += 1
    return sf

class TPMesh:
    """A tensor product mesh described by a sequence of knot vectors."""
    def __init__(self, kvs):
        self.kvs = tuple(kvs)
        self.dim = len(kvs)
        self.numspans = [kv.numspans for kv in kvs]
        self.numel = np.prod(self.numspans)
        self.numdofs = [kv.numdofs for kv in kvs]
        self.numbf = np.prod(self.numdofs)
        self.meshsupp = tuple(kv.mesh_support_idx_all() for kv in self.kvs)
        self.suppfunc = tuple(_compute_supported_functions(kv,ms) for (kv,ms) in zip(self.kvs, self.meshsupp))

    def refine(self):
        return TPMesh([kv.refine() for kv in self.kvs])

    def cells(self):
        """Return a list of all cells in this mesh."""
        return list(itertools.product(
            *(range(n) for n in self.numspans)))

    def cell_extents(self, c):
        """Return the extents (as a tuple of pairs) of the cell `c`."""
        return tuple((kv.mesh[cd], kv.mesh[cd+1]) for (kv,cd) in zip(self.kvs, c))

    def functions(self):
        """Return a list of all basis functions defined on this mesh."""
        return list(itertools.product(
            *(range(n) for n in self.numdofs)))

    def _support_1d(self, dim, j):
        ms = self.meshsupp[dim]
        return range(ms[j,0], ms[j,1])

    def _supported_in_1d(self, dim, k):
        sf = self.suppfunc[dim]
        return range(sf[k,0], sf[k,1])

    def support(self, indices):
        """Return the list of cells where any of the given functions does not vanish."""
        supp = []
        for jj in indices:
            supp.extend(itertools.product(
                *(self._support_1d(d, j) for (d,j) in enumerate(jj))))
        supp.sort()
        return _make_unique(supp)

    def supported_in(self, cells):
        """Return the list of functions whose support intersects the given cells."""
        funcs = []
        for kk in cells:
            funcs.extend(itertools.product(
                *(self._supported_in_1d(d, k) for (d,k) in enumerate(kk))))
        funcs.sort()
        return _make_unique(funcs)

    def neighbors(self, indices):
        """Return all functions which have nontrivial support intersection with the given ones."""
        return self.supported_in(self.support(indices))

class HMesh:
    """A hierarchical mesh built on a sequence of uniformly refined tensor product meshes."""
    def __init__(self, mesh):
        self.dim = mesh.dim
        self.meshes = [mesh]
        self.active = [set(mesh.cells())]
        self.deactivated = [set()]
        self.P = []

    def add_level(self):
        self.meshes.append(self.meshes[-1].refine())
        self.active.append(set())
        self.deactivated.append(set())
        self.P.append(tuple(
            bspline.prolongation(k0, k1).tocsc() for (k0,k1)
            in zip(self.meshes[-2].kvs, self.meshes[-1].kvs)))

    def cell_children(self, lv, cells):
        assert 0 <= lv < len(self.meshes) - 1, 'Invalid level'
        children = []
        for c in cells:
            children.extend(itertools.product(
                *(range(2*ci, 2*(ci + 1)) for ci in c)))
        return children

    def cell_parent(self, lv, cells):
        assert 1 <= lv < len(self.meshes), 'Invalid level'
        parents = []
        for c in cells:
            parents.append(tuple(ci // 2 for ci in c))
        parents.sort()
        return _make_unique(parents)

    def _function_children_1d(self, lv, dim, j):
        assert 0 <= lv < len(self.meshes) - 1, 'Invalid level'
        return list(self.P[lv][dim].getcol(j).nonzero()[0])

    def _function_parents_1d(self, lv, dim, j):
        assert 0 < lv < len(self.meshes), 'Invalid level'
        return list(self.P[lv-1][dim].getrow(j).nonzero()[1])

    def function_children(self, lv, indices):
        children = []
        for jj in indices:
            children.extend(itertools.product(
                *(self._function_children_1d(lv, d, j) for (d,j) in enumerate(jj))))
        children.sort()
        return _make_unique(children)

    def function_parents(self, lv, indices):
        parents = []
        for jj in indices:
            parents.extend(itertools.product(
                *(self._function_parents_1d(lv, d, j) for (d,j) in enumerate(jj))))
        parents.sort()
        return _make_unique(parents)

    def refine(self, marked):
        new_cells = dict()
        if marked.get(len(self.meshes) - 1):
            # if any cells on finest mesh marked, add a new level
            self.add_level()
        for lv in range(len(self.meshes) - 1):
            cells = set(marked.get(lv, []))
            # deactivate refined cells
            self.active[lv] -= cells
            self.deactivated[lv] |= cells
            # add children
            new_cells[lv+1] = self.cell_children(lv, cells)
            self.active[lv+1] |= set(new_cells[lv+1])
        return new_cells


class HSpace:
    """Represents a HB-spline or THB-spline space over an adaptively refined mesh.

    Arguments:
        kvs: a sequence of :class:`pyiga.bspline.KnotVector` instances, representing
            the tensor product B-spline space on the coarsest level
    """
    def __init__(self, kvs):
        tp = TPMesh(kvs)
        hmesh = HMesh(tp)
        assert len(hmesh.meshes) == 1
        self.dim = hmesh.dim
        self.hmesh = hmesh
        self.actfun = [set(hmesh.meshes[0].functions())]
        self.deactfun = [set()]

    def _add_level(self):
        self.actfun.append(set())
        self.deactfun.append(set())
        assert len(self.actfun) == len(self.deactfun) == len(self.hmesh.meshes)

    @property
    def numlevels(self):
        """The number of levels in this hierarchical space."""
        return len(self.hmesh.meshes)

    @property
    def numdofs(self):
        """The total number of active basis functions in this hierarchical space."""
        return sum(self.numactive)

    @property
    def numactive(self):
        """A tuple containing the number of active basis functions per level."""
        return tuple(len(af) for af in self.actfun)

    def mesh(self, lvl):
        """Return the underlying :class:`TPMesh` on the given level."""
        return self.hmesh.meshes[lvl]

    def active_cells(self, lvl=None):
        """If `lvl` is specified, return the set of active cells on that level.
        Otherwise, return a list containing, for each level, the set of active cells.
        """
        if lvl is not None:
            return self.hmesh.active[lvl]
        else:
            return [self.active_cells(lv) for lv in range(self.numlevels)]

    def deactivated_cells(self, lvl=None):
        """If `lvl` is specified, return the set of deactivated cells on that level.
        Otherwise, return a list containing, for each level, the set of deactivated cells.
        """
        if lvl is not None:
            return self.hmesh.deactivated[lvl]
        else:
            return [self.deactivated_cells(lv) for lv in range(self.numlevels)]

    def _ravel_indices(self, indexsets):
        return tuple(
            (np.ravel_multi_index(np.array(sorted(indexsets[lv])).T, self.mesh(lv).numdofs, order='C')
                if len(indexsets[lv])
                else np.arange(0))
            for lv in range(self.numlevels)
        )

    def active_indices(self):
        """Return a tuple which contains, per level, the raveled (sequential) indices of
        active basis functions.
        """
        return self._ravel_indices(self.actfun)

    def deactivated_indices(self):
        """Return a tuple which contains, per level, the raveled (sequential) indices of
        deactivated basis functions.
        """
        return self._ravel_indices(self.deactfun)

    def function_support(self, lv, jj):
        """Return the support (as a tuple of pairs) of the function on level `lv` with index `jj`."""
        kvs = self.mesh(lv).kvs
        meshsupps = (kv.mesh_support_idx(j) for (kv,j) in zip(kvs, jj))
        return tuple((kv.mesh[lohi[0]], kv.mesh[lohi[1]]) for (kv,lohi) in zip(kvs,meshsupps))

    def _functions_to_deactivate(self, marked):
        mf = dict()
        # for now assuming marked cells, not functions
        for lv in range(len(self.hmesh.meshes)):
            m = marked.get(lv)
            if not m:
                mf[lv] = set()
            else:
                # can only deactivate active functions
                mfuncs = set(self.mesh(lv).supported_in(m)) & self.actfun[lv]
                # A function is deactivated when all the cells of its level within
                # the support are deactivated.
                mf[lv] = set(f for f in mfuncs
                        if not (set(self.mesh(lv).support([f])) & self.hmesh.active[lv]))
        return mf

    def refine(self, marked):
        """Refine the given cells; `marked` is a dictionary which has the
        levels as indices and the list of marked cells on that level as values.
        """
        new_cells = self.hmesh.refine(marked)
        mf = self._functions_to_deactivate(marked)

        if len(self.hmesh.meshes) > len(self.actfun):
            self._add_level()

        for lv in range(len(self.hmesh.meshes) - 1):
            mfuncs = mf[lv]
            # deactivate the marked functions
            self.actfun[lv] -= mfuncs
            self.deactfun[lv] |= mfuncs
            # find candidate functions on fine level to be activated
            candidate_funcs = self.mesh(lv+1).supported_in(new_cells[lv+1])
            # ignore functions that are already active
            candidate_funcs = set(candidate_funcs) - self.actfun[lv+1]

            # set of active or deactivated cells on finer level
            fine_cells = self.hmesh.active[lv+1] | self.hmesh.deactivated[lv+1]
            # keep only those functions whose support is contained in those fine cells
            newfuncs = set(f for f in candidate_funcs if
                set(self.mesh(lv+1).support([f])).issubset(fine_cells))
            # activate them on the finer level
            self.actfun[lv+1] |= newfuncs

    def represent_fine(self, truncate=False):
        """Compute a matrix which represents all active HB-spline basis functions on the fine level.

        The returned matrix has size `N_fine x N_act`, where `N_fine` is the number of degrees
        of freedom in the finest tensor product mesh and `N_act` is the total number of active
        basis functions across all levels.

        If `truncate` is True, the representation of the THB-spline (truncated) basis functions
        is computed instead.
        """
        # raveled indices for active functions per level
        act_indices = self.active_indices()
        blocks = []

        for lv in reversed(range(self.numlevels)):
            Nj = self.mesh(lv).numbf

            if lv == self.numlevels - 1:
                P = scipy.sparse.eye(Nj)
            else:
                Pj = utils.multi_kron_sparse(self.hmesh.P[lv], format='lil')
                if truncate:
                    Pj[act_indices[lv+1], :] = 0
                P = P.dot(Pj)

            act_to_all = scipy.sparse.eye(Nj, format='csr')[:, act_indices[lv]]

            blocks.append(P.dot(act_to_all))

        blocks.reverse()
        return scipy.sparse.bmat([blocks], format='csr')

    def split_coeffs(self, x):
        """Given a coefficient vector `x` of length `numdofs`, split it into `numlevels` vectors
        which contain the contributions from each individual level.
        """
        j = 0
        result = []
        for af in self.actfun:
            nk = len(af)
            result.append(x[j : j+nk])
            j += nk
        assert j == x.shape[0], 'Wrong length of input vector'
        return result
