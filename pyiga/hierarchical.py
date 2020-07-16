# -*- coding: utf-8 -*-
"""This module contains support for dealing with hierarchical spline spaces and
truncated hierarchical B-splines (THB-splines).

The main user-facing class is :class:`HSpace`, which describes a hierarchical
spline space and supports HB- and THB-spline representations. In order to
compute the stiffness matrix and right-hand side vector for the Galerkin
discretization of a variational problem in such a hierarchical spline space,
use the :class:`HDiscretization` class.

The implementation is loosely based on the approach described in [GV2018]_ and
the corresponding implementation in [GeoPDEs]_.

A tensor product B-spline basis function is usually referred to by a
multi-index represented as a tuple `(i_1, ..., i_d)`, where `d` is the space
dimension and `i_k` is the index of the univariate B-spline function used in
the `k`-th coordinate direction. Similarly, cells in the underlying tensor
product mesh are indexed by multi-indices `(j_1, .., j_d)`, where `j_k` is the
index of the knot span along the `k`-th axis.

Whenever an ordering of the degrees of freedom in a hierarchical spline space
is required, for instance when assembling a stiffness matrix, we use the
following **canonical order**: first, all active basis function on the coarsest
level, then all active basis functions on the next finer level, and so on until
the finest level. Within each level, the functions are ordered
lexicographically with respect to their tensor product multi-index `(i_1, ...,
i_d)`.

The canonical order on active cells is defined in the same way.

.. [GV2018] `Garau, Vazquez: "Algorithms for the implementation of adaptive
    isogeometric methods using hierarchical B-splines", 2018.
    <https://doi.org/10.1016/j.apnum.2017.08.006>`_
.. [GeoPDEs] http://rafavzqz.github.io/geopdes/
"""
import numpy as np
import scipy.sparse
import itertools

from . import bspline, utils, assemble
from ._hdiscr import HDiscretization

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

def _reindex(n, Idx, u):
    """Functionally identical to eye(n)[:, Idx].dot(u)."""
    result = np.zeros(n, dtype=u.dtype)
    result[Idx] = u
    return result

def _position_index(suplist, sublist):
    """Takes two sorted lists of unique integers (e.g., two raveled index
    lists), where `suplist` contains `sublist`. Returns list of position
    indices of matching entries."""
    out = []
    k = 0
    for candidate in sublist:
        k = suplist.index(candidate, k)
        out.append(k)
    return np.array(out)

def _drop_empty_items(d):
    """Returns a copy of the dict `d` with entries with empty values removed"""
    return {lv: c for (lv, c) in d.items() if c}

def _dict_union(dA, dB):
    """Takes two dicts of sets and returns a new dict `d` where `d[k]` is the
    union of `dA[k]` and `dB[k]`. Non-existent keys are treated as the empty set.
    """
    return { k: dA.get(k, set()) | dB.get(k, set())
            for k in dA.keys() | dB.keys() }

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
        """Return the extents (as a tuple of min/max pairs) of the cell `c`."""
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
    """A hierarchical mesh built on a sequence of uniformly refined tensor product meshes.

    This class is an implementation detail and should not be used in user-facing code.
    """
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

    def cell_babys(self, lv, cells, targetlv = None):
        if not targetlv:
            targetlv = len(self.meshes) - 1
        assert 0 <= lv < len(self.meshes) - 1, 'Invalid level'
        assert lv < targetlv < len(self.meshes), 'Invalid target level'
        if targetlv - lv == 1:
            return self.cell_children(lv, cells)
        else:
            return self.cell_babys(lv+1, self.cell_children(lv, cells), targetlv)
        #babys = []
        #for c in cells:
        #    babys.extend(itertools.product(
        #        *(range((targetlv - lv)*2*ci, (targetlv - lv)*2*(ci + 1)) for ci in c)))
        #return babys

    def cell_parent(self, lv, cells):
        assert 1 <= lv < len(self.meshes), 'Invalid level'
        parents = []
        for c in cells:
            parents.append(tuple(ci // 2 for ci in c))
        parents.sort()
        return _make_unique(parents)

    def cell_grandparent(self, lv, cells, targetlv = None):
        if not targetlv:
            targetlv = 0
        assert 1 <= lv < len(self.meshes), 'Invalid level'
        assert 0 <= targetlv < lv, 'Invalid target level'
        if lv - targetlv == 1:
            return self.cell_parent(lv, cells)
        else:
            return self.cell_grandparent(lv-1, self.cell_parent(lv, cells), targetlv)
        #grandparents = []
        #for c in cells:
        #    grandparents.append(tuple(ci // (lv - targetlv)*2 for ci in c))
        #grandparents.sort()
        #return _make_unique(grandparents)

    def _TP_to_HMesh_cells_up(self, lv, cells):
        """Return dictionary of hierarchical cells of levels >= `lv` contributing to the
        `cells` of level `lv`."""
        assert 0 <= lv < len(self.meshes), 'Invalid level'
        out = dict()
        aux_cells = set(cells)
        L = len(self.meshes)
        for l in range(lv, L):
            out[l] = aux_cells & self.active[l]
            aux_cells -= self.active[l]
            if l < L - 1:
                aux_cells = set(self.cell_children(l, aux_cells))
        assert not aux_cells, 'Invalid cells detected: {}'.format(aux_cells)
        return out

    def _TP_to_HMesh_cells_down(self, lv, cells):
        """Return dictionary of hierarchical cells of level <= `lv` contributing to the
        `cells` of level `lv`."""
        assert 0 <= lv < len(self.meshes), 'Invalid level'
        out = dict()
        aux_cells = set(cells)
        for l in reversed(range(lv + 1)):
            out[l] = aux_cells & self.active[l]
            aux_cells -= self.active[l]
            if l > 0:
                aux_cells = set(self.cell_parent(l, aux_cells))
        assert not aux_cells, 'Invalid cells detected: {}'.format(aux_cells)
        return out

    def _TP_to_HMesh_cells(self, lv, cells):
        """Return dictionary of hierarchical cells contributing to the
        `cells` of level `lv`."""
        assert 0 <= lv < len(self.meshes), 'Invalid level'
        cells = set(cells)
        act_deact_lv = self.active[lv] | self.deactivated[lv]
        out_up   = self._TP_to_HMesh_cells_up(lv,   cells & act_deact_lv)
        out_down = self._TP_to_HMesh_cells_down(lv, cells - act_deact_lv)
        return _dict_union(out_down, out_up)

    def hmesh_cells(self, cells):
        """Return the smallest dictionary of active hierarchical cells containing `cells`."""
        if isinstance(cells, dict):
            # convert from dict to list
            c = [[] for _ in range(len(self.meshes))]
            for lv, cls in cells.items():
                c[lv] = cls
            cells = c
        out = dict()
        for lv in range(len(self.meshes)):
            out = _dict_union(out, self._TP_to_HMesh_cells(lv, cells[lv]))
        return _drop_empty_items(out)

    def _function_children_1d(self, lv, dim, j):
        assert 0 <= lv < len(self.meshes) - 1, 'Invalid level'
        #return list(self.P[lv][dim].getcol(j).nonzero()[0])
        # access CSC datastructure directly for speed
        P = self.P[lv][dim]
        return P.indices[P.indptr[j]:P.indptr[j+1]]

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

    def function_babys(self, lv, indices, targetlv = None):
        if not targetlv:
            targetlv = len(self.meshes) - 1
        assert lv < targetlv < len(self.meshes), 'Invalid target level'
        if targetlv - lv == 1:
            return self.function_children(lv, indices)
        else:
            return self.function_babys(lv+1, self.function_children(lv, indices), targetlv)

    def function_parents(self, lv, indices):
        parents = []
        for jj in indices:
            parents.extend(itertools.product(
                *(self._function_parents_1d(lv, d, j) for (d,j) in enumerate(jj))))
        parents.sort()
        return _make_unique(parents)

    def function_grandparents(self, lv, indices, targetlv = None):
        if not targetlv:
            targetlv = 0
        assert 0 <= targetlv < lv, 'Invalid target level'
        if lv - targetlv == 1:
            return self.function_parents(lv, indices)
        else:
            return self.function_grandparents(lv-1, self.function_parents(lv, indices), targetlv)

    def ensure_levels(self, L):
        """Make sure that the hierarchical space has at least `L` levels."""
        while len(self.meshes) < L:
            self.add_level()

    def refine(self, marked):
        # if necessary, add new fine levels to the data structure
        # NB: if refining on lv 0, we need 2 levels (0 and 1) -- hence the +2
        max_lv = max(lv for (lv,cells) in marked.items() if cells)
        self.ensure_levels(max_lv + 2)

        new_cells = dict()
        for lv in range(len(self.meshes) - 1):
            cells = set(marked.get(lv, []))
            # deactivate refined cells
            self.active[lv] -= cells
            self.deactivated[lv] |= cells
            # add children
            new_cells[lv+1] = self.cell_children(lv, cells)
            self.active[lv+1] |= set(new_cells[lv+1])
        return new_cells

    def get_virtual_mesh(self, level):
        assert 0 <= level < len(self.meshes)
        out = HMesh(self.meshes[0])
        for i in range(level+1):
            out.refine({i: self.deactivated[i]})
        return out


class HSpace:
    """Represents a HB-spline or THB-spline space over an adaptively refined mesh.

    Arguments:
        kvs: a sequence of `d` :class:`.KnotVector` instances, representing
            the tensor product B-spline space on the coarsest level
        disparity (int): the mesh level disparity, meaning the maximum number of
            coarser levels with which any active function on a given level may
            have interactions with. This disparity is respected when calling
            :meth:`refine`. If no restriction on the number of overlapping
            mesh levels is desired, pass `np.inf` (which is the default).
        bdspecs: optionally, a list of boundary specifications on which degrees
            of freedom should be eliminated (usually for treating Dirichlet
            boundary conditions). See :func:`.assemble.compute_dirichlet_bc`
            for the precise format.
    """
    def __init__(self, kvs, disparity=np.inf, bdspecs=None):
        tp = TPMesh(kvs)
        hmesh = HMesh(tp)
        assert len(hmesh.meshes) == 1
        self.dim = hmesh.dim
        self.hmesh = hmesh
        self.actfun = [set(hmesh.meshes[0].functions())]
        self.deactfun = [set()]
        self.disparity = disparity
        if bdspecs is not None:
            bdspecs = [assemble._parse_bdspec(bd, self.dim) for bd in bdspecs]
        self.bdspecs = bdspecs
        self._clear_cache()

    def _clear_cache(self):
        self.__ravel_global = None
        self.__index_dirichlet = None
        self.__ravel_dirichlet = None

    def _add_level(self):
        self.hmesh.add_level()
        self.actfun.append(set())
        self.deactfun.append(set())
        assert len(self.actfun) == len(self.deactfun) == len(self.hmesh.meshes)

    def _ensure_levels(self, L):
        """Make sure we have at least `L` levels."""
        while self.numlevels < L:
            self._add_level()

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

    def mesh(self, lv):
        """Return the underlying :class:`TPMesh` on the given level."""
        return self.hmesh.meshes[lv]

    def knotvectors(self, lv):
        """Return a tuple of :class:`.KnotVector` instances describing the
        tensor product space on level `lv`.
        """
        return self.hmesh.meshes[lv].kvs

    def active_cells(self, lv=None, flat=False):
        """If `lv` is specified, return the set of active cells on that level.
        Otherwise, return a list containing, for each level, the set of active cells.

        If `lv=None` and `flat=True`, return a flat list of `(lv, (j_1, ..., j_d))`
        pairs of all active cells in canonical order, where the first entry is the level
        and the second entry is the multi-index of the cell on that level.
        """
        if lv is not None:
            return self.hmesh.active[lv]
        else:
            if flat:
                return [(l, ac)
                        for l in range(self.numlevels)
                        for ac in sorted(self.active_cells(l))]
            else:
                return [self.active_cells(lv) for lv in range(self.numlevels)]

    @property
    def total_active_cells(self):
        """The total number of active cells in the hierarchical mesh."""
        return sum(len(ac) for ac in self.active_cells())

    def active_functions(self, lv=None, flat=False):
        """If `lv` is specified, return the set of active functions on that level.
        Otherwise, return a list containing, for each level, the set of active functions.

        If `lv=None` and `flat=True`, return a flat list of `(lv, (i_1, ..., i_d))`
        pairs of all active functions in canonical order, where the first entry is the level
        and the second entry is the multi-index of the function on that level.
        """
        if lv is not None:
            return self.actfun[lv]
        else:
            if flat:
                return [(l, af)
                        for l in range(self.numlevels)
                        for af in sorted(self.actfun[l])]
            else:
                return self.actfun

    def deactivated_cells(self, lv=None):
        """If `lv` is specified, return the set of deactivated cells on that level.
        Otherwise, return a list containing, for each level, the set of deactivated cells.
        """
        if lv is not None:
            return self.hmesh.deactivated[lv]
        else:
            return [self.deactivated_cells(lv) for lv in range(self.numlevels)]

    def cell_extents(self, lv, c):
        """Return the extents (as a tuple of min/max pairs) of the cell `c` on level `lv`."""
        return self.hmesh.meshes[lv].cell_extents(c)

    def _ravel_indices(self, indexsets):
        indices = [sorted(ix) if isinstance(ix, set) else ix for ix in indexsets]
        return tuple(
            (np.ravel_multi_index(np.array(indices[lv]).T, self.mesh(lv).numdofs, order='C')
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

    def _compute_single_axis_single_level_dirichlet_indices(self, lv, bdspec):
        assert 0 <= lv < self.numlevels, 'Invalid level.'
        bdax, bdside = bdspec
        N = tuple(kv.numdofs for kv in self.hmesh.meshes[lv].kvs)
        return set(tuple(iter) for iter in assemble.slice_indices(bdax, 0 if bdside==0 else -1, N, ravel=False))

    def _dirichlet_indices(self):
        # Compute tensor product boundary indices (specified by self.bdspecs)
        TPbindices = list()
        for lv in range(self.numlevels):
            aux = set()
            for bdspec in self.bdspecs:
                aux |= self._compute_single_axis_single_level_dirichlet_indices(lv, bdspec)
            TPbindices.append(aux)

        # Compute boundary indices for virtual hierarchy
        out = []
        out_index = []
        for lv in range(self.numlevels):
            aux = []
            for i in range(self.numlevels):
                # the deactivated functions on level `lv` will be added later
                if 0 <= i <= lv:
                    aux.append(self.actfun[i] & TPbindices[i])
                else:
                    aux.append(set())
            out.append(list(self._ravel_indices(aux)))
            out_index.append(aux)

        # deactivated boundary ravel
        ravel_bddeact = self._ravel_indices(
            [self.deactfun[lv] & TPbindices[lv]
                for lv in range(self.numlevels)])

        # insert deactivated functions (separate step to preserve canonical order)
        for lv in range(self.numlevels):
            out_index[lv][lv] |= self.deactfun[lv] & TPbindices[lv]
            out[lv][lv] = np.concatenate((out[lv][lv], ravel_bddeact[lv]))

        self.__ravel_dirichlet = tuple(out)
        self.__index_dirichlet = tuple(out_index)

    @property
    def index_dirichlet(self):
        if not self.__index_dirichlet:
            self._dirichlet_indices()
        return self.__index_dirichlet

    @property
    def ravel_dirichlet(self):
        if not self.__ravel_dirichlet:
            self._dirichlet_indices()
        return self.__ravel_dirichlet

    @property
    def ravel_global(self):
        if not self.__ravel_global:
            indices = self.global_indices()
            self.__ravel_global = [self._ravel_indices(idx) for idx in indices]
        return self.__ravel_global

    @property
    def cell_dirichlet(self):
        return self.compute_virtual_supports(self.index_dirichlet)

    @property
    def cell_new(self):
        return self.compute_virtual_supports(self.new_indices())

    @property
    def cell_trunc(self):
        return self.compute_virtual_supports(self.trunc_indices())

    @property
    def cell_func_supp(self):
        return self.compute_virtual_supports(self.func_supp_indices())

    @property
    def cell_cell_supp(self):
        return self.compute_virtual_supports(self.cell_supp_indices())

    @property
    def cell_global(self):
        return self.compute_virtual_supports(self.global_indices())

    def dirichlet_dofs(self, lv=None):
        """Matrix indices which lie on the specified Dirichlet boundaries."""
        if lv is None:
            lv = self.numlevels - 1
        return self.raveled_to_virtual_matrix_indices(lv, self.ravel_dirichlet[lv])

    def non_dirichlet_dofs(self):
        """Matrix indices which do not lie on the specified Dirichlet boundaries."""
        return sorted(set(range(self.numdofs)) - set(self.dirichlet_dofs()))

    def new_indices(self):
        """Return a tuple which contains tuples which contain, per level, the raveled
        (sequential) indices of newly added basis functions in the virtual hierarchy per level."""
        return [
                [ ( sorted(self.actfun[i] - self.index_dirichlet[lv][i])
                  + sorted(self.deactfun[i] - self.index_dirichlet[lv][i]))
                    if i == lv
                    else []
                    for i in range(self.numlevels)]
                for lv in range(self.numlevels)]

    def trunc_indices(self):
        """Return a tuple which contains tuples which contain, per level, the raveled
        (sequential) indices of ``trunc`` basis functions in the virtual hierarchy per level."""
        indices = self.new_indices()        # start with only the newly added indices
        out = list()
        out_index = list()
        aux_dict = dict()
        for lv in range(self.numlevels):
            aux = list()
            for i in range(self.numlevels):
                if i == lv:
                    aux_act = list(self.actfun[lv])
                    aux_dict[lv] = dict(zip(aux_act, aux_act))
                elif lv - self.disparity <= i < lv:
                    aux_indices = list()
                    for j in aux_dict[i]:
                        if isinstance(aux_dict[i][j], tuple):
                            aux_dict[i][j] = {aux_dict[i][j]}
                        aux_dict[i][j] = set(self.hmesh.function_children(lv-1, aux_dict[i][j]))
                        if aux_dict[i][j] & (self.actfun[lv] | self.deactfun[lv]):
                            aux_dict[i][j] -= (self.actfun[lv] | self.deactfun[lv])
                            aux_indices.append(j)
                    indices[lv][i] = sorted(set(aux_indices) - self.index_dirichlet[lv][i])
        return indices

    def func_supp_indices(self):
        """Return a tuple which contains tuples which contain, per level, the raveled
        (sequential) indices of ``func_supp`` basis functions in the virtual hierarchy per level."""
        indices = self.new_indices()        # start with only the newly added indices
        for lv in range(self.numlevels):
            for i in range(self.numlevels):
                if lv - self.disparity <= i < lv:
                    funcs = set(self.hmesh.function_grandparents(lv, self.actfun[lv], i)) & self.actfun[i]
                    indices[lv][i] = sorted(funcs - self.index_dirichlet[lv][i])

        return indices

    def cell_supp_indices(self, remove_dirichlet=True):
        """Return a tuple which contains tuples which contain, per level, the raveled
        (sequential) indices of ``cell_supp`` basis functions in the virtual hierarchy per level."""
        indices = self.new_indices()        # start with only the newly added indices
        for lv in range(self.numlevels):    # loop over virtual hierarchy levels
            for i in range(self.numlevels):
                if lv - self.disparity <= i < lv:
                    funcs = set(
                            self.hmesh.meshes[i].supported_in(
                                self.hmesh.cell_grandparent(
                                    lv,
                                    self.hmesh.meshes[lv].support(self.actfun[lv]),
                                    i))) & self.actfun[i]
                    if remove_dirichlet:
                        indices[lv][i] = sorted(funcs - self.index_dirichlet[lv][i])
                    else:
                        indices[lv][i] = sorted(funcs)
        return indices

    def global_indices(self):
        """Return a tuple which contains tuples which contain, per level, the raveled
        (sequential) indices of ``global`` basis functions in the virtual hierarchy per level."""
        indices = [ [[] for i in range(self.numlevels)] for j in range(self.numlevels) ]
        for lv in range(self.numlevels):
            for i in range(self.numlevels):
                if i == lv:
                    indices[lv][i] = sorted(self.actfun[i]) + sorted(self.deactfun[i])
                elif i < lv:
                    indices[lv][i] = sorted(self.actfun[i])
        return indices

    def indices_to_smooth(self, strategy='func_supp'):
        assert strategy in ("new", "trunc", "func_supp", "cell_supp"), "Invalid smoothing strategy"
        # get smoothing indices in TP form
        chosen_indices = getattr(self, strategy + '_indices')()
        # convert them to raveled form
        chosen_indices = [self._ravel_indices(idx) for idx in chosen_indices]
        # convert them to matrix indices
        return [self.raveled_to_virtual_matrix_indices(lv, chosen_indices[lv])
                for lv in range(self.numlevels)]

    def raveled_to_virtual_matrix_indices(self, lv, indices):
        # convert indices from levelwise raveled TP indices to matrix indices within the
        # stiffness matrix on the corresponding virtual multigrid hierarchy level
        available_indices = self.ravel_global
        out = []
        n_lv = 0
        for l in range(self.numlevels):
            out += list(n_lv + _position_index(list(available_indices[lv][l]), indices[l]))
            n_lv += len(available_indices[lv][l])
        return np.array(out, dtype=int)

    def compute_supports(self, functions):
        """Compute the union of the supports, in terms of active mesh cells, of
        the given list-of-seqs of functions and return the active cells as a
        dict-of-sets.
        """
        supports = [set(self.hmesh.meshes[l].support(funcs))
                for (l, funcs) in enumerate(functions)]
        return self.hmesh.hmesh_cells(supports)

    def compute_virtual_supports(self, tuplelistset):
        return tuple(
                self.get_virtual_space(lv).compute_supports(functions)
                for (lv, functions) in enumerate(tuplelistset))

    def function_support(self, lv, jj):
        """Return the support (as a tuple of pairs) of the function on level `lv` with multi-index `jj`."""
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

    def cell_support_extension(self, l, cells, k):
        assert 0 <= k <= l, 'Invalid level.'
        if k == l:
            aux = cells
        else:
            aux = self.hmesh.cell_grandparent(l, cells, k)
        supported_functions = self.hmesh.meshes[k].supported_in(aux)
        return set(self.hmesh.meshes[k].support(supported_functions))

    def function_support_extension(self, l, functions, k):
        assert 0 <= k <= l, 'Invalid level.'
        aux = self.hmesh.meshes[l].support(functions)
        if k == l:
            pass
        else:
            aux = self.hmesh.cell_grandparent(l, aux, k)
        return self.hmesh.meshes[k].supported_in(aux)

    def _cell_neighborhood(self, l, cells, truncate=False):
        if l - self.disparity < 0:
            return set()
        else:
            if truncate:
                return self.hmesh.active[l-self.disparity] & \
                        set(self.hmesh.cell_parent(l-self.disparity+1, self.cell_support_extension(l, cells, l-self.disparity+1)))
            else:
                return self.hmesh.active[l-self.disparity] & \
                        set(self.cell_support_extension(l, cells, l-self.disparity))

    def _mark_recursive(self, l, marked, truncate=False):
        neighbors = self._cell_neighborhood(l, marked.get(l, set()), truncate=truncate)
        if neighbors:
            marked[l-self.disparity] = marked.get(l-self.disparity, set()) | neighbors
            self._mark_recursive(l-self.disparity, marked)

    def refine(self, marked, truncate=False):
        """Refine the given cells; `marked` is a dictionary which has the
        levels as indices and the list of marked cells on that level as values.

        The refinement procedure preserves the mesh level disparity, following
        the method described in [Bracco, Giannelli, Vazquez, 2018].

        Returns:
            the actually refined cells in the same format as `marked`; if
            disparity is less than infinity, this is a superset of the
            input cells
        """
        max_lv = max(lv for (lv,cells) in marked.items() if cells)
        self._ensure_levels(max_lv + 2)

        if self.disparity < np.inf:
            marked = marked.copy()
            for l in range(self.numlevels):
                self._mark_recursive(l, marked, truncate=truncate)

        new_cells = self.hmesh.refine(marked)
        mf = self._functions_to_deactivate(marked)

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

        self._clear_cache()
        return marked       # return the actual refined cells

    def get_virtual_space(self, level):
        assert 0 <= level < self.numlevels, 'Invalid level.'
        out = HSpace(self.hmesh.meshes[0].kvs)
        for i in range(level): # refine level-1 times
            out.refine({i: self.hmesh.deactivated[i]})
        return out

    def refine_region(self, lv, region_function):
        """Refine all active cells on level `lv` whose cell center satisfies `region_function`.

        `region_function` should be a function of `dim` scalar arguments (e.g., `(x,y)`)
        which returns True if the point is within the refinement region.
        """
        self._ensure_levels(lv + 2)

        def cell_center(c):
            return tuple(0.5*(lo+hi) for (lo,hi) in reversed(self.cell_extents(lv, c)))
        self.refine({
            lv: tuple(c for c in self.active_cells(lv) if region_function(*cell_center(c)))
        })

    def represent_fine(self, lv=None, truncate=False, rows=None, restrict=False):
        """Compute a matrix which represents HB- or THB-spline basis functions in terms of
        their coefficients in the finest tensor product spline space.

        By default, all active HB-spline functions are represented on the finest tensor
        product mesh. The returned matrix has size `N_fine × N_act`, where `N_fine` is
        the number of degrees of freedom in the tensor product mesh of the finest level
        and `N_act` = :attr:`numdofs` is the total number of active basis functions.

        If `lv` is specified, only active functions up to level `lv` are represented in terms
        of their coefficients on level `lv`.

        If `truncate` is True, the representation of the THB-spline (truncated) basis
        functions is computed instead of that of the HB-splines.

        If `rows` is given, only those rows are kept in the output. In other
        words, only the representation with respect to these tensor product
        basis functions on the fine level is computed. If `restrict=False`,
        then the shape of the matrix is not changed, but only the corresponding
        rows are filled.  If `restrict=True`, the matrix is restricted to the
        submatrix having only these rows.
        """
        if lv == None:
            lv = self.numlevels - 1
        assert 0 <= lv < self.numlevels, "Invalid level."
        act_indices = list(self.active_indices()[:lv+1])
        deact_indices = self.deactivated_indices()[lv]
        # generate list of raveled active indices of virtual level lv
        act_indices[lv] = np.concatenate((act_indices[lv],deact_indices))

        # Intermediate matrix format; if truncating, we need a format which
        # allows efficient changing of the sparsity structure due to setting
        # some rows to 0.
        fmt = 'lil' if truncate else 'csr'

        # In every iteration, needed_rows keeps track of the rows which are
        # needed from the prolongation matrix of the next coarser level. This
        # is purely an optimization so that not the complete tensor product
        # prolongation matrix has to be assembled.

        blocks = []
        for k in reversed(range(lv+1)):
            Nj = self.mesh(k).numbf

            if k == lv:
                if rows is None:
                    P = scipy.sparse.eye(Nj, format='csc')
                    needed_rows = None
                else:
                    if restrict:
                        # construct restricted slice of identity matrix
                        n = len(rows)
                        P = scipy.sparse.coo_matrix(
                                (np.ones(n), (np.arange(n), rows)),
                                shape=(n, Nj)).tocsc()
                    else:
                        # construct partial identity matrix which is 1 only on the given rows
                        n = len(rows)
                        P = scipy.sparse.coo_matrix(
                                (np.ones(n), (rows, rows)),
                                shape=(Nj, Nj)).tocsc()
                    needed_rows = rows
            else:
                # check what percentage of rows we need from the prolongator
                if needed_rows is not None:
                    if len(needed_rows) / P.shape[1] > 0.5:
                        # use the more efficient full Kronecker product if we need most rows
                        needed_rows = None

                if needed_rows is None:
                    Pj = utils.multi_kron_sparse(self.hmesh.P[k], format=fmt)
                else:
                    Pj = utils.kron_partial(self.hmesh.P[k], needed_rows, format=fmt)
                if truncate:
                    Pj[act_indices[k+1], :] = 0
                P = P.dot(Pj)

                # determine needed rows for the next step from the nonzero columns of P
                nnz_per_col = P.getnnz(axis=0)              # count nnz_j per column
                needed_rows = nnz_per_col.nonzero()[0]      # find columns with nnz_j > 0

            blocks.append(P[:, act_indices[k]])

        blocks.reverse()
        return scipy.sparse.bmat([blocks], format='csr')

    def truncate_one_level(self, k, num_rows=None, inverse=False):
        """Compute the matrix which realizes truncation from level `k` to `k+1`."""
        nt = np.cumsum(self.numactive)  # nt[k]: total active dofs up to level k
        actidx = self.active_indices()  # TP indices of active functions per level

        if num_rows is None:
            num_rows = nt[-1]
        T = scipy.sparse.eye(num_rows, format='lil')
        A = self.represent_fine(lv=k+1, rows=actidx[k+1], restrict=True)    # rep act(0..k+1) as act(k+1)
        # truncation: subtract the components of the coarse functions which can
        # be represented by the active functions on level k+1
        if inverse:
            T[nt[k]:nt[k+1], 0:nt[k]] = A[:, 0:nt[k]]
        else:
            T[nt[k]:nt[k+1], 0:nt[k]] = -A[:, 0:nt[k]]
        return T.tocsr()

    def thb_to_hb(self):
        """Return a sparse square matrix of size :attr:`numdofs` which
        transforms THB-spline coefficients into the corresponding HB-spline
        coefficients.
        """
        if self.numlevels == 1:
            return scipy.sparse.eye(self.numdofs, format='csr')
        else:
            T = self.truncate_one_level(0)
            for k in range(1, self.numlevels - 1):
                T = self.truncate_one_level(k) @ T
            return T

    def hb_to_thb(self):
        """Return a sparse square matrix of size :attr:`numdofs` which
        transforms HB-spline coefficients into the corresponding THB-spline
        coefficients.
        """
        if self.numlevels == 1:
            return scipy.sparse.eye(self.numdofs, format='csr')
        else:
            T = self.truncate_one_level(0, inverse=True)
            for k in range(1, self.numlevels - 1):
                T = T @ self.truncate_one_level(k, inverse=True)
            return T

    def split_coeffs(self, x):
        """Given a coefficient vector `x` of length :attr:`numdofs` in
        canonical order, split it into :attr:`numlevels` vectors which contain
        the contributions from each individual level.
        """
        j = 0
        result = []
        for af in self.actfun:
            nk = len(af)
            result.append(x[j : j+nk])
            j += nk
        assert j == x.shape[0], 'Wrong length of input vector'
        return result

    def tp_prolongation(self, lv, kron=False):
        """Return the prolongation operator for the underlying tensor product mesh from level
        `lv` to `lv+1`.

        If `kron` is True, the prolongation is returned as a sparse matrix. Otherwise, the
        prolongation is returned as a tuple of sparse matrices, one per space dimension,
        whose Kronecker product represents the prolongation operator.

        .. note::
            This method, particularly with `kron=True`, is inherently
            inefficient since it deals with the full tensor product spaces, not
            merely the active basis functions.
        """
        Ps = self.hmesh.P[lv]
        return utils.multi_kron_sparse(Ps) if kron else Ps

    def incidence_matrix(self):
        """Compute the incidence matrix which contains one row per active basis
        function and one column per active cell in the hierarchical mesh. An
        entry `(i,j)` is 1 if the function `i` is nonzero in the cell `j`, and 0
        otherwise.
        """
        naf = tuple(len(ii) for ii in self.actfun)
        nac = tuple(len(ii) for ii in self.hmesh.active)
        ndc = tuple(len(ii) for ii in self.hmesh.deactivated)

        L = self.numlevels

        # data structure which maps cell multi-indices to sequential ones;
        # on each level, we first number the active and then the deactivated cells
        cell_index = [
                utils.BijectiveIndex(sorted(self.hmesh.active[k])
                                   + sorted(self.hmesh.deactivated[k]))
                for k in range(L)
        ]

        def incidence_1level(k):
            # Compute the incidence matrix for functions/cells on a single level.
            # Size: naf[k] x (nac[:k] + nac[k] + ndc[k])
            n0 = sum(nac[:k])
            Z = scipy.sparse.lil_matrix((naf[k], n0 + nac[k] + ndc[k]), dtype=int)

            for (i, f) in enumerate(sorted(self.actfun[k])):
                cells = self.hmesh.meshes[k].support([f])
                for c in cells:
                    Z[i, n0 + cell_index[k].index(c)] = 1
            return Z.tocsr()

        def cell_prolongation(k):
            # Matrix which "prolongs" deactivated cells on level k to the next
            # finer level.
            # Size: (nac[:k+1] + nac[k+1] + ndc[k+1]) x (nac[:k+1] + ndc[k])

            # P is the matrix which prolongs deactive cells on k to cells on k+1
            P = scipy.sparse.lil_matrix((nac[k+1] + ndc[k+1], ndc[k]), dtype=int)
            for i in range(ndc[k]):
                I = cell_index[k][nac[k] + i]
                children = self.hmesh.cell_children(k, [I])
                for c in children:
                    P[cell_index[k+1].index(c), i] = 1

            # extend P to preserve active cells on all earlier levels
            I_k = scipy.sparse.eye(sum(nac[:k+1]), dtype=int)
            return scipy.sparse.bmat(
                    [[ I_k,  None ],
                     [ None, P    ]], format='csr')

        # start with the one-level incidence matrices
        # initially, the k-th matrix has size naf[k] x (nac[:k+1] + ndc[k])
        result = [ incidence_1level(k) for k in range(L) ]

        # prolong the deactivated cells on each level
        for k in range(L - 1):
            P = cell_prolongation(k)
            for j in range(k+1):    # only levels j with j <= k need prolongation
                result[j] = result[j].dot(P.T)
        return scipy.sparse.vstack(result, format='csr')

    def virtual_hierarchy_prolongators(self):
        # compute tensor product prolongators
        Ps = tuple(self.tp_prolongation(lv, kron=False) for lv in range(self.numlevels-1))

        # indices of active and deactivated basis functions per level
        IA = self.active_indices()
        ID = self.deactivated_indices()
        # indices of all functions in the refinement region per level
        IR = tuple(np.concatenate((iA,iD)) for (iA,iD) in zip(IA,ID))

        # total number of active dofs up to a given level
        nt = np.cumsum(tuple(len(ii) for ii in IA))

        prolongators = []
        for lv in range(self.numlevels - 1):
            P_rd = utils.kron_partial(Ps[lv], rows=IR[lv+1], restrict=True)[:, ID[lv]]
            P_hb = scipy.sparse.bmat((
              (scipy.sparse.eye(nt[lv]), None),
              (None,                     P_rd)
            ), format='csc')
            prolongators.append(P_hb)
        return prolongators

    def coeffs_to_levelwise_funcs(self, coeffs, truncate=False):
        """Compute the levelwise contributions of the hierarchical spline
        function given by the coefficient vector `coeffs` as a list containing
        one :class:`.BSplineFunc` per level.

        If `truncate=True`, the coefficients are interpreted as THB-spline
        coefficients, otherwise as HB-splines.
        """
        if truncate:
            coeffs = self.thb_to_hb() @ coeffs
        # construct the level-wise B-spline functions
        u_lv = self.split_coeffs(coeffs)
        n_tp = tuple(self.mesh(k).numbf for k in range(self.numlevels))
        IA = self.active_indices()
        return tuple(
                bspline.BSplineFunc(self.knotvectors(lv), _reindex(n_tp[lv], IA[lv], uj))
                for (lv,uj) in enumerate(u_lv)
                )

    def grid_eval(self, coeffs, gridaxes, truncate=False):
        """Evaluate an HB-spline function with the given coefficients over a
        tensor product grid.
        """
        # evaluate them and sum the result
        return sum(f.grid_eval(gridaxes)
                for f in self.coeffs_to_levelwise_funcs(coeffs, truncate=truncate))

class HSplineFunc:
    """A function living in a hierarchical spline space.

    Args:
        hspace (:class:`HSpace`): the hierarchical spline space
        u (array): the vector of coefficients corresponding to the active basis
            functions, with length :attr:`HSpace.numdofs`, in canonical order
        truncate (bool): if true, the coefficients are interpreted as THB-spline
            coefficients; otherwise, as HB-spline coefficients
    """
    def __init__(self, hspace, u, truncate=False):
        self.hs = hspace
        self.coeffs = u
        self.sdim = hspace.dim
        self.dim = 1        # for now only scalar functions
        self.truncate = truncate

    def grid_eval(self, gridaxes):
        """Evaluate the function on a tensor product grid.

        See :meth:`.BSplineFunc.grid_eval` for details.
        """
        return self.hs.grid_eval(self.coeffs, gridaxes, truncate=self.truncate)

    def grid_jacobian(self, gridaxes):
        """Evaluate the Jacobian on a tensor product grid.

        See :meth:`.BSplineFunc.grid_jacobian` for details.
        """
        return sum(f.grid_jacobian(gridaxes)
                for f in self.hs.coeffs_to_levelwise_funcs(self.coeffs, truncate=self.truncate))

    def grid_hessian(self, gridaxes):
        """Evaluate the Hessian matrix on a tensor product grid.

        See :meth:`.BSplineFunc.grid_hessian` for details.
        """
        return sum(f.grid_hessian(gridaxes)
                for f in self.hs.coeffs_to_levelwise_funcs(self.coeffs, truncate=self.truncate))

    @property
    def support(self):
        """Return a sequence of pairs `(lower,upper)`, one per source dimension,
        which describe the extent of the support in the parameter space."""
        return tuple(kv.support() for kv in self.hs.knotvectors(0))
