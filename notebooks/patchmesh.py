from pyiga import bspline, vis, assemble
import copy

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def corners(geo):
    """Return an array containing the locations of the 2^d corners of the given
    geometry."""
    return geo.grid_eval(geo.support).reshape((-1, geo.dim))

def split_dirichlet_data(p, n, dir_data, axis = None):
    """Splits the Dirichlet information of a patch `p`
    
    Args:
        p: number of the patch to be split
        n: total number of patches
        dir_data: information of the Dirichlet condition over the whole PatchMesh structure
            
    Returns:
        Modified list `dir_data` with split patches.
    """
    dir_data_p = [(patch , bd, g) for (patch, bd, g) in dir_data if patch == p]
    
    if not dir_data_p:
        return
    
    if axis == None:
        split_dirichlet_data(p, n, dir_data, 1)
        split_dirichlet_data(p, n+1, dir_data, 0)
        split_dirichlet_data(n, n+2, dir_data, 0)
        return
    
    for (patch, bd, g) in dir_data_p:
        if axis == 0:
            if bd == 'left' or bd == 'right':
                dir_data.append((n, bd, g))
            if bd == 'top':
                dir_data.remove((patch, bd, g))
                dir_data.append((n, bd, g))
        if axis == 1:
            if bd == 'bottom' or bd == 'top':
                dir_data.append((n, bd, g))
            if bd == 'right':
                dir_data.remove((patch, bd, g))
                dir_data.append((n, bd, g))    


# Data structures:
#
# - vertices is a list of topological vertices, the entries are their locations
#
# - patches is a list of tuples:
#       ((kvs, geo), boundaries)
#   - kvs are the tensor product knot vectors
#   - geo is the geometry map
#   - boundaries is a list of the four boundaries of the patch; each boundary
#     is a list describing the boundary segments on that boundary as a sequence
#     of vertex indices. Each boundary has at least length 2 (a single
#     segment).  Boundaries are always stored in order of increasing
#     patch-local coordinates.
#
# - interfaces is a dict which stores connectivity information between boundary
#   segments on neighboring patches: an entry of the form
#     interfaces[(p0, b0, s0)] == ((p1, b1, s1), flip)
#   means that segment s0 of boundary b0 of patch p0 (all integer indices) is
#   connected to segment s1 of boundary b1 of patch p1. flip is a
#   (d-1)-dimensional tuple indicating whether each coordinate axis runs in
#   opposite direction in the two patches. This information is symmetrical, so
#   each entry like the above has a corresponding second entry with the roles
#   of (p0,b0,s0) and (p1,b1,s1) reversed.
#

class PatchMesh:
    def __init__(self, patches = None):
        self.vertices = []
        self.patches = []
        self.interfaces = dict()
        self.C_nodes = dict()
        self.T_nodes = dict()

        if patches:
            
            # add interfaces between patches
            conn, interfaces = assemble.detect_interfaces(patches)
            assert conn, 'patch graph is not connected!'
            
            for p,patch in enumerate(patches):
                kvs, geo = patch
                # add/get vertices (checks for duplicates)
                vtx = [self.add_vertex(c) for c in corners(geo)]
                # add boundaries in fixed order
                self.add_patch(patch, (
                    [vtx[0], vtx[1]],    # bottom
                    [vtx[2], vtx[3]],    # top
                    [vtx[0], vtx[2]],    # left
                    [vtx[1], vtx[3]],    # right
                ))
                for i, v in enumerate(vtx):
                    if v in self.C_nodes:
                        self.C_nodes[v][p] = (i//2,i%2)
                    else:
                        self.C_nodes[v]={p : (i//2,i%2)}

            def bdspec_to_int(bdspec):
                return 2 * bdspec[0] + bdspec[1]    # convert to a boundary index (0..3)

            for (p0, bd0, p1, bd1, flip) in interfaces:
                self.add_interface(p0, bdspec_to_int(bd0), 0, p1, bdspec_to_int(bd1), 0, flip)

            self.sanity_check()

    @property
    def numpatches(self):
        return len(self.patches)
            
    def add_vertex(self, pos):
        """Add a new vertex at `pos` or return its index if one already exists there."""
        if self.vertices:
            distances = [np.linalg.norm(vtxpos - pos) for vtxpos in self.vertices]
            i_min = np.argmin(distances)
            if distances[i_min] < 1e-14:
                return i_min
        self.vertices.append(pos)
        return len(self.vertices) - 1

    def get_vertex_index(self, pos):
        if self.vertices:
            distances = [np.linalg.norm(vtxpos - pos) for vtxpos in self.vertices]
            i_min = np.argmin(distances)
            if distances[i_min] < 1e-14:
                return i_min
        raise ValueError('no vertex found at %s' % (pos,))

    def add_patch(self, patch, boundaries):
        self.patches.append((patch, boundaries))

    def add_interface(self, p0, b0, s0, p1, b1, s1, flip):
        """Join segment s0 of boundary b0 of patch p0 with segment i1
        of boundary b1 of p1.
        """
        #bds0 = self.boundaries(p0)
        #bds1 = self.boundaries(p1)
        #assert b0 < len(bds0) and b1 < len(bds1)
        #assert s0 < len(bds0[b0]) and s1 < len(bds1[b1])
        S0 = (p0, b0, s0)
        S1 = (p1, b1, s1)
        self.interfaces[S0] = (S1, flip)
        self.interfaces[S1] = (S0, flip)
        
    def refine(self, patches = None):
        if patches==None:
            patches = np.arange(self.numpatches)
        if np.isscalar(patches):
            patches=(patches,)
        for p in patches:
            (kvs,geo), b = self.patches[p]
            new_kvs = tuple([kv.refine() for kv in kvs])
            self.patches[p]=((new_kvs, geo), b)

    def _reindex_interfaces(self, p, b, old_s, ofs, new_p=None):
        old_s = list(old_s)
        new_s = [s + ofs for s in old_s]
        assert len(old_s) == len(new_s)
        if new_p is None:
            new_p = p
        S_old = [(p, b, s) for s in old_s]
        S_new = [(new_p, b, s) for s in new_s]
        old_intf = [self.interfaces.pop(S, None) for S in S_old]
        for Sn, intf in zip(S_new, old_intf):
            if intf:
                self.interfaces[Sn] = intf
                self.interfaces[intf[0]] = (Sn, intf[1])

    def split_boundary_segment(self, p, b, s, new_vtx):
        """Split the boundary segment `s` on boundary `b` of patch `p` by
        inserting the new vertex with index `new_vtx`.

        If the segment interfaces with another patch, also splits the
        corresponding segment on the other side and updates the interface
        information.
        """
        bd = self.boundaries(p)[b]
        assert 0 <= s < len(bd) - 1
        bd.insert(s + 1, new_vtx)
        # shift all later interfaces up by one
        self._reindex_interfaces(p, b, range(s + 1, len(bd) - 2), +1)

        # also split the matching boundary segment on neighboring patch, if any
        other = self.get_matching_interface(p, b, s)
        if other:
            (p1, b1, s1) = other
            bd1 = self.boundaries(p1)[b1]
            bd1.insert(s1 + 1, new_vtx)

            # shift all later interfaces up by one
            self._reindex_interfaces(p1, b1, range(s1 + 1, len(bd1) - 2), +1)

            # fix the new interfaces to point to each other
            flip = self.interfaces[(p, b, s)][1]
            if flip[0]:
                # indices are running in opposite directions on the two sides
                self.interfaces[(p, b, s)] = ((p1, b1, s1+1), flip)
                self.interfaces[(p1, b1, s1+1)] = ((p, b, s), flip)

                self.interfaces[(p, b, s+1)] = ((p1, b1, s1), flip)
                self.interfaces[(p1, b1, s1)] = ((p, b, s+1), flip)
            else:
                # these are already correctly indexed from the old interface
                #self.interfaces[(p, b, s)] = ((p1, b1, s1), flip)
                #self.interfaces[(p1, b1, s1)] = ((p, b, s), flip)

                self.interfaces[(p, b, s+1)] = ((p1, b1, s1+1), flip)
                self.interfaces[(p1, b1, s1+1)] = ((p, b, s+1), flip)

    def split_patch_boundary(self, p, b, xi, vtxpos, new_p):
        """Split the boundary `b` of patch `p` at a vertex which lies at
        parameter value `xi` of the boundary curve and has coordinates
        `vtxpos`.

        Returns the index of the first segment after the new vertex.

        It is valid to pass a vertex which is already contained in the
        boundary, in which case nothing is inserted and the correct index is
        returned.
        """
        new_vtx = self.add_vertex(vtxpos)
        
        vtx1, vtx2 = self.boundaries(p)[b][::len(self.boundaries(p)[b])-1]

        if vtx1 in self.C_nodes:
            corner1 = self.C_nodes[vtx1][p]
        else:
            [corner1] = [c for patch, c in self.T_nodes[vtx1][1:] if patch==p]
        
        if vtx2 in self.C_nodes:
            corner2 = self.C_nodes[vtx2][p]
        else:
            [corner2] = [c for patch, c in self.T_nodes[vtx2][1:] if patch==p]
            
        try:
            # is the vertex already contained in the boundary?
            vtx_idx = self.boundaries(p)[b].index(new_vtx)
        except ValueError:
            # otherwise, we need to insert it, split the segments and insert a new T_node (or corner at the boundary of the domain)
            seg = self._find_boundary_split_index(p, b, xi, new_vtx)
            self.split_boundary_segment(p, b, seg, new_vtx)
            
            if (p,b,0) in self.interfaces:
                self.T_nodes[new_vtx] = ((self.interfaces[(p,b,0)][0][:-1],self.interfaces[(p,b,0)][1]), (p, corner2), (new_p, corner1))
            else:
                self.C_nodes[new_vtx] = dict()
                self.C_nodes[new_vtx][p] = corner2
                self.C_nodes[new_vtx][new_p] = corner1
            return seg + 1  # we want the segment just after the newly inserted vertex
        else:
            if new_vtx not in self.C_nodes:
                self.C_nodes[new_vtx] = dict()
            for patch, c in self.T_nodes[new_vtx][1:]:
                self.C_nodes[new_vtx][patch] = c
            self.C_nodes[new_vtx][p] = corner2
            self.C_nodes[new_vtx][new_p] = corner1
            del self.T_nodes[new_vtx]
            return vtx_idx

    def _find_boundary_split_index(self, p, bdidx, xi_split, vtx_idx):
        (kvs, geo), boundaries = self.patches[p]
        segments = boundaries[bdidx]
        # simple case: if a single interval covers the boundary, we split it
        if len(segments) == 2:
            return 0
        bd_geo = geo.boundary((bdidx // 2, bdidx % 2))
        bd_vtx_xi = [bd_geo.find_inverse(self.vertices[j])[0] for j in segments]
        # find segment where xi_split would need to be inserted to maintain order
        return np.searchsorted(bd_vtx_xi, xi_split) - 1

    def split_patch(self, p, axis = None):
        if axis == None:
            p1_, p2_ = self.split_patch(p, 1)
            p1, p3 = self.split_patch(p1_, 0)
            p2, p4 = self.split_patch(p2_, 0)
            return p1, p2, p3, p4
        
        (kvs, geo), boundaries = self.patches[p]
        kv = kvs[axis].refine()

        #split_xi = sum(kv.support())/2.0
        #split_idx = kv.findspan(split_xi)+1
     
        split_idx = len(kv.kv) // 2
        split_xi = kv.kv[split_idx]    # parameter value where we split the KV
        new_knots1 = np.concatenate((kv.kv[:split_idx], (kv.p+1) * (split_xi,)))

        #new_knots2 = np.concatenate(((kv.p+1) * (split_xi,), kv.kv[split_idx:]))
        
        new_knots2 = np.concatenate(((kv.p) * (split_xi,), kv.kv[split_idx:]))
            
        # create new kvs and geo for first patch
        kvs1 = list(kvs)
        kvs1[axis] = bspline.KnotVector(new_knots1, kv.p)
        geo1 = copy.copy(geo)
        geo1.support = tuple(kv.support() for kv in kvs1)
        kvs1 = tuple(kvs1)
        
        # create new kvs and geo for second patch
        kvs2 = list(kvs)
        kvs2[axis] = bspline.KnotVector(new_knots2, kv.p) 
        geo2 = copy.copy(geo)
        geo2.support = tuple(kv.support() for kv in kvs2)
        kvs2 = tuple(kvs2)

        # dimension-independent description of bottom/left and top/right edge
        lower, upper = 2 * axis, 2 * axis + 1

        # copy existing boundaries, they will be corrected below
        boundaries = list(boundaries)
        new_boundaries = [list(bd) for bd in boundaries]

        new_p =  self.numpatches

        if axis == 0:     # y-axis
            split_boundaries = [2, 3]     # left and right were split
            new_vertices = [self.add_vertex(c) for c in corners(geo1)[2::1]]
        elif axis == 1:   # x-axis
            split_boundaries = [0, 1]     # bottom and top were split
            new_vertices = [self.add_vertex(c) for c in corners(geo1)[1::2]]
        else:
            assert False, 'unimplemented'

        # move existing interfaces from upper side of old to upper of new patch
        self._reindex_interfaces(p, upper, range(0, len(boundaries[upper]) - 1), 0, new_p=new_p)

        boundaries[upper]     = list(new_vertices)      # upper edge of new lower patch
        new_boundaries[lower] = list(new_vertices)      # lower edge of new upper patch

        # add interface between the two new patches
        self.add_interface(p, upper, 0, new_p, lower, 0, (False,))

        for sb, new_vtx in zip(split_boundaries, new_vertices):
            i_new = self.split_patch_boundary(p, sb, split_xi, self.vertices[new_vtx], new_p)

            # split the boundaries of the new patches at this vertex
            new_bd = self.boundaries(p)[sb]
            boundaries[sb] = list(new_bd[:i_new+1])
            new_boundaries[sb] = list(new_bd[i_new:])

            # change patch index for all interfaces from the split part of the boundary
            self._reindex_interfaces(p, sb, range(i_new, len(new_bd) - 1), -i_new, new_p=new_p)
            
        # change patch index for all corner nodes and T nodes on the upper edge of old patch   
        for vtx in new_boundaries[upper]:
                if vtx in self.C_nodes:
                    c = self.C_nodes[vtx][p]
                    del self.C_nodes[vtx][p]
                    self.C_nodes[vtx][new_p] = c
                else:
                    ((p0, b0), flip), (p1, c1), (p2, c2) = self.T_nodes[vtx]
                    if p0 == p: p0 = new_p
                    if p1 == p: p1 = new_p
                    if p2 == p: p2 = new_p
                    self.T_nodes[vtx] = (((p0, b0), flip), (p1, c1), (p2, c2))
        
        #also change patch index of possible T_nodes at the new boundaries in the different axis direction (left and right)
        for sb in split_boundaries:
            for vtx in new_boundaries[sb][1:-1]:
                if vtx in self.T_nodes:
                    ((p0, b0), flip), (p1, c1), (p2, c2) = self.T_nodes[vtx]
                    self.T_nodes[vtx] = (((new_p, b0), flip), (p1, c1), (p2, c2))
            
        self.patches[p] = ((kvs1, geo1), tuple(boundaries))
        self.patches.append(((kvs2, geo2), tuple(new_boundaries)))
        
        return p, new_p     # return the two indices of the split patches
    
            
    def split_patches(self, dir_data, patches=None, axis=None):
        if np.isscalar(patches):
            patches=(patches,)
        if patches==None:
            patches = np.arange(self.numpatches)
        if np.isscalar(axis):
            axis=len(patches)*(axis,)
        if axis==None:
            axis = len(patches)*(None,)
        assert len(patches)==len(axis), 'Splitting information does not match with the number of given patches to be split.'
        
        for ax, p in zip(np.array(axis), patches):
            split_dirichlet_data(p, self.numpatches, dir_data, axis=ax), self.split_patch(p,axis=ax)

    def boundaries(self, p):
        """Get the boundaries for patch `p`.

        A 2D patch has four boundaries, and each one is a list of vertex
        indices describing individual segments of the boundary.
        """
        return self.patches[p][1]

    def get_matching_interface(self, p, boundary, segment):
        """Get the boundary/interface which is connected to the given boundary/interface."""
        assert 0 <= p < len(self.patches)
        bdrs = self.boundaries(p)
        assert 0 <= boundary < len(bdrs)
        assert 0 <= segment < len(bdrs[boundary]) - 1
        matching = self.interfaces.get((p, boundary, segment))
        if matching:
            return matching[0]
        else:
            return None     # no matching segment - must be on the boundary

    def draw(self, vertex_idx = False, patch_idx = False, nodes=False, figsize=(8,8)):
        
        fig=plt.figure(figsize=figsize)
        for ((kvs, geo),_) in self.patches:
            vis.plot_geo(geo, gridx=kvs[0].mesh,gridy=kvs[1].mesh, color='lightgray')
            for x,y in [(x,y) for x in range(2) for y in range(2)]:
                vis.plot_geo(geo.boundary((x,y)), grid=2)
           
        if nodes:
            plt.scatter(*np.transpose(self.vertices))

        if patch_idx:
            for p in range(len(self.patches)):        # annotate patch indices
                geo = self.patches[p][0][1]
                center_xi = np.flipud(np.mean(geo.support, axis=1))
                center = geo(*center_xi)
                plt.annotate(str(p), center, fontsize=18, color='green')
            
        if vertex_idx:
            for i, vtx in enumerate(self.vertices):   # annotate vertex indices
                plt.annotate(str(i), vtx, fontsize=18, color='red')
        

    def sanity_check(self):
        for (p, b, s), ((p1, b1, s1), flip) in self.interfaces.items():
            # check that all interface indices make sense
            def bdspec_to_int(bdspec):
                return 2 * bdspec[0] + bdspec[1]    # convert to a boundary index (0..3)

            assert 0 <= p < len(self.patches) and 0 <= p1 < len(self.patches)
            bd, bd1 = self.boundaries(p), self.boundaries(p1)
            assert 0 <= b < len(bd) and 0 <= b1 < len(bd1)
            assert 0 <= s < len(bd[b]) - 1 and 0 <= s1 < len(bd1[b1]) - 1

        outer_boundaries = set()

        for p in range(len(self.patches)):
            # check topology of corner vertices
            kvs, geo = self.patches[p][0]
            (B,T,L,R) = bdrs = self.boundaries(p)
            assert B[0]  == L[0]     # bottom/left have same starting point
            assert B[-1] == R[0]     # bottom/right match in corner
            assert T[0]  == L[-1]    # top/left match in corner
            assert T[-1] == R[-1]    # top/right match in corner

            # check geometric position of corner vertices
            crns = corners(geo)
            assert np.allclose(self.vertices[B[0]],  crns[0])
            assert np.allclose(self.vertices[B[-1]], crns[1])
            assert np.allclose(self.vertices[T[0]],  crns[2])
            assert np.allclose(self.vertices[T[-1]], crns[3])

            # check that there are no duplicate vertices in any segment
            for bd in bdrs:
                assert len(np.unique(bd)) == len(bd)

            # check that connectivity of interfaces is consistent
            for b, bd in enumerate(bdrs):
                assert len(bd) >= 2    # each boundary has at least one segment
                for s in range(len(bd) - 1):    # check each boundary segment
                    other = self.get_matching_interface(p, b, s)
                    if other:
                        # if not on the boundary, make sure the other one refers back to this one
                        matching = self.get_matching_interface(*other)
                        #print((p,b,s), '->', other, '->', matching)
                        assert matching == (p, b, s)

                        # check that the two segments refer to the same two vertices
                        (p1, b1, s1) = other
                        assert sorted(bd[s:s+2]) == sorted(self.boundaries(p1)[b1][s1:s1+2])
                    else:
                        segment = tuple(sorted(bd[s:s+2]))
                        if segment in outer_boundaries:
                            assert False, str(segment) + ' is non-linked boundary segment for two patches!'
                        outer_boundaries.add(segment)
