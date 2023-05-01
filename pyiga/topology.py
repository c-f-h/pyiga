import itertools as it
from pyiga import bspline, vis, assemble
import copy

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def face_indices(n, m, zipped=True):
    """Returns all m-dimensional boundary manifolds of an n-dimensional hypercube.
    
    The manifolds are represented by a list of pairs ('ax', 'side') where 'ax' is an integer 
    describing which axis we set to 'side' which is 0 or 1.
    
    Optionally a parameter 'zipped' can be passed, which is by default 'True'. If 'zipped' is set to False,
    the manifolds are given as 2 tuples ('axis', 'sides') which saves the 'axis' and corresponding 'sides' information seperately.
    """
    S=[]
    for comb in it.combinations(range(n),n-m):
        for i in it.product(*(n-m)*((0,1),)):
            #print(list(zip(comb,i)))
            if zipped:
                S.append(tuple(zip(comb,i)))
            else:
                S.append((comb, i))
    return S

def bdspec_to_int(bdspec):
    return 2 * bdspec[0][0] + bdspec[0][1]    # convert to a boundary index (0..3)
    
def corners(geo, ravel=False):
    """Return an array containing the locations of the 2^d corners of the given
    geometry."""
    vtx = geo.grid_eval(geo.support)
    if ravel:
        return vtx.reshape((-1,geo.dim))
    return vtx

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
        

        self.outer_boundaries = {0:set()}

        
        self.Nodes = {'T0':dict(), 'T1':dict()}
        #self.Nodes['T1'] = dict()

        if patches:
            # add interfaces between patches
            conn, interfaces = assemble.detect_interfaces(patches)
            assert conn, 'patch graph is not connected!'
            
            for p, patch in enumerate(patches):
                kvs, geo = patch
                # add/get vertices (checks for duplicates)
                vtx = [self.add_vertex(c) for c in corners(geo, ravel = True)]
                # add boundaries in fixed order
                self.add_patch(patch, (
                    [vtx[0], vtx[1]],    # bottom
                    [vtx[2], vtx[3]],    # top
                    [vtx[0], vtx[2]],    # left
                    [vtx[1], vtx[3]],    # right
                ))
                    
                for cspec, v in zip(face_indices(2,0), vtx):
                    if v in self.Nodes['T0']:
                        self.Nodes['T0'][v][p] = cspec
                    else:
                        self.Nodes['T0'][v]={p : cspec}

            for (p0, bd0, p1, bd1, (perm, flip)) in interfaces:
                self.add_interface(p0, bdspec_to_int(bd0), p1, bdspec_to_int(bd1), (perm,flip))
            
            for p in range(len(patches)):
                for b in range(4):
                    if (p,b) not in self.interfaces:
                        self.outer_boundaries[0].add((p,b))
                                
            #self.sanity_check()

    @property
    def numpatches(self):
        return len(self.patches)
    
    def geos(self):
        return [geo for ((_, geo),_) in self.patches]
    
    def kvs(self):
        return [kvs for ((kvs, _),_) in self.patches]
            
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

    def add_interface(self, p0, b0, p1, b1, flip):
        """Join segment s0 of boundary b0 of patch p0 with segment s1
        of boundary b1 of p1.
        """
        #bds0 = self.boundaries(p0)
        #bds1 = self.boundaries(p1)
        #assert b0 < len(bds0) and b1 < len(bds1)
        #assert s0 < len(bds0[b0]) and s1 < len(bds1[b1])
        S0 = (p0, b0)
        S1 = (p1, b1)
        self.interfaces[S0] = (S1, flip)
        self.interfaces[S1] = (S0, flip)        

    def set_boundary_id(self, boundary_id):
        marked = set().union(*boundary_id.values())
        for key in self.outer_boundaries.keys():
            self.outer_boundaries[key]=self.outer_boundaries[key]-marked
        self.outer_boundaries.update(boundary_id)
        
    def rename_boundary(self, idx,new_idx):
        assert new_idx not in self.outer_boundaries
        self.outer_boundaries[new_idx] = self.outer_boundaries.pop(idx)

    def boundaries(self, p):
        """Get the boundaries for patch `p`.

        A 2D patch has four boundaries, and each one is a list of vertex
        indices describing individual segments of the boundary.
        
        """
        return self.patches[p][1]

    def get_matching_interface(self, p, boundary):
        """Get the boundary/interface which is connected to the given boundary/interface."""
        assert 0 <= p < len(self.patches)
        bdrs = self.boundaries(p)
        assert 0 <= boundary < len(bdrs)
        matching = self.interfaces.get((p, boundary))
        if matching:
            return matching[0]
        else:
            return None     # no matching segment - must be on the boundary

    def draw(self, knots=True, vertex_idx = False, patch_idx = False, nodes=False, figsize=(8,8)):
        """draws a visualization of the patchmesh in 2D."""
        fig=plt.figure(figsize=figsize)
        for ((kvs, geo),_) in self.patches:
            if knots:
                vis.plot_geo(geo, gridx=kvs[0].mesh,gridy=kvs[1].mesh, color='lightgray')
            vis.plot_geo(geo, grid=2)
           
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
        for (p, b), ((p1, b1), flip) in self.interfaces.items():
            # check that all interface indices make sense
            assert 0 <= p < len(self.patches) and 0 <= p1 < len(self.patches)
            bd, bd1 = self.boundaries(p), self.boundaries(p1)
            assert 0 <= b < len(bd) and 0 <= b1 < len(bd1)
                                        
        for p in range(len(self.patches)):
                       
            # check topology of corner vertices
            kvs, geo = self.patches[p][0]
            crns = corners(geo)
            
            (B,T,L,R) = bdrs = self.boundaries(p)
            assert B[0] == L[0]    # bottom/left match in corner
            assert B[1] == R[0]    # bottom/right match in corner
            assert T[0] == L[1]    # top/left match in corner
            assert T[1] == R[1]    # top/right match in corner

            # check geometric position of corner vertices

            assert np.allclose(self.vertices[B[0]],  crns[0])
            assert np.allclose(self.vertices[B[1]], crns[1])
            assert np.allclose(self.vertices[T[0]],  crns[2])
            assert np.allclose(self.vertices[T[1]], crns[3])

            # check that there are no duplicate vertices in any segment
            for bd in bdrs:
                assert len(np.unique(bd)) == len(bd)

            # check that connectivity of interfaces is consistent
            for b, bd in enumerate(bdrs):
                assert len(bd) >= 2    # each boundary has at least one segment
                other = self.get_matching_interface(p, b)
                if other:
                    # if not on the boundary, make sure the other one refers back to this one
                    matching = self.get_matching_interface(*other)
                    #print((p,b,s), '->', other, '->', matching)
                    assert matching == (p, b)

                    # check that the two segments refer to the same two vertices
                    (p1, b1) = other
                    assert sorted(bd[0:2]) == sorted(self.boundaries(p1)[b1][0:2])
                else:
                    segment = tuple(sorted(bd[0:2]))
                    assert (p, b) in outer_boundaries, str(segment) + ' is non-linked boundary segment for two patches!'
                    #assert False, str(segment) + ' is non-linked boundary segment for two patches!'
                    #self.outer_boundaries.add(segment)
