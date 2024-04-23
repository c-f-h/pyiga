import itertools as it
from pyiga import bspline, vis, assemble
import copy
import time

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
    if isinstance(bdspec, int):
        return bdspec
    return 2 * bdspec[0][0] + bdspec[0][1]    # convert to a boundary index (0..3)

def int_to_bdspec(bdspec):
    if isinstance(bdspec,tuple) and len(bdspec)==2:
        return bdspec
    else:
        return (bdspec//2,bdspec%2)
    
def corners(geo, ravel=False):
    """Return an array containing the locations of the 2^d corners of the given
    geometry."""
    vtx = geo.grid_eval(geo.support)
    if ravel:
        return vtx.reshape((-1,geo.dim))
    return vtx

def edges(corners):
    C=np.array(corners).reshape(2,2,2)
    return np.vstack((C.transpose((0,1,2)).reshape(-1,2), C.transpose((0,2,1)).reshape(-1,2), C.transpose((1,2,0)).reshape(-1,2)))

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
    def __init__(self, patches = None, interfaces = None, domains=None):
        self.vertices = []
        self.patches = []
        self.interfaces = dict()
        self.outer_boundaries = {0:set()}
        self.domains = {0:set()}
        self.patch_domains = dict()
        
        # self.Nodes = {'T0':dict(), 'T1':dict()}
        # self.Nodes['T1'] = dict()

        if patches:
            if domains:
                self.domains=domains
                self.patch_domains=dict()
                for idx in domains:
                    for p in domains[idx]:
                        self.patch_domains[p]=idx
            else:
                self.domains[0]=set(np.arange(len(patches)))
                self.patch_domains = {p:0 for p in range(len(patches))}
            for p, patch in enumerate(patches):
                kvs, geo = patch
                # add/get vertices (checks for duplicates)
                vtx = [self.add_vertex(c) for c in corners(geo, ravel = True)]
                self.add_patch(patch)
                
                # for cspec, v in zip(face_indices(2,0), vtx):
                #     if v in self.Nodes['T0']:
                #         self.Nodes['T0'][v][p] = cspec
                #     else:
                #         self.Nodes['T0'][v]={p : cspec}
                
            # add interfaces between patches
            conn, conf_interfaces = assemble.detect_interfaces(patches)
            #assert conn, 'patch graph is not connected!'
            for (p0, bd0, p1, bd1, (perm, flip)) in conf_interfaces:
                self.add_interface(p0, bdspec_to_int(bd0), 0, p1, bdspec_to_int(bd1), 0, flip)
            if interfaces:
                D={}
                for (p, b, s),(p1, b1, s1),flip in interfaces:
                    self.add_interface(p, b, s, p1, b1, s1, (flip,))
                    if (p,b) in D:
                        D[(p,b)][s]=self.boundaries(p1)[0][b1]
                    else:
                        D[(p,b)]={s:self.boundaries(p1)[0][b1]}
                    if (p1,b1) in D:
                        D[(p1,b1)][s1]=self.boundaries(p)[0][b]
                    else:
                        D[(p1,b1)]={s1:self.boundaries(p)[0][b]}
                #print(D)
                for (p,b) in D:
                    if len(D[(p,b)])>1:
                        bd=D[(p,b)][0].copy()
                        for i in range(1,len(D[(p,b)])):
                            bd.append(D[(p,b)][i][1])
                        self.patches[p][1][0][b]=bd
                        
            for p in range(len(patches)):
                for b in range(4):
                    if (p,b,0) not in self.interfaces:
                        self.outer_boundaries[0].add((p,b))
                                
            #self.sanity_check()

    @property
    def numpatches(self):
        return len(self.patches)
    
    @property
    def geos(self):
        return [geo for ((_, geo),_) in self.patches]
    
    @property
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

    def add_patch(self, patch):
        kvs, geo = patch
        vtx = [self.add_vertex(c) for c in corners(geo, ravel = True)]
        bds = [[vtx[0], vtx[1]],    # bottom
               [vtx[2], vtx[3]],    # top
               [vtx[0], vtx[2]],    # left
               [vtx[1], vtx[3]]]    # right
        S=geo.support
        bds_par = [[S[1][0], S[1][1]],
                   [S[1][0], S[1][1]],
                   [S[0][0], S[0][1]],
                   [S[0][0], S[0][1]]]
        self.patches.append((patch, [bds,bds_par]))

    def add_interface(self, p0, b0, s0, p1, b1, s1, flip):
        """Join segment s0 of boundary b0 of patch p0 with segment s1
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
        
    def p_refine(self, patches = None):
        if isinstance(patches,int):
            patches = {p:patches for p in range(self.numpatches)}     

        for p in patches:
            (kvs,geo), b = self.patches[p]
            new_kvs = tuple([kv.p_refine(patches[p]) for kv in kvs])
            self.patches[p]=((new_kvs, geo), b)
            
        return {p:(p,) for p in patches}
            
    def set_boundary_id(self, boundary_id):
        marked = set().union(*boundary_id.values())
        empty_keys =[]
        for key in self.outer_boundaries:
            self.outer_boundaries[key]=self.outer_boundaries[key]-marked
            if len(self.outer_boundaries[key])==0: empty_keys.append(key)
        for key in empty_keys:
            del self.outer_boundaries[key]
        for key in boundary_id:
            if key in self.outer_boundaries:
                self.outer_boundaries[key]=self.outer_boundaries[key].union(boundary_id[key])
            else:
                self.outer_boundaries[key]=boundary_id[key]
        #self.outer_boundaries.update(boundary_id)
        
    def set_domain_id(self, domain_id):
        marked = set().union(*domain_id.values())
        empty_keys =[]
        for key in self.domains.keys():
            self.domains[key]=self.domains[key]-marked
            if len(self.domains[key])==0: empty_keys.append(key)
        for key in empty_keys:
            del self.domains[key]
        for key in domain_id.keys():
            if key in self.domains:
                self.domains[key]=self.domains[key].union(domain_id[key])
            else:
                self.domains[key]=domain_id[key]
            for p in domain_id[key]:
                self.patch_domains[p]=key
        #self.domains.update(domain_id)
        
    def rename_boundary(self, idx, new_idx):
        assert new_idx not in self.outer_boundaries
        self.outer_boundaries[new_idx] = self.outer_boundaries.pop(idx)
        
    def rename_domain(self, idx, new_idx):
        assert new_idx not in self.domains
        self.domains[new_idx] = self.domains.pop(idx)
        for p in self.patch_domains:
            if self.patch_domains[p]==idx:
                self.patch_domains[p]=new_idx
        
    def remove_boundary(self, bds):
        bds=set(bds)
        for key in self.outer_boundaries:
            self.outer_boundaries[key] = self.outer_boundaries[key]-bds
            
    #def split_mesh(self, patches):
        #patches=np.unique(patches)
        #self.patches=self.patches[patches]
        
    def _reindex_interfaces(self, p, b, old_s, ofs, new_p=None):
        old_s = list(old_s)
        new_s = [s + ofs for s in old_s]
        assert len(old_s) == len(new_s)
        if new_p is None:
            new_p = p
        # S_old = [(p, b, s) for s in old_s]
        S_new = [(new_p, b, s) for s in new_s]
        old_intf = [self.interfaces.pop((p, b, s), None) for s in old_s]
        for Sn, intf in zip(S_new, old_intf):
            if intf:
                self.interfaces[Sn] = intf
                self.interfaces[intf[0]] = (Sn, intf[1])

    def split_boundary_segment(self, p, b, s, new_vtx, split_xi, new_p):
        """Split the boundary segment `s` on boundary `b` of patch `p` by
        inserting the new vertex with index `new_vtx`.

        If the segment interfaces with another patch, also splits the
        corresponding segment on the other side and updates the interface
        information.
        """
        bd = self.boundaries(p)[0][b]
        bd_par = self.boundaries(p)[1][b]
        assert 0 <= s < len(bd) - 1
        bd.insert(s + 1, new_vtx)
        bd_par.insert(s + 1, split_xi)
        # shift all later interfaces up by one
        #print(list(range(s + 1, len(bd) - 2)))
        self._reindex_interfaces(p, b, range(s + 1, len(bd) - 2), +1)

        # also split the matching boundary segment on neighboring patch, if any
        other = self.get_matching_interface(p, b, s)
        if other:
            (p1, b1, s1) = other
            bd1 = self.boundaries(p1)[0][b1]
            bd_par1 = self.boundaries(p1)[1][b1]
            #print(bd1)
            #print(new_vtx)
            bd1.insert(s1 + 1, new_vtx)
            bd_par1.insert(s1 + 1, split_xi)

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
#             else:
#                 self.outer_boundaries.add((new_p, b))

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
        
        vtx1, vtx2 = self.boundaries(p)[0][b][::len(self.boundaries(p)[0][b])-1]

#         if vtx1 in self.Nodes['T0']:
#             corner1 = self.Nodes['T0'][vtx1][p]
#         else:
#             [corner1] = [c for patch, c in self.Nodes['T1'][vtx1][1:] if patch==p]
        
#         if vtx2 in self.Nodes['T0']:
#             corner2 = self.Nodes['T0'][vtx2][p]
#         else:
#             [corner2] = [c for patch, c in self.Nodes['T1'][vtx2][1:] if patch==p]
            
        try:
            # is the vertex already contained in the boundary?
            vtx_idx = self.boundaries(p)[1][b].index(xi)
            #print(1)
        except ValueError:
            # otherwise, we need to insert it, split the segments and insert a new T_node (or corner at the boundary of the domain)
            seg = self._find_boundary_split_index(p, b, xi, new_vtx)
            #print(seg)
            self.split_boundary_segment(p, b, seg, new_vtx, xi, new_p)
            
            # if (p,b,0) in self.interfaces:
            #     self.Nodes['T1'][new_vtx] = ((self.interfaces[(p,b,0)][0][:-1],self.interfaces[(p,b,0)][1]), (p, corner2), (new_p, corner1))
            # else:
            #     self.Nodes['T0'][new_vtx] = dict()
            #     self.Nodes['T0'][new_vtx][p] = corner2
            #     self.Nodes['T0'][new_vtx][new_p] = corner1
            return seg + 1  # we want the segment just after the newly inserted vertex
        else:
            # if new_vtx not in self.Nodes['T0']:
            #     self.Nodes['T0'][new_vtx] = dict()
            # for patch, c in self.Nodes['T1'][new_vtx][1:]:
            #     self.Nodes['T0'][new_vtx][patch] = c
            # self.Nodes['T0'][new_vtx][p] = corner2
            # self.Nodes['T0'][new_vtx][new_p] = corner1
            # del self.Nodes['T1'][new_vtx]
            return vtx_idx

    def _find_boundary_split_index(self, p, bdidx, xi_split, vtx_idx):
        (kvs, geo), (bds, bds_par) = self.patches[p]
        segments = bds[bdidx]
        # simple case: if a single interval covers the boundary, we split it
        if len(segments) == 2:
            return 0
        bd_geo = geo.boundary((bdidx // 2, bdidx % 2))
        #print(bd_geo.support)
        bd_vtx_xi = [bd_geo.find_inverse(self.vertices[j])[0] for j in segments]
        # find segment where xi_split would need to be inserted to maintain order
        return np.searchsorted(bd_vtx_xi, xi_split) - 1

    def split_patch(self, p, axis = None, mult=1):
        if axis == None:
            (p1, p2) = self.split_patch(p,  axis=1, mult=mult)
            (p1, p3) = self.split_patch(p1, axis=0, mult=mult)
            (p2, p4) = self.split_patch(p2, axis=0, mult=mult)
            
            #new_kvs = (new_kvs1[0], new_kvs0[1])
            return (p1, p2, p3, p4)
        
        (kvs, geo), (bds, bds_par) = self.patches[p]
        kv = kvs[axis].h_refine(mult=mult)
        dim = len(kvs)
        
        #split_xi = sum(kv.support())/2.0
        #split_idx = kv.findspan(split_xi)+1
     
        m_idx = len(kv.mesh)//2
        mesh_ofs = kv.mesh_span_indices()
        split_idx = mesh_ofs[m_idx]
        split_mult = mesh_ofs[m_idx]-mesh_ofs[m_idx-1]
        split_xi = kv.kv[split_idx]    # parameter value where we split the KV
        new_knots1 = np.concatenate((kv.kv[:split_idx], (kv.p+1-(mult-1)) * (split_xi,)))
        new_knots2 = np.concatenate(((kv.p) * (split_xi,), kv.kv[split_idx:]))
        new_kvs = tuple([bspline.KnotVector(np.concatenate((kv.kv[:split_idx], (kv.p-1) * (split_xi,), kv.kv[split_idx:])),kv.p) if d==axis else kvs[d] for d in range(dim)])
            
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
        sides = (lower+2)%4, (upper+2)%4

        # copy existing boundaries, they will be corrected below
        bds = list(bds)
        new_bds = [list(bd) for bd in bds]
        bds_par = list(bds_par)
        new_bds_par = [list(par) for par in bds_par]

        new_p =  self.numpatches
        
        if axis == 0:                     # y-axis
            split_boundaries = [2, 3]     # left and right were split
            new_vertices = [self.add_vertex(c) for c in corners(geo1)[1,:,:]]
        elif axis == 1:                   # x-axis
            split_boundaries = [0, 1]     # bottom and top were split
            new_vertices = [self.add_vertex(c) for c in corners(geo1)[:,1,:]]
        else:
            assert False, 'unimplemented'
        
        # print(new_vertices)
        # print(split_boundaries)

        # move existing interfaces from upper side of old to upper of new patch ###
        self._reindex_interfaces(p, upper, range(0, len(bds[upper]) - 1), 0, new_p=new_p)
        # reindex existing outer boundary to new patch
        for s in self.outer_boundaries.keys():
            if (p, upper) in self.outer_boundaries[s]:
                self.outer_boundaries[s].remove((p, upper))
                self.outer_boundaries[s].add((new_p, upper))
            for bd in sides:
                if (p, bd) in self.outer_boundaries[s]:
                    self.outer_boundaries[s].add((new_p, bd))
                    
        bds[upper]         = list(new_vertices)      # upper edge of new lower patch
        bds_par[upper]     = [bds_par[upper][0], bds_par[upper][-1]]
        new_bds[lower]     = list(new_vertices)      # lower edge of new upper patch
        new_bds_par[lower] = [bds_par[upper][0], bds_par[upper][-1]]
        # add interface between the two new patches
        self.add_interface(p, upper, 0, new_p, lower, 0, (False,))

        for sb, new_vtx in zip(split_boundaries, new_vertices):
            i_new = self.split_patch_boundary(p, sb, split_xi, self.vertices[new_vtx], new_p)
            #print(i_new)
            # split the boundaries of the new patches at this vertex
            new_bd = self.boundaries(p)[0][sb]
            new_bd_par = self.boundaries(p)[1][sb]
            
            # print(new_bd)

            bds[sb] = list(new_bd[:i_new+1])
            new_bds[sb] = list(new_bd[i_new:])
            bds_par[sb] = list(new_bd_par[:i_new+1])
            new_bds_par[sb] = list(new_bd_par[i_new:])
            #bds_par[sb] = list()
            # if len(bds[sb])==1:
            #     print(bds[sb])
            if len(new_bds[sb])==1:
                print(new_bd)
                print(i_new)
                print(new_bds[sb])

            # change patch index for all interfaces from the split part of the boundary
            self._reindex_interfaces(p, sb, range(i_new, len(new_bd) - 1), -i_new, new_p=new_p)
            
        self.patches[p] = ((kvs1, geo1), [bds, bds_par])
        self.patches.append(((kvs2, geo2), [new_bds, new_bds_par]))
        
        domain_idx=self.patch_domains[p]
        self.patch_domains[new_p]=domain_idx
        self.domains[domain_idx].add(new_p)
        
        return (p, new_p)     # return the two indices of the split patches
    
    def split_boundary_idx(self, p, n, axis=None):
        if axis==None:
            axis=(0,1)
        axis=np.unique(axis)
        if len(axis)==1: axis=axis[0]
        for s in self.outer_boundaries.keys():
            
            b_idx_p = [(patch , bd) for (patch, bd) in self.outer_boundaries[s] if patch == p]
            
            if b_idx_p:
                if not np.isscalar(axis):
                    for k,ax in enumerate(axis[::-1]):
                        self.split_boundary_idx(p, n+2**k-1, axis=ax)
                        for i in range(2**k-1):
                            self.split_boundary_idx(n+i, n+2**k+i, axis=ax)
                else:
                    for (patch, bd) in b_idx_p:
                        if axis == 0:
                            if bd == 'top':
                                self.outer_boundaries[s].remove((patch, bd))
                                self.outer_boundaries[s].append((n, bd))
                            if bd == 'left' or bd == 'right':
                                self.outer_boundaries[s].append((n, bd))
                        if axis == 1:
                            if bd == 'right':
                                self.outer_boundaries[s].remove((patch, bd))
                                self.outer_boundaries[s].append((n, bd))
                            if bd == 'top' or bd == 'bottom':
                                self.outer_boundaries[s].append((n, bd))
    
            
    def h_refine(self, patches=None, mult=1):
        if isinstance(patches, dict):
            if len(patches)>0:
                assert max(patches.keys())<self.numpatches and min(patches.keys())>=0, "patch index out of bounds."
        elif isinstance(patches,int):
            assert patches >=-1 and patches < 2, "dimension error."
            patches = {p:patches for p in range(self.numpatches)}
        elif isinstance(patches, (list, set, np.ndarray)):
            if len(patches)>0:
                assert max(patches)<self.numpatches and min(patches)>=0, "patch index out of bounds."
            patches = {p:None for p in patches}
        elif patches==None:
            patches = {p:None for p in range(self.numpatches)}
        else:
            assert 0, "unknown input type"

        new_patches = dict()
        new_kvs = dict()
        
        for p in patches.keys():
            #self.split_boundary_idx(p, self.numpatches, axis=patches[p])
            (kvs,geo), (b, b_par) = self.patches[p]
            if patches[p]==-1:
                new_kvs = tuple([kv.h_refine(mult=mult) for kv in kvs])
                self.patches[p]=((new_kvs, geo), (b, b_par))
                new_p=(p,)
            elif patches[p]=='q':
                new_kvs = tuple([kv.b_refine(q=0.75) for kv in kvs])
                self.patches[p]=((new_kvs, geo), (b, b_par))
                new_p=(p,)
            else:    
                t=time.time()
                new_p = self.split_patch(p, axis=patches[p], mult=mult)
                #print(time.time()-t)
            new_patches[p] = new_p
            #new_kvs[p] = new_kvs_  
        return new_patches

    def boundaries(self, p):
        """Get the boundaries for patch `p`.

        A 2D patch has four boundaries, and each one is a list of vertex
        indices describing individual segments of the boundary.
        
        A 3D patch has six boundaries, and each one is a list of 4 tuples 
        of vertex indices each describing the edges enclosing the boundary. 
        """
        return self.patches[p][1]

    def get_matching_interface(self, p, boundary, segment):
        """Get the boundary/interface which is connected to the given boundary/interface."""
        assert 0 <= p < len(self.patches)
        bdrs = self.boundaries(p)[0]
        assert 0 <= boundary < len(bdrs)
        assert 0 <= segment < len(bdrs[boundary]) - 1
        matching = self.interfaces.get((p, boundary, segment))
        if matching:
            return matching[0]
        else:
            return None     # no matching segment - must be on the boundary

    def draw(self, vertex_idx = False, patch_idx = False, nodes=False, bwidth=1, color=None, bcolor=None, axis='scaled', **kwargs):
        """draws a visualization of the patchmesh in 2D."""
        
        if color is not None:
            if isinstance(color,dict):
                for d in (set(self.domains) - set(color)):
                    color[d] = 'white'
            elif isinstance(color, str):
                color = {d: color for d in self.domains}
        
        if kwargs.get('fig'):
            fig, ax = kwargs['fig']
        else:
            fig=plt.figure(figsize=kwargs.get('figsize'))
            ax = plt.axes()
            
        if nodes:
            plt.scatter(*np.transpose(self.vertices),zorder=100)
        
        for p,((kvs, geo),_) in enumerate(self.patches):
            if color is not None:
                c=color[self.patch_domains[p]]
            else:
                c=None
            if kwargs.get('knots'):
                vis.plot_geo(geo, gridy=kvs[0].mesh, gridx=kvs[1].mesh, lcolor='lightgray', color=c, boundary=True)
            else:
                vis.plot_geo(geo, grid=2, color=c, boundary=True)
        
        for key in self.outer_boundaries:
            bcol=None
            if bcolor is not None:
                bcol=bcolor[key]
            for (p,b) in self.outer_boundaries[key]:
                vis.plot_geo(self.geos[p].boundary([assemble.int_to_bdspec(b)]), linewidth=bwidth, color=bcol)

        if patch_idx:
            for p in range(len(self.patches)):        # annotate patch indices
                geo = self.patches[p][0][1]
                center_xi = np.flipud(np.mean(geo.support, axis=1))
                center = geo(*center_xi)
                plt.annotate(str(p), center, fontsize=18, color='green')
            
        if vertex_idx:
            for i, vtx in enumerate(self.vertices):   # annotate vertex indices
                plt.annotate(str(i), vtx, fontsize=18, color='red')
                
        plt.axis(axis);
        

    def sanity_check(self):
        # for (p, b, s), ((p1, b1, s1), flip) in self.interfaces.items():
        #     # check that all interface indices make sense
        #     assert 0 <= p < len(self.patches) and 0 <= p1 < len(self.patches)
        #     bd, bd1 = self.boundaries(p), self.boundaries(p1)
        #     assert 0 <= b < len(bd) and 0 <= b1 < len(bd1)
        #     assert 0 <= s < len(bd[b]) - 1 and 0 <= s1 < len(bd1[b1]) - 1
            
        I1 = list(self.interfaces.keys())
        I2 = [i[0] for i in self.interfaces.values()]
        
        assert set(I1)==set(I2), "interface information not compatible"
        I1=np.array(I1)
        I2=np.array(I2)
        
        V = np.array(self.vertices)
        I = np.array([[b[::len(b)-1] for b in self.boundaries(p)[0]] for p in range(self.numpatches)])
        bdrs = [[b for b in self.boundaries(p)[0]] for p in range(self.numpatches)]
        S=np.array([[len(b)-1 for b in B] for B in bdrs])
        
        assert np.all(S>0), "some boundary does not have starting- or endpoint"
        
        assert np.all((I1[:,0]>=0) | (I1[:,0]<self.numpatches)), "patch index is negative or exceeds number of patches"
        assert np.all((I1[:,1]>=0) | (I1[:,1]<4)), "boundary index is invalid"
        assert np.all(np.array(I1)[:,2]<S[np.array(I1)[:,0],np.array(I1)[:,1]]), "segment index exceeds number of segments on that side of the interface"
        
        assert np.all(np.isclose(V[I[:,0:2]],np.array([corners(geo) for geo in self.geos]))), "corners do not match vertex information"
        
        BL = (I[:,0,0]==I[:,2,0])
        BR = (I[:,0,1]==I[:,3,0])
        TL = (I[:,1,0]==I[:,2,1])
        TR = (I[:,1,1]==I[:,3,1])
        
        assert np.all(BL), "boundary information does not match on lower left corner on patches: " + str(np.where((BL==False)[0]))
        assert np.all(BR), "boundary information does not match on lower right corner on patches: " + str(np.where((BR==False)[0]))
        assert np.all(TL), "boundary information does not match on upper left corner on patches: " + str(np.where((TL==False)[0]))
        assert np.all(TR), "boundary information does not match on upper right corner on patches: " + str(np.where((TR==False)[0]))
        
        # for p in range(self.numpatches):
        #     for b in range(4):
        #         for s in range(S[p,b]):
        #             other = self.get_matching_interface(p, b, s)
        #             if other:
        #                 assert (p,b,s)==self.interfaces[self.interfaces[(p,b,s)][0]][0], "interface connectivity invalid"
        #             else:
        #                 assert np.any([(p, b) in self.outer_boundaries[idx] for idx in self.outer_boundaries]),'A segment is neither a linked interface segment for two patches nor an outer boundary!'
                 
################################################################################################################
###  3D PATCHMESH
################################################################################################################
            
class PatchMesh3D:
    def __init__(self, patches = None, domains = None):
        self.vertices = []
        self.edges=[]
        self.patches = []
        self.interfaces = dict()
        self.outer_boundaries = {0:set()}
        
        self.domains = {0:set()}
        self.patch_domains = dict()
        
        # self.Nodes = {'T0':dict(), 'T1':dict(), 'T2':dict()}
        # self.Edges = {'T0':dict(), 'T1':dict()}
        #self.T_nodes = dict()

        if patches:
            # add interfaces between patches
            conn, interfaces = assemble.detect_interfaces(patches)
            assert conn, 'patch graph is not connected!'
            
            if domains:
                self.domains=domains
                for idx in domains:
                    for p in domains[idx]:
                        self.patch_domains[p]=idx
            else:
                self.domains[0]=set(np.arange(len(patches)))
                self.patch_domains = {p:0 for p in range(len(patches))}
            
            for p, patch in enumerate(patches):
                kvs, geo = patch
                # add/get vertices (checks for duplicates)
                vtx = [self.add_vertex(c)[0] for c in corners(geo, ravel=True)]
                edg = [self.add_edge(vtx1, vtx2) for vtx1, vtx2 in edges(vtx)]
                #edges = [(self.add_vertex(vtx1), self.add_vertex(vtx2)) for (vtx1, vtx2) in edges(geo)]
                #edges = [self.add_edge(e) for e in edges]
                
                # add boundaries in fixed order
                self.add_patch(patch, (
                    BSegments([(vtx[0], vtx[1]), (vtx[2], vtx[3]), (vtx[0], vtx[2]), (vtx[1], vtx[3])], 0),    #front
                    BSegments([(vtx[4], vtx[5]), (vtx[6], vtx[7]), (vtx[4], vtx[6]), (vtx[5], vtx[7])], 0),    #back
                    BSegments([(vtx[0], vtx[1]), (vtx[4], vtx[5]), (vtx[0], vtx[4]), (vtx[1], vtx[5])], 1),    #bottom
                    BSegments([(vtx[2], vtx[3]), (vtx[6], vtx[7]), (vtx[2], vtx[6]), (vtx[3], vtx[7])], 1),    #top
                    BSegments([(vtx[0], vtx[2]), (vtx[4], vtx[6]), (vtx[0], vtx[4]), (vtx[2], vtx[6])], 2),    #left
                    BSegments([(vtx[1], vtx[3]), (vtx[5], vtx[7]), (vtx[1], vtx[5]), (vtx[3], vtx[7])], 2),    #right
                ))
                    
#                 for c_spec, v in zip(face_indices(3,0), vtx):
#                     if v in self.Nodes['T0']:
#                         self.Nodes['T0'][v][p] = c_spec
#                     else:
#                         self.Nodes['T0'][v]={p : c_spec}
                        
#                 for e_spec, e in zip(face_indices(3,1), edg):
#                     if e in self.Edges['T0']:
#                         self.Edges['T0'][e][p] = e_spec
#                     else:
#                         self.Edges['T0'][e]={p: e_spec}

            for (p0, bd0, p1, bd1, conn_info) in interfaces:
                self.add_interface(p0, bdspec_to_int(bd0), tuple(), p1, bdspec_to_int(bd1), tuple(), conn_info)
                
            for p in range(len(patches)):
                for b in range(6):
                    if (p,b,tuple()) not in self.interfaces:
                        self.outer_boundaries[0].add((p,b))

            #self.sanity_check()

    @property
    def numpatches(self):
        return len(self.patches)
    
    @property
    def geos(self):
        return [geo for ((_, geo),_) in self.patches]
    
    @property
    def kvs(self):
        return [kvs for ((kvs, _),_) in self.patches]
            
    def add_vertex(self, pos):
        """Add a new vertex at `pos` at return new index and `False` or return its index and `True` if one already exists there."""
        if self.vertices:
            distances = [np.linalg.norm(vtxpos - pos) for vtxpos in self.vertices]
            i_min = np.argmin(distances)
            if distances[i_min] < 1e-14:
                return i_min, False
        self.vertices.append(pos)
        return len(self.vertices) - 1, True
    
    def add_edge(self, vtx1, vtx2): 
        """Add a new edge from vtx1 to vtx2 or return its index if one already exists there."""
        if (vtx1, vtx2) in self.edges:
            return self.edges.index((vtx1, vtx2))
        elif (vtx2, vtx1) in self.edges:
            return self.edges.index((vtx2, vtx1))
        else:
            self.edges.append((vtx1, vtx2))
            return len(self.edges) - 1
    
    def remove_edge(self,vtx1,vtx2):
        if (vtx1, vtx2) in self.edges:
            self.edges.remove((vtx1, vtx2))
        elif (vtx2, vtx1) in self.edges:
            self.edges.remove((vtx2, vtx1))
        else:
            return

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
        """Join segment s0 of boundary b0 of patch p0 with segment s1
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
        
    def p_refine(self, patches = None):
        if isinstance(patches,int):
            patches = {p:patches for p in range(self.numpatches)}     

        for p in patches:
            (kvs,geo), b = self.patches[p]
            new_kvs = tuple([kv.p_refine(patches[p]) for kv in kvs])
            self.patches[p]=((new_kvs, geo), b)
            
        return {p:(p,) for p in patches}
              
    def set_boundary_id(self, boundary_id):
        marked = set().union(*boundary_id.values())
        empty_keys =[]
        for key in self.outer_boundaries:
            self.outer_boundaries[key]=self.outer_boundaries[key]-marked
            if len(self.outer_boundaries[key])==0: empty_keys.append(key)
        for key in empty_keys:
            del self.outer_boundaries[key]
        for key in boundary_id:
            if key in self.outer_boundaries:
                self.outer_boundaries[key]=self.outer_boundaries[key].union(boundary_id[key])
            else:
                self.outer_boundaries[key]=boundary_id[key]
        #self.outer_boundaries.update(boundary_id)
        
    def set_domain_id(self, domain_id):
        marked = set().union(*domain_id.values())
        empty_keys =[]
        for key in self.domains.keys():
            self.domains[key]=self.domains[key]-marked
            if len(self.domains[key])==0: empty_keys.append(key)
        for key in empty_keys:
            del self.domains[key]
        for key in domain_id.keys():
            if key in self.domains:
                self.domains[key]=self.domains[key].union(domain_id[key])
            else:
                self.domains[key]=domain_id[key]
            for p in domain_id[key]:
                self.patch_domains[p]=key
        #self.domains.update(domain_id)
        
    def rename_boundary(self, idx, new_idx):
        assert new_idx not in self.outer_boundaries
        self.outer_boundaries[new_idx] = self.outer_boundaries.pop(idx)
        
    def rename_domain(self, idx, new_idx):
        assert new_idx not in self.domains
        self.domains[new_idx] = self.domains.pop(idx)
        for p in self.patch_domains:
            if self.patch_domains[p]==idx:
                self.patch_domains[p]=new_idx
        
    def remove_boundary(self, bds):
        bds=set(bds)
        for key in self.outer_boundaries:
            self.outer_boundaries[key] = self.outer_boundaries[key]-bds

    def _reindex_interfaces(self, p, b, old_s, ofs = tuple(), r = 0, new_p=None):
        #old_s = list(old_s)
        if r > 0:
            new_s = [ofs + s[r:] for s in old_s]
        else:
            new_s = [ofs + s for s in old_s]
        if not new_p:
            new_p = p
        S_old = [(p, b, s) for s in old_s]
        S_new = [(new_p, b, s) for s in new_s]
        old_intf = [self.interfaces.pop(S, None) for S in S_old]
        for Sn, intf in zip(S_new, old_intf):
            if intf:
                self.interfaces[Sn] = intf
                self.interfaces[intf[0]] = (Sn, intf[1])

    def split_boundary_segment(self, p, b, axis, s, new_edge):
        """Split the boundary segment `s` on boundary `b` of patch `p` by
        inserting the new vertex with index `new_vtx`.

        If the segment interfaces with another patch, also splits the
        corresponding segment on the other side and updates the interface
        information.
        """
        bd = self.boundaries(p)[b] 
        #assert 0 <= s < len(bd) - 1
        bd.split_segment(ax=axis, seg=s, edge=new_edge)
        # shift all later interfaces up by one
        #self._reindex_interfaces(p, b, range(s + 1, len(bd) - 2), 1)

        # also split the matching boundary segment on neighboring patch, if any
        other = self.get_matching_interface(p, b, s)
        if other:
            (p1, b1, s1) = other
            bd1 = self.boundaries(p1)[b1]
            #bd1.insert(s1 + 1, new_vtx)
            
            ### need to check axis and possibly change it here if boundarys have different normal axis (if a the neighbouring patch was rotated e.g.)
            bd1.split_segment(ax=axis, seg=s1, edge=new_edge) 
            
            # shift all later interfaces up by one
            #self._reindex_interfaces(p1, b1, range(s1 + 1, len(bd1) - 2), +1)

            # fix the new interfaces to point to each other
            flip = self.interfaces[(p, b, s)][1]
            
            self.interfaces.pop((p, b, s))
            self.interfaces.pop((p1, b1, s1))
            
            bd_axis = axis - 1*(axis > self.boundaries(p)[b].normal_axis)
            if flip[bd_axis]:
                # indices are running in opposite directions on the two sides
                self.interfaces[(p, b, s + (0,))] = ((p1, b1, s1 + (1,)), flip)
                self.interfaces[(p1, b1, s1 + (1,))] = ((p, b, s + (0,)), flip)

                self.interfaces[(p, b, s + (1,))] = ((p1, b1, s1 + (0,)), flip)
                self.interfaces[(p1, b1, s1 + (0,))] = ((p, b, s + (1,)), flip)
            else:
                self.interfaces[(p, b, s + (0,))] = ((p1, b1, s1 + (0,)), flip)
                self.interfaces[(p1, b1, s1 + (0,))] = ((p, b, s + (0,)), flip)

                self.interfaces[(p, b, s + (1,))] = ((p1, b1, s1 + (1,)), flip)
                self.interfaces[(p1, b1, s1 + (1,))] = ((p, b, s + (1,)), flip)

    def split_patch_boundary(self, p, b, xi, axis, new_edge, new_p):
        """Split the boundary `b` of patch `p` at a vertex which lies at
        parameter value `xi` of the boundary curve and has coordinates
        `vtxpos`.

        Returns the index of the first segment after the new vertex.

        It is valid to pass a vertex which is already contained in the
        boundary, in which case nothing is inserted and the correct index is
        returned.
        """

        #vtx1, vtx2 = self.boundaries(p)[b][::len(self.boundaries(p)[b])-1]
        #new_edge = self.edges[new_edge]
        try:
            # is the new edge already contained in the boundary?
            if self.boundaries(p)[b].axis != axis:
                raise ValueError
        except ValueError:
            # otherwise, we need to insert it, split the segment and insert a new T_node (or corner at the boundary of the domain)
            seg = self._find_boundary_split_index(p, b, xi, new_edge)
            self.split_boundary_segment(p, b, axis, seg, new_edge)
            self.add_edge(*new_edge) ###T_edge
        else:
            return 

    def _find_boundary_split_index(self, p, bdidx, xi_split, vtx_idx):
        (kvs, geo), boundaries = self.patches[p]
        segments = boundaries[bdidx]
        # simple case: if a single interval covers the boundary, we split it
        if segments.is_leaf():
            return tuple()
        #bd_geo = geo.boundary((bdidx // 2, bdidx % 2))
        #bd_vtx_xi = [bd_geo.find_inverse(self.vertices[j])[0] for j in segments]
        # find segment where xi_split would need to be inserted to maintain order
        #return np.searchsorted(bd_vtx_xi, xi_split) - 1

    def split_patch(self, p, axis = None, mult=1):
        if axis == None:
            
            (p1, p2), new_kvs0 = self.split_patch(p,  axis=2, mult=mult)
            (p1, p3), new_kvs1 = self.split_patch(p1, axis=1, mult=mult)
            (p2, p4), _        = self.split_patch(p2, axis=1, mult=mult)
            (p1, p5), new_kvs2 = self.split_patch(p1, axis=0, mult=mult)
            (p2, p6), _        = self.split_patch(p2, axis=0, mult=mult)
            (p3, p7), _        = self.split_patch(p3, axis=0, mult=mult)
            (p4, p8), _        = self.split_patch(p4, axis=0, mult=mult)
                
            new_kvs = (new_kvs2[0], new_kvs1[1], new_kvs0[2])
            return (p1, p2, p3, p4, p5, p6, p7, p8), new_kvs
            #self.split_patch(p1_, 1, mult=mult)
        
        (kvs, geo), boundaries = self.patches[p]
        kv = kvs[axis].refine(mult=mult)
        
        #split_xi = sum(kv.support())/2.0
        #split_idx = kv.findspan(split_xi)+1
     
        m_idx = len(kv.mesh)//2
        mesh_ofs = kv.mesh_span_indices()
        split_idx = mesh_ofs[m_idx]
        split_mult = mesh_ofs[m_idx]-mesh_ofs[m_idx-1]
        split_xi = kv.kv[split_idx]    # parameter value where we split the KV
        new_knots1 = np.concatenate((kv.kv[:split_idx], (kv.p+1-(mult-1)) * (split_xi,)))
        new_knots2 = np.concatenate(((kv.p) * (split_xi,), kv.kv[split_idx:]))
        new_kvs = tuple([bspline.KnotVector(np.concatenate((kv.kv[:split_idx], (kv.p-1) * (split_xi,), kv.kv[split_idx:])),kv.p) if d==axis else kvs[d] for d in range(3)])
            
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

        # dimension-independent description of front/bottom/left and back/top/right edge
        lower, upper = 2 * axis, 2 * axis + 1
        sides = (lower+2)%6, (upper+2)%6, (lower+4)%6, (upper+4)%6 

        # copy existing boundaries, they will be corrected below
        boundaries = list(boundaries)
        new_boundaries = [bd for bd in boundaries]

        new_p =  self.numpatches
        new_vertices = []
        
        if axis == 0:                           # z-axis
            split_boundaries = [2, 3, 4, 5]     # bottom, top, left and right were split
            splitC = corners(geo1)[1,:,:,:].reshape(-1,3)
            C = [self.add_vertex(c)[0] for c in corners(geo).transpose((0,1,2,3)).reshape(-1,3)]
        elif axis == 1:                         # y-axis
            split_boundaries = [0, 1, 4, 5]     # front, back, left and right were split
            splitC = corners(geo1)[:,1,:,:].reshape(-1,3)
            C = [self.add_vertex(c)[0] for c in corners(geo).transpose((1,0,2,3)).reshape(-1,3)]
        elif axis == 2:                         # x-axis
            split_boundaries = [0, 1, 2, 3]     # front, back, bottom and top were split
            splitC = corners(geo1)[:,:,1,:].reshape(-1,3)
            C = [self.add_vertex(c)[0] for c in corners(geo).transpose((2,0,1,3)).reshape(-1,3)]
        else:
            assert False, 'This axis does not exist.'
            
        for i, c in enumerate(splitC):
            vtx, is_new = self.add_vertex(c)
            new_vertices.append(vtx)
            if is_new:
                self.remove_edge(C[i], C[i+4])
                self.add_edge(C[i], vtx)
                self.add_edge(vtx, C[i+4])

        #new_edges = [(new_vertices[0], new_vertices[1]), (new_vertices[2], new_vertices[3]), (new_vertices[0], new_vertices[2]), (new_vertices[1], new_vertices[3])]
        new_edges = [self.add_edge(new_vertices[i1], new_vertices[i2]) for i1, i2 in zip([0,2,0,1],[1,3,2,3])]
        # move existing interfaces from upper side of old to upper of new patch 
        self._reindex_interfaces(p, upper, boundaries[upper].return_segments(), new_p=new_p)
        
        for s in self.outer_boundaries.keys():
            if (p, upper) in self.outer_boundaries[s]:
                self.outer_boundaries[s].remove((p, upper))
                self.outer_boundaries[s].add((new_p, upper))
            for bd in sides:
                if (p, bd) in self.outer_boundaries[s]:
                    self.outer_boundaries[s].add((new_p, bd))
                    
        boundaries[upper]     =  BSegments([self.edges[e] for e in new_edges], axis)   # upper edge of new lower patch
        new_boundaries[lower] =  BSegments([self.edges[e] for e in new_edges], axis)   # lower edge of new upper patch

        # add interface between the two new patches
        self.add_interface(p, upper, tuple(), new_p, lower, tuple(), (False, False))

        for sb, new_edge in zip(split_boundaries, new_edges):
            #print(self.boundaries(p)[sb].normal_axis)
            #bd_axis = axis - 1*(axis > self.boundaries(p)[sb].normal_axis)
            #print(self.edges[new_edge])
            self.split_patch_boundary(p, sb, split_xi, axis, self.edges[new_edge], new_p)
            
            # split the boundaries of the new patches at this edge
            new_bd = self.boundaries(p)[sb]
            boundaries[sb] = new_bd.lower
            new_boundaries[sb] = new_bd.upper

            # change patch index for all interfaces from the split part of the boundary
            self._reindex_interfaces(p, sb, [(1,) + s for s in new_boundaries[sb].return_segments()], ofs=tuple(), r = 1, new_p=new_p)
            self._reindex_interfaces(p, sb, [(0,) + s for s in boundaries[sb].return_segments()], ofs=tuple(), r = 1)
            
        # change patch index for all corner nodes and T nodes on the upper edge of old patch   
        
        #also change patch index of possible T_nodes at the new boundaries in the different axis direction (left and right)
            
        self.patches[p] = ((kvs1, geo1), tuple(boundaries))
        self.patches.append(((kvs2, geo2), tuple(new_boundaries)))
        
        return (p, new_p), new_kvs     # return the two indices of the split patches and the joined knot mesh over the 2 patches
    
    def split_boundary_idx(self, p, n, axis=None):
        if axis==None:
            axis=(0,1,2)
        axis=np.unique(axis)
        if len(axis)==1: axis=axis[0]
        for s in self.outer_boundaries.keys():
            
            b_idx_p = [(patch , bd) for (patch, bd) in self.outer_boundaries[s] if patch == p]
            
            if b_idx_p:
                if not np.isscalar(axis):
                    for k,ax in enumerate(axis[::-1]):
                        self.split_boundary_idx(p, n+2**k-1, axis=ax)
                        for i in range(2**k-1):
                            self.split_boundary_idx(n+i, n+2**k+i, axis=ax)
                else:
                    for (patch, bd) in b_idx_p:
                        if axis == 0:
                            if bd == 'back':
                                self.outer_boundaries[s].remove((patch, bd))
                                self.outer_boundaries[s].append((n, bd))
                            if bd == 'left' or bd == 'right' or bd == 'bottom' or bd=='top':
                                 self.outer_boundaries[s].append((n, bd))
                        if axis == 1:
                            if bd == 'top':
                                self.outer_boundaries[s].remove((patch, bd))
                                self.outer_boundaries[s].append((n, bd))
                            if bd == 'left' or bd == 'right':
                                self.outer_boundaries[s].append((n, bd))
                            if self.dim == 3:
                                if bd == 'front' or bd == 'back':
                                    self.outer_boundaries[s].append((n, bd)) 
                        if axis == 2:
                            if bd == 'right':
                                self.outer_boundaries[s].remove((patch, bd))
                                self.outer_boundaries[s].append((n, bd))   
                            if bd == 'bottom' or bd == 'top':
                                self.outer_boundaries[s].append((n, bd))
                            if self.dim == 3:
                                if bd == 'front' or bd == 'back':
                                    self.outer_boundaries[s].append((n, bd)) 
            
    def h_refine(self, patches=None, mult=1):
        if isinstance(patches, dict):
            if len(patches)>0:
                assert max(patches.keys())<self.numpatches and min(patches.keys())>=0, "patch index out of bounds."
        elif isinstance(patches,int):
            assert patches >=-1 and patches < 2, "dimension error."
            patches = {p:patches for p in range(self.numpatches)}
        elif isinstance(patches, (list, set, np.ndarray)):
            if len(patches)>0:
                assert max(patches)<self.numpatches and min(patches)>=0, "patch index out of bounds."
            patches = {p:None for p in patches}
        elif patches==None:
            patches = {p:None for p in range(self.numpatches)}
        else:
            assert 0, "unknown input type"

        new_patches = dict()
        new_kvs = dict()
        
        for p in patches.keys():
            #self.split_boundary_idx(p, self.numpatches, axis=patches[p])
            (kvs,geo), b = self.patches[p]
            if patches[p]==-1:
                new_kvs = tuple([kv.h_refine(mult=mult) for kv in kvs])
                self.patches[p]=((new_kvs, geo), b)
                new_p=(p,)
            elif patches[p]=='q':
                new_kvs = tuple([kv.b_refine(q=0.75) for kv in kvs])
                self.patches[p]=((new_kvs, geo), b)
                new_p=(p,)
            else:    
                t=time.time()
                new_p = self.split_patch(p, axis=patches[p], mult=mult)
                #print(time.time()-t)
            new_patches[p] = new_p
            #new_kvs[p] = new_kvs_  
        return new_patches

    def boundaries(self, p):
        """Get the boundaries for patch `p`.

        A 2D patch has four boundaries, and each one is a list of vertex
        indices describing individual segments of the boundary.
        
        A 3D patch has six boundaries, and each one is a list of 4 tuples 
        of vertex indices each describing the edges enclosing the boundary. 
        """
        return self.patches[p][1]

    def get_matching_interface(self, p, boundary, segment):
        """Get the boundary/interface which is connected to the given boundary/interface."""
        assert 0 <= p < len(self.patches)
        bdrs = self.boundaries(p)
        assert 0 <= boundary < len(bdrs)
        #assert 0 <= segment < len(bdrs[boundary]) - 1
        matching = self.interfaces.get((p, boundary, segment))
        if matching:
            return matching[0]
        else:
            return None     # no matching segment - must be on the boundary
        
    def draw(self, knots = True, vertex_idx = False, edge_idx=False, patch_idx = False, nodes=False, figsize=(8,8)):
        """draws a visualization of the patchmesh in 2D."""
        fig=plt.figure(figsize=figsize)
        ax = plt.axes(projection='3d')
        ax.grid(False)
        
        for p,((kvs, geo),_) in enumerate(self.patches):
            if knots:
                vis.plot_geo(geo, gridx=kvs[0].mesh, gridy=kvs[1].mesh, gridz=kvs[2].mesh, lcolor='lightgray')   
            vis.plot_geo(geo, grid=2,lcolor='black')
           
        if nodes:
            ax.scatter(*np.transpose([vtx[[0,1,2]] for vtx in self.vertices]), color='red')
            #ax.scatter(*np.transpose([vtx[[0,1,2]] for vtx in self.vertices]), color='red')

        if patch_idx:
            for p in range(len(self.patches)):        # annotate patch indices
                geo = self.patches[p][0][1]
                center_xi = np.flipud(np.mean(geo.support, axis=1))
                center = geo(*center_xi)
                ax.text(*(center[[0,1,2]]), str(p), fontsize=18, color='green')
            
        if vertex_idx:
            for i, vtx in enumerate(self.vertices):   # annotate vertex indices
                ax.text(*(vtx[[0,1,2]]), str(i), fontsize=18, color='red')
                
        if edge_idx:    ### still need to compute real center of the geometry edge
            for i, (vtx1, vtx2) in enumerate(self.edges):    #annotate edge indices
                ax.text(*((self.vertices[vtx1] + self.vertices[vtx2])/2)[[0,1,2]], str(i), fontsize=18, color='royalblue')
                
        #ax.invert_xaxis()
        #ax.invert_yaxis()  
        #ax.invert_zaxis()  
        
        #ax.view_init(azim=-30, elev=-25, roll=-180, vertical_axis='y')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        #ax.set_aspect('equal')
        plt.show()
            
    def sanity_check(self):
        for (p, b, s), ((p1, b1, s1), flip) in self.interfaces.items():
            # check that all interface indices make sense
            assert 0 <= p < len(self.patches) and 0 <= p1 < len(self.patches)
            bd, bd1 = self.boundaries(p), self.boundaries(p1)
            assert 0 <= b < len(bd) and 0 <= b1 < len(bd1)
            #assert all([0 <= s_ < len(bd[b]) - 1 and 0 <= s1_ < len(bd1[b1]) - 1 for s_,s1_ in zip(s,s1)]

        outer_boundaries = set()   
                                        
        for p in range(len(self.patches)):
            # check topology of corner vertices
            kvs, geo = self.patches[p][0]
            crns = corners(geo)
            

            # check that there are no duplicate vertices in any segment
            #for bd in bdrs:
                #assert len(np.unique(bd)) == len(bd)

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
                        
class BSegments:
    def __init__(self, bds, normal_axis=None):
        self.lower = None
        self.upper  = None
        self.axis  = None
        self.boundaries = bds
        self.normal_axis = normal_axis
        
    def split_segment(self, ax, seg=tuple(), edge = (0,0)):
        S = self
        for i in seg:
            if i==0:
                if S.lower:
                    S=S.lower
                else:
                    assert 0, "There is no segment here."
            elif i==1:
                if S.upper:
                    S=S.upper
                else:
                    assert 0, "There is no segment here."
            else:
                assert 0, "wrong type of input for segment."
        
        S.axis = ax
        P0,P1,P2,P3 = S.boundaries[0][0],S.boundaries[0][1],S.boundaries[1][0],S.boundaries[1][1]
        S.boundaries = None
        
        assert self.normal_axis!=ax, "cannot split in this axis since boundary segment is orthogonal."
        axes = np.setdiff1d(np.arange(3), self.normal_axis)
        if ax==min(axes):
            bds_lower = [(P0,P1),edge,(P0,edge[0]),(P1,edge[1])]
            bds_upper = [edge,(P2,P3),(edge[0],P2),(edge[1],P3)]
        if ax==max(axes):
            bds_lower = [(P0,edge[0]),(P2, edge[1]),(P0,P2),edge]
            bds_upper = [(edge[0],P1),(edge[1],P3),edge,(P1,P3)]

        S.lower = BSegments(bds_lower, self.normal_axis)
        S.upper = BSegments(bds_upper, self.normal_axis)
            
    def return_segments(self):
        if not self.lower and not self.upper:
            return [tuple()]
        elif not self.lower and self.upper:
            return [(1,) + seg for seg in self.upper.return_segments()]
        elif self.lower and not self.upper:
            return [(0,) + seg for seg in self.lower.return_segments()]
        else:
            return [(0,) + seg for seg in self.lower.return_segments()] + [(1,) + seg for seg in self.upper.return_segments()]
        
    def is_leaf(self):
        return (self.upper == None and self.lower==None)
    
    def print(self):
        if not self.lower and not self.upper:
            return str(self.boundaries)
        else:
            return "(" + self.lower.print() + "," + self.upper.print() + ")"
