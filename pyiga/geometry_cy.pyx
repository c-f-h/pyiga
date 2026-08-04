# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

cimport cython

import numpy as np
cimport numpy as np
import networkx as nx
import itertools

#from . import geometry, bspline

# @cython.cdivision(False)
# @cython.boundscheck(False)
# @cython.wraparound(False)
# cpdef tuple pyx_detect_interfaces(geos):
#     """Automatically detect matching interfaces between patches.

#     Args:
#         patches: a list of patches in the form `(kvs, geo)`

#     Returns:
#         A pair `(connected, interfaces)`, where `connected` is a `bool`
#         describing whether the detected patch graph is connected, and
#         `interfaces` is a list of the detected interfaces, each entry of
#         which is suitable for passing to :meth:`join_boundaries`.
#     """
#     cdef int d = geos[0].dim
#     cdef int i,j,p1, p2
#     cdef list interfaces = []
#     cdef list matches
#     cdef int n = len(geos)
#     cdef double mindist, maxdiam
#     cdef double[:,:,:] bbs = np.empty((d,2,n), dtype=float)
#     cdef double[:] diams = np.zeros(n, dtype=float)
#     for i in range(n):
#         bb = geos[i].bounding_box()
#         bbs[0,0,i] = bb[0][0]
#         bbs[0,1,i] = bb[0][1]
#         bbs[1,0,i] = bb[1][0]
#         bbs[1,1,i] = bb[1][1]
#         for j in range(d):
#             diams[i] = diams[i] + (bbs[j,1,i]-bbs[j,0,i])**2
#         diams[i] = np.sqrt(diams[i])

#     # set up a graph of patch connectivity for later checking
#     patch_graph = nx.Graph()
#     patch_graph.add_nodes_from(range(n))

#     for p1 in range(n):
#         for p2 in range(p1 + 1, n):
#             mindist = pyx_min_distance_rectangle(bbs[:,:,p1],bbs[:,:,p2])
#             maxdiam = max(diams[p1], diams[p2])
#             if mindist < 1e-10 * maxdiam:    # do the bounding boxes touch?
#                 matches = _pyx_find_matching_boundaries(geos[p1], geos[p2])
#                 if matches:
#                     for (bd1, bd2, conn_info) in matches:
#                         interfaces.append((p1, bd1, p2, bd2, conn_info))
#                     patch_graph.add_edge(p1, p2)

#     return nx.is_connected(patch_graph), interfaces
#     #return (0,0)

# @cython.cdivision(False)
# @cython.boundscheck(False)
# @cython.wraparound(False)
# cdef list _pyx_find_matching_boundaries(G1, G2):
#     # find all interfaces which match between G1 and G2
#     #all_bds = list(itertools.product(range(G1.sdim), (0,1)))
#     all_bds = list(itertools.product(range(G1.sdim), (0,1)))
#     cdef list matches = []
#     cdef int i, j, k, m
#     cdef int d = G1.sdim
#     cdef tuple conn_info

#     for j in range(d):
#         for i in range(2):
#             bd1 = G1.boundary(((j,i),))
#             for k in range(d):
#                 for m in range(2):
#                     bd2 = G2.boundary(((k,m),))
#                     match, conn_info = _pyx_check_geo_match(bd1, bd2)
#                     if match:
#                         matches.append((((j,i),), ((k,m),), conn_info))
#     # for bdspec1 in all_bds:
#     #     bd1 = G1.boundary((bdspec1,))
#     #     for bdspec2 in all_bds:
#     #         bd2 = G2.boundary((bdspec2,))
#     #         match, conn_info = _pyx_check_geo_match(bd1, bd2)
#     #         if match:
#     #             matches.append(((bdspec1,), (bdspec2,), conn_info))
#     return matches

# @cython.cdivision(False)
# @cython.boundscheck(False)
# @cython.wraparound(False)
# cdef tuple _pyx_check_geo_match(G1, G2, int gridn=4):
#     # check if the two geos match with any possible flip
#     # if not np.allclose(G1.support, G2.support):
#     #     return False, (None, None)
#     cdef list grid = [np.linspace(s[0], s[1], gridn) for s in G1.support]
#     cdef int sdim = G1.sdim
#     cdef np.ndarray[np.float32_t, dim=sdim+1] X1 = G1.grid_eval(grid)
#     cdef np.ndarray[np.float32_t, dim=sdim+1] X2

#     for k, perm in enumerate(itertools.permutations(np.arange(G1.sdim))):
#         all_flips = itertools.product(*(G2.sdim * [(False, True)]))
#         for flip in all_flips:  # try all 2^d possible flips
#             flipped_grid = list(grid)
#             for (i, f) in enumerate(flip):
#                 if f: flipped_grid[i] = np.ascontiguousarray(np.flip(flipped_grid[i]))
#             X2 = G2.grid_eval(flipped_grid).transpose(perm + (G2.sdim,))
#             if np.allclose(X1, X2):
#                 #if G1.sdim == 1 or k==0:
#                     #return True, (None, flip)
#                 return True, (perm, flip)
#     return False, (None, None)

# @cython.cdivision(False)
# @cython.boundscheck(False)
# @cython.wraparound(False)
# cdef double pyx_min_distance_rectangle(double[:,:] bb1, double[:,:] bb2):
#     cdef int j
#     cdef double m
#     cdef int d = bb1.shape[0]
#     cdef double out = 0.0

#     for j in range(d):
#         out = out + max(max(bb2[j,0]-bb1[j,1],bb1[j,0]-bb2[j,1]),0.)**2
#     return np.sqrt(out)

# @cython.cdivision(False)
# @cython.boundscheck(False)
# @cython.wraparound(False)
# cdef double max(double a, double b):
#     if a > b: 
#         return a
#     else: 
#         return b
    
# @cython.cdivision(False)
# @cython.boundscheck(False)
# @cython.wraparound(False)
# cdef double min(double a, double b):
#     if a < b: 
#         return a
#     else: 
#         return b
