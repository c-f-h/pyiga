#
# Compute basis for the subspace S-tilde
# (splines with vanishing odd derivatives at the boundary)
#
# Algorithm is documented in
#   C. Hofreither and S. Takacs:
#   "Robust Multigrid for Isogeometric Analysis Based on Stable Splittings of
#     Spline Spaces"
#   http://www.numa.uni-linz.ac.at/publications/List/2016/2016-02-r1.pdf
#

import scipy.linalg
import numpy as np

from . import bspline

def Stilde_basis_side(kv, side):
    p = kv.p
    u = kv.kv[0] if side==0 else kv.kv[-1]
    # rows correspond to derivatives
    # columns correspond to basis functions
    derivs = bspline.active_deriv(kv, u, p-1) #shape: (p,p+1)

    # remove p+1st basis function since it's always in the nullspace
    if side==0:
        derivs = derivs[:, :p]
    else:
        derivs = derivs[:, 1:]

    # normalize with h^(deriv)
    h = kv.meshsize_avg()
    derivs = (np.repeat(h, p) ** range(p))[:,np.newaxis] * derivs

    n_tilde = (p + 1) // 2
    # use only odd derivatives
    evenderivs = range(0,p,2)
    assert n_tilde == len(evenderivs)
    derivs[evenderivs, :] = 0

    U, S, Vt = scipy.linalg.svd(derivs)
    # return nullspace and its orthogonal complement
    return (Vt.T[:, -n_tilde:], Vt.T[:, :-n_tilde])

def Stilde_basis(kv):
    """Compute a basis for S-tilde and one for its orthogonal complement"""
    p = kv.p
    (b_L, b_compl_L) = Stilde_basis_side(kv, 0)
    (b_R, b_compl_R) = Stilde_basis_side(kv, 1)

    n = kv.numdofs
    n_L = b_L.shape[1]
    n_R = b_R.shape[1]
    n_I = n - 2*p
    n_c_L = b_compl_L.shape[1]
    n_c_R = b_compl_R.shape[1]

    P_tilde = np.zeros((n, n_L + n_I + n_R))
    P_compl = np.zeros((n, n_c_L + n_c_R))

    P_tilde[:p,:n_L] = b_L
    P_tilde[p:-p, n_L:-n_R] = np.eye(n_I)
    P_tilde[-p:,-n_R:] = b_R

    P_compl[:p, :n_c_L] = b_compl_L
    P_compl[-p:, -n_c_R:] = b_compl_R

    return (P_tilde, P_compl)

