from pyiga.stilde import *

def test_Stilde_basis():
    kv = bspline.make_knots(4, 0.0, 1.0, 10)
    P_tilde, P_compl = Stilde_basis(kv)
    n = kv.numdofs
    assert n == P_tilde.shape[0]
    assert n == P_compl.shape[0]
    assert n == P_tilde.shape[1] + P_compl.shape[1]
    assert P_tilde.shape[1] == 10
    assert abs(P_tilde.T.dot(P_compl)).max() < 1e-14
