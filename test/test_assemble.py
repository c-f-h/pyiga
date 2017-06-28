from pyiga.assemble import *
from pyiga import geometry
import os.path
from pyiga.utils import read_sparse_matrix

def test_mass():
    kv = bspline.KnotVector(np.array([ 0., 0., 0., 0., 0., 0.25, 0.35, 0.45, 0.55, 0.65, 0.9, 0.9, 0.9, 0.9, 0.9 ]), 4)
    test_M = np.array(
        [[  2.77777778e-02,   1.57607941e-02,   5.40162735e-03,   9.88223305e-04, 7.15774525e-05,   0.            ,   0.            ,   0.            , 0.            ,   0.            ],
         [  1.57607941e-02,   2.57855523e-02,   1.98353624e-02,   7.52169657e-03, 1.09644036e-03,   1.54256714e-07,   0.            ,   0.            , 0.            ,   0.            ],
         [  5.40162735e-03,   1.98353624e-02,   3.18639867e-02,   2.49975292e-02, 7.81923076e-03,   8.21448165e-05,   1.18765551e-07,   0.            , 0.            ,   0.            ],
         [  9.88223305e-04,   7.52169657e-03,   2.49975292e-02,   4.21935343e-02, 3.12182455e-02,   3.00733082e-03,   7.33215324e-05,   1.18765551e-07, 0.            ,   0.            ],
         [  7.15774525e-05,   1.09644036e-03,   7.81923076e-03,   3.12182455e-02, 6.01755079e-02,   2.65293681e-02,   3.00733082e-03,   8.21448165e-05, 1.54256714e-07,   0.            ],
         [  0.            ,   1.54256714e-07,   8.21448165e-05,   3.00733082e-03, 2.65293681e-02,   6.01755079e-02,   3.12182455e-02,   7.81923076e-03, 1.09644036e-03,   7.15774525e-05],
         [  0.            ,   0.            ,   1.18765551e-07,   7.33215324e-05, 3.00733082e-03,   3.12182455e-02,   4.21935343e-02,   2.49975292e-02, 7.52169657e-03,   9.88223305e-04],
         [  0.            ,   0.            ,   0.            ,   1.18765551e-07, 8.21448165e-05,   7.81923076e-03,   2.49975292e-02,   3.18639867e-02, 1.98353624e-02,   5.40162735e-03],
         [  0.            ,   0.            ,   0.            ,   0.            , 1.54256714e-07,   1.09644036e-03,   7.52169657e-03,   1.98353624e-02, 2.57855523e-02,   1.57607941e-02],
         [  0.            ,   0.            ,   0.            ,   0.            , 0.            ,   7.15774525e-05,   9.88223305e-04,   5.40162735e-03, 1.57607941e-02,   2.77777778e-02]])
    M = bsp_mass_1d(kv).A
    assert np.abs(test_M - M).max() < 1e-10

def test_stiffness():
    kv = bspline.KnotVector(np.array([ 0., 0., 0., 0., 0., 0.25, 0.35, 0.45, 0.55, 0.65, 0.9, 0.9, 0.9, 0.9, 0.9 ]), 4)
    test_K = np.array(
          [[ 9.1428571429, -5.4777176177, -2.807060844 , -0.7756214559, -0.0824572253,  0.          ,  0.          ,  0.          ,  0.          ,  0.          ],
           [-5.4777176177,  6.3440233236,  0.8890945645, -1.2538584045, -0.5004312176, -0.0011106483,  0.          ,  0.          ,  0.          ,  0.          ],
           [-2.807060844 ,  0.8890945645,  2.9602480448,  0.4981488415, -1.3996319341, -0.1399435607, -0.000855112 ,  0.          ,  0.          ,  0.          ],
           [-0.7756214559, -1.2538584045,  0.4981488415,  3.1130787412,  0.0527464473, -1.5087149732, -0.1249240845, -0.000855112 ,  0.          ,  0.          ],
           [-0.0824572253, -0.5004312176, -1.3996319341,  0.0527464473,  4.9470212327, -1.3674781207, -1.5087149732, -0.1399435607, -0.0011106483,  0.          ],
           [ 0.          , -0.0011106483, -0.1399435607, -1.5087149732, -1.3674781207,  4.9470212327,  0.0527464473, -1.3996319341, -0.5004312176, -0.0824572253],
           [ 0.          ,  0.          , -0.000855112 , -0.1249240845, -1.5087149732,  0.0527464473,  3.1130787412,  0.4981488415, -1.2538584045, -0.7756214559],
           [ 0.          ,  0.          ,  0.          , -0.000855112 , -0.1399435607, -1.3996319341,  0.4981488415,  2.9602480448,  0.8890945645, -2.807060844 ],
           [ 0.          ,  0.          ,  0.          ,  0.          , -0.0011106483, -0.5004312176, -1.2538584045,  0.8890945645,  6.3440233236, -5.4777176177],
           [ 0.          ,  0.          ,  0.          ,  0.          ,  0.          , -0.0824572253, -0.7756214559, -2.807060844 , -5.4777176177,  9.1428571429]])
    K = bsp_stiffness_1d(kv).A
    assert np.abs(test_K - K).max() < 1e-10

def test_mass_asym():
    kv1 = bspline.make_knots(4, 0.0, 1.0, 10)
    kv2 = bspline.make_knots(1, 0.0, 1.0, 20)
    M_12 = bsp_mass_1d_asym(kv1, kv2, quadgrid=kv2.kv[kv2.p:-kv2.p])
    assert(M_12.shape[0] == kv1.numdofs)
    assert(M_12.shape[1] == kv2.numdofs)

def test_stiffness_asym():
    kv1 = bspline.make_knots(4, 0.0, 1.0, 10)
    kv2 = bspline.make_knots(1, 0.0, 1.0, 20)
    K_12 = bsp_stiffness_1d_asym(kv1, kv2, quadgrid=kv2.kv[kv2.p:-kv2.p])
    assert(K_12.shape[0] == kv1.numdofs)
    assert(K_12.shape[1] == kv2.numdofs)

def test_mixed_deriv_biform():
    kv = bspline.make_knots(4, 0.0, 1.0, 20)
    DxxD0 = bsp_mixed_deriv_biform_1d(kv, 2, 0)
    DxxDx = bsp_mixed_deriv_biform_1d(kv, 2, 1)
    from pyiga.approx import interpolate
    u = interpolate(kv, lambda x: x)
    # second derivative of linear function x -> x is 0
    assert abs(DxxD0.dot(u)).max() < 1e-10
    assert abs(DxxDx.dot(u)).max() < 1e-10

def test_stiffness_2d():
    kvs = (bspline.make_knots(4, 0.0, 1.0, 10),
           bspline.make_knots(3, 0.0, 1.0, 12))
    assert(np.allclose(
        bsp_stiffness_2d(kvs, geo=None).A,
        bsp_stiffness_2d(kvs, geo=geometry.unit_square()).A,
        rtol=0, atol=1e-14)
    )

def test_stiffness_3d():
    kvs = (bspline.make_knots(3, 0.0, 1.0, 4),
           bspline.make_knots(3, 0.0, 1.0, 5),
           bspline.make_knots(3, 0.0, 1.0, 6))
    assert(np.allclose(
        bsp_stiffness_3d(kvs, geo=None).A,
        bsp_stiffness_3d(kvs, geo=geometry.unit_cube()).A,
        rtol=0, atol=1e-14)
    )

################################################################################
# Test full Gauss quadrature assemblers with geometry transforms
################################################################################

def test_mass_geo_2d():
    kv = bspline.make_knots(3, 0.0, 1.0, 15)
    geo = geometry.bspline_quarter_annulus()
    M = mass((kv,kv), geo)
    M_ref = read_sparse_matrix(os.path.join(os.path.dirname(__file__),
        "poisson_neu_d2_p3_n15_mass.mtx.gz"))
    assert abs(M - M_ref).max() < 1e-14

def test_stiffness_geo_2d():
    kv = bspline.make_knots(3, 0.0, 1.0, 15)
    geo = geometry.bspline_quarter_annulus()
    A = stiffness((kv,kv), geo)
    A_ref = read_sparse_matrix(os.path.join(os.path.dirname(__file__),
        "poisson_neu_d2_p3_n15_stiff.mtx.gz"))
    assert abs(A - A_ref).max() < 1e-14

def test_mass_geo_3d():
    kv = bspline.make_knots(2, 0.0, 1.0, 10)
    geo = geometry.twisted_box()
    M = mass((kv,kv,kv), geo)
    M_ref = read_sparse_matrix(os.path.join(os.path.dirname(__file__),
        "poisson_neu_d3_p2_n10_mass.mtx.gz"))
    assert abs(M - M_ref).max() < 1e-14

def test_stiffness_geo_3d():
    kv = bspline.make_knots(2, 0.0, 1.0, 10)
    geo = geometry.twisted_box()
    A = stiffness((kv,kv,kv), geo)
    A_ref = read_sparse_matrix(os.path.join(os.path.dirname(__file__),
        "poisson_neu_d3_p2_n10_stiff.mtx.gz"))
    assert abs(A - A_ref).max() < 1e-14

################################################################################
# Test fast ACA assemblers with geometry transforms
################################################################################

def test_fast_mass_geo_2d():
    kv = bspline.make_knots(3, 0.0, 1.0, 15)
    geo = geometry.bspline_quarter_annulus()
    M = mass_fast((kv,kv), geo, verbose=0)
    M_ref = read_sparse_matrix(os.path.join(os.path.dirname(__file__),
        "poisson_neu_d2_p3_n15_mass.mtx.gz"))
    assert abs(M - M_ref).max() < 1e-9

def test_fast_stiffness_geo_2d():
    kv = bspline.make_knots(3, 0.0, 1.0, 15)
    geo = geometry.bspline_quarter_annulus()
    A = stiffness_fast((kv,kv), geo, verbose=0)
    A_ref = read_sparse_matrix(os.path.join(os.path.dirname(__file__),
        "poisson_neu_d2_p3_n15_stiff.mtx.gz"))
    assert abs(A - A_ref).max() < 1e-9

def test_fast_mass_geo_3d():
    kv = bspline.make_knots(2, 0.0, 1.0, 10)
    geo = geometry.twisted_box()
    M = mass_fast((kv,kv,kv), geo, verbose=0)
    M_ref = read_sparse_matrix(os.path.join(os.path.dirname(__file__),
        "poisson_neu_d3_p2_n10_mass.mtx.gz"))
    assert abs(M - M_ref).max() < 1e-9

def test_fast_stiffness_geo_3d():
    kv = bspline.make_knots(2, 0.0, 1.0, 10)
    geo = geometry.twisted_box()
    A = stiffness_fast((kv,kv,kv), geo, verbose=0)
    A_ref = read_sparse_matrix(os.path.join(os.path.dirname(__file__),
        "poisson_neu_d3_p2_n10_stiff.mtx.gz"))
    assert abs(A - A_ref).max() < 1e-9

################################################################################
# Test right-hand side
################################################################################

def test_inner_products():
    kvs = [bspline.make_knots(p, 0.0, 1.0, 8+p) for p in range(3,6)]
    def f(x,y,z): return np.cos(x) * np.exp(y) * np.sin(z)
    inp = inner_products(kvs, f)
    assert inp.shape == tuple(kv.numdofs for kv in kvs)
    inp2 = inner_products(kvs, f, geo=geometry.unit_cube())
    assert np.allclose(inp, inp2)
