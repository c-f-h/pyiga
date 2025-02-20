import unittest
from pyiga.assemble import *
from pyiga import geometry
from pyiga.utils import read_sparse_matrix
from pyiga.approx import interpolate

import os.path
from scipy.sparse import kron as spkron

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
    M = bsp_mass_1d(kv).toarray()
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
    K = bsp_stiffness_1d(kv).toarray()
    assert np.abs(test_K - K).max() < 1e-10

def test_mass_asym():
    kv1 = bspline.make_knots(4, 0.0, 1.0, 10)
    kv2 = bspline.make_knots(1, 0.0, 1.0, 20)
    M_12 = bsp_mass_1d_asym(kv1, kv2, quadgrid=kv2.mesh)
    assert(M_12.shape[0] == kv2.numdofs)
    assert(M_12.shape[1] == kv1.numdofs)
    u = interpolate(kv1, lambda x: x**4)
    itg = M_12.dot(u).dot(np.ones(kv2.numdofs))
    assert abs(itg - 1.0/5) < 1e-10     # int(x^4, 0, 1) = 1/5

def test_stiffness_asym():
    kv1 = bspline.make_knots(4, 0.0, 1.0, 10)
    kv2 = bspline.make_knots(1, 0.0, 1.0, 20)
    K_12 = bsp_stiffness_1d_asym(kv1, kv2, quadgrid=kv2.mesh)
    assert(K_12.shape[0] == kv2.numdofs)
    assert(K_12.shape[1] == kv1.numdofs)
    u = interpolate(kv1, lambda x: x**4)
    v = interpolate(kv2, lambda x: x)
    itg = K_12.dot(u).dot(v)
    assert abs(itg - 1.0) < 1e-10     # int(4*x^3, 0, 1) = 1

def test_assemble_asym():
    kv1 = bspline.make_knots(4, 0.0, 1.0, 10)
    kv2 = bspline.make_knots(1, 0.0, 1.0, 20)
    K_12 = bsp_mixed_deriv_biform_1d_asym(kv1, kv2, 1, 0, quadgrid=kv2.mesh)
    assert(K_12.shape[0] == kv2.numdofs)
    assert(K_12.shape[1] == kv1.numdofs)
    u = interpolate(kv1, lambda x: x**4)
    v = interpolate(kv2, lambda x: 1.0)
    itg = K_12.dot(u).dot(v)
    assert abs(itg - 1.0) < 1e-10     # int(4*x^3, 0, 1) = 1

def test_mixed_deriv_biform():
    kv = bspline.make_knots(4, 0.0, 1.0, 20)
    DxxD0 = bsp_mixed_deriv_biform_1d(kv, 2, 0)
    DxxDx = bsp_mixed_deriv_biform_1d(kv, 2, 1)
    u = interpolate(kv, lambda x: x)
    # second derivative of linear function x -> x is 0
    assert abs(DxxD0.dot(u)).max() < 1e-10
    assert abs(DxxDx.dot(u)).max() < 1e-10

def test_stiffness_2d():
    kvs = (bspline.make_knots(4, 0.0, 1.0, 10),
           bspline.make_knots(3, 0.0, 1.0, 12))
    assert(np.allclose(
        bsp_stiffness_2d(kvs, geo=None).toarray(),
        bsp_stiffness_2d(kvs, geo=geometry.unit_square()).toarray(),
        rtol=0, atol=1e-14)
    )

def test_stiffness_3d():
    kvs = (bspline.make_knots(3, 0.0, 1.0, 4),
           bspline.make_knots(3, 0.0, 1.0, 5),
           bspline.make_knots(3, 0.0, 1.0, 6))
    assert(np.allclose(
        bsp_stiffness_3d(kvs, geo=None).toarray(),
        bsp_stiffness_3d(kvs, geo=geometry.unit_cube()).toarray(),
        rtol=0, atol=1e-14)
    )

def test_heat_st_2d():
    T_end = 2.0
    geo = geometry.unit_cube(dim=1).cylinderize(0.0, T_end, support=(0.0,T_end))
    kv_t = bspline.make_knots(2, 0.0, T_end, 6)    # time axis
    kv   = bspline.make_knots(3, 0.0,   1.0, 8)    # space axis
    kvs = (kv_t, kv)

    M = mass(kv)
    M_t = mass(kv_t)
    DxDx = stiffness(kv)
    DtD0 = bsp_mixed_deriv_biform_1d(kv_t, 1, 0)
    A_ref = (spkron(DtD0, M) + spkron(M_t, DxDx)).tocsr()

    A = assemble_entries(assemblers.HeatAssembler_ST2D(kvs, geo))
    assert abs(A_ref - A).max() < 1e-14

def test_wave_st_2d():
    T_end = 2.0
    geo = geometry.unit_cube(dim=1).cylinderize(0.0, T_end, support=(0.0,T_end))
    kv_t = bspline.make_knots(2, 0.0, T_end, 6)    # time axis
    kv   = bspline.make_knots(3, 0.0,   1.0, 8)    # space axis
    kvs = (kv_t, kv)

    M = mass(kv)
    DxDx = stiffness(kv)
    D0Dt = bsp_mixed_deriv_biform_1d(kv_t, 0, 1)
    DttDt = bsp_mixed_deriv_biform_1d(kv_t, 2, 1)
    A_ref = (spkron(DttDt, M) + spkron(D0Dt, DxDx)).tocsr()

    A = assemble_entries(assemblers.WaveAssembler_ST2D(kvs, geo))
    assert abs(A_ref - A).max() < 1e-12

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

def test_divdiv_geo_2d():
    kv = bspline.make_knots(3, 0.0, 1.0, 15)
    geo = geometry.bspline_quarter_annulus()
    A = divdiv((kv,kv), geo, layout='packed', format='bsr')
    # construct divergence-free function
    u = interpolate((kv,kv), lambda x,y: (x,-y), geo=geo)
    assert abs(A.dot(u.ravel())).max() < 1e-12

    # test blocked layout
    A = divdiv((kv,kv), geo, layout='blocked')
    u_blocked = np.moveaxis(u, -1, 0)   # move last axis to the front
    assert abs(A.dot(u_blocked.ravel())).max() < 1e-12

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

    # compare to autogenerated assemblers when using geometry
    geo = geometry.twisted_box()
    import pyiga.assemblers

    # physical=False
    asm = pyiga.assemblers.L2FunctionalAssembler3D(kvs, geo, f=f)
    inp = asm.assemble_vector()
    inp2 = inner_products(kvs, f, geo=geo)
    assert np.allclose(inp, inp2)

    # physical=True
    asm = pyiga.assemblers.L2FunctionalAssemblerPhys3D(kvs, geo, f=f)
    inp = asm.assemble_vector()
    inp2 = inner_products(kvs, f, f_physical=True, geo=geo)
    assert np.allclose(inp, inp2)

################################################################################
# Test boundary condition functions
################################################################################

def test_boundary_dofs():
    kvs = (bspline.make_knots(3, 0.0, 1.0, 5),
           bspline.make_knots(2, 0.0, 1.0, 8))
    assert np.array_equal(
            boundary_dofs(kvs, 'bottom', ravel=True),
            np.arange(10))
    assert np.array_equal(
            boundary_dofs(kvs, 'right'),
            [ (i,9) for i in range(8) ])
    assert np.array_equal(
            boundary_dofs(kvs, 'left', ravel=True),
            np.arange(0, 80, 10))

def test_dirichlet_bc():
    kvs = (bspline.make_knots(3, 0.0, 1.0, 5), bspline.make_knots(2, 0.0, 1.0, 3))
    # dof indices are (0, ..., 7) x (0, ..., 4)
    geo = geometry.identity(kvs)
    def f(x,y): return 1.0
    ny, nx = (kv.numdofs for kv in kvs)
    indices, values = compute_dirichlet_bc(kvs, geo, (0,0), f)
    assert np.array_equal(indices, range(nx))
    assert np.allclose(values, np.ones(nx))
    indices, values = compute_dirichlet_bc(kvs, geo, (0,1), f)
    assert np.array_equal(indices, range((ny-1)*nx, ny*nx))
    assert np.allclose(values, np.ones(nx))
    indices, values = compute_dirichlet_bc(kvs, geo, (1,0), f)
    assert np.array_equal(indices, range(0, ny*nx, nx))
    assert np.allclose(values, np.ones(ny))
    indices, values = compute_dirichlet_bc(kvs, geo, (1,1), f)
    assert np.array_equal(indices, range(nx-1, nx-1+ny*nx, nx))
    assert np.allclose(values, np.ones(ny))

################################################################################
# Test assembling from VForms
################################################################################

def test_assemble_vf():
    # set up vform
    from pyiga.vform import VForm, grad, inner, dx
    vf = VForm(2)
    u, v = vf.basisfuns()
    vf.add(inner(grad(u), grad(v)) * dx)

    # set up space and geo
    kvs = 2 * (bspline.make_knots(3, 0.0, 1.0, 10),)
    geo = geometry.quarter_annulus()

    # assemble
    A = assemble_vf(vf, kvs, geo=geo)
    A_ref = stiffness(kvs, geo)
    assert np.allclose(A.toarray(), A_ref.toarray())

    # right-hand side vform
    vf_f = VForm(2, arity=1)
    f = vf_f.input('f')
    v = vf_f.basisfuns()
    vf_f.add(f * v * dx)

    def f(x, y): return np.exp(x + y)
    f1 = assemble_vf(vf_f, kvs, geo=geo, f=f)
    f2 = inner_products(kvs, f, geo=geo)
    assert np.allclose(f1, f2)

def test_assemble_surface_vf():
    # set up vform
    from pyiga.vform import VForm, ds
    vf = VForm(2, geo_dim=3, arity=1)
    v = vf.basisfuns()
    vf.add(v * ds)      # compute surface area by summing integrals of basis functions

    # set up space and geo
    kvs = 2 * (bspline.make_knots(3, 0.0, 1.0, 10),)

    # cylinder built on a quarter annulus
    geo_3d = geometry.tensor_product(
            geometry.line_segment(0.0, 1.0),
            geometry.quarter_annulus())

    # assemble
    f = assemble_vf(vf, kvs, geo=geo_3d.boundary('left')) # inner mantle
    assert np.allclose(f.sum(), (2 * 1 * np.pi) / 4)    # r=1 at inner mantle, one quarter of the full circle
    f = assemble_vf(vf, kvs, geo=geo_3d.boundary('right')) # outer mantle
    assert np.allclose(f.sum(), (2 * 2 * np.pi) / 4)    # r=2 at outer mantle, one quarter of the full circle

def test_assemble_boundary_vector():
    # set up space and geo
    kvs = 3 * (bspline.make_knots(3, 0.0, 1.0, 3),)

    # cylinder built on a quarter annulus
    geo_3d = geometry.tensor_product(
            geometry.line_segment(0.0, 1.0),
            geometry.quarter_annulus())
    f = assemble('v * ds', kvs, geo=geo_3d, boundary='left')
    assert f.shape == (6, 6, 1)
    # r=1 at inner mantle, one quarter of the full circle
    assert np.allclose(f.sum(), (2 * 1 * np.pi) / 4)
    # r=2 at outer mantle, one quarter of the full circle
    assert np.allclose(assemble('v * ds', kvs, geo=geo_3d, boundary='right').sum(), (2 * 2 * np.pi) / 4)
    # bottom and top: squares on x/z and y/z planes
    assert np.allclose(assemble('v * ds', kvs, geo=geo_3d, boundary='bottom').sum(), 1.0)
    assert np.allclose(assemble('v * ds', kvs, geo=geo_3d, boundary='top').sum(),    1.0)
    # front and back: quarter annulus
    assert np.allclose(assemble('v * ds', kvs, geo=geo_3d, boundary='front').sum(), (2**2-1**2) * np.pi/4)
    assert np.allclose(assemble('v * ds', kvs, geo=geo_3d, boundary='back').sum(),  (2**2-1**2) * np.pi/4)
    #
    # compute average normal vectors over different faces
    nv = assemble('inner(v, n) * ds', kvs, bfuns=[('v', 3)], geo=geo_3d, boundary='left', layout='packed')
    assert np.allclose(nv.sum(axis=(0,1,2)), [-1, -1, 0])
    nv = assemble('inner(v, n) * ds', kvs, bfuns=[('v', 3)], geo=geo_3d, boundary='right', layout='packed')
    assert np.allclose(nv.sum(axis=(0,1,2)), [2, 2, 0])
    nv = assemble('inner(v, n) * ds', kvs, bfuns=[('v', 3)], geo=geo_3d, boundary='bottom', layout='packed')
    assert np.allclose(nv.sum(axis=(0,1,2)), [0, -1, 0])
    nv = assemble('inner(v, n) * ds', kvs, bfuns=[('v', 3)], geo=geo_3d, boundary='top', layout='packed')
    assert np.allclose(nv.sum(axis=(0,1,2)), [-1, 0, 0])
    nv = assemble('inner(v, n) * ds', kvs, bfuns=[('v', 3)], geo=geo_3d, boundary='front', layout='packed')
    assert np.allclose(nv.sum(axis=(0,1,2)), (2**2-1**2) * np.pi/4 * np.array([0, 0, -1]))
    nv = assemble('inner(v, n) * ds', kvs, bfuns=[('v', 3)], geo=geo_3d, boundary='back', layout='packed')
    assert np.allclose(nv.sum(axis=(0,1,2)), (2**2-1**2) * np.pi/4 * np.array([0, 0, 1]))
    ####
    # check normal vectors in 2D
    kvs = 2 * (bspline.make_knots(3, 0.0, 1.0, 3),)
    geo = geometry.unit_square()
    nv = assemble('inner(v, n) * ds', kvs, bfuns=[('v', 2)], geo=geo, boundary='left', layout='packed')
    assert np.allclose(nv.sum(axis=(0,1)), [-1, 0])
    nv = assemble('inner(v, n) * ds', kvs, bfuns=[('v', 2)], geo=geo, boundary='right', layout='packed')
    assert np.allclose(nv.sum(axis=(0,1)), [+1, 0])
    nv = assemble('inner(v, n) * ds', kvs, bfuns=[('v', 2)], geo=geo, boundary='bottom', layout='packed')
    assert np.allclose(nv.sum(axis=(0,1)), [0, -1])
    nv = assemble('inner(v, n) * ds', kvs, bfuns=[('v', 2)], geo=geo, boundary='top', layout='packed')
    assert np.allclose(nv.sum(axis=(0,1)), [0, +1])

def test_assemble_boundary_matrix():
    # set up space and geo
    kvs = (bspline.make_knots(3, 0.0, 1.0, 3),
           bspline.make_knots(3, 0.0, 1.0, 4),
           bspline.make_knots(3, 0.0, 1.0, 5))

    # cylinder built on a quarter annulus
    geo_3d = geometry.tensor_product(
            geometry.line_segment(0.0, 1.0),
            geometry.quarter_annulus())
    A = assemble('inner(grad(u), grad(v)) * ds', kvs, geo=geo_3d, boundary='left')
    assert A.shape == (6 * 7, 6 * 7)
    A = assemble('inner(grad(u), grad(v)) * ds', kvs, geo=geo_3d, boundary='top')
    assert A.shape == (6 * 8, 6 * 8)
    # assemble only the tangential components (equivalent to 2D Laplacian since 'front' is plane)
    A = assemble('inner(cross(n, grad(u)), cross(n, grad(v))) * ds', kvs, geo=geo_3d, boundary='front')
    assert A.shape == (7 * 8, 7 * 8)
    A2 = stiffness(kvs[1:], geo=geometry.quarter_annulus())
    assert np.allclose(A.toarray(), A2.toarray())

def test_assemble_vf_with_params():
    geo = geometry.quarter_annulus()
    kvs = 2 * (bspline.make_knots(3, 0.0, 1.0, 10),)
    f = assemble('a * inner(grad(u), b) * dx', kvs, geo=geo, a=1.8, b=(-1.5, 0.7))
    f2 = assemble('1.8 * inner(grad(u), (-1.5, 0.7)) * dx', kvs, geo=geo)
    assert np.allclose(f, f2)

def test_assemble_string():
    kvs = 2 * (bspline.make_knots(2, 0.0, 1.0, 10),)
    geo = geometry.quarter_annulus()
    A1 = assemble('inner(grad(u), grad(v)) * dx', kvs, geo=geo)
    A2 = stiffness(kvs, geo)
    assert np.allclose(A1.toarray(), A2.toarray())

    # use Assembler class
    asm = Assembler('inner(grad(u), grad(v)) * dx', kvs, geo=geo, symmetric=True, updatable=['geo'])
    A3 = asm.assemble()
    assert np.allclose(A2.toarray(), A3.toarray())

    with unittest.TestCase().assertRaises(RuntimeError):
        asm.assemble(f=geo)       # not an updatable field
    with unittest.TestCase().assertRaises(ValueError):
        asm = Assembler('inner(grad(u), grad(v)) * dx', kvs, geo=geo, updatable=['f'])  # f is not an input

    f = lambda x, y: x * y**2
    f1 = assemble('f * v * dx', kvs, geo=geo, f=f)
    f2 = inner_products(kvs, f, geo=geo, f_physical=True)
    assert np.allclose(f1, f2)

    # vector-valued problems
    f1 = assemble('f * div(v) * dx', kvs, bfuns=[('v',2)], geo=geo, f=f, layout='packed')
    f2 = assemble('f * div(v) * dx', kvs, bfuns=[('v',2)], geo=geo, f=f, layout='blocked')
    assert np.allclose(f1.transpose(2, 0, 1), f2)

    # 1D problem - matrix
    geo = geometry.unit_cube(dim=1)
    A1 = assemble('inner(grad(u), grad(v)) * dx', kvs[:1], geo=geo)
    A2 = stiffness(kvs[0])
    assert np.allclose(A1.toarray(), A2.toarray())
    # 1D problem - rhs
    f = lambda x: 1 + x**2
    f1 = assemble('f * v * dx', kvs[:1], geo=geo, f=f)
    f2 = inner_products(kvs[0], f=f, f_physical=True, geo=geo)
    assert np.allclose(f1, f2)

    # test error handling
    with unittest.TestCase().assertRaises(ValueError) as cm:
        assemble('inner(grad(u), grad(v)) * dx', kvs)
    assert str(cm.exception) == "required input parameter 'geo' missing"

def test_assemble_nonsym_vec():
    kvs = 2 * (bspline.make_knots(2, 0.0, 1.0, 5),)
    geo = geometry.quarter_annulus()

    problem = 'inner(as_matrix([[2,1],[0,0]]).dot(u), v) * dx'
    A = assemble(problem, kvs, geo=geo, bfuns=[('u',2), ('v',2)], layout='packed', format='bsr')
    u = interpolate(kvs, lambda x, y: (x*y, -2*x*y), geo=geo)
    assert np.allclose(A @ u.ravel(), 0)

    # test the blockwise assembling using multi_blocks
    asm = instantiate_assembler(problem, kvs, args={'geo': geo}, bfuns=[('u',2), ('v',2)])
    blocks = np.array(asm.multi_blocks([(0,0), (0,1), (2,1)]))
    AA = A.toarray()        # bsr does not support slicing
    assert np.array_equal(blocks[0], AA[0:2, 0:2])
    assert np.array_equal(blocks[1], AA[0:2, 2:4])
    assert np.array_equal(blocks[2], AA[4:6, 2:4])

    # test blocked layout
    A = assemble(problem, kvs, geo=geo, bfuns=[('u',2), ('v',2)], layout='blocked')
    u_blocked = np.moveaxis(u, -1, 0)   # move last axis to the front
    assert np.allclose(A @ u_blocked.ravel(), 0)

################################################################################
# Test integrals
################################################################################

def test_integrate():
    def one(*X): return 1.0

    geo = geometry.unit_square()
    assert abs(1.0 - integrate(geo.kvs, one)) < 1e-8
    assert abs(1.0 - integrate(geo.kvs, one, f_physical=True, geo=geo)) < 1e-8

    geo = geometry.unit_cube()
    assert abs(1.0 - integrate(geo.kvs, one)) < 1e-8
    assert abs(1.0 - integrate(geo.kvs, one, f_physical=True, geo=geo)) < 1e-8

    kvs = 2 * (bspline.make_knots(3, 0.0, 1.0, 10),)
    geo = geometry.quarter_annulus()
    assert abs(3*np.pi/4 - integrate(kvs, one, geo=geo)) < 1e-8

################################################################################
# Test solution
################################################################################

def test_solution_1d():
    # solve -u''(x) = 1, u(0) = 0, u(1) = 1
    kv = bspline.make_knots(2, 0.0, 1.0, 10)
    A = stiffness(kv)
    f = inner_products(kv, lambda x: 1.0)
    LS = RestrictedLinearSystem(A, f, [(0,kv.numdofs-1), (0.0,1.0)])
    u = LS.complete(np.linalg.solve(LS.A.toarray(), LS.b))
    u_ex = interpolate(kv, lambda x: 0.5*x*(3-x))
    assert np.linalg.norm(u - u_ex) < 1e-12

################################################################################
# Test multipatch
################################################################################

def _make_Lshape():
    kvs = 2 * (bspline.make_knots(2, 0.0, 1.0, 8),)
    squ = geometry.unit_square()
    geos = (squ, squ.translate((1,0)), squ.scale((-1,1)).translate((2,1)))
    patches = [(kvs, g) for g in geos]
    MP = Multipatch(patches)
    MP.join_boundaries(0, 'right', 1, 'left')
    MP.join_boundaries(1, 'top', 2, 'bottom', flip=(True,))
    MP.finalize()
    return MP

def test_multipatch():
    MP = _make_Lshape()
    assert MP.numpatches == 3
    assert MP.numdofs == 90 + 81 + 90 + 2*10 - 1
    ### test patch-to-global indexing
    idx1 = MP.patch_to_global_idx(1)
    assert idx1.size == 100
    idx1 = idx1.reshape((10, 10))
    assert np.array_equal(idx1[:-1, 1:].ravel(), 90 + np.arange(9*9))   # local
    assert np.array_equal(idx1[:, 0], 90+81+90 + np.arange(10))         # left shared
    assert np.array_equal(idx1[-1, 1:], 90+81+90 + 10 + np.arange(9))   # top shared
    ### test transfer matrices
    u1 = np.arange(100)
    P1 = MP.patch_to_global(1)
    assert scipy.sparse.linalg.norm(MP.global_to_patch(1) @ P1 - scipy.sparse.eye(100)) == 0
    ug = P1 @ u1
    u0 = (MP.global_to_patch(0) @ ug).reshape((10, 10))
    assert np.allclose(u0[:, :-1], 0)
    assert np.array_equal(u0[:, -1], np.arange(0, 100, 10))
    u2 = (MP.global_to_patch(2) @ ug).reshape((10, 10))
    assert np.allclose(u2[1:, :], 0)
    assert np.array_equal(u2[0, :], np.arange(99, 89, -1))
    ### test Dirichlet BCs
    bcidx, bcvals = MP.compute_dirichlet_bcs([(0, 'top', lambda x,y: 1.0)])
    assert np.array_equal(bcidx, list(range(9*9, 10*9)) + [90+81+90 + 9])
    assert np.allclose(bcvals, 1.0)

def test_detect_interfaces():
    MP = _make_Lshape()
    MP2 = Multipatch(MP.patches, automatch=True)
    assert MP2.numdofs == MP.numdofs
    assert MP2.shared_per_patch == MP.shared_per_patch

def test_multipatch_assemble():
    kvs = 2 * (bspline.make_knots(2, 0.0, 1.0, 8),)
    geos = [geometry.unit_square(), geometry.unit_square().translate((1,0))]
    MP = Multipatch([(kvs, g) for g in geos], automatch=True)
    from pyiga import vform
    def f(x, y): return np.sin(2*x) + np.exp(y)
    A, b = MP.assemble_system(
            vform.stiffness_vf(2),
            vform.L2functional_vf(2, physical=True), f=f)
    # assemble same problem as a single patch
    knots_x = np.array(2 * [0.0] + list(np.linspace(0, 1.0, 9))
            + list(np.linspace(1.0, 2.0, 9)) + 2 * [2.0])
    kvs2 = (kvs[0], bspline.KnotVector(knots_x, 2))
    geo2 = geometry.identity(kvs2)
    A2 = assemble(vform.stiffness_vf(2), kvs2, geo=geo2)
    b2 = assemble(vform.L2functional_vf(2, physical=True), kvs2, geo=geo2, f=f)

    # construct index which maps between the two discretizations
    Ix = np.arange(b.size)
    Ix = np.hstack((
        Ix[:9*10].reshape((10,9)),      # first patch
        Ix[2*9*10:].reshape((10,1)),    # second patch
        Ix[9*10:2*9*10].reshape((10,9)))).ravel()   # interface
    assert np.allclose(b[Ix], b2.ravel())
    assert np.allclose(A.toarray()[Ix][:, Ix], A2.toarray())
