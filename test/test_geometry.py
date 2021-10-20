from pyiga.geometry import *
from pyiga import approx, bspline

def geos_roughly_equal(geo1, geo2, n=25):
    supp = geo1.support
    grid = tuple(np.linspace(s[0], s[1], n) for s in supp)
    f1 = geo1.grid_eval(grid)
    f2 = geo2.grid_eval(grid)
    return np.allclose(f1, f2)

def test_creation():
    geo = unit_square()
    assert geo.sdim == geo.dim == 2
    geo = perturbed_square(noise=0.05)
    assert geo.sdim == geo.dim == 2
    geo = bspline_quarter_annulus()
    assert geo.sdim == geo.dim == 2
    geo = unit_cube()
    assert geo.sdim == geo.dim == 3
    geo = unit_cube(dim=4)
    assert geo.sdim == geo.dim == 4
    geo = twisted_box()
    assert geo.sdim == geo.dim == 3

def test_cube():
    cube2 = unit_cube(dim=2)
    cube3 = unit_cube(dim=3)
    cube4 = unit_cube(dim=4)
    assert np.allclose(cube2.coeffs, unit_square().coeffs)
    assert np.allclose(cube4.coeffs, cube3.cylinderize(0.0, 1.0).coeffs)

def test_identity():
    geo = identity([(3.0,4.0), (5.0,6.0)])
    assert geo.dim == geo.sdim == 2
    assert np.allclose(geo.eval(5,3), (5,3))
    assert np.allclose(geo.eval(6,4), (6,4))
    assert np.allclose(geo.eval(5.87,3.21), (5.87,3.21))
    geo2 = identity([bspline.make_knots(3, 3.0, 4.0, 10),
                     bspline.make_knots(3, 5.0, 6.0, 5)])
    assert geos_roughly_equal(geo, geo2)

def test_evaluation():
    geo = bspline_quarter_annulus()
    x = np.asarray([0.0, 0.5, 1.0])
    y = np.asarray([0.0, 0.3, 0.7, 1.0])
    values = geo.grid_eval((y,x))
    exact = np.array(
      [[[ 1.   ,  0.   ],
        [ 1.5  ,  0.   ],
        [ 2.   ,  0.   ]],
       [[ 0.91 ,  0.51 ],
        [ 1.365,  0.765],
        [ 1.82 ,  1.02 ]],
       [[ 0.51 ,  0.91 ],
        [ 0.765,  1.365],
        [ 1.02 ,  1.82 ]],
       [[ 0.   ,  1.   ],
        [ 0.   ,  1.5  ],
        [ 0.   ,  2.   ]]])
    assert abs(exact[1,1] - geo.eval(0.5, 0.3)).max() < 1e-14
    assert abs(exact - values).max() < 1e-14
    # test pointwise_eval
    mesh_x, mesh_y = np.meshgrid(x, y, indexing='xy')
    values2 = geo.pointwise_eval((mesh_x, mesh_y))
    assert values2.shape == mesh_x.shape + (2,)
    assert np.allclose(values, values2)
    # test pointwise_eval for NURBS
    geo = quarter_annulus()
    values = geo.grid_eval((y, x))
    values2 = geo.pointwise_eval((mesh_x, mesh_y))
    assert values2.shape == mesh_x.shape + (2,)
    assert np.allclose(values, values2)
    # test calling a geometry function with mixed scalar/array arguments
    x = 0.7
    y = [0.1, 0.33, 0.72]
    z = np.linspace(0.0, 0.5, 4)
    # BSplineFunc
    geo = bspline_quarter_annulus().cylinderize(0, 1)
    assert np.allclose(geo(x, y, z), geo.grid_eval((z, y, [x]))[:, :, 0])
    # NurbsFunc
    geo = twisted_box()
    assert np.allclose(geo(x, y, z), geo.grid_eval((z, y, [x]))[:, :, 0])

def test_jacobian():
    geo = bspline_quarter_annulus()
    x = np.asarray([0.0, 0.3, 0.7, 1.0])
    y = np.asarray([0.75])
    jac = geo.grid_jacobian((y,x))
    exact = np.array([[[[ 0.4375, -1.5   ],
                        [ 0.9375,  0.5   ]],
                       [[ 0.4375, -1.95  ],
                        [ 0.9375,  0.65  ]],
                       [[ 0.4375, -2.55  ],
                        [ 0.9375,  0.85  ]],
                       [[ 0.4375, -3.    ],
                        [ 0.9375,  1.    ]]]])
    assert abs(exact - jac).max() < 1e-14

def test_unitsquare():
    S1 = unit_square()
    S2 = unit_square(num_intervals=10)
    assert geos_roughly_equal(S1, S2)

def test_boundary():
    geo = twisted_box()
    bd = geo.boundary((2, 1))
    assert bd.sdim == geo.sdim - 1
    assert bd.dim == geo.dim
    assert np.allclose(geo.eval(1,1,0), bd.eval(1,0))

def test_trf_gradient():
    geo = bspline_quarter_annulus()
    u_coeffs = approx.interpolate(geo.kvs, lambda x,y: x-y, geo=geo)
    u = BSplineFunc(geo.kvs, u_coeffs)
    u_grad = u.transformed_jacobian(geo)
    grd = 2 * (np.linspace(0, 1, 10),)
    grads = u_grad.grid_eval(grd)
    assert np.allclose(grads[:, :, 0], 1) and np.allclose(grads[:, :, 1], -1)

def test_nurbs():
    kv = bspline.make_knots(2, 0.0, 1.0, 1)
    r = 2.0
    # construct quarter circle using NURBS
    coeffs = np.array([
            [  r, 0.0, 1.0],
            [  r,   r, 1.0 / np.sqrt(2.0)],
            [0.0,   r, 1.0]])

    grid = (np.linspace(0.0, 1.0, 20),)

    nurbs = NurbsFunc((kv,), coeffs.copy(), weights=None)
    vals = nurbs.grid_eval(grid)
    assert abs(r - np.linalg.norm(vals, axis=-1)).max() < 1e-12

    nurbs = NurbsFunc((kv,), coeffs[:, :2], weights=coeffs[:, -1])
    vals = nurbs.grid_eval(grid)
    assert abs(r - np.linalg.norm(vals, axis=-1)).max() < 1e-12

    assert nurbs.output_shape() == (2,)
    assert (not nurbs.is_scalar()) and nurbs.is_vector()
    nurbsx = nurbs[0]
    assert nurbsx.output_shape() == ()
    assert nurbsx.is_scalar() and (not nurbsx.is_vector())
    assert nurbsx.grid_eval(grid).shape[1:] == ()
    assert nurbsx.grid_jacobian(grid).shape[1:] == (1,)
    assert nurbsx.grid_hessian(grid).shape[1:] == (1,)

def _num_hess(f, x, h=1e-3):
    def delta(i, diffi, j, diffj):
        y = list(x)
        y[i] += diffi
        y[j] += diffj
        return y
    def pd2(i, j):
        # see https://dlmf.nist.gov/3.4#E25
        return (f(delta(i,h, j,h)) + f(delta(i,-h, j,-h))
              - f(delta(i,h, j,-h)) - f(delta(i,-h, j,h))) / (4*h**2)
    return np.array([pd2(0,0), pd2(1,0), pd2(1,1)])

def test_bspline_hessian():
    geo = bspline_quarter_annulus()
    def f1(xy): return geo.eval(*xy)[0]
    def f2(xy): return geo.eval(*xy)[1]
    X = np.linspace(0, 1, 4)[1:-1]
    H = geo.grid_hessian((X,X))
    H_num = np.array([[
            [_num_hess(f1, (X[i],X[j])),
             _num_hess(f2, (X[i],X[j]))]
            for i in range(len(X))]
            for j in range(len(X))])
    assert np.allclose(H, H_num)

def test_nurbs_hessian():
    geo = quarter_annulus()
    def f1(xy): return geo.eval(*xy)[0]
    def f2(xy): return geo.eval(*xy)[1]
    X = np.linspace(0, 1, 4)[1:-1]
    H = geo.grid_hessian((X,X))
    H_num = np.array([[
            [_num_hess(f1, (X[i],X[j])),
             _num_hess(f2, (X[i],X[j]))]
            for i in range(len(X))]
            for j in range(len(X))])
    assert np.allclose(H, H_num)

def test_nurbs_boundary():
    geo = quarter_annulus()
    assert geos_roughly_equal(geo.boundary('left'),
                              circular_arc(np.pi/2, 1.0))
    assert geos_roughly_equal(geo.boundary('right'),
                              circular_arc(np.pi/2, 2.0))

def test_reduced_support():
    ## test B-spline patch
    geo = unit_square()
    supp = ((0.2, 0.7), (0.4, 0.6))
    geo.support = supp
    assert np.allclose(geo.bounding_box(), list(reversed(supp)))    # bounding box is in xy order
    #
    bd = geo.boundary('right')
    assert geos_roughly_equal(bd, line_segment((0.6, 0.2), (0.6, 0.7), support=(0.2, 0.7)))
    ## test NURBS patch
    geo = quarter_annulus()
    geo.support = supp
    bd = geo.boundary('top')
    assert np.allclose(bd.bounding_box(),
            ((0.6177743988536184, 0.7060278844041353), (1.2563259099935216, 1.4358010399925962)))

def test_line_segment():
    L1 = line_segment((1,0), (4,2), support=(1,2))
    assert L1.sdim == 1
    assert L1.dim == 2
    assert np.allclose(L1.eval(1.5), (2.5, 1.0))
    ###
    assert line_segment(3, 5).dim == 1

def test_circular_arc():
    alpha, r = 2./3.*np.pi, 2.0
    def pt(s): return (r * np.cos(s * alpha), r * np.sin(s * alpha))
    geo = circular_arc(alpha, r=r)   # 3-point arc
    grd = np.linspace(0., 1., 51)
    v = geo(grd)
    assert np.allclose(v[0],  pt(0.0))
    assert np.allclose(v[25], pt(0.5))
    assert np.allclose(v[50], pt(1.0))
    assert np.allclose(r, np.linalg.norm(v, axis=-1))
    ## 7-point arc
    alpha, r = 5.2/3.*np.pi, 1.7
    geo = circular_arc(alpha, r=r)
    grd = np.linspace(0., 1., 61)
    v = geo(grd)
    for k in range(0, 61, 10):
        assert np.allclose(v[k],  pt(k / 60))
    assert np.allclose(r, np.linalg.norm(v, axis=-1))

def test_circle():
    r = 1.75
    geo = circle(r=r)
    grd = np.linspace(0., 1., 50)
    v = geo.grid_eval((grd,))
    assert np.allclose(v[0],  [r,0])
    assert np.allclose(v[-1], [r,0])
    assert abs(r - np.linalg.norm(v, axis=-1)).max() < 1e-12

def test_semicircle():
    r = 2.09
    geo = semicircle(r=r)
    assert np.allclose(geo(0.0), (r,0))
    assert np.allclose(geo(0.5), (0,r))
    assert np.allclose(geo(1.0), (-r,0))
    v = geo.grid_eval((np.linspace(0., 1., 50),))
    assert abs(r - np.linalg.norm(v, axis=-1)).max() < 1e-12

def test_outer():
    ### outer_sum
    Gy, Gx = line_segment([0,1],[0,2]), line_segment([2,0],[3,0])
    G1 = outer_sum(Gy, Gx)
    G2 = unit_square().translate((2,1))
    assert geos_roughly_equal(G1, G2)
    Y, X = np.linspace(0, 1, 10), np.linspace(0, 1, 10)
    assert np.allclose(G1.grid_eval((Y,X)),
            Gy.grid_eval((Y,))[:, None, ...] + Gx.grid_eval((X,))[None, :, ...])
    ### outer_product
    Gy, Gx = line_segment([1,1],[1,2]), line_segment([3,1],[4,1])
    G1 = outer_product(Gy, Gx)
    G2 = unit_square().translate((3,1))
    assert geos_roughly_equal(G1, G2)
    assert np.allclose(G1.grid_eval((Y,X)),
            Gy.grid_eval((Y,))[:, None, ...] * Gx.grid_eval((X,))[None, :, ...])
    ### NURBS outer_sum
    # create two circular arcs going from (0,0) to (0,1) and to (1,0), respectively
    Gy = circular_arc(np.pi/3).translate((-1,0)).rotate_2d(-np.pi/6)
    Gx = Gy.scale(-1).rotate_2d(np.pi/2)
    G1 = outer_sum(Gy, Gx)
    assert np.allclose(G1.grid_eval((Y,X)),
            Gy.grid_eval((Y,))[:, None, ...] + Gx.grid_eval((X,))[None, :, ...])
    ### NURBS outer_sum
    Gy = Gy.translate((1,1))
    Gx = Gx.translate((1,1))
    G1 = outer_product(Gy, Gx)
    assert np.allclose(G1.grid_eval((Y,X)),
            Gy.grid_eval((Y,))[:, None, ...] * Gx.grid_eval((X,))[None, :, ...])

def test_tensorproduct():
    Gy = circular_arc(np.pi / 2)                    # 2D function
    kv = bspline.make_knots(2, 0.0, 1.0, 1)
    Gx = NurbsFunc(kv, [0,.5,1], [1,1.0/3.0,1])     # scalar function
    G = tensor_product(Gy, Gx)
    assert G.sdim == Gy.sdim + Gx.sdim == 2
    assert G.dim  == Gy.dim + Gx.dim   == 3
    Y,X = np.linspace(0,1,7), np.linspace(0,1,7)
    Vy, Vx = Gy.grid_eval((Y,)), Gx.grid_eval((X,))[..., np.newaxis]  # convert Gx to vector
    V = G.grid_eval((Y,X))
    for i in range(len(Y)):
        for j in range(len(X)):
            assert np.allclose(V[i,j], np.concatenate((Vx[j], Vy[i])))

def test_translate():
    # translation of B-spline functions
    G = unit_cube().translate((1,2,3))
    assert geos_roughly_equal(G,
            tensor_product(line_segment(3,4), line_segment(2,3), line_segment(1,2)))
    # translation of NURBS patch
    G = quarter_annulus().translate((1,2))
    values = G.grid_eval((np.linspace(0,1,20), [0.]))
    assert np.allclose(np.linalg.norm(values - (1,2), axis=-1), 1.0)

def test_scale():
    G = unit_square().scale(2)
    assert geos_roughly_equal(G,
            tensor_product(line_segment(0,2), line_segment(0,2)))
    G = unit_square().scale((3,1))
    assert geos_roughly_equal(G,
            tensor_product(line_segment(0,1), line_segment(0,3)))
    # scaling a NURBS curve
    G1 = circular_arc(np.pi / 2)
    G2 = G1.scale((2,1))
    grid = (np.linspace(0, 1, 20),)
    V1, V2 = G1.grid_eval(grid), G2.grid_eval(grid)
    assert np.allclose(V1 * (2,1), V2)

def test_rotation():
    # rotation of a rectangle
    G = unit_square().scale((2,1)).rotate_2d(np.pi / 4)
    d = 1.0 / np.sqrt(2)
    assert geos_roughly_equal(G,
            outer_sum(line_segment([0,0], [-d,d]),
                      line_segment([0,0], [2*d,2*d])))
    # rotation of NURBS patch
    G = quarter_annulus().rotate_2d(np.pi / 4)
    values = G.grid_eval((np.linspace(0,1,20), [0.,1.]))
    assert np.allclose(np.linalg.norm(values[:,0], axis=-1), 1.0) # inner arc r=1
    assert np.allclose(np.linalg.norm(values[:,1], axis=-1), 2.0) # outer arc r=2
    assert np.allclose(values[0,0],  (d,d))      # check that the 4 corners are correct
    assert np.allclose(values[-1,0], (-d,d))
    assert np.allclose(values[0,1],  (2*d,2*d))
    assert np.allclose(values[-1,1], (-2*d,2*d))

def test_userfunction():
    def f(x, y):
        r = 1 + x
        w = 1 - (2 - np.sqrt(2)) * y * (1-y)
        return (r * (1 - y**2/w), r * (1 - (1-y)**2/w))
    F = UserFunction(f, [[0,1],[0,1]])
    assert F.sdim == F.dim == 2
    assert geos_roughly_equal(F, quarter_annulus())

def test_as_nurbs():
    G = bspline_quarter_annulus()
    G2 = G.as_nurbs()
    assert isinstance(G, BSplineFunc) and isinstance(G2, NurbsFunc)
    assert geos_roughly_equal(G, G2)

def test_as_vector():
    grid = 2 * (np.linspace(0.0, 1.0, 10),)

    F = bspline_quarter_annulus()[0]
    assert F.is_scalar()
    Fv = F.as_vector()
    assert Fv.is_vector()
    f = F.grid_eval(grid)
    fv = Fv.grid_eval(grid)
    assert f.shape + (1,) == fv.shape
    assert np.allclose(f, fv[..., 0])

    # same test with NURBS
    F = quarter_annulus()[0]
    assert F.is_scalar()
    Fv = F.as_vector()
    assert Fv.is_vector()
    f = F.grid_eval(grid)
    fv = Fv.grid_eval(grid)
    assert f.shape + (1,) == fv.shape
    assert np.allclose(f, fv[..., 0])

def test_getitem():
    grid = 2 * (np.linspace(0.0, 1.0, 10),)
    # BSplineFunc
    G = bspline_quarter_annulus()
    f = G.grid_eval(grid)
    fx = G[0].grid_eval(grid)
    fy = G[1].grid_eval(grid)
    assert np.allclose(f[..., 0], fx)
    assert np.allclose(f[..., 1], fy)
    # NurbsFunc
    G = quarter_annulus()
    f = G.grid_eval(grid)
    fx = G[0].grid_eval(grid)   # NB: currently, these are 1D vector-valued functions
    fy = G[1].grid_eval(grid)
    assert np.allclose(f[..., 0], fx)
    assert np.allclose(f[..., 1], fy)

def test_bounding_box():
    bb = quarter_annulus().bounding_box()
    assert np.allclose(bb, [(0,2), (0,2)])

def test_inverse():
    def check_inverse(geo, x):
        xi = geo.find_inverse(x)
        assert np.allclose(x, geo(*xi))
    check_inverse(quarter_annulus(), [1.2, 1.5])
    check_inverse(circle(), np.ones(2) / np.sqrt(2))
    try:
        # should raise an exception
        circle().find_inverse((0.7, 0.7))
        assert False, 'find_inverse should fail for point outside the geometry'
    except ValueError:
        pass
