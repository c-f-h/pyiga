from pyiga.geometry import *
from pyiga import approx, bspline

def geos_roughly_equal(geo1, geo2, n=25):
    x = np.linspace(0.0, 1.0, n)
    grid = geo1.sdim * (x,)
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
    assert geos_roughly_equal(cube2, unit_square())
    assert geos_roughly_equal(cube4, cube3.extrude(0.0, 1.0))

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
    bd = geo.boundary(axis=2, side=1)
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

def test_nurbs_boundary():
    geo = quarter_annulus()
    assert geos_roughly_equal(geo.boundary(1, 0),
                              circular_arc(np.pi/2, 1.0))
    assert geos_roughly_equal(geo.boundary(1, 1),
                              circular_arc(np.pi/2, 2.0))

def test_circular_arc():
    alpha = 2./3.*np.pi
    geo = circular_arc(alpha, r=2)
    grd = np.linspace(0., 1., 50)
    v = geo.grid_eval((grd,))
    assert np.allclose(v[0], [2,0])
    assert np.allclose(v[-1], [2*np.cos(alpha), 2*np.sin(alpha)])
    assert abs(2. - np.linalg.norm(v, axis=-1)).max() < 1e-12

def test_circle():
    r = 1.75
    geo = circle(r=r)
    grd = np.linspace(0., 1., 50)
    v = geo.grid_eval((grd,))
    assert np.allclose(v[0],  [r,0])
    assert np.allclose(v[-1], [r,0])
    assert abs(r - np.linalg.norm(v, axis=-1)).max() < 1e-12
