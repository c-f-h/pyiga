# -*- coding: utf-8 -*-

from pyiga.bspline import *

def test_eval():
    # create random spline
    kv = make_knots(4, 0.0, 1.0, 25)
    n = kv.numdofs
    coeffs = np.random.rand(n)
    # evaluate it, B-spline by B-spline
    x = np.linspace(0.0, 1.0, 100)
    values = sum((coeffs[j] * single_ev(kv, j, x) for j in range(n)))
    # evaluate all at once and compare
    values2 = ev(kv, coeffs, x)
    assert np.linalg.norm(values - values2) < 1e-10
    # evaluate through collocation matrix and compare
    values3 = collocation(kv, x).dot(coeffs)
    assert np.linalg.norm(values - values3) < 1e-10

def test_interpolation():
    kv = make_knots(3, 0.0, 1.0, 10)
    # create random spline
    coeffs = np.random.rand(kv.numdofs)
    def f(x): return ev(kv, coeffs, x)
    # interpolate it at Gréville points and check that result is the same
    result = interpolate(kv, f)
    assert np.allclose(coeffs, result)
    ##
    # test for p=0
    kv = make_knots(0, 0.0, 1.0, 10)
    coeffs = np.random.rand(kv.numdofs)
    def f(x): return ev(kv, coeffs, x)
    # interpolate it at Gréville points and check that result is the same
    result = interpolate(kv, f)
    assert np.allclose(coeffs, result)


def test_L2_projection():
    kv = make_knots(3, 0.0, 1.0, 10)
    def f(x): return np.sin(2*np.pi * x**2)
    x = np.linspace(0.0, 1.0, 100)
    coeffs = project_L2(kv, f)
    assert np.linalg.norm(f(x) - ev(kv, coeffs, x)) / np.sqrt(len(x)) < 1e-3

def test_deriv():
    # create linear spline
    kv = make_knots(4, 0.0, 1.0, 25)
    coeffs = interpolate(kv, lambda x: 1.0 + 2.5*x)
    # check that derivative is 2.5
    x = np.linspace(0.0, 1.0, 100)
    drv = deriv(kv, coeffs, 1, x)
    assert np.linalg.norm(drv - 2.5) < 1e-10

    # create random spline
    coeffs = np.random.rand(kv.numdofs)
    # compare derivatives by two methods
    derivs1 = deriv(kv, coeffs, 1, x)
    derivs2 = deriv(kv, coeffs, 2, x)
    allders = collocation_derivs(kv, x, derivs=2)
    assert np.linalg.norm(derivs1 - allders[1].dot(coeffs), np.inf) < 1e-10
    assert np.linalg.norm(derivs2 - allders[2].dot(coeffs), np.inf) < 1e-10

def test_refine():
    kv = make_knots(2, 0.0, 1.0, 4)
    kv2 = kv.refine([0.1])
    assert kv2.p == kv.p and np.array_equal(kv2.kv,
            [0.0, 0.0, 0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0])
    kv2 = kv.refine()
    assert kv2.p == kv.p and np.array_equal(kv2.kv, make_knots(2, 0.0, 1.0, 8).kv)

def test_prolongation():
    # create random spline
    kv = make_knots(3, 0.0, 1.0, 10)
    coeffs = np.random.rand(kv.numdofs)
    # compute a refined knot vector and prolongation matrix
    kv2 = kv.refine()
    P = prolongation(kv, kv2)
    coeffs2 = P.dot(coeffs)
    # check that they evaluate to the same function
    x = np.linspace(0.0, 1.0, 100)
    val1 = ev(kv, coeffs, x)
    val2 = ev(kv2, coeffs2, x)
    assert np.linalg.norm(val1 - val2) < 1e-10

def test_knot_insertion():
    kv = KnotVector(np.array([0.0, 0.0, 0.0, 0.0, 0.0,
        0.05, 0.12, 0.33, 0.51, 0.51, 0.51, 0.74, 0.88, 0.91,
        1.0, 1.0, 1.0, 1.0, 1.0]), 4)
    u = np.random.rand(kv.numdofs)
    x = np.linspace(0, 1, 100)
    for newknot in (0.01, 0.2, 0.33, 0.44, 0.6, 0.99):
        P = knot_insertion(kv, newknot)
        kv1 = kv.refine([newknot])
        assert np.allclose(ev(kv, u, x), ev(kv1, P @ u, x))

def test_mesh_span_indices():
    kv = make_knots(3, 0.0, 1.0, 4)
    assert np.array_equal(kv.mesh_span_indices(), [3, 4, 5, 6])
    kv = make_knots(3, 0.0, 1.0, 4, mult=3)
    assert np.array_equal(kv.mesh_span_indices(), [3, 6, 9, 12])

def test_hessian():
    from pyiga.approx import interpolate

    # 2D test
    kvs = 2 * (make_knots(3, 0.0, 1.0, 4),)
    grid = 2 * (np.linspace(0, 1, 7),)
    u = BSplineFunc(kvs, interpolate(kvs, lambda x,y: x**2 + 4*x*y + 3*y**2))
    hess = u.grid_hessian(grid)
    assert np.allclose(hess, [2.0, 4.0, 6.0])   # (xx, xy, yy)

    # 3D test
    kvs = 3 * (make_knots(3, 0.0, 1.0, 4),)
    grid = 3 * (np.linspace(0, 1, 5),)
    u = BSplineFunc(kvs, interpolate(kvs, lambda x,y,z: x**2 + 3*x*z + 2*y*z))
    hess = u.grid_hessian(grid)
    assert np.allclose(hess, [2.0, 0.0, 3.0, 0.0, 2.0, 0.0])   # (xx, xy, xz, yy, yz, zz)

    # 2D vector test
    kvs = 2 * (make_knots(3, 0.0, 1.0, 4),)
    grid = 2 * (np.linspace(0, 1, 7),)
    u = BSplineFunc(kvs, interpolate(kvs, lambda x,y: (x**2 + 4*x*y, 3*y**2)))
    hess = u.grid_hessian(grid)
    assert np.allclose(hess, [[2.0, 4.0, 0.0], [0.0, 0.0, 6.0]])   # (xx, xy, yy)

    # 3D vector test
    kvs = 3 * (make_knots(3, 0.0, 1.0, 4),)
    grid = 3 * (np.linspace(0, 1, 5),)
    u = BSplineFunc(kvs, interpolate(kvs, lambda x,y,z: (x**2, 3*x*z, 2*y*z)))
    hess = u.grid_hessian(grid)
    assert np.allclose(hess,                    # (xx, xy, xz, yy, yz, zz)
            [[2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 3.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 2.0, 0.0]])
