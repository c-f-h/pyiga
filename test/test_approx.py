from pyiga.approx import *
from pyiga import bspline, geometry
import numpy as np

def _test_approx(approx_fun, extra_dims):
    kvs = [bspline.make_knots(p, 0.0, 1.0, 8+p) for p in range(3,6)]
    N = [kv.numdofs for kv in kvs]
    coeffs = np.random.random_sample(N + extra_dims)
    func = geometry.BSplineFunc(kvs, coeffs)
    result = approx_fun(kvs, func)
    assert np.allclose(coeffs, result)

    # try also with direct function call
    def f(X, Y, Z):
        return func.grid_eval([np.squeeze(w) for w in (Z,Y,X)])
    result = approx_fun(kvs, f)
    assert np.allclose(coeffs, result)


def test_project_L2():
    _test_approx(project_L2, [])    # scalar-valued

def test_project_L2_vector():
    _test_approx(project_L2, [3])   # vector-valued

def test_project_L2_matrix():
    _test_approx(project_L2, [2,2]) # matrix-valued


def test_project_L2_geo():
    f = lambda x,y,z: np.cos(x)*np.sin(y)*np.exp(z)
    kvs = 3 * (bspline.make_knots(3, 0.0, 1.0, 10),)
    x1 = project_L2(kvs, f)
    x2 = project_L2(kvs, f, geo=geometry.unit_cube())
    assert np.allclose(x1, x2)


def test_interpolate():
    _test_approx(interpolate, [])    # scalar-valued

def test_interpolate_vector():
    _test_approx(interpolate, [3])   # vector-valued

def test_interpolate_matrix():
    _test_approx(interpolate, [2,2]) # matrix-valued
