from pyiga.spline import *

def _random_kv(p, n):
    steps = np.random.rand(n) * 0.75 + 0.25
    knots = np.cumsum(steps)
    knots -= knots.min()
    knots /= knots.max()
    return bspline.KnotVector(
        np.concatenate((p * [knots[0]], knots, p * [knots[-1]])), p)

def test_derivative():
    kv = _random_kv(4, 20)
    s = Spline(kv, np.random.rand(kv.numdofs))
    s1 = s.derivative()
    x = np.linspace(0.0, 1.0, 50)
    d1 = s.deriv(x, 1)
    d2 = s1.eval(x)
    assert abs(d1-d2).max() < 1e-10

