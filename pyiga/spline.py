import numpy as np

from . import bspline

class Spline:
    def __init__(self, kv, coeffs):
        """Create a spline function with the given knot vector and coefficients."""
        coeffs = np.asarray(coeffs)
        assert coeffs.shape == (kv.numdofs(),)
        self.kv = kv
        self.coeffs = coeffs

    def eval(self, x):
        """Evaluate the spline at all points of the vector x"""
        return bspline.ev(self.kv, self.coeffs, x)

    def deriv(self, x, deriv=1):
        """Evaluate a derivative of the spline at all points of the vector x"""
        return bspline.deriv(self.kv, self.coeffs, deriv, x)

    def derivative(self):
        """Return the derivative of this spline as a spline"""
        p = self.kv.p
        diffcoeffs = p / (self.kv.kv[p+1:-1] - self.kv.kv[1:-(p+1)]) * np.diff(self.coeffs)
        diffkv = bspline.KnotVector(self.kv.kv[1:-1], p-1)
        return Spline(diffkv, diffcoeffs)

