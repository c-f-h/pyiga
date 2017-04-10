from pyiga.relaxation import *
import numpy as np

def test_gauss_seidel():
    from numpy.random import rand
    A = abs(rand(10,10)) + np.eye(10) # try to make it not too badly conditioned
    b = rand(10)

    for sweep in ('forward', 'backward', 'symmetric'):
        x1 = rand(10)
        x2 = x1.copy()

        gauss_seidel(scipy.sparse.csr_matrix(A), x1, b, iterations=2, sweep=sweep)
        gauss_seidel(A, x2, b, iterations=2, sweep=sweep)
        assert abs(x1-x2).max() < 1e-12

