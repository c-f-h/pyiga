from pyiga.solvers import *
from pyiga import bspline, assemble

def test_fastdiag_solver():
    kvs = [
        bspline.make_knots(4, 0.0, 1.0, 3),
        bspline.make_knots(3, 0.0, 1.0, 4),
        bspline.make_knots(2, 0.0, 1.0, 5)
    ]
    # compute Dirichlet matrices
    KM = [(assemble.stiffness(kv)[1:-1, 1:-1].toarray(), assemble.mass(kv)[1:-1, 1:-1].toarray()) for kv in kvs]
    solver = fastdiag_solver(KM)

    def multikron(*Xs):
        return reduce(np.kron, Xs)
    A = ( multikron(KM[0][0], KM[1][1], KM[2][1]) +
          multikron(KM[0][1], KM[1][0], KM[2][1]) +
          multikron(KM[0][1], KM[1][1], KM[2][0]) )
    f = np.random.rand(A.shape[0])
    assert np.allclose(f, solver.dot(A.dot(f)))

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

def test_gauss_seidel_indexed():
    from numpy.random import rand
    A = abs(rand(10,10)) + np.eye(10) # try to make it not too badly conditioned
    b = rand(10)
    indices = [3, 6, 9]

    for sweep in ('forward', 'backward', 'symmetric'):
        x1 = rand(10)
        x2 = x1.copy()

        gauss_seidel(scipy.sparse.csr_matrix(A), x1, b, iterations=2, indices=indices, sweep=sweep)
        gauss_seidel(A, x2, b, iterations=2, indices=indices, sweep=sweep)
        assert abs(x1-x2).max() < 1e-12

def test_twogrid():
    kv_c = bspline.make_knots(3, 0.0, 1.0, 50)
    kv = kv_c.refine()
    P = bspline.prolongation(kv_c, kv)
    A = assemble.mass(kv) + assemble.stiffness(kv)
    f = bspline.load_vector(kv, lambda x: 1.0)
    S = SequentialSmoother((GaussSeidelSmoother(), OperatorSmoother(1e-6*np.eye(len(f)))))
    x = twogrid(A, f, P, S)
    assert np.linalg.norm(f - A.dot(x)) < 1e-6

def test_newton():
    def F(x): return np.array([np.sin(x[0]) - 1/2])
    def J(x): return np.array([[np.cos(x[0])]])
    x = newton(F, J, [0.0])
    assert np.allclose(x, np.pi / 6)

def test_ode():
    # simple stiff ODE
    A = np.array([
        [0.0, 1.0],
        [-1000.0, -1001.0]
    ])
    M = np.eye(2)
    def F(x): return A.dot(x)
    def J(x): return A
    x0 = np.array([1.0, 0.0])

    def exsol(t): return -1/999 * np.exp(-1000*t) + 1000/999 * np.exp(-t)

    t_end = 1.0
    sol_1 = exsol(t_end)

    # constant step Crank-Nicolson
    sols = crank_nicolson(M, F, J, x0, 1e-2, t_end)
    assert np.isclose(sols[1][-1][0], sol_1, rtol=1e-4)

    # constant step DIRK method
    sols = sdirk3(M, F, J, x0, 1e-2, t_end)
    assert np.isclose(sols[1][-1][0], sol_1, rtol=1e-4)

    # constant step Rosenbrock method
    sols = ros3p(M, F, J, x0, 1e-2, t_end, tol=None)
    assert np.isclose(sols[1][-1][0], sol_1, rtol=1e-4)

    # adaptive step DIRK method
    sols = esdirk34(M, F, J, x0, 1e-2, t_end, tol=1e-5)
    ts = sols[0]
    xs = sols[1]
    assert ts[-2] <= t_end <= ts[-1]
    from scipy.interpolate import interp1d
    x_end = interp1d(ts, xs, kind='cubic', axis=0)(t_end)
    #print(len(ts), 'steps, error =', abs(x_end[0] - sol_1))
    assert np.isclose(x_end[0], sol_1, rtol=1e-4)
