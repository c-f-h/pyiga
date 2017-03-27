import pyamg
import scipy.linalg

def gauss_seidel(A, x, b, iterations=1, sweep='forward'):
    if scipy.sparse.issparse(A):
        return pyamg.relaxation.relaxation.gauss_seidel(A, x, b, iterations=iterations, sweep=sweep)
    else:
        if sweep == 'symmetric':
            for i in range(iterations):
                gauss_seidel(A, x, b, iterations=1, sweep='forward')
                gauss_seidel(A, x, b, iterations=1, sweep='backward')
            return
        elif sweep == 'forward':
            lower = True
        elif sweep == 'backward':
            lower = False
        else:
            raise ValueError("valid sweep directions are 'forward', 'backward', and 'symmetric'")
        for i in range(iterations):
            r = b - A.dot(x)
            x += scipy.linalg.solve_triangular(A, r, lower=lower, overwrite_b=True, check_finite=False)

