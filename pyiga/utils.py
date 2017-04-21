import numpy as np
import scipy.sparse
import scipy.sparse.linalg

def grid_eval(f, grid):
    if hasattr(f, 'grid_eval'):
        return f.grid_eval(grid)
    else:
        mesh = np.meshgrid(*grid, sparse=True, indexing='ij')
        mesh.reverse() # convert order ZYX into XYZ
        values = f(*mesh)
        return values


def make_solver(B, symmetric=False):
    """Construct a linear solver for the (dense or sparse) square matrix B.
    
    Returns a LinearOperator."""
    if scipy.sparse.issparse(B):
        spLU = scipy.sparse.linalg.splu(B.tocsc(), permc_spec='NATURAL')
        return scipy.sparse.linalg.LinearOperator(B.shape, spLU.solve)
    else:
        if symmetric:
            chol = scipy.linalg.cho_factor(B, check_finite=False)
            return scipy.sparse.linalg.LinearOperator(B.shape,
                lambda x: scipy.linalg.cho_solve(chol, x, check_finite=False))
        else:
            LU = scipy.linalg.lu_factor(B, check_finite=False)
            return scipy.sparse.linalg.LinearOperator(B.shape,
                lambda x: scipy.linalg.lu_solve(LU, x, check_finite=False))


def read_sparse_matrix(fname):
    I,J,vals = np.loadtxt(fname, skiprows=1, unpack=True)
    I -= 1
    J -= 1
    return scipy.sparse.coo_matrix((vals, (I,J))).tocsr()
