import numpy as np
from . import assemble

class HDiscretization:
    def __init__(self, hspace, truncate=False):
        self.hs = hspace
        self._I_hb = hspace.represent_fine(truncate=truncate)

    def assemble_matrix(self):
        kvs_fine = self.hs.knotvectors(-1)
        A_fine = assemble.stiffness(kvs_fine)
        return (self._I_hb.T @ A_fine @ self._I_hb).tocsr()

    def assemble_rhs(self, f):
        kvs_fine = self.hs.knotvectors(-1)
        f_fine = assemble.inner_products(kvs_fine, f).ravel()
        return self._I_hb.T @ f_fine
