import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as linalg

class LU_Factorization:

    def __init__(self, A):
        lu = linalg.splu(A)

        self.L = sp.csc_matrix(lu.L, dtype=complex)
        self.U = sp.csc_matrix(lu.U, dtype=complex)
        n = self.L.shape[0]

        Pr = sp.lil_matrix((n, n))
        Pr[lu.perm_r, np.arange(n)] = 1
        self.Pr = Pr.tocsc()

        Pc = sp.lil_matrix((n, n))
        Pc[np.arange(n), lu.perm_c] = 1
        self.Pc = Pc.tocsc()

    def solve(self, b):
        tmp = self.Pr.dot(b)
        tmp = linalg.spsolve(self.L, tmp)
        tmp = linalg.spsolve(self.U, tmp)
        return self.Pc.dot(tmp)
