import numpy as np

import matrices

class SolverDebug:

    def test_extend_src_f(self):
        exp_src_f = np.zeros((self.N-1)**2, dtype=complex)

        for i, j in self.Kplus:
            x, y = self.get_coord(i, j)

            l = matrices.get_index(self.N, i, j)
            exp_src_f[l] = self.problem.eval_f(x, y)

        return np.max(np.abs(exp_src_f - self.src_f)) 
