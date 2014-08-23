import numpy as np

import matrices
import solver

class SolverDebug:

    def test_extend_src_f(self):
        error = []
        
        for i, j in self.Kplus:
            x, y = self.get_coord(i, j)

            l = matrices.get_index(self.N, i, j)         
            diff = abs(self.problem.eval_f(x, y) - self.src_f[l])
            error.append(diff)

        result = solver.Result()
        result.error = max(error)
        return result
