import numpy as np
import math

import matrices

class PsInhomo:

    def extend_inhomogeneous_f(self, *args):
        ext = np.zeros(len(self.gamma), dtype=complex)
        return ext

    def extend_src_f(self):
        #NOTE
        return
        p = self.problem
        if p.homogeneous:
            return

        for i,j in self.Kplus - self.Mplus:
            x, y = self.get_coord(i, j)
            delta = 0

            derivs = (0, 0)

            v = p.eval_f(x, y)
            #for l in range(1, len(derivs)+1):
            #    v += derivs[l-1] / math.factorial(l) * delta**l

            self.src_f[matrices.get_index(self.N,i,j)] = v
