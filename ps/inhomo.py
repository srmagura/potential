import numpy as np
import math

class PsInhomo:

    def extend_inhomogeneous_f(self, *args):
        ext = np.zeros(len(self.gamma), dtype=complex)
        return ext

    def extend_src_f(self):
        #FIXME
        return 
        p = self.problem
        if p.homogeneous:
            return

        for i,j in self.Kplus - self.Mplus:
            R = self.R

            r, th = self.get_polar(i, j)
            x, y = self.get_coord(i, j)

            derivs = [p.eval_f(x, y), 0, 0]

            v = 0
            for l in range(len(derivs)):
                v += derivs[l] / math.factorial(l) * (r - R)**l

            self.src_f[matrices.get_index(self.N,i,j)] = v
