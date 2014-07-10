import numpy as np
import math

import matrices

class PsInhomo:

    def extend_inhomogeneous_f(self, *args):
        ext = np.zeros(len(self.gamma), dtype=complex)
        return ext

    def extend_src_f(self):
        p = self.problem
        if p.homogeneous:
            return

        for i,j in self.Kplus - self.Mplus:
            x, y = self.get_coord(i, j)
            r, th = self.get_polar(i, j)
            etype = self.get_etype(i, j)

            derivs = [0, 0]

            if etype == self.etypes['circle']:
                R = self.R
                delta = r - R
                x0 = R * np.cos(th)
                y0 = R * np.sin(th)
                derivs[0] = p.eval_d_f_r(R, th)

            v = p.eval_f(x0, y0)
            for l in range(1, len(derivs)+1):
                v += derivs[l-1] / math.factorial(l) * delta**l

            self.src_f[matrices.get_index(self.N,i,j)] = v
