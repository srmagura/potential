import math
import numpy as np

from solver import Solver
import matrices

class CircleSolver(Solver):

    def __init__(self, problem, N, scheme_order, **kwargs):
        super().__init__(problem, N, scheme_order, **kwargs)
        problem.R = self.R

    def is_interior(self, i, j):
        return self.get_polar(i, j)[0] <= self.R

    def extend_inhomogeneous(self, r, th):
        p = self.problem
        if p.homogeneous:
            return 0

        R = self.R
        k = self.k

        x = R * np.cos(th)
        y = R * np.sin(th)

        derivs = [0, 0]
        derivs.append(p.eval_f(x, y))
        derivs.append(p.eval_d_f_r(R, th) - p.eval_f(x, y) / R)
        derivs.append(p.eval_d2_f_r(R, th) - 
            p.eval_d2_f_th(R, th) / R**2 - 
            p.eval_d_f_r(R, th) / R + 
            (3/R**2 - k**2) * p.eval_f(x, y))

        v = 0
        for l in range(len(derivs)):
            v += derivs[l] / math.factorial(l) * (r - R)**l

        return v

    def extend_inhomogeneous_f(self):
        ext = np.zeros(len(self.gamma), dtype=complex)
        for l in range(len(self.gamma)):
            i, j = self.gamma[l]
            r, th = self.get_polar(i, j)
            ext[l] = self.extend_inhomogeneous(r, th)

        return ext

    def extend_src_f(self):
        p = self.problem
        if p.homogeneous:
            return

        for i,j in self.Kplus - self.Mplus:
            R = self.R

            r, th = self.get_polar(i, j)
            x = R * np.cos(th)
            y = R * np.sin(th)

            derivs = [p.eval_f(x, y), p.eval_d_f_r(R, th),
                p.eval_d2_f_r(R, th)]

            v = 0
            for l in range(len(derivs)):
                v += derivs[l] / math.factorial(l) * (r - R)**l

            self.src_f[matrices.get_index(self.N,i,j)] = v
