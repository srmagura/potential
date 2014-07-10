import math
import numpy as np

class SolverExtend:

    def extend_circle(self, r, xi0, xi1,
        d2_xi0_th, d2_xi1_th, d4_xi0_th):

        R = self.R
        k = self.k

        derivs = []
        derivs.append(xi0) 
        derivs.append(xi1)
        derivs.append(-xi1 / R - d2_xi0_th / R**2 - k**2 * xi0)

        derivs.append(2 * xi1 / R**2 + 3 * d2_xi0_th / R**3 -
            d2_xi1_th / R**2 + k**2 / R * xi0 - k**2 * xi1)

        derivs.append(-6 * xi1 / R**3 + 
            (2*k**2 / R**2 - 11 / R**4) * d2_xi0_th +
            6 * d2_xi1_th / R**3 + d4_xi0_th / R**4 -
            (3*k**2 / R**2 - k**4) * xi0 +
            2 * k**2 / R * xi1)

        v = 0
        for l in range(len(derivs)):
            v += derivs[l] / math.factorial(l) * (r - R)**l

        return v

    def extend_inhomogeneous_circle(self, r, th):
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
