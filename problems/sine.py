import numpy as np
from numpy import cos, sin

from .problem import Problem, PizzaProblem

def eval_d_u_r(k, R, th):
    return k*cos(th) * cos(k*R*cos(th))

class Sine(PizzaProblem):

    k = 1
    homogeneous = True
    expected_known = True

    n_basis_dict = {
        16: (21, 9),
        32: (28, 9),
        64: (34, 17),
        128: (40, 24),
        256: (45, 29),
        None: (53, 34)
    }

    def eval_expected(self, x, y):
        return sin(self.k*x)

    def eval_bc_extended(self, arg, sid):
        r, th = self.arg_to_polar(arg, sid)
        x, y = r*cos(th), r*sin(th)

        return self.eval_expected(x, y)

    def eval_d_u_outwards(self, arg, sid):
        a = self.a
        k = self.k
        R = self.R

        r, th = self.arg_to_polar(arg, sid)
        x = r*cos(th)

        if sid == 0:
            return eval_d_u_r(k, R, th)
        elif sid == 1:
            return 0
        elif sid == 2:
            return k*cos(k*x)*cos(a - np.pi/2)
