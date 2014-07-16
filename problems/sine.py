import numpy as np
from numpy import cos, sin

from solver import cart_to_polar
from cs.csf import CsFourier

from .problem import Problem, Pizza

class Sine(Problem):
    k = 1

    solver_class = CsFourier
    homogeneous = True

    def eval_f(self, x, y):
        return 0

    def eval_expected(self, x, y):
        return sin(self.k*x)

    def eval_d_u_r(self, th):
        k = self.k
        R = self.R
        return k*cos(th) * cos(k*R*cos(th))
        

class SinePizza(Pizza, Sine):

    def eval_d_u_outwards(self, x, y, **kwargs):
        r, th = cart_to_polar(x, y)

        if 'sid' in kwargs:
            sid = kwargs['sid']
        else:
            sid = self.get_sid(th)

        a = self.a
        k = self.k

        if sid == 0:
            return super().eval_d_u_r(th)
        elif sid == 1:
            return 0
        elif sid == 2:
            return k*cos(k*x)*cos(a - np.pi/2)
