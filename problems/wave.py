import numpy as np

from solver import cart_to_polar
from cs.csf import CsFourier

from .problem import Problem, Pizza

class Wave(Problem):
    k = 1 
    kx = .8*k
    ky = .6*k

    solver_class = CsFourier
    homogeneous = True

    def eval_f(self, x, y):
        return 0

    def eval_expected(self, x, y):
        return np.exp(complex(0, self.kx*x + self.ky*y))

    def eval_d_u_r(self, th):
        a = self.kx*cos(th) + self.ky*sin(th)
        return 1j*a*np.exp(1j*self.R*a)


class WavePizza(Pizza, Wave):
    pass
