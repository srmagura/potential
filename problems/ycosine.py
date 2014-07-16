import numpy as np
from numpy import cos, sin

from solver import cart_to_polar
from cs.csf import CsFourier

from .problem import Problem, Pizza

class YCosine(Problem):
    k = 1
    solver_class = CsFourier

    def eval_expected(self, x, y):
        k = self.k
        return y*cos(x)

    def eval_f(self, x, y):
        k = self.k
        return k**2*y*cos(x) - y*cos(x)

    def eval_d_f_r(self, r, th):
        k = self.k
        return -k**2*r*sin(th)*sin(r*cos(th))*cos(th) + k**2*sin(th)*cos(r*cos(th)) + r*sin(th)*sin(r*cos(th))*cos(th) - sin(th)*cos(r*cos(th))

    def eval_d2_f_r(self, r, th):
        k = self.k
        return (-k**2*r*cos(th)*cos(r*cos(th)) - 2*k**2*sin(r*cos(th)) + r*cos(th)*cos(r*cos(th)) + 2*sin(r*cos(th)))*sin(th)*cos(th)

    def eval_d_f_th(self, r, th):
        k = self.k
        return k**2*r**2*sin(th)**2*sin(r*cos(th)) + k**2*r*cos(th)*cos(r*cos(th)) - r**2*sin(th)**2*sin(r*cos(th)) - r*cos(th)*cos(r*cos(th))

    def eval_d2_f_th(self, r, th):
        k = self.k
        return r*(-k**2*r**2*sin(th)**2*cos(r*cos(th)) + 3*k**2*r*sin(r*cos(th))*cos(th) - k**2*cos(r*cos(th)) + r**2*sin(th)**2*cos(r*cos(th)) - 3*r*sin(r*cos(th))*cos(th) + cos(r*cos(th)))*sin(th)

    def eval_d2_f_r_th(self, r, th):
        k = self.k
        return k**2*r**2*sin(th)**2*cos(th)*cos(r*cos(th)) + 2*k**2*r*sin(th)**2*sin(r*cos(th)) - k**2*r*sin(r*cos(th))*cos(th)**2 + k**2*cos(th)*cos(r*cos(th)) - r**2*sin(th)**2*cos(th)*cos(r*cos(th)) - 2*r*sin(th)**2*sin(r*cos(th)) + r*sin(r*cos(th))*cos(th)**2 - cos(th)*cos(r*cos(th))

    def eval_grad_f(self, x, y):
        k = self.k
        return (-k**2*y*sin(x) + y*sin(x), k**2*cos(x) - cos(x))

    def eval_hessian_f(self, x, y):
        k = self.k
        return ((y*(-k**2 + 1)*cos(x), (-k**2 + 1)*sin(x)), ((-k**2 + 1)*sin(x), 0))

