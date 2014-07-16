import numpy as np
from numpy import cos, sin

from solver import cart_to_polar
from ps.ps import PizzaSolver

class Problem:
    homogeneous = False

    def get_solver(self, *args, **kwargs):
        return self.solver_class(self, *args, **kwargs)

    def eval_bc(self, th):
        return self.eval_expected(self.R*cos(th), self.R*sin(th))

    def eval_expected(self, x, y):
        return self.eval_expected_polar(*cart_to_polar(x, y))

    def eval_f(self, x, y):
        return self.eval_f_polar(*cart_to_polar(x, y))

    def eval_f_polar(self, r, th):
        x = r * cos(th)
        y = r * sin(th)
        return self.eval_f(x, y)

       
class Pizza:
    a = np.pi / 6
    solver_class = PizzaSolver

    def eval_bc(self, x, y):
        return self.eval_expected(x, y)

    def get_sid(self, th):
        tol = 1e-12

        if th > self.a:
            return 0
        elif abs(th) < tol:
            return 1
        elif abs(th - self.a) < tol:
            return 2

        assert False