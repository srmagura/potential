import numpy as np
from numpy import cos, sin

from solver import cart_to_polar
from ps.ps import PizzaSolver

class Problem:
    homogeneous = False
    expected_known = True

    def get_solver(self, *args, **kwargs):
        return self.solver_class(self, *args, **kwargs)
        
    def get_n_basis(self, N):
        if N in self.n_basis_dict:
            return self.n_basis_dict[N]
        else:
            return self.n_basis_dict[None]

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
    
    n_basis_dict = {16: (17, 12), 32: (23, 19), 64: (32, 27), 
        128: (45, 37), None: (55, 50)}   

    def eval_bc(self, x, y):
        return self.eval_expected(x, y)
        
    def eval_bc_extended(self, x, y, sid):
        return self.eval_bc(x, y)

    @staticmethod
    def get_sid(th):
        tol = 1e-12
        a = Pizza.a

        if th >= 2*np.pi:
            return 1
        elif th > a:
            return 0
        elif abs(th - a) < tol:
            return 2

        assert False
