import numpy as np
from numpy import cos, sin

from cs.csf import CsFourier
from .problem import Problem, PizzaProblem

def eval_d_u_r(k, R, th):
    return k*cos(th) * cos(k*R*cos(th))

class Sine(Problem):
    
    k = 1
    solver_class = CsFourier
    homogeneous = True
    expected_known = True
    
    def eval_bc(self, th):
        return self.eval_expected(self.R*cos(th), self.R*sin(th))

    def eval_expected(self, x, y):
        return sin(self.k*x)

    def eval_d_u_r(self, th):
        return eval_d_u_r(self.k, self.R, th)
        

class SinePizza(PizzaProblem):
    
    k = 1
    homogeneous = True
    expected_known = True
    
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
