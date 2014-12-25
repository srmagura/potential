import numpy as np
from numpy import cos, sin

from solver import cart_to_polar
from cs.csf import CsFourier

from .problem import Problem, PizzaProblem

class Sine(Problem):
    
    k = 1
    solver_class = CsFourier
    homogeneous = True
    
    def eval_bc(self, th):
        return self.eval_expected(self.R*cos(th), self.R*sin(th))

    def eval_expected(self, x, y):
        return sin(self.k*x)

    def eval_d_u_r(self, th):
        k = self.k
        R = self.R
        return k*cos(th) * cos(k*R*cos(th))
        

class SinePizza(PizzaProblem):
    
    k = 1
    homogeneous = True
    
    def eval_expected(self, x, y):
        return sin(self.k*x)
    
    def eval_bc_extended(self, arg, sid):
        arg, sid = self.wrap_func(arg, sid)
        r, th = self.arg_to_polar(arg, sid) 
 
        x, y = r*cos(th), r*sin(th)    
        return self.eval_expected(x, y)
    
    # FIXME    
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
