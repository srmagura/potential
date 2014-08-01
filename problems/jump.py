from sympy import *
import numpy as np

from solver import cart_to_polar
from ps.ps import PizzaSolver

from .problem import Problem, Pizza
import problems.sympy_problem as sympy_problem

def get_L(R):
    return 2*R + 11/6*pi*R

def eval_bc_extended(R, r, th, sid): 
    L = get_L(R)

    if sid == 0:
        return R / L * (1 + th - pi/6)
    elif sid == 1:
        return 1 - r / L
    elif sid == 2:
        return r / L

class JumpNoCorrection(Pizza, Problem):
    k = 1

    solver_class = PizzaSolver
    homogeneous = True
    expected_known = False

    def eval_f(self, x, y):
        return 0
        
    def eval_bc(self, x, y):
        r, th = cart_to_polar(x, y)
        sid = Pizza.get_sid(th)
        return eval_bc_extended(self.R, r, th, sid)
        
    def eval_bc_extended(self, x, y, sid):
        r, th = cart_to_polar(x, y)
        return eval_bc_extended(self.R, r, th, sid)

def get_u04_expr():
    k, R, r, th = symbols('k R r th')
    l = 1 / get_L(R)       
    k2 = k**2
    r2 = r**2
    
    phi = th - pi/6
    
    u04 = 44*k2*r2*pi*sqrt(3) * (k2*r2 - 12) * sin(2*phi)
    u04 += -11*pi*sqrt(3)*sin(4*phi) * k**4*r**4
    u04 += -396*pi*sin(phi)*l*r*(k2*r2 - 8)*sqrt(3)
    u04 += -1584*pi*sin(phi)**2 * cos(phi) * k2*l*r**3
    u04 += 1584*l*(k2*r2*cos(phi)**2 - 3/4*k2*r2 + 4)*pi*r*sin(phi)
    u04 += 3168*pi*cos(phi)*l*r
    u04 += 27*phi*(k2*r2 - 8)**2
    
    u04 *= 1 / (3168*pi)
    return u04   
        
class JumpReg(Pizza, Problem):
    k = 1

    expected_known = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def eval_bc(self, x, y):
        r, th = cart_to_polar(x, y)
        sid = Pizza.get_sid(th)
        return self.eval_bc_extended(x, y, sid)
        
    def eval_bc_extended(self, x, y, sid):
        a = self.a
        r, th = cart_to_polar(x, y)
        
        if sid != 1 and not 'from1':
            return 1
        
        if sid == 0 and th < a/2:
            th += 2*pi 
        elif sid == 1 and x < 0:    
            return self.eval_bc_extended(-x*cos(a), -x*sin(a), 2)
        elif sid == 2 and x < 0:
            return self.eval_bc_extended(r, 0, 1)
        
        bc_nc = eval_bc_extended(self.R, r, th, sid)
        u04 = self.u04_lambda(self.k, self.R, r, th)

        return float(bc_nc - u04)
        #return bc_nc
        #return u04
                 
        
class JumpReg0(sympy_problem.SympyProblem, JumpReg):

    def __init__(self, **kwargs): 
        kwargs['u_expr'] = -get_u04_expr()
        super().__init__(**kwargs)

