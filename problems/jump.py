from sympy import *
import numpy as np

from solver import cart_to_polar
from ps.ps import PizzaSolver

from .problem import Problem, Pizza
import problems.sympy_f as sympy_f

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
    k, R, r, phi = symbols('k R r phi')
    l = 1 / get_L(R)       
    k2 = k**2
    
    r, phi = symbols('r phi')
    r2 = r**2
    
    u04 = 44*k2*r2*pi*sqrt(3) * (k2*r2 - 12) * sin(2*phi)
    u04 += -11*pi*sqrt(3)*sin(4*phi) * k**4*r**4
    u04 += -396*pi*sin(phi)*l*r*(k2*r2 - 8)*sqrt(3)
    u04 += -1584*pi*sin(phi)**2 * cos(phi) * k2*l*r**3
    u04 += 1584*l*(k2*r2*cos(phi)**2 - 3/4*k2*r2 + 4)*pi*r*sin(phi)
    u04 += 3168*pi*cos(phi)*l*r
    u04 += 27*phi*(k2*r2 - 8)**2
    
    u04 *= 1 / (3168*pi)
    return u04
    
def get_reg_f_expr():
    k, R, r, phi = symbols('k R r phi')
    l = 1 / get_L(R)
    k2 = k**2
    r2 = r**2
    
    f = k**4*r**3*pi*sqrt(3)*sin(2*phi)
    f += -1/4*pi*sqrt(3)*sin(4*phi)*k**4*r**3
    f += -9*pi*sin(phi)*l*r2*k2*sqrt(3)
    f += 72*pi*sin(phi)**3*l
    f += -36*pi*cos(phi)*l*(k2*r2 + 2)*sin(phi)**2 
    f += 36*pi*l*((k2*r2 + 2)*cos(phi)**2 - 3/4*k2*r2 - 2)*sin(phi)
    f += 27*k**4*phi*r**3/44
    f += -72*pi*cos(phi)**3*l 
    f += 72*pi*cos(phi)*l

    f *= - k2*r/(72*pi)
    return f
    
def eval_phi(th):
    return th - pi/6
    
        
class JumpReg(Pizza, Problem):
    k = 1

    expected_known = False
    
    def __init__(self):
        super().__init__()
        
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
        u04 = self.u04_lambda(self.k, self.R, r, eval_phi(th))

        return float(bc_nc - u04)
        #return bc_nc
        #return u04
        
    
class RegF0(sympy_f.SympyF):

    def __init__(self):
        kwargs = {'f_expr': get_reg_f_expr()}
        super().__init__(**kwargs)
        
        
class JumpReg0(RegF0, JumpReg):

    def __init__(self): 
        super().__init__()
             
        k, R, r, phi = symbols('k R r phi')     
        self.u04_lambda = lambdify((k, R, r, phi), get_u04_expr())

