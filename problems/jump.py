import numpy as np
from numpy import sin, cos, pi, sqrt

from solver import cart_to_polar
from ps.ps import PizzaSolver

from .problem import Problem, Pizza

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
        

def eval_u04(k, R, r, th):
    if r == 0:
        print('u04 received r=0')
        return None

    l = 1 / get_L(R)
    phi = th + pi / 6
    k2 = k**2
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
    
def eval_reg_f(k, R, r, th):
    l = 1 / get_L(R)
    phi = th + pi / 6
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
    
    
class RegF:

    def eval_f_polar(self, r, th):
        return eval_reg_f(self.k, self.R, r, th)
        
    def eval_d_f_r(self, r, th):
        return 0
        
    def eval_d2_f_r(self, r, th):
        return 0
        
    def eval_d_f_th(self, r, th):
        return 0
        
    def eval_d2_f_th(self, r, th):
        return 0
        
    def eval_grad_f(self, x, y):
        return np.array((0, 0))
        
    def eval_hessian_f(self, x, y):
        return np.array(((0, 0), (0, 0)))
        
  
class u04TestProblem(Pizza, RegF, Problem): 
    k = 1

    def eval_expected(self, x, y):
        r, th = cart_to_polar(x, y)
        return -eval_u04(self.k, self.R, r, th)
        
    
class JumpReg(Pizza, RegF, Problem):
    k = 1

    expected_known = False
        
    def eval_bc(self, x, y):
        sid = Pizza.get_sid(th)
        return self.eval_bc_extended(x, y, sid)
        
    def eval_bc_extended(self, x, y, sid):
        r, th = cart_to_polar(x, y)
        
        bc_nc = eval_bc_extended(self.R, r, th, sid)
        u04 = eval_u04(self.k, self.R, r, th)
        
        return bc_nc - u04

