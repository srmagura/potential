from sympy import *
import numpy as np
from scipy.special import jv

from solver import cart_to_polar
from ps.ps import PizzaSolver

from .problem import Problem, Pizza
import problems.sympy_problem as sympy_problem

shared_k = 1

def get_u_asympt_expr():
    k, R, r, th = symbols('k R r th')
    nu = sympify('6/11')
    
    kr2 = k*r/2
    
    u_asympt = 1/gamma(nu+1)
    u_asympt += -1/gamma(nu+2)*kr2**2
    u_asympt += 1/(2*gamma(nu+3))*kr2**4
    u_asympt *= sin(nu*(th-pi/6)) * kr2**nu
    
    return u_asympt
    
def get_reg_f_expr():
    k, R, r, th = symbols('k R r th')
    nu = sympify('6/11')
    
    f = 1331/(182784*gamma(nu)) * cos(nu*th+9*pi/22)
    f *= 2**(5/11) * k**(72/11) * r**(50/11)
    
    return f

def eval_expected_polar(k, R, r, th):    
    nu_float = 6/11
    return jv(nu_float, k*r) * np.sin(nu_float*(th-np.pi/6))

def eval_bc_nc_extended(x, y, sid, a, k, R):
    r, th = cart_to_polar(x, y)

    if sid == 0 and th < a/2:
        th += 2*np.pi 
    elif sid == 1 and x < 0:    
        return eval_bc_nc_extended(-x*cos(a), -x*sin(a), 2, a, k, R)
    elif sid == 2 and x < 0:
        return eval_bc_nc_extended(r, 0, 1, a, k, R) 
        
    if sid == 0:
        bc_nc = eval_expected_polar(k, R, r, th) 
    elif sid == 1 or sid == 2:
        bc_nc = 0
        
    return bc_nc


class BesselNoCorrection(Pizza, Problem):

    k = shared_k
    homogeneous = True

    def eval_f(self, x, y):
        return 0
                
    def eval_expected_polar(self, r, th):
        return eval_expected_polar(self.k, self.R, r, th)
                
    def eval_bc_extended(self, x, y, sid):
        return eval_bc_nc_extended(x, y, sid, self.a, self.k, self.R)
    
    
class BesselReg(sympy_problem.SympyProblem, Pizza, Problem):
    
    k = shared_k

    def __init__(self, **kwargs): 
        kwargs['f_expr'] = get_reg_f_expr()
        super().__init__(**kwargs)
        
        self.u_asympt_lambda = lambdify(symbols('k R r th'), get_u_asympt_expr())
                
    def eval_expected_polar(self, r, th):
        u_asympt = float(self.u_asympt_lambda(self.k, self.R, r, th))
        return eval_expected_polar(self.k, self.R, r, th) - u_asympt
        
    def eval_bc_extended(self, x, y, sid):
        a = self.a
        k = self.k
        R = self.R
        
        r, th = cart_to_polar(x, y)
        
        bc_nc = eval_bc_nc_extended(x, y, sid, a, k, R)
        u_asympt = self.u_asympt_lambda(k, R, r, th)
        
        return bc_nc - float(u_asympt)
        
class Bessel(BesselReg):
    
    def eval_expected_polar(self, r, th):
        return eval_expected_polar(self.k, self.R, r, th)
        
    def get_restore_polar(self, r, th):
        return self.u_asympt_lambda(self.k, self.R, r, th)
