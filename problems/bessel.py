from sympy import *
import numpy as np
from scipy.special import jv

from solver import cart_to_polar
from ps.ps import PizzaSolver

from .problem import Problem, Pizza
import problems.sympy_problem as sympy_problem


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
   
    
class BesselReg(sympy_problem.SympyProblem, Pizza, Problem):

    k = 1

    def __init__(self, **kwargs): 
        kwargs['f_expr'] = get_reg_f_expr()
        super().__init__(**kwargs)
        
        self.u_asympt_lambda = lambdify(symbols('k R r th'), get_u_asympt_expr())
        
    def eval_expected(self, x, y):
        r, th = cart_to_polar(x, y)
        nu_float = 6/11
        return jv(nu_float, self.k*self.R) * np.sin(nu_float*(th-np.pi/6))
                
    def eval_bc(self, x, y):
        r, th = cart_to_polar(x, y)
        sid = self.get_sid(th)
        return self.eval_bc_extended(x, y, sid)
        
    def eval_bc_extended(self, x, y, sid):
        a = self.a
        r, th = cart_to_polar(x, y)

        if sid == 0 and th < a/2:
            th += 2*pi
        
        if sid == 0:
            bc_nc = self.eval_expected(x, y) 
        elif sid == 1 or sid == 2:
            bc_nc = 0
        
        u_asympt = self.u_asympt_lambda(self.k, self.R, r, th)
        return bc_nc - float(u_asympt)
