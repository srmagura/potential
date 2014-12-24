from sympy import *
import numpy as np
from scipy.special import jv

from solver import cart_to_polar

from .problem import Problem, Pizza
import problems.sympy_problem as sympy_problem

def get_u_asympt_expr():
    '''
    Returns SymPy expression for the inhomogeneous part of the
    asymptotic expansion.
    '''
    k, R, r, th = symbols('k R r th')    
    kr2 = k*r/2
    
    u_asympt = 1/gamma(14/11)
    u_asympt += -1/gamma(25/11)*kr2**2
    u_asympt += 1/(2*gamma(36/11))*kr2**4
    
    u_asympt *= sin(3/11*(th-pi/6)) * kr2**(3/11)
    
    return u_asympt
    
def get_reg_f_expr():
    '''
    Returns SymPy expression that represents the RHS of the regularized
    problem. This expression is the result of applying the Helmholtz
    operator to the inhomogeneous asymptotic expansion; see this module's
    get_u_asympt_expr().
    '''
    k, R, r, th = symbols('k R r th')
    
    f = -1331/(67200 * gamma(3/11))
    f *= 2**(8/11) * k**(69/11) * r**(47/11)
    f *= sin(-3*th/11 + pi/22)
    
    return f
    
class ShcBesselAbstract(sympy_problem.SympyProblem, Pizza, Problem):
    
    k = 1

    def __init__(self, **kwargs): 
        kwargs['f_expr'] = get_reg_f_expr()
        super().__init__(**kwargs)
        
        self.u_asympt_lambda = lambdify(symbols('k R r th'), get_u_asympt_expr())
                
    def eval_expected_polar(self, r, th):
        k = self.k
        R = self.R
        
        nu = 6/11
        return jv(nu/2, k*r) * np.sin(nu*(th-np.pi/6)/2)
        
    def _eval_bc_extended(self, x, y, sid, d0):
        k = self.k
        R = self.R
        
        r, th = cart_to_polar(x, y)
        
        if sid == 0:
            u_asympt = self.u_asympt_lambda(k, R, R, th)
            bc = jv(3/11, k*R) / (11*np.pi/6) * (th - np.pi/6)
            bc -= np.sin(3/11 * (th-np.pi/6)) * u_asympt
            return bc
            
        elif sid == 1:
            u_asympt = self.u_asympt_lambda(k, R, r, th)
            return jv(3/11, k*r) - u_asympt
            
        elif sid == 2:
            return 0
        
    def get_restore_polar(self, r, th):
        return self.u_asympt_lambda(self.k, self.R, r, th)

class ShcBesselKnown(ShcBessel):
    
    def eval_bc_extended(self, x, y, sid):
        return self._eval_bc_extended(x, y, sid, #)
        
class ShcBessel(ShcBesselAbstract):
                
    def eval_bc_extended(self, x, y, sid):
        return self._eval_bc_extended(x, y, sid, [0])
