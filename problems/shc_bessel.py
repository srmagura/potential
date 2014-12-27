from sympy import *
import numpy as np
from scipy.special import jv
from scipy.integrate import quad

from solver import cart_to_polar

from .problem import PizzaProblem
from .sympy_problem import SympyProblem

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
    
    f = 1331/(67200 * gamma(3/11))
    f *= 2**(8/11) * k**(69/11) * r**(47/11)
    f *= sin(-3*th/11 + pi/22)
    
    return f
            
class ShcBesselAbstract(SympyProblem, PizzaProblem):
    
    k = 1

    def __init__(self, **kwargs): 
        kwargs['f_expr'] = get_reg_f_expr()
        super().__init__(**kwargs)
        
        self.u_asympt_lambda = lambdify(symbols('k R r th'), get_u_asympt_expr())
        self.nu = np.pi / (2*np.pi - self.a)
                
    def eval_expected_polar(self, r, th):
        k = self.k
        R = self.R
        nu = self.nu
    
        return jv(nu/2, k*r) * np.sin(nu*(th-np.pi/6)/2)
        
    def _eval_bc_extended(self, arg, sid, shc_coef):
        k = self.k
        R = self.R
        a = self.a
        nu = self.nu
        
        r, th = self.arg_to_polar(arg, sid)
        
        if sid == 0:
            u_asympt = float(self.u_asympt_lambda(k, R, R, th))
            bc = jv(3/11, k*R) / (11*np.pi/6) * (th - np.pi/6)
            bc -= np.sin(3/11 * (th-np.pi/6)) * u_asympt
            
            for m in range(len(shc_coef)):
                bc -= shc_coef[m] * jv(m*nu, k*R) * np.sin(m*nu*(th - a))
            
            return bc
            
        elif sid == 1:
            u_asympt = float(self.u_asympt_lambda(k, R, r, th))
            bc = jv(3/11, k*r) - u_asympt
            
            if th == 2*np.pi:
                return bc
            elif th == np.pi:
                return 0
            else:
                assert False
            
        elif sid == 2:
            return 0
        
    def get_restore_polar(self, r, th):
        k = self.k
        R = self.R
        a = self.a
        nu = self.nu
        
        v = self.u_asympt_lambda(k, R, r, th)
        
        for m in range(len(self.shc_coef)):
            v += self.shc_coef[m] * jv(m*nu, k*r) * np.sin(m*nu*(th - a))
    
        return v
        
    def calc_exact_shc_coef(self):
        k = self.k
        R = self.R
        
        def eval_integrand(th):
            phi = (th - np.pi/6) / (11*np.pi/6)
            phi -= np.sin(3/11*(th - np.pi/6))
            phi *= jv(3/11, k*R)
            
            return phi * np.sin(6*m/11*(th - np.pi/6))
    
        self.shc_coef = np.zeros(self.shc_coef_len)
    
        for m in range(self.shc_coef_len):
            self.shc_coef[m] = quad(eval_integrand, np.pi/6, 2*np.pi)[0]
            self.shc_coef[m] *= 12 / (11*np.pi * jv(6*m/11, k*R))

class ShcBesselKnown(ShcBesselAbstract):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calc_exact_shc_coef()
    
    def eval_bc_extended(self, arg, sid):
        return self._eval_bc_extended(arg, sid, self.shc_coef)
        
        
class ShcBessel(ShcBesselAbstract):
                
    def eval_bc_extended(self, x, y, sid):
        return self._eval_bc_extended(x, y, sid, [0])