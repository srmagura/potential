from sympy import *
import numpy as np
from scipy.special import jv
from scipy.integrate import quad

from solver import cart_to_polar

from .problem import PizzaProblem
from .sympy_problem import SympyProblem

def get_v_asympt_expr():
    '''
    Returns SymPy expression for the inhomogeneous part of the
    asymptotic expansion.
    '''
    k, R, r, th = symbols('k R r th')    
    kr2 = k*r/2
    
    v_asympt = 1/gamma(14/11)
    v_asympt += -1/gamma(25/11)*kr2**2
    v_asympt += 1/(2*gamma(36/11))*kr2**4
    
    v_asympt *= sin(3/11*(th-pi/6)) * kr2**(3/11)
    
    return v_asympt
    
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
    expected_known = True
    many_shc_coef_len = 50

    def __init__(self, **kwargs): 
        kwargs['f_expr'] = get_reg_f_expr()
        super().__init__(**kwargs)
        
        self.v_asympt_lambda = lambdify(symbols('k R r th'), get_v_asympt_expr())
        self.nu = np.pi / (2*np.pi - self.a)
                
    def eval_expected_polar(self, r, th):
        k = self.k
        R = self.R
        nu = self.nu
    
        # v_reg
        u_reg = jv(nu/2, k*r) * np.sin(nu*(th-np.pi/6)/2)
        u_reg -= self.v_asympt_lambda(k, R, r, th)

        return u_reg

    def eval_phi0(self, th):
        k = self.k
        R = self.R

        return jv(3/11, k*R) / (11*np.pi/6) * (th - np.pi/6)

        # Degenerate
        #return jv(3/11, k*R) * np.sin(3/11*(th - np.pi/6))
        
    def _eval_bc_extended(self, arg, sid, shc_coef):
        k = self.k
        R = self.R
        a = self.a
        nu = self.nu
        
        r, th = self.arg_to_polar(arg, sid)
        
        if sid == 0:
            v_asympt = float(self.v_asympt_lambda(k, R, R, th))
            bc = self.eval_phi0(th)
            bc -= v_asympt
            
            for m in range(len(shc_coef)):
                bc -= shc_coef[m] * jv(m*nu, k*R) * np.sin(m*nu*(th - a))
            
            return bc
            
        elif sid == 1:
            v_asympt = float(self.v_asympt_lambda(k, R, r, th))
            bc = jv(3/11, k*r) - v_asympt
            
            if th == 2*np.pi:
                return bc
            elif th == np.pi:
                return 0
            else:
                assert False
            
        elif sid == 2:
            return 0
        
    def calc_exact_shc_coef(self):
        k = self.k
        R = self.R
        
        def eval_integrand(th):
            phi = self.eval_phi0(th)
            phi -= jv(3/11, k*R) * np.sin(3/11*(th - np.pi/6))
            
            return phi * np.sin(6*m/11*(th - np.pi/6))
    
        self.shc_coef = np.zeros(self.shc_coef_len)
        self.many_shc_coef = np.zeros(self.many_shc_coef_len)
    
        for m in range(1, self.many_shc_coef_len+1):
            coef = quad(eval_integrand, np.pi/6, 2*np.pi)[0]
            coef *= 12 / (11*np.pi * jv(6*m/11, k*R))

            self.many_shc_coef[m-1] = coef

            if m < self.shc_coef_len + 1:
                self.shc_coef[m-1] = coef


class ShcBesselKnown(ShcBesselAbstract):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calc_exact_shc_coef()
        print(self.many_shc_coef)
    
    def eval_bc_extended(self, arg, sid):
        return self._eval_bc_extended(arg, sid, self.shc_coef)
        
        
class ShcBessel(ShcBesselAbstract):
                
    def eval_bc_extended(self, arg, sid):
        return self._eval_bc_extended(arg, sid, [0])
