from sympy import *
import numpy as np
from scipy.special import jv

from solver import cart_to_polar

from .problem import PizzaProblem
from .sympy_problem import SympyProblem

import problems.bdata

def get_v_asympt_expr():
    '''
    Returns SymPy expression for the inhomogeneous part of the
    asymptotic expansion.
    '''
    k, R, r, th = symbols('k R r th')    
    kr2 = k*r/2
    
    v_asympt = 1/gamma(sympify('14/11'))
    v_asympt += -1/gamma(sympify('25/11'))*kr2**2
    v_asympt += 1/(2*gamma(sympify('36/11')))*kr2**4
    
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
    
    f = 1331/(67200 * gamma(sympify('3/11')))
    f *= 2**(8/11) * k**(69/11) * r**(47/11)
    f *= sin(-3*th/11 + pi/22)
    
    return f

class MyBData(problems.bdata.BData):

    def __init__(self, k, R, a, nu):
        self.k = k
        self.R = R
        self.a = a
        self.nu = nu

    def eval_phi0(self, th):
        k = self.k
        R = self.R
        a = self.a
        nu = self.nu

        phi0 = (th - a) / (2*np.pi - a)
        phi0 -= np.sin(nu*(th - a)/2)
        phi0 *= jv(nu/2, k*R)
        return phi0

class ShcBesselAbstract(SympyProblem, PizzaProblem):
    
    k = 1
    expected_known = False
    
    M = 7

    n_basis_dict = problems.bdata.shc_n_basis_dict

    def __init__(self, **kwargs): 
        kwargs['f_expr'] = get_reg_f_expr()
        super().__init__(**kwargs)
        
        self.v_asympt_lambda = lambdify(symbols('k R r th'), get_v_asympt_expr())
        self.nu = np.pi / (2*np.pi - self.a)

        self.bdata = MyBData(self.k, self.R, self.a, self.nu)

    def _eval_bc_extended(self, arg, sid, b_coef):
        k = self.k
        R = self.R
        a = self.a
        nu = self.nu
        
        r, th = self.arg_to_polar(arg, sid)
        
        if sid == 0:
            bc = self.bdata.eval_phi0(th)
            
            for m in range(1, self.M+1):
                bc -= b_coef[m-1] * np.sin(m*nu*(th - a))
            
            return bc
            
        elif sid == 1:
            bc = jv(nu/2, k*r)
            bc -= self.v_asympt_lambda(k, R, r, th)
            return bc

        elif sid == 2:
            return 0


class ShcBesselKnown(ShcBesselAbstract):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.b_coef = self.bdata.calc_coef(self.M)
    
    def eval_bc_extended(self, arg, sid):
        return self._eval_bc_extended(arg, sid, self.b_coef)
        
        
class ShcBessel(ShcBesselAbstract):
                
    def eval_bc_extended(self, arg, sid):
        return self._eval_bc_extended(arg, sid, [0])
