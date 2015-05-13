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

    def __init__(self, problem):
        self.problem = problem

    def eval_phi0(self, th):
        R = self.problem.R
        phi0 = self.problem.eval_phi0(R, th)
        phi0 -= self.problem.eval_v(R, th)
        return phi0

class ShcBesselAbstract(SympyProblem, PizzaProblem):
    
    k = 1
    expected_known = False
    
    M = 7

    n_basis_dict = {
        16: (14, 7), 
        32: (26, 13), 
        64: (30, 19), 
        128: (29, 11), 
        256: (65, 37),
        512: (90, 50),
    }

    def __init__(self, **kwargs): 
        kwargs['f_expr'] = get_reg_f_expr()
        super().__init__(**kwargs)
        
        self.v_asympt_lambda = lambdify(symbols('k R r th'), get_v_asympt_expr())
        self.nu = np.pi / (2*np.pi - self.a)

        self.bdata = MyBData(self)

    def eval_v(self, r, th):
        k = self.k
        nu = self.nu
        a = self.a

        return jv(nu/2, k*r) * np.sin(nu*(th - a)/2)

    def eval_phi0(self, r, th):
        nu = self.nu
        a = self.a

        phi0 = self.eval_v(r, th)
        phi0 += np.sin(2*nu*(th - a))
        phi0 += np.sin(8*nu*(th - a))
        return phi0

    def _eval_bc_extended(self, arg, sid, b_coef):
        k = self.k
        R = self.R
        a = self.a
        nu = self.nu
        
        r, th = self.arg_to_polar(arg, sid)
        
        if sid == 0:
            bc = self.eval_phi0(r, th)
            bc -= self.v_asympt_lambda(k, R, r, th)
            
            for m in range(1, self.M+1):
                bc -= b_coef[m-1] * np.sin(m*nu*(th - a))
            
            return bc
            
        elif sid == 1:
            if th < 1.5*np.pi:
                return 0

            bc = self.eval_v(r, th)
            bc -= self.v_asympt_lambda(k, R, r, th)
            return bc

        elif sid == 2:
            return 0


class ShcBesselKnown(ShcBesselAbstract):

    expected_known = True
    m_max = 13

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.expected_known:
            m = self.m_max
        else:
            m = self.M

        self.b_coef = self.bdata.calc_coef(m)
        #print(self.b_coef)

        if self.expected_known:
            self.bessel_R = np.zeros(len(self.b_coef))

            for m in range(1, len(self.b_coef)+1):
                self.bessel_R[m-1] = jv(m*self.nu, self.k*self.R)

        #    print('ShcBesselKnown: b_coef[m_max] =', self.b_coef[self.m_max-1])

    def calc_bessel_ratio(self, m, r):
        k = self.k
        R = self.R
        nu = self.nu

        ratio = jv(m*nu, k*r) / self.bessel_R[m-1]
        return float(ratio)

    def eval_expected_polar(self, r, th):
        a = self.a
        k = self.k
        R = self.R
        nu = self.nu

        v = self.eval_v(r, th)
        u = v - self.v_asympt_lambda(k, R, r, th)

        for m in range(self.M+1, self.m_max+1):
            b = self.b_coef[m-1]
            ratio = self.calc_bessel_ratio(m, r)
            u += b * ratio * np.sin(m*nu*(th-a))

        return u
    
    def eval_bc_extended(self, arg, sid):
        return self._eval_bc_extended(arg, sid, self.b_coef[:self.M])
        
        
class ShcBessel(ShcBesselAbstract):
                
    def eval_bc_extended(self, arg, sid):
        return self._eval_bc_extended(arg, sid, [0])
