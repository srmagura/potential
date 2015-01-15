import numpy as np
from scipy.integrate import quad
from scipy.special import jv

from solver import cart_to_polar

from .problem import PizzaProblem
from .sympy_problem import SympyProblem
    
class FourierBesselSimple(PizzaProblem):
    
    k = 1
    homogeneous = True
    expected_known = True
    
    shc_coef_len = 7

    def __init__(self, **kwargs): 
        super().__init__(**kwargs)

        self.nu = np.pi / (2*np.pi - self.a)

    def eval_expected_polar(self, r, th):
        k = self.k
        R = self.R
        a = self.a
        nu = self.nu

        m = self.shc_coef_len + 1

        u = jv(m*nu, k*r)*np.sin(m*nu*(th-a))
        u /= jv(m*nu, k*R)
        
        return u

    def eval_phi0(self, th):
        a = self.a
        nu = self.nu

        m = self.shc_coef_len + 1
        return np.sin(m*nu*(th - a))
        
    def eval_bc_extended(self, arg, sid):
        a = self.a
        nu = self.nu
        
        r, th = self.arg_to_polar(arg, sid)
        
        if sid == 0:
            return self.eval_phi0(th)
        elif sid == 1:
            return 0 
        elif sid == 2:
            return 0
