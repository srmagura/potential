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
    
    m_values = [25]

    def __init__(self, **kwargs): 
        super().__init__(**kwargs)

        self.nu = np.pi / (2*np.pi - self.a)

    def get_b(self, m):
        return 1
        #if m % 2 == 0:
        #    return 0
        #else:
        #    return 8/(2*np.pi - self.a) * 1/(m*self.nu)**3

    def eval_expected_polar(self, r, th):
        k = self.k
        R = self.R
        a = self.a
        nu = self.nu

        u = 0

        for m in self.m_values:
            b = self.get_b(m)
            u += jv(m*nu, k*r)*b*np.sin(m*nu*(th-a))/jv(m*nu, k*R)
        
        return u

    def eval_phi0(self, th):
        a = self.a
        nu = self.nu

        phi0 = 0

        for m in self.m_values:
            b = self.get_b(m)
            phi0 += b*np.sin(m*nu*(th - a))

        return phi0
        
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
