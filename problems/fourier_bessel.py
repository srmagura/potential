import numpy as np
from scipy.special import jv

from solver import cart_to_polar

from .problem import PizzaProblem
from .sympy_problem import SympyProblem

import problems.bdata
    
class FourierBessel(PizzaProblem):
    
    k = 1
    homogeneous = True
    #expected_known = True
    #m_max = 99
    
    M = 7

    def __init__(self, **kwargs): 
        super().__init__(**kwargs)

        self.nu = np.pi / (2*np.pi - self.a)
        #print('b(m_max) =', self.get_b(self.m_max))

        self.bdata = problems.bdata.Hat()
        self.b_coef = self.bdata.calc_coef(self.M)

    def eval_expected_polar(self, r, th):
        k = self.k
        R = self.R
        a = self.a
        nu = self.nu

        u = 0

        for m in range(self.M+1, self.m_max):
            b = self.get_b(m)
            u += jv(m*nu, k*r)*b*np.sin(m*nu*(th-a))/jv(m*nu, k*R)
        
        return u

    def eval_phi0(self, th):
        return self.bdata.eval_phi0(th) 
        
    def eval_bc_extended(self, arg, sid):
        a = self.a
        nu = self.nu
        
        r, th = self.arg_to_polar(arg, sid)
        
        if sid == 0:
            bc = self.eval_phi0(th)
            
            for m in range(1, self.M+1):
                bc -= self.b_coef[m-1] * np.sin(m*nu*(th - a))
            
            return bc
            
        elif sid == 1:
            return 0 
        elif sid == 2:
            return 0
