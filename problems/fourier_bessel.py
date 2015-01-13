from sympy import *
import numpy as np
from scipy.special import jv
from scipy.integrate import quad

from solver import cart_to_polar

from .problem import PizzaProblem
from .sympy_problem import SympyProblem

mpmath.mp.dps = 60
    
class FourierBessel(PizzaProblem):
    
    k = 1
    homogeneous = True
    expected_known = False
    
    shc_coef_len = 15

    def __init__(self, **kwargs): 
        super().__init__(**kwargs)

        self.nu = np.pi / (2*np.pi - self.a)
        self.calc_exact_shc_coef()

    def eval_phi0(self, th):
        return -(th - np.pi/6) * (th - 2*np.pi) 
        
    def eval_bc_extended(self, arg, sid):
        a = self.a
        nu = self.nu
        
        r, th = self.arg_to_polar(arg, sid)
        
        if sid == 0:
            bc = self.eval_phi0(th)
            
            for m in range(1, len(self.shc_coef)+1):
                bc -= self.shc_coef[m-1] * np.sin(m*nu*(th - a))
            
            return bc
            
        elif sid == 1:
            return 0 
        elif sid == 2:
            return 0
        
    def calc_exact_shc_coef(self):
        k = self.k
        R = self.R
        
        def eval_integrand(th):
            phi = self.eval_phi0(th)
            return phi * np.sin(6*m/11*(th - np.pi/6))
    
        self.shc_coef = np.zeros(self.shc_coef_len)
    
        quad_error = []

        for m in range(1, len(self.shc_coef)+1):
            quad_result = quad(eval_integrand, np.pi/6, 2*np.pi, epsabs=1e-10)
            quad_error.append(quad_result[1])
            
            coef = quad_result[0]
            coef *= 12 / (11*np.pi)

            self.shc_coef[m-1] = coef

        print('quad_error:', max(quad_error))
