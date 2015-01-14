from sympy import mpmath
import numpy as np

from solver import cart_to_polar

from .problem import PizzaProblem
from .sympy_problem import SympyProblem

mpmath.mp.dps = 15
mpf = mpmath.mpf
    
class FourierBesselMpMath(PizzaProblem):
    
    k = mpf('1')
    homogeneous = True
    expected_known = False
    
    shc_coef_len = 20
    many_shc_coef_len = 200

    def __init__(self, **kwargs): 
        super().__init__(**kwargs)

        self.mp_a = mpmath.pi / 6
        self.nu = mpmath.pi / (2*mpmath.pi - self.mp_a)
        self.calc_exact_shc_coef()

    def eval_expected_polar(self, r, th):
        k = self.k
        R = self.R
        a = self.mp_a
        nu = self.nu
    
        u_reg = 0

        for m in range(len(self.shc_coef)+1, len(self.many_shc_coef)+1):
            u_reg += self.a_coef[m-1] * mpmath.besselj(m*nu, k*r)* mpmath.sin(m*nu*(th - a))

        return u_reg

    def eval_phi0(self, th):
        return -(th - mpmath.pi/6) * (th - 2*mpmath.pi) 
        
    def eval_bc_extended(self, arg, sid):
        a = self.a
        nu = self.nu
        
        r, th = self.arg_to_polar(arg, sid)
        
        if sid == 0:
            bc = float(self.eval_phi0(th))
            
            for m in range(1, len(self.shc_coef)+1):
                bc -= float(self.shc_coef[m-1]) * np.sin(m*float(nu)*(th - a))
            
            return bc
            
        elif sid == 1:
            return 0 
        elif sid == 2:
            return 0
        
    def calc_exact_shc_coef(self):
        k = self.k
        R = self.R
        a = self.mp_a
        nu = self.nu
        
        def eval_integrand(th):
            phi = self.eval_phi0(th)
            return phi * mpmath.sin(m*nu*(th - a))
    
        self.shc_coef = [0] * self.shc_coef_len
        self.many_shc_coef = [0] * self.many_shc_coef_len
        self.a_coef = [0] * self.many_shc_coef_len
    
        quad_error = []

        if self.expected_known:
            count = self.many_shc_coef_len
        else:
            count = self.shc_coef_len

        for m in range(1, count+1):
            quad_result = mpmath.quad(eval_integrand,
                [mpmath.pi/6, 2*mpmath.pi], 
                maxdegree=10, error=True)
            quad_error.append(quad_result[1])
            
            coef = quad_result[0]
            coef *= 2 / (2*mpmath.pi - a)

            self.many_shc_coef[m-1] = coef
            self.a_coef[m-1] = coef / mpmath.besselj(m*nu, k*R)

            if m < self.shc_coef_len + 1:
                self.shc_coef[m-1] = coef

        print('quad error:', max(quad_error))
