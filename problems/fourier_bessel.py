import numpy as np
from scipy.special import jv, jvp
from sympy import mpmath

from solver import cart_to_polar

from .problem import PizzaProblem
from .sympy_problem import SympyProblem

import problems.bdata

mpmath.mp.dps = 15
    
class FourierBessel(PizzaProblem):
    
    k = 1
    homogeneous = True
    expected_known = True
    m_max = 99
    
    M = 7

    def __init__(self, **kwargs): 
        super().__init__(**kwargs)

        self.nu = np.pi / (2*np.pi - self.a)

        self.bdata = problems.bdata.Hat()

        if self.expected_known:
            m = self.m_max
        else:
            m = self.M

        self.b_coef = self.bdata.calc_coef(m)

        if self.expected_known:
            self.bessel_R = np.zeros(len(self.b_coef))

            for m in range(1, len(self.b_coef)+1):
                self.bessel_R[m-1] = jv(m*self.nu, self.k*self.R)

            print('b[m_max] =', self.b_coef[self.m_max-1])

    def calc_bessel_ratio(self, m, r):
        k = self.k
        R = self.R
        nu = self.nu

        #ratio = mpmath.besselj(m*nu, k*r) / mpmath.besselj(m*nu, k*R)
        ratio = jv(m*nu, k*r) / self.bessel_R[m-1]
        return float(ratio)

    def eval_expected_polar(self, r, th):
        a = self.a
        nu = self.nu

        u = 0

        for m in range(self.M+1, self.m_max+1):
            b = self.b_coef[m-1]
            ratio = self.calc_bessel_ratio(m, r)
            u += b * ratio * np.sin(m*nu*(th-a))
        
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

    def calc_d_bessel_R(self):
        self.d_bessel_R = np.zeros(len(self.b_coef))

        for m in range(1, len(self.b_coef)+1):
            self.d_bessel_R[m-1] = self.k * jvp(m*self.nu, self.k*self.R, 1)

    def eval_d_u_outwards(self, arg, sid):
        if not hasattr(self, 'd_bessel_R'):
            self.calc_d_bessel_R()

        a = self.a
        nu = self.nu
        k = self.k

        r, th = self.arg_to_polar(arg, sid)

        d_u_n = 0
        if sid == 0:
            for m in range(self.M+1, self.m_max+1):
                ratio = self.d_bessel_R[m-1] / self.bessel_R[m-1]
                b = self.b_coef[m-1]
                d_u_n += ratio * b * np.sin(m*nu*(th-a))

        else:
            for m in range(self.M+1, self.m_max+1):
                ratio = jv(m*nu, k*r) / self.bessel_R[m-1]
                b = self.b_coef[m-1]
                d_u_n += ratio * b * (m*nu)

            if sid == 2:
                d_u_n *= -1

        return d_u_n
