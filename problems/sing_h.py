import numpy as np
from scipy.special import jv, jvp

from solver import cart_to_polar

from .problem import PizzaProblem
from .sympy_problem import SympyProblem

import problems.bdata

    
class SingH(PizzaProblem):

    #R = 10
    #AD_len = 24
    R = 2.3
    AD_len = 2*np.pi
    
    k = 1
    homogeneous = True
    expected_known = False
    force_relative = True
    
    M = 7

    n_basis_dict = {
        16: (14, 7), 
        32: (26, 13), 
        64: (30, 19), 
        128: (45, 34), 
        256: (65, 34), 
        512: (90, 34), 
    }

    def __init__(self, **kwargs): 
        super().__init__(**kwargs)

        self.nu = np.pi / (2*np.pi - self.a)
        
        if self.expected_known:
            m = self.m_max
        else:
            m = self.M

        self.b_coef = self.bdata.calc_coef(m)

        if self.expected_known:
            self.bessel_R = np.zeros(len(self.b_coef))

            for m in range(1, len(self.b_coef)+1):
                self.bessel_R[m-1] = jv(m*self.nu, self.k*self.R)

            print('FourierBessel: b_coef[m_max] =', self.b_coef[self.m_max-1])

    def calc_bessel_ratio(self, m, r):
        k = self.k
        R = self.R
        nu = self.nu

        ratio = jv(m*nu, k*r) / self.bessel_R[m-1]
        return float(ratio)

    def eval_expected_polar(self, r, th, restored=False, setype=None):
        a = self.a
        nu = self.nu

        if setype is not None:
            if setype[0] == 1:
                if th < a:
                    th += 2*np.pi

        u = 0

        if restored:
            m_start = 1
        else:
            m_start = self.M+1

        for m in range(m_start, self.m_max+1):
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
            #for m in range(1, self.m_max+1):
            for m in range(self.M+1, self.m_max+1):
                ratio = self.calc_bessel_ratio(m, r)
                b = self.b_coef[m-1]
                d_u_n += ratio * b * (m*nu) * np.cos(m*nu*(th-a))
            
            d_u_n /= r

            if sid == 2:
                d_u_n *= -1


        return d_u_n


class SingHSine(SingH):

    n_basis_dict = {
        16: (20, 5), 
        32: (24, 11), 
        64: (41, 18), 
        128: (53, 28), 
        256: (65, 34), 
        512: (80, 34), 
    }

    expected_known = True
    m_max = 8

    def __init__(self, **kwargs): 
        self.bdata = problems.bdata.Sine()
        super().__init__(**kwargs)


class SingHHat(SingH):

    n_basis_dict = {
        16: (24, 6), 
        32: (33, 8), 
        64: (42, 12), 
        128: (65, 18), 
        256: (80, 24), 
        512: (100, 30), 
    }

    expected_known = True
    m_max = 199

    def __init__(self, **kwargs): 
        self.bdata = problems.bdata.Hat()
        super().__init__(**kwargs)


class SingHParabola(SingH):

    expected_known = False

    def __init__(self, **kwargs): 
        self.bdata = problems.bdata.Parabola()
        super().__init__(**kwargs)
