import numpy as np
from scipy.special import jv, jvp

from solver import cart_to_polar

from .problem import PizzaProblem
from .sympy_problem import SympyProblem
from .bdata import BData
    
class SingH(PizzaProblem):

    homogeneous = True
    expected_known = False
    
    M = 7

    def __init__(self, **kwargs): 
        super().__init__(**kwargs)
        
        if self.expected_known:
            m = self.m_max
        else:
            m = self.M

        self.b_coef = self.bdata.calc_coef(m)

        if self.expected_known:
            self.bessel_R = np.zeros(len(self.b_coef))

            for m in range(1, len(self.b_coef)+1):
                self.bessel_R[m-1] = jv(m*self.nu, self.k*self.R)

            print('[sing-h] b_coef[m_max] =', self.b_coef[self.m_max-1])

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

    ## Begin debugging functions ##
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


###### sing-h-sine ######

class SineBData(BData):

    def eval_phi0(self, th):
        a = PizzaProblem.a
        nu = PizzaProblem.nu

        return np.sin(8*nu*(th-a))


class SingHSine(SingH):

    k = 1.75

    n_basis_dict = {
        16: (20, 5), 
        32: (24, 11), 
        64: (41, 18), 
        128: (53, 28), 
        256: (65, 34), 
        None: (80, 34), 
    }

    expected_known = True
    m_max = 8

    def __init__(self, **kwargs): 
        self.bdata = SineBData()
        super().__init__(**kwargs)


###### sing-h-hat ######

def eval_hat_x(x):
    if abs(abs(x)-1) < 1e-15 or abs(x) > 1:
        return 0
    else:
        arg = -1/(1-x**2)
        return np.exp(arg)

def eval_hat_th(th):
    a = PizzaProblem.a
    x = (2*th-(2*np.pi+a))/(2*np.pi-a)
    return eval_hat_x(x)

class HatBData(BData):
    
    def eval_phi0(self, th):
        return eval_hat_th(th)


class SingHHat(SingH):

    k = 5.5

    n_basis_dict = {
        16: (24, 6), 
        32: (33, 8), 
        64: (42, 12), 
        128: (65, 18), 
        256: (80, 24), 
        512: (100, 30), 
        1024: (120, 35), 
    }

    expected_known = False
    m_max = 199

    def __init__(self, **kwargs): 
        self.bdata = HatBData() 
        super().__init__(**kwargs)


###### sing-h-parabola ######

class ParabolaBData(BData):

    def eval_phi0(self, th):
        return -(th - np.pi/6) * (th - 2*np.pi) 

    def calc_coef_analytic(self, M):
        a = PizzaProblem.a
        nu = PizzaProblem.nu

        coef = np.zeros(M)

        for m in range(1, M+1):
            if m % 2 == 0:
                coef[m-1] = 0
            else:
                coef[m-1] = 8/(2*np.pi - a) * 1/(m*nu)**3

        return coef


class SingHParabola(SingH):

    k = 5.5

    n_basis_dict = {
        16: (17, 3), 
        32: (33, 7), 
        64: (36, 15), 
        128: (39, 21), 
        256: (42, 23), 
        512: (65, 43), 
        1024: (80, 45), 
    }

    expected_known = False
    m_max = 199

    def __init__(self, **kwargs): 
        self.bdata = ParabolaBData()
        super().__init__(**kwargs)
