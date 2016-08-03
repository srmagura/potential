import numpy as np
from math import factorial
from scipy.special import jv, gamma
import scipy
import sympy

import domain_util

from .singular import SingularKnown, HReg
from .sympy_problem import SympyProblem
from .problem import PizzaProblem

# These are very different than "real" problems since zmethod is used

class SingIH_Problem(PizzaProblem):

    zmethod = True
    k = 3

    n_basis_dict = {
        16: (24, 6),
        32: (33, 8),
        64: (42, 12),
        128: (65, 18),
        256: (80, 30),
        512: (100, 40),
        1024: (120, 45),
    }

nterms = 8
print('FS nterms:', nterms)

def get_v_asympt0(K, k, nu):
    v_asympt0 = 0
    r, th = sympy.symbols('r th')
    kr2 = k*r/2


    for l in range(nterms):
        x = (K+1/2)*nu + l + 1
        v_asympt0 += (-1)**l/(factorial(l)*gamma(x)) * kr2**(2*l)

    v_asympt0 *= kr2**((K+1/2)*nu)
    return v_asympt0


class IH_Bessel(SingIH_Problem):
    """ Abstract class. """

    expected_known = False


    def __init__(self, **kwargs):
        a = self.a
        nu = self.nu
        k = self.k

        r, th = sympy.symbols('r th')

        v_asympt01 = get_v_asympt0(0, k, nu)

        self.v_asympt = (
            get_v_asympt0(0, k, nu) * sympy.sin(nu/2*(th-a))
        )
        self.g = (th - a) / (2*np.pi - a) * (sympy.besselj(nu/2, k*r) - v_asympt01)


    def eval_v(self, r, th):
        """
        Using this function is "cheating".
        """
        a = self.a
        nu = self.nu
        k = self.k

        return jv(nu/2, k*r) * np.sin(nu/2* (th-a))

    #def eval_expected__no_w(self, r, th):
    #    return self.eval_v(r, th)


class IZ_Bessel(IH_Bessel):
    """
    a coefficients are all 0
    """

    m1_dict = {
        'arc': 7,
        'outer-sine': 134,
        'inner-sine': 220,
        'cubic': 210,
        'sine7': 200,
    }

    def eval_phi0(self, th):
        r = self.boundary.eval_r(th)
        return self.eval_v(r, th)

    def eval_phi1(self, r):
        if r > 0:
            return self.eval_v(r, 2*np.pi)
        else:
            return 0

    def eval_phi2(self, r):
        return 0

    def eval_expected_polar(self, r, th):
        return self.eval_v(r, th)


class IHZ_Bessel_Line(IH_Bessel):

    m1_dict = {
        'arc': 8,
        'outer-sine': 210,
        'inner-sine': 236,
        'cubic': 200,
        'sine7': 204,
    }

    def eval_phi0(self, th):
        a = self.a
        nu = self.nu
        k = self.k

        r = self.boundary.eval_r(th)
        return jv(nu/2, k*r) * (th - a) / (2*np.pi - a)

    def eval_phi1(self, r):
        if r > 0:
            return jv(self.nu/2, self.k*r)
        else:
            return 0

    def eval_phi2(self, r):
        return 0


class IHZ_Bessel_Quadratic(SingIH_Problem):

    expected_known = False

    m1_dict = {
        'arc': 7,
        'outer-sine': 200,
        'inner-sine': 220,
        'cubic': 140,
        'sine7': 218,
    }

    def __init__(self, **kwargs):
        a = self.a
        nu = self.nu
        k = self.k

        r, th = sympy.symbols('r th')

        v_asympt01 = get_v_asympt0(0, k, nu)
        v_asympt03 = get_v_asympt0(1, k, nu)

        self.v_asympt = (
            v_asympt01 * sympy.sin(nu/2*(th-a)) +
            v_asympt03 * sympy.sin(3*nu/2*(th-2*np.pi))
        )

        self.g = (
            (th - a) / (2*np.pi - a) * (sympy.besselj(nu/2, k*r) - v_asympt01)
            + (th - 2*np.pi) / (a - 2*np.pi) * (sympy.besselj(3*nu/2, k*r) - v_asympt03)
        )

    def eval_v(self, r, th):
        """
        Using this function is "cheating".
        """
        a = self.a
        nu = self.nu
        k = self.k

        return (
            jv(nu/2, k*r) * np.sin(nu/2 * (th-a)) +
            jv(3*nu/2, k*r) * np.sin(3*nu/2 * (th-2*np.pi))
        )

    def eval_phi0(self, th):
        a = self.a
        nu = self.nu
        k = self.k

        r = self.boundary.eval_r(th)
        p = lambda th: (th-a) * (th - (2*np.pi-a)) / (a*(2*np.pi - a))

        return jv(3*nu/2, k*r) + (jv(nu/2, k*r) - jv(3*nu/2, k*r)) * p(th)
        #return self.eval_v(r, th) #FIXME

    def eval_phi1(self, r):
        if r > 0:
            return jv(self.nu/2, self.k*r)
        else:
            return 0

    def eval_phi2(self, r):
        if r > 0:
            return jv(3*self.nu/2, self.k*r)
        else:
            return 0
