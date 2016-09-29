import numpy as np
from math import factorial
from scipy.special import jv, gamma
import scipy
import sympy

import domain_util

import problems.functions as functions
from .singular import SingularKnown, HReg
from .sympy_problem import SympyProblem
from .problem import PizzaProblem

# These are very different than "real" problems since zmethod is used

class SingIH_Problem(PizzaProblem):

    zmethod = True

    k = 6.75

    n_basis_dict = {
        16: (25, 5),
        32: (30, 17),
        64: (40, 21),
        128: (65, 18),
        256: (80, 30),
        512: (80, 45),
        1024: (80, 45),
    }

    m1_dict = {
        'arc': 7,
        'outer-sine': 200,
        'inner-sine': 200,
        'cubic': 200,
        'sine7': 200,
    }

_nterms = 3
print('FS nterms:', _nterms)

alt_phi1_ext = False
print('ihz-bessel-line alternate phi1 extension:', alt_phi1_ext)

def get_v_asympt0(K, k, nu, nterms=_nterms):
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
    K = 0

    def __init__(self, **kwargs):
        a = self.a
        nu = self.nu
        k = self.k
        K = self.K
        R = self.R

        r, th = sympy.symbols('r th')

        v_asympt01 = get_v_asympt0(K, k, nu)

        self.v_asympt = (
            get_v_asympt0(K, k, nu) * sympy.sin((K+1/2)*nu*(th-a))
        )

        self.v_series = get_v_asympt0(K, k, nu, nterms=10)* sympy.sin((K+1/2)*nu*(th-a))

        phi1 = sympy.besselj((K+1/2)*nu, k*r)
        if alt_phi1_ext:
            phi1 += sympy.Piecewise((0, r<R), ((r-R)**5, True))

        self.g = (th - a) / (2*np.pi - a) * (phi1)

    def eval_v(self, r, th):
        """
        Using this function is "cheating".
        """
        a = self.a
        nu = self.nu
        k = self.k
        K = self.K

        return jv((K+1/2)*nu, k*r) * np.sin((K+1/2)*nu* (th-a))



class IZ_Bessel(IH_Bessel):
    """
    a coefficients are all 0
    """

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

    def eval_phi0(self, th):
        a = self.a
        nu = self.nu
        k = self.k

        r = self.boundary.eval_r(th)
        return jv((self.K+1/2)*nu, k*r) * (th - a) / (2*np.pi - a)

    def eval_phi1(self, r):
        R = self.R
        bc = jv((self.K+1/2)*self.nu, self.k*r)

        if r > R and alt_phi1_ext:
            return bc + (r-R)**5
        elif r > 0:
            return bc
        else:
            return 0

    def eval_phi2(self, r):
        return 0


class IHZ_Bessel_Quadratic(SingIH_Problem):

    expected_known = False

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
