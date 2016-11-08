import numpy as np
from math import factorial
from scipy.special import jv, gamma
import scipy
import sympy

import domain_util

import problems.functions as functions
from .singular import SingularKnown
from .sympy_problem import SympyProblem
from .problem import PizzaProblem

from ps.zmethod import ZMethod

# These are very different than "real" problems since zmethod is used


_n_basis_dict = {
    16: (25, 5),
    32: (30, 20),
    64: (40, 30),
    128: (55, 40),
    256: (70, 60),
    #512: (90, 60),
    #1024: (110, 60),
}

'''_n_basis_dict  = {
    16: (21, 9),
    32: (28, 9),
    64: (34, 17),
    128: (40, 24),
    256: (45, 29),
#    None: (53, 50)
    #None: (65, 34)
}'''

def get_tapering_func(R):
    """
    A function that smoothly decreases from 1 to 0 between values r1
    and r2. Used to prevent v_asympt from becoming very large near
    outer boundary
    """
    r, x = sympy.symbols('r x')
    p = sympy.Piecewise(
        (0, x < 0),
        (x**7*(924*x**6 - 6006*x**5 + 16380*x**4 -
            24024*x**3 + 20020*x**2 - 9009*x + 1716), x <= 1),
        (1, x > 1),
    )

    r1 = ZMethod.R0
    r2 = R
    return p.subs(x, 1-(r-r1)/(r2-r1))

class SingIH_Problem(PizzaProblem):

    zmethod = True

    n_basis_dict = _n_basis_dict

    m1_dict = {
        'arc': 7,
        'outer-sine': 200,
        'inner-sine': 200,
        'cubic': 200,
        'sine7': 200,
    }

_nterms = 2
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

    k = 6.75
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
        ) * get_tapering_func(R)

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

    k = 10.75
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
        ) * get_tapering_func(self.R)

        self.v_series = (
            get_v_asympt0(0, k, nu, nterms=10) * sympy.sin(nu/2*(th-a)) +
            get_v_asympt0(1, k, nu, nterms=10) * sympy.sin(3*nu/2*(th-2*np.pi))
        )

        self.g = (
            (th - a) / (2*np.pi - a) * sympy.besselj(nu/2, k*r)
            + (th - 2*np.pi) / (a - 2*np.pi) * sympy.besselj(3*nu/2, k*r)
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
