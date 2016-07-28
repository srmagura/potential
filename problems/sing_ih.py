import numpy as np
from math import factorial
from scipy.special import jv, gamma
import scipy
import sympy

import domain_util

from .singular import SingularKnown, HReg
from .sympy_problem import SympyProblem


class SingIH_Problem(SympyProblem, SingularKnown):

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

    m1_dict = {
        'arc': 8,
        'outer-sine': 210,
        'inner-sine': 236,
        'cubic': 200,
        'sine7': 204,
    }



class IH_Bessel(SingIH_Problem):
    """ Abstract class. """

    expected_known = False
    K = 0

    def __init__(self, **kwargs):
        a = self.a
        nu = self.nu
        k = self.k
        R = self.R
        K = self.K

        r, th = sympy.symbols('r th')
        kr2 = k*r/2

        v_asympt0 = 0

        # Recommended: 10 terms
        for l in range(3):
            x = (K+1/2)*nu + l + 1
            v_asympt0 += (-1)**l/(factorial(l)*gamma(x)) * kr2**(2*l)

        v_asympt0 *= kr2**((K+1/2)*nu)

        self.v_asympt = v_asympt0 * sympy.sin(nu*(K+1/2)*(th-a))
        self.g = (th - a) / (2*np.pi - a) * (sympy.besselj(nu/2, k*r) - v_asympt0)

        # Derivatives of RHS are calculated in zmethod
        kwargs['f_expr'] = sympy.sympify(0)

        super().__init__(**kwargs)

    def set_boundary(self, boundary):
        super().set_boundary(boundary)

        if boundary.name == 'arc':
            self.expected_known = True

    def eval_v(self, r, th):
        """
        Using this function is "cheating".
        """
        a = self.a
        nu = self.nu
        k = self.k
        K = self.K

        return jv(nu*(K+1/2), k*r) * np.sin(nu*(K+1/2)* (th-a))

    def eval_expected__no_w(self, r, th):
        if r == 0:
            return 0

        return self.eval_v(r, th) - self.eval_regfunc(r, th)


class I_Bessel(IH_Bessel):
    """
    a coefficients are all 0
    """

    def eval_bc(self, arg, sid):
        a = self.a
        nu = self.nu
        k = self.k

        r, th = domain_util.arg_to_polar(self.boundary, self.a, arg, sid)
        if sid == 0:
            return self.eval_v(r, th)
        elif sid == 1:
            if r > 0:
                return self.eval_v(r, th)
            else:
                return 0
        elif sid == 2:
            return 0


class I_Bessel3(I_Bessel):

    K = 3


class IH_Bessel_Line(IH_Bessel):

    def __init__(self, **kwargs):
        a = self.a
        nu = self.nu
        k = self.k
        R = self.R

        def to_dst(th):
            return jv(nu/2, k*R) * (th - a) / (2*np.pi - a) - self.eval_v(R, th)

        kwargs['to_dst'] = to_dst
        super().__init__(**kwargs)

    def eval_bc(self, arg, sid):
        a = self.a
        nu = self.nu
        k = self.k

        r, th = domain_util.arg_to_polar(self.boundary, self.a, arg, sid)
        if sid == 0:
            return jv(nu/2, k*r) * (th - a) / (2*np.pi - a)
        elif sid == 1:
            if r > 0:
                return jv(nu/2, k*r)
            else:
                return 0
        elif sid == 2:
            return 0



# TODO change data on wedge
"""class IH_Bessel_Quadratic(IH_Bessel):

    def eval_bc(self, arg, sid):
        a = self.a
        nu = self.nu
        k = self.k
        R = self.R

        th = arg
        return jv(nu/2, k*R) * (th - a)**2 / (2*np.pi - a)**2"""
