import numpy as np
from scipy.special import jv
import scipy
import sympy

import domain_util

from .singular import SingularKnown
from .sympy_problem import SympyProblem


class SingIH_Problem(SympyProblem, SingularKnown):

    k = 5.5

    n_basis_dict = {
        16: (24, 6),
        32: (33, 8),
        64: (42, 12),
        128: (65, 18),
        256: (80, 30),
        512: (100, 40),
        1024: (120, 45),
    }

    def __init__(self, **kwargs):
        g = kwargs.pop('g')
        v_asympt = kwargs.pop('v_asympt')

        k, r, th = sympy.symbols('k r th')

        diff = sympy.diff
        gv = -(g + v_asympt)
        f = diff(gv, r, 2) + diff(gv, r) / r + diff(gv, th, 2) / r**2 + k**2 * gv

        kwargs['f_expr'] = f

        self.build_sympy_subs_dict()
        subs_dict = self.sympy_subs_dict
        lambdify_modules = SympyProblem.lambdify_modules

        self.eval_v_asympt = sympy.lambdify((r, th), v_asympt.subs(subs_dict),
            modules=lambdify_modules)
        self.eval_g = sympy.lambdify((r, th), g.subs(subs_dict),
            modules=lambdify_modules)

        super().__init__(**kwargs)

    def eval_bc(self, arg, sid):
        if sid == 0:
            r = self.R
            th = arg
            return (self.eval_bc__noreg(arg, sid) - self.eval_g(r, th) -
                self.eval_v_asympt(r, th))
        else:
            return 0


class IH_Bessel(SingIH_Problem):

    expected_known = False

    def __init__(self, **kwargs):
        R = self.R

        def to_dst(th):
            return self.eval_bc__noreg(th, 0) - self.eval_v(R, th)

        kwargs['to_dst'] = to_dst

        gamma = sympy.gamma
        sympify = sympy.sympify
        sin = sympy.sin
        besselj = sympy.besselj
        pi = sympy.pi

        k, r, th, a, nu = sympy.symbols('k r th a nu')
        kr2 = k*r/2

        v_asympt0 = 1/gamma(sympify('14/11'))
        v_asympt0 += -1/gamma(sympify('25/11'))*kr2**2
        v_asympt0 += 1/(2*gamma(sympify('36/11')))*kr2**4
        #v_asympt0 += -1/(6*gamma(sympify('47/11')))*kr2**6
        #v_asympt0 += 1/(24*gamma(sympify('58/11')))*kr2**8
        v_asympt0 *= kr2**(sympify('3/11'))

        v_asympt = v_asympt0 * sin(nu/2*(th-a))

        g = (th - a) / (2*pi - a) * (besselj(nu/2, k*r) - v_asympt0)

        kwargs['g'] = g
        kwargs['v_asympt'] = v_asympt
        super().__init__(**kwargs)

    def eval_v(self, r, th):
        """
        Using this function is "cheating".
        """
        a = self.a
        nu = self.nu
        k = self.k

        return jv(nu/2, k*r) * np.sin(nu/2 * (th-a))

    def eval_expected__no_w(self, r, th):
        return (self.eval_v(r, th) - self.eval_g(r, th) -
            self.eval_v_asympt(r, th))

    def eval_bc__noreg(self, arg, sid):
        a = self.a
        nu = self.nu
        k = self.k
        R = self.R

        th = arg
        return jv(nu/2, k*R) * (th - a) / (2*np.pi - a)
