import numpy as np
from math import factorial
from scipy.special import jv, gamma
import scipy
import sympy

import domain_util

from .singular import SingularKnown, HReg
from .sympy_problem import SympyProblem


class SingIH_Problem(SympyProblem, SingularKnown):

    hreg = HReg.ode

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

        diff = sympy.diff
        pi = sympy.pi
        k, a, r, th, x = sympy.symbols('k a r th x')

        def do_lapacian(u):
            return diff(u, r, 2) + diff(u, r) / r + diff(u, th, 2) / r**2 + k**2 * u

        f1 = -do_lapacian(g + v_asympt)

        logistic = 1 / (1 + sympy.exp(-90*(x-0.5)))

        # TODO insert q2
        q1 = (r**2 * 1/2 * (th-2*pi)**2 * f1.subs(th, 2*pi) *
            logistic.subs(x, (th-2*pi)/(2*pi-a)+1))

        f = f1 - do_lapacian(q1)

        kwargs['f_expr'] = f

        self.build_sympy_subs_dict()
        subs_dict = self.sympy_subs_dict
        lambdify_modules = SympyProblem.lambdify_modules

        #r_data = np.arange(.01, 2.3, .01)
        #_f_lambda = sympy.lambdify([r], f.subs(subs_dict).subs('th', 2*np.pi),
        #   modules=lambdify_modules)
        #f_data = np.array([_f_lambda(r) for r in r_data])
        #print('fmax: ', np.max(np.abs(f_data)))
        #import sys; sys.exit(0)

        self.eval_regfunc = sympy.lambdify((r, th),
            (v_asympt + g + q1).subs(subs_dict),
            modules=lambdify_modules)

        super().__init__(**kwargs)

    def eval_bc(self, arg, sid):
        if sid == 0:
            r = self.R
            th = arg
            return self.eval_bc__noreg(arg, sid) - self.eval_regfunc(r, th)
        else:
            return 0


class IH_Bessel(SingIH_Problem):

    expected_known = False

    def __init__(self, **kwargs):
        a = self.a
        nu = self.nu
        k = self.k
        R = self.R

        def to_dst(th):
            return self.eval_bc__noreg(th, 0) - self.eval_v(R, th)

        kwargs['to_dst'] = to_dst

        r, th = sympy.symbols('r th')
        kr2 = k*r/2

        v_asympt0 = 0
        for l in range(3):
            x = 3/11 + l + 1
            v_asympt0 += (-1)**l/(factorial(l)*gamma(x)) * kr2**(2*l)

        v_asympt0 *= kr2**(3/11)

        v_asympt = v_asympt0 * sympy.sin(nu/2*(th-a))

        g = (th - a) / (2*np.pi - a) * (sympy.besselj(nu/2, k*r) - v_asympt0)

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
        return self.eval_v(r, th) - self.eval_regfunc(r, th)

    def eval_bc__noreg(self, arg, sid):
        a = self.a
        nu = self.nu
        k = self.k
        R = self.R

        th = arg
        return jv(nu/2, k*R) * (th - a) / (2*np.pi - a)


class IH_Bessel(SingIH_Problem):

    expected_known = False

    def __init__(self, **kwargs):
        a = self.a
        nu = self.nu
        k = self.k
        R = self.R

        def to_dst(th):
            return self.eval_bc__noreg(th, 0) - self.eval_v(R, th)

        kwargs['to_dst'] = to_dst

        r, th = sympy.symbols('r th')
        kr2 = k*r/2

        v_asympt0 = 0
        for l in range(4):
            x = 3/11 + l + 1
            v_asympt0 += (-1)**l/(factorial(l)*gamma(x)) * kr2**(2*l)

        v_asympt0 *= kr2**(3/11)

        v_asympt = v_asympt0 * sympy.sin(nu/2*(th-a))

        g = (th - a) / (2*np.pi - a) * (sympy.besselj(nu/2, k*r) - v_asympt0)

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
        return self.eval_v(r, th) - self.eval_regfunc(r, th)


class IH_Bessel_Line(IH_Bessel):

    def eval_bc__noreg(self, arg, sid):
        a = self.a
        nu = self.nu
        k = self.k
        R = self.R

        th = arg
        return jv(nu/2, k*R) * (th - a) / (2*np.pi - a)


# TODO change data on wedge
class IH_Bessel_Quadratic(IH_Bessel):

    def eval_bc__noreg(self, arg, sid):
        a = self.a
        nu = self.nu
        k = self.k
        R = self.R

        th = arg
        return jv(nu/2, k*R) * (th - a)**2 / (2*np.pi - a)**2
