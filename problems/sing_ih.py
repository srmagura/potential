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

    # FIXME
    k = 1 #5.5

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

    def __init__(self, **kwargs):
        g = kwargs.pop('g')
        v_asympt = kwargs.pop('v_asympt')

        diff = sympy.diff
        pi = sympy.pi
        k, a, r, th, x = sympy.symbols('k a r th x')

        def do_laplacian(u):
            d_u_r = diff(u, r)
            return diff(d_u_r, r) + d_u_r / r + diff(u, th, 2) / r**2 + k**2 * u

        f = -do_laplacian(g + v_asympt)
        kwargs['f_expr'] = f

        logistic = 1 / (1 + sympy.exp(-90*(x-0.5)))

        # TODO insert q2
        q1 = (r**2 * 1/2 * (th-2*pi)**2 * f.subs(th, 2*pi) *
            logistic.subs(x, (th-2*pi)/(2*pi-a)+1))

        f1 = f - do_laplacian(q1)

        self.build_sympy_subs_dict()
        subs_dict = self.sympy_subs_dict
        lambdify_modules = SympyProblem.lambdify_modules


        #plt.plot(r_data, f1_data)
        #plt.show()

        #_f_lambda = sympy.lambdify(r, f.subs(subs_dict).subs('th', 2*np.pi),
        #    modules=lambdify_modules)
        #f_data = np.array([_f_lambda(r) for r in r_data])
        #print('fmax: ', np.max(np.abs(f_data)))

        #plt.plot(r_data, f_data)

        #q1_lambda = sympy.lambdify(r, q1.subs(subs_dict).subs('th', np.pi),
        #    modules=lambdify_modules)
        #q1_data = np.array([q1_lambda(r) for r in r_data])
        #print('qmax: ', np.max(np.abs(q1_data)))

        #plt.plot(r_data, q1_data)
        #plt.show()

        #_lambda = sympy.lambdify(r,
        #    (abs(diff(q1, u, r) + .subs(subs_dict).subs('th', 2*np.pi),
        #    modules=lambdify_modules)
        #f_data = np.array([_f_lambda(r) for r in r_data])

        #plt.show()
        #import sys; sys.exit(0)

        def my_lambdify(expr):
            return sympy.lambdify((r, th),
                expr.subs(subs_dict),
                modules=lambdify_modules)

        self.eval_regfunc = my_lambdify(g+v_asympt)
        self.eval_q = my_lambdify(q1)
        self.eval_f1 = my_lambdify(f1)

        r_data = np.arange(self.R/256, self.R, .0001)

        f1_data = np.array([self.eval_f1(r, 2*np.pi) for r in r_data])
        print('f1_max:', np.max(np.abs(f1_data)))

        super().__init__(**kwargs)

    def eval_bc(self, arg, sid):
        if sid == 0:
            th = arg
            r = self.boundary.eval_r(th)
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
        if r == 0:
            return 0

        return self.eval_v(r, th) - self.eval_regfunc(r, th)

class I_Bessel(IH_Bessel):

    # Expected is known only for the true arc
    expected_known=True

    def eval_bc__noreg(self, arg, sid):
        th = arg
        r = self.boundary.eval_r(th)
        return self.eval_v(r, th)


class IH_Bessel_Line(IH_Bessel):

    # Expected is known only for the true arc
    expected_known=True

    def __init__(self, **kwargs):
        a = self.a
        nu = self.nu
        k = self.k
        R = self.R

        def to_dst(th):
            return jv(nu/2, k*R) * (th - a) / (2*np.pi - a) - self.eval_v(R, th)

        kwargs['to_dst'] = to_dst
        super().__init__(**kwargs)

    def eval_bc__noreg(self, arg, sid):
        a = self.a
        nu = self.nu
        k = self.k

        th = arg
        r = self.boundary.eval_r(th)
        return jv(nu/2, k*r) * (th - a) / (2*np.pi - a)


# TODO change data on wedge
"""class IH_Bessel_Quadratic(IH_Bessel):

    def eval_bc__noreg(self, arg, sid):
        a = self.a
        nu = self.nu
        k = self.k
        R = self.R

        th = arg
        return jv(nu/2, k*R) * (th - a)**2 / (2*np.pi - a)**2"""
