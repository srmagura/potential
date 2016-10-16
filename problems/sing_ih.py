import numpy as np
from math import factorial
from scipy.special import jv, gamma, jvp
import scipy
import sympy

import domain_util

from .singular import SingularKnown
from .sympy_problem import SympyProblem

from .sing_ihz import _n_basis_dict, _nterms, get_tapering_func

class SingIH_Problem(SympyProblem, SingularKnown):

    k = 6.75

    n_basis_dict = _n_basis_dict

    def __init__(self, **kwargs):
        #g = kwargs.pop('g')
        v_asympt = kwargs.pop('v_asympt')

        diff = sympy.diff
        pi = sympy.pi
        k, a, r, th, x = sympy.symbols('k a r th x')

        def apply_helmholtz_op(u):
            d_u_r = diff(u, r)
            return diff(d_u_r, r) + d_u_r / r + diff(u, th, 2) / r**2 + k**2 * u

        f = -apply_helmholtz_op(v_asympt)
        kwargs['f_expr'] = f

        self.build_sympy_subs_dict()
        subs_dict = self.sympy_subs_dict
        lambdify_modules = SympyProblem.lambdify_modules

        def my_lambdify(expr):
            return sympy.lambdify((r, th),
                expr.subs(subs_dict),
                modules=lambdify_modules)

        self.eval_regfunc = my_lambdify(v_asympt)
        super().__init__(**kwargs)

    def eval_bc(self, arg, sid):
        """ Evaluate regularized BC """
        r, th = domain_util.arg_to_polar(self.boundary, self.a, arg, sid)

        if r <= 0:
            return -self.eval_bc(-arg, sid)

        return self.eval_bc__noreg(arg, sid) - self.eval_regfunc(r, th)

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

        for l in range(_nterms):
            x = (K+1/2)*nu + l + 1
            v_asympt0 += (-1)**l/(factorial(l)*gamma(x)) * kr2**(2*l)

        v_asympt0 *= kr2**((K+1/2)*nu)
        v_asympt = v_asympt0 * sympy.sin(nu*(K+1/2)*(th-a))

        tapering_func = get_tapering_func(R)
        v_asympt = tapering_func*v_asympt

        kwargs['v_asympt'] = v_asympt
        super().__init__(**kwargs)

    def eval_v(self, r, th):
        """
        Using this function is "cheating".
        """
        a = self.a
        nu = self.nu
        k = self.k
        K = self.K

        return jv(nu*(K+1/2), k*r) * np.sin(nu*(K+1/2)* (th-a))


class I_Bessel(IH_Bessel):
    """
    a coefficients are all 0
    """

    expected_known = True

    def eval_bc__noreg(self, arg, sid):
        # r > 0 only
        a = self.a
        nu = self.nu
        k = self.k

        r, th = domain_util.arg_to_polar(self.boundary, self.a, arg, sid)
        return self.eval_v(r, th)

    def eval_expected_polar(self, r, th):
        #if r == 0:
        #    return 0

        return self.eval_v(r, th) - self.eval_regfunc(r, th)
