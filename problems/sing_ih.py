from sympy import *
import numpy as np
from scipy.special import jv

from solver import cart_to_polar

from .problem import PizzaProblem
from .sympy_problem import SympyProblem

import problems.functions as functions
from .singular import SingularProblem

def get_v_asympt_expr():
    '''
    Returns SymPy expression for the inhomogeneous part of the
    asymptotic expansion.
    '''
    k, R, r, th = symbols('k R r th')
    kr2 = k*r/2

    v_asympt = 1/gamma(sympify('14/11'))
    v_asympt += -1/gamma(sympify('25/11'))*kr2**2
    v_asympt += 1/(2*gamma(sympify('36/11')))*kr2**4

    v_asympt *= sin(3/11*(th-pi/6)) * kr2**(3/11)

    return v_asympt

def get_reg_f_expr():
    '''
    Returns SymPy expression that represents the RHS of the regularized
    problem. This expression is the result of applying the Helmholtz
    operator to the inhomogeneous asymptotic expansion; see this module's
    get_u_asympt_expr().
    '''
    k, R, r, th = symbols('k R r th')

    f = 1331/(67200 * gamma(sympify('3/11')))
    f *= 2**(8/11) * k**(69/11) * r**(47/11)
    f *= sin(-3*th/11 + pi/22)

    return f

def eval_v(k, r, th):
    a = np.pi/6
    nu = np.pi / (2*np.pi - a)

    return jv(nu/2, k*r) * np.sin(nu*(th - a)/2)


class SingIH(SympyProblem, SingularProblem):

    def __init__(self, **kwargs):
        k = self.k
        R = self.R

        kwargs['f_expr'] = get_reg_f_expr()
        kwargs['eval_u_asympt'] = lambdify(symbols('r th'),
            get_v_asympt_expr().subs({'k': k, 'R': R}))
        kwargs['to_dst'] = lambda th: (self.eval_bc__no_reg(th, 0) -
            eval_v(k, R, th))
        super().__init__(**kwargs)

    def eval_expected_polar__no_reg(self, r, th):
        return eval_v(self.k, r, th)

    def eval_bc__no_reg(self, arg, sid):
        k = self.k
        R = self.R
        nu = self.nu
        a = self.a

        if sid == 0:
            th = arg
            return self.eval_phi0(th)
        elif sid == 1:
            r = arg
            if r >= 0:
                return eval_v(k, r, 2*np.pi)
            else:
                return 0
        elif sid == 2:
            return 0


class SingIH_Sine8(SingIH):

    expected_known = True
    k = 2/3

    n_basis_dict = {
        16: (20, 5),
        32: (24, 11),
        64: (41, 18),
        128: (53, 28),
        256: (65, 34),
        None: (80, 34),
    }

    def eval_phi0(self, th):
        a = self.a
        nu = self.nu
        k = self.k
        R = self.R

        phi0 = eval_v(k, R, th)
        phi0 += np.sin(8*nu*(th-a))
        return phi0



class SingIH_Hat(SingIH):

    # k = 5.5  # 4th order
    k = 1 # 2nd order

    expected_known = True
    m_max = 199

    n_basis_dict = {
        16: (24, 6),
        32: (33, 8),
        64: (42, 12),
        128: (65, 18),
        256: (80, 24),
        512: (100, 30),
        1024: (120, 35),
    }

    def eval_phi0(self, th):
        k = self.k
        R = self.R
        nu = self.nu
        a = self.a

        phi0 = eval_v(k, R, th)
        phi0 += functions.eval_hat_th(th)
        return phi0


class SingIH_Parabola(SingIH):

    #k = 5.5  # 4th order
    k = 1.5 # 2nd order

    expected_known = True
    m_max = 200

    n_basis_dict = {
        16: (17, 3),
        32: (33, 7),
        64: (36, 15),
        128: (39, 21),
        256: (42, 23),
        512: (65, 43),
        1024: (80, 45),
    }

    def eval_phi0(self, th):
        k = self.k
        R = self.R
        a = self.a

        phi0 = eval_v(k, R, th)
        phi0 += -(th - a) * (th - 2*np.pi)
        return phi0


class SingIH_Line(SingIH):

    # k = 5.5  # 4th order
    k = 1.5 # 2nd order

    n_basis_dict = {
        16: (19, 6),
        32: (25, 8),
        64: (28, 12),
        128: (28, 16),
        256: (32, 20),
        512: (60, 30),
        1024: (80, 35),
    }

    #expected_known = True
    m_max = 149

    def eval_phi0(self, th):
        k = self.k
        R = self.R
        nu = self.nu
        a = self.a

        phi0 = jv(nu/2, k*R) / (2*np.pi - a) * (th - a)
        return phi0
