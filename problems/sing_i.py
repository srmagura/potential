from sympy import *
import numpy as np
from scipy.special import jv

from .problem import PizzaProblem
from .singular import SingularProblem
from .sympy_problem import SympyProblem

def get_u_asympt_expr():
    k, R, r, th = symbols('k R r th')
    nu = sympify('6/11')

    kr2 = k*r/2

    u_asympt = 1/gamma(nu+1)
    u_asympt += -1/gamma(nu+2)*kr2**2
    u_asympt += 1/(2*gamma(nu+3))*kr2**4
    u_asympt *= sin(nu*(th-pi/6)) * kr2**nu

    return u_asympt

def get_reg_f_expr():
    k, R, r, th = symbols('k R r th')
    nu = sympify('6/11')

    f = 1331/(182784*gamma(nu)) * cos(nu*th+9*pi/22)
    f *= 2**(5/11) * k**(72/11) * r**(50/11)

    return f

def eval_expected_polar(k, R, r, th):
    nu_float = 6/11
    return jv(nu_float, k*r) * np.sin(nu_float*(th-np.pi/6))


class SingI_NoCorrection(SingularProblem):

    k = 1
    expected_known = True
    homogeneous = True

    n_basis_dict = {
        16: (21, 9),
        32: (28, 9),
        64: (34, 17),
        128: (40, 24),
        256: (45, 29),
        None: (53, 34)
    }

    def eval_expected_polar__no_reg(self, r, th):
        return eval_expected_polar(self.k, self.R, r, th)

    def eval_bc__no_reg(self, arg, sid):
        r, th = self.arg_to_polar(arg, sid)
        return eval_expected_polar(self.k, self.R, r, th)


class SingI(SympyProblem, SingI_NoCorrection):

    homogeneous = False

    def __init__(self, **kwargs):
        kwargs['f_expr'] = get_reg_f_expr()
        kwargs['eval_u_asympt'] = lambdify(symbols('r th'),
            get_u_asympt_expr().subs({'k': self.k, 'R': self.R}))
        super().__init__(**kwargs)
