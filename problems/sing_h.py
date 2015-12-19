import numpy as np
from scipy.special import jv, jvp


from solver import cart_to_polar

from .problem import PizzaProblem
from .sympy_problem import SympyProblem
import problems.functions as functions
from .singular import SingularProblem, RegularizeBc

class SingH(SingularProblem):
    homogeneous = True

    def __init__(self, **kwargs):
        kwargs['to_dst'] = self.eval_phi0
        super().__init__(**kwargs)

    def eval_expected_polar__no_reg(self, r, th):
        return 0

    def eval_bc__no_reg(self, arg, sid):
        if sid == 0:
            return self.eval_phi0(arg)
        else:
            return 0

class SingH_Sine(SingH):

    k = 1.75

    expected_known = True
    silent = True

    n_basis_dict = {
        16: (20, 5),
        32: (24, 11),
        64: (41, 18),
        128: (53, 28),
        256: (65, 34),
        512: (80, 34),
        None: (90, 45),
    }

    def __init__(self, **kwargs):
        self.m = kwargs.pop('m')
        self.m_max = self.m

        super().__init__(**kwargs)

    def eval_phi0(self, th):
        a = self.a
        nu = self.nu
        m = self.m
        return np.sin(m*nu*(th-a))


class SingH_Sine4(SingH_Sine):

    def __init__(self, **kwargs):
        super().__init__(m=4)


class SingH_Sine8(SingH_Sine):

    def __init__(self, **kwargs):
        super().__init__(m=8)


class SingH_SineRange(SingH):

    k = 1.75

    expected_known = True
    silent = True

    m_max = 9

    n_basis_dict = {
        16: (20, 5),
        32: (24, 11),
        64: (41, 18),
        128: (53, 28),
        256: (65, 34),
        512: (80, 34),
        None: (90, 45),
    }

    def eval_phi0(self, th):
        a = self.a
        nu = self.nu

        phi0 = 0
        for m in range(1, self.m_max+1):
            phi0 += np.sin(m*nu*(th-a))

        return phi0


class SingH_Hat(SingH):

    #k = 5.5  # 4th order
    k = 1  # 2nd order

    n_basis_dict = {
        16: (24, 6),
        32: (33, 8),
        64: (42, 12),
        128: (65, 18),
        256: (80, 24),
        512: (100, 30),
        1024: (120, 35),
    }

    expected_known = True
    m_max = 199

    def eval_phi0(self, th):
        return functions.eval_hat_th(th)


class SingH_Parabola(SingH):

    k = 5.5

    n_basis_dict = {
        16: (17, 3),
        32: (33, 7),
        64: (36, 15),
        128: (39, 21),
        256: (42, 23),
        512: (65, 43),
        1024: (80, 45),
    }

    expected_known = False
    m_max = 199

    def eval_phi0(self, th):
        return -(th - np.pi/6) * (th - 2*np.pi)
