import numpy as np
from scipy.special import jv, jvp


from solver import cart_to_polar

from .problem import PizzaProblem
from .sympy_problem import SympyProblem
import problems.functions as functions
from .singular import SingularProblem, RegularizeBc

class SingH(SingularProblem):
    homogeneous = True

    k = 1.75

    n_basis_dict = {
        16: (20, 5),
        32: (24, 11),
        64: (41, 18),
        128: (65, 28),
        256: (80, 34),
        512: (100, 34),
        1024: (120, 45),
    }

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

    expected_known = True
    silent = True

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

    expected_known = True
    silent = True

    m_max = 9

    def eval_phi0(self, th):
        a = self.a
        nu = self.nu

        phi0 = 0
        for m in range(1, self.m_max+1):
            phi0 += np.sin(m*nu*(th-a))

        return phi0


class SingH_Hat(SingH):

    expected_known = False
    m_max = 199

    def eval_phi0(self, th):
        return functions.eval_hat_th(th)


class SingH_Parabola(SingH):

    expected_known = False
    m_max = 5

    def eval_phi0(self, th):
        return -(th - np.pi/6) * (th - 2*np.pi)
