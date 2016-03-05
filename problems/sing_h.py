import numpy as np
from scipy.special import jv

from .problem import PizzaProblem
import problems.functions as functions
from .singular import SingularProblem

class SingH(SingularProblem):
    homogeneous = True

    
    # 4th order
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



    '''
    # 2nd order
    k = 1.75

    n_basis_dict = {
        16: (16, 5),
        32: (27, 7),
        64: (35, 11),
        128: (41, 15),
        256: (60, 22),
        512: (80, 30),
        1024: (100, 40),
        2048: (120, 45)
    }
    '''

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


class H_Sine(SingH):

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


class H_Sine8(H_Sine):

    def __init__(self, **kwargs):
        super().__init__(m=8)


class H_SineRange(SingH):

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


class H_Hat(SingH):

    expected_known = False
    m_max = 200

    def eval_phi0(self, th):
        return functions.eval_hat_th(th)


class H_Parabola(SingH):

    expected_known = False
    m_max = 200

    def eval_phi0(self, th):
        a = self.a
        return -(th - a) * (th - 2*np.pi)


class H_LineSine(SingH):

    expected_known = False
    m_max = 200

    def eval_phi0(self, th):
        k = self.k
        R = self.R
        a = self.a
        nu = self.nu

        return jv(nu/2, k*R) * ((th - a)/(2*np.pi - a) - np.sin(nu/2*(th-a)))
