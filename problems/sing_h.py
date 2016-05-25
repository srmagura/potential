import numpy as np
from scipy.special import jv, jvp

import domain_util

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

    # 2nd order
    '''k = 1.75

    n_basis_dict = {
        16: (16, 5),
        32: (27, 7),
        64: (35, 11),
        128: (41, 15),
        256: (60, 22),
        512: (80, 30),
        1024: (100, 40),
        2048: (120, 45)
    }'''


    def __init__(self, **kwargs):
        kwargs['to_dst'] = self.eval_phi0
        super().__init__(**kwargs)

    #def eval_expected_polar__no_reg(self, r, th):
    #    return 0

    def eval_bc__no_reg(self, arg, sid):
        if sid == 0:
            #r, th = domain_util.arg_to_polar(self.boundary, self.a, arg, sid)
            #return self.eval_expected_polar(r, th)

            return self.eval_phi0(arg)
        else:
            return 0

    def eval_d_u_outwards(self, arg, sid):
        # Only for true arc. At least at the moment
        k = self.k
        R = self.R
        a = self.a
        nu = self.nu

        r, th = domain_util.arg_to_polar(self.boundary, a, arg, sid)

        d_u = 0

        for m in range(1, self.m_max+1):
            ac = self.fft_a_coef[m-1]
            if sid == 0:
                d_u += ac * k * jvp(m*nu, k*r, 1) * np.sin(m*nu*(th-a))
            elif sid == 1:
                d_u += ac * jv(m*nu, k*r) * m * nu * np.cos(m*nu*(th-a))
            elif sid == 2:
                d_u += -ac * jv(m*nu, k*r) * m * nu * np.cos(m*nu*(th-a))

        return d_u


class H_Sine(SingH):

    expected_known = True
    silent = True

    def __init__(self, **kwargs):
        self.m = kwargs.pop('m')
        self.m1 = self.m

        super().__init__(**kwargs)

    def eval_phi0(self, th):
        a = self.a
        nu = self.nu
        m = self.m

        return np.sin(m*nu*(th-a))


class H_Sine8(H_Sine):

    k = 5.5

    def __init__(self, **kwargs):
        super().__init__(m=8)


class H_SineRange(SingH):

    expected_known = True
    silent = True

    m_range_max = 9
    m1 = 9

    def eval_phi0(self, th):
        a = self.a
        nu = self.nu

        phi0 = 0
        for m in range(1, self.m_range_max+1):
            phi0 += np.sin(m*nu*(th-a))

        return phi0


class H_Hat(SingH):

    expected_known = True

    m1_dict = {
        'arc': 128,
        'outer-sine': 134,
        'inner-sine': 220,
        'cubic': 210,
        'sine7': 200,
    }

    def eval_phi0(self, th):
        return functions.eval_hat_th(th)


class H_Parabola(SingH):

    # Needs to be False for finer grids
    expected_known = True

    #k = 5.5

    '''n_basis_dict = {
        16: (16, 3),
        32: (34, 9),
        64: (34, 17),
        128: (40, 25),
        256: (43, 31),
        512: (55, 35),
        1024: (65, 40),
    }'''

    m1_dict = {
        'arc': 48,
        'outer-sine': 200,
        'inner-sine': 220,
        'cubic': 210,
        'sine7': 218,
    }

    def eval_phi0(self, th):
        a = self.a
        return -(th - a) * (th - 2*np.pi)


class H_LineSine(SingH):

    m1_dict = {
        'arc': 8,
        'outer-sine': 210,
        'inner-sine': 236,
        'cubic': 200,
        'sine7': 204,
    }

    def eval_phi0(self, th):
        k = self.k
        R = self.R
        a = self.a
        nu = self.nu

        return (th - a)/(2*np.pi - a) - np.sin(nu/2*(th-a))
