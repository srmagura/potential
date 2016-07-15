import numpy as np
from scipy.special import jv, jvp

import domain_util

from .problem import PizzaProblem
import problems.functions as functions
from .singular import SingularKnown, HReg

#_k = 5.5
_k = 10.75

_n_basis_dict = {
    16: (24, 6),
    32: (33, 8),
    64: (42, 12),
    128: (65, 18),
    256: (80, 30),
    512: (100, 40),
    1024: (120, 45),
}

class SingH_Trace(SingularKnown):

    homogeneous = True
    hreg = HReg.linsys

    expected_known = True
    no_fourier = False

    k = _k
    n_basis_dict = _n_basis_dict


    def __init__(self, **kwargs):
        kwargs['to_dst'] = self.eval_phi0
        super().__init__(**kwargs)

    def eval_bc(self, arg, sid):
        if sid == 0:
            if not self.no_fourier:
                r, th = domain_util.arg_to_polar(self.boundary, self.a, arg, sid)
                # MEGA FIXME
                m = 8
                a = self.a
                nu = self.nu
                k = self.k
                return self.eval_expected_polar(r, th) + jv(-m*nu, k*r) * np.sin(m*nu*(th-a))
            else:
                return self.eval_phi0(arg)
        else:
            return 0

    def eval_d_u_outwards(self, arg, sid):
        # Debug function
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


class Trace_Sine(SingH_Trace):

    expected_known = True

    def __init__(self, **kwargs):
        self.m = kwargs.pop('m')
        self.m1 = self.m

        super().__init__(**kwargs)

    def eval_phi0(self, th):
        a = self.a
        nu = self.nu
        m = self.m

        return np.sin(m*nu*(th-a))


class Trace_Sine8(Trace_Sine):

    def __init__(self, **kwargs):
        super().__init__(m=8)


class Trace_SineRange(SingH_Trace):

    expected_known = True

    m_range_max = 9
    m1 = 9

    def eval_phi0(self, th):
        a = self.a
        nu = self.nu

        phi0 = 0
        for m in range(1, self.m_range_max+1):
            phi0 += np.sin(m*nu*(th-a))

        return phi0


class Trace_Hat(SingH_Trace):

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


class Trace_Parabola(SingH_Trace):

    m1_dict = {
        'arc': 48,
        'outer-sine': 200,
        'inner-sine': 220,
        'cubic': 210,
        'sine7': 218,
    }

    def eval_phi0(self, th):
        return functions.eval_parabola(th, self.a)


class Trace_LineSine(SingH_Trace):

    m1_dict = {
        'arc': 8,
        'outer-sine': 210,
        'inner-sine': 236,
        'cubic': 200,
        'sine7': 204,
    }

    def eval_phi0(self, th):
        return functions.eval_linesine(th, self.a, self.nu)
