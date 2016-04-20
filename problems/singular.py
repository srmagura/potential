import numpy as np

from scipy.fftpack import dst
from scipy.special import jv

import domain_util
import fourier
import singular_util

from .problem import PizzaProblem

class SingularProblem(PizzaProblem):
    """
    A problem that should be treated as if it has a singularity.
    """

    regularize = True
    m_max = 100

    def __init__(self, **kwargs):
        if 'eval_u_asympt' in kwargs:
            self.eval_u_asympt = kwargs.pop('eval_u_asympt')
        else:
            self.eval_u_asympt = lambda r, th: 0

        if 'to_dst' in kwargs:
            self.outside_bdata = kwargs.pop('to_dst')
        else:
            self.outside_bdata = lambda th: 0

        self.fft_b_coef = fourier.arc_dst(self.a, self.outside_bdata)[:self.m_max]
        self.fft_a_coef = singular_util.b_to_a(self.fft_b_coef, self.k,
            self.R, self.nu)

    # deprecate?
    def print_b(self):
        np.set_printoptions(precision=4)

        n = 5
        print('Last {} of {} b coefficients:'.
            format(n, len(self.fft_b_coef)))

        print(self.fft_b_coef[-n:])
        print()

    def eval_expected_polar(self, r, th):
        k = self.k
        a = self.a
        nu = self.nu
        R = self.R

        u = self.eval_expected_polar__no_reg(r, th)
        for m in range(1, self.m_max+1):
            ac = self.fft_a_coef[m-1]
            u += ac * jv(m*nu, k*r) * np.sin(m*nu*(th-a))

        return u - self.eval_u_asympt(r, th)

    def eval_bc(self, arg, sid):
        """
        Evaluate extended boundary condition.
        """
        a = self.a
        nu = self.nu
        k = self.k
        R = self.R

        bc = self.eval_bc__no_reg(arg, sid)
        r, th = domain_util.arg_to_polar(R, a, arg, sid)

        if sid == 0:
            return bc - self.eval_u_asympt(r, th)

        if sid == 1:
            if th < 1.5*np.pi:
                return 0
            else:
                return bc - self.eval_u_asympt(r, th)

        elif sid == 2:
            return 0
