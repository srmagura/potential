import enum
import numpy as np

from scipy.fftpack import dst
from scipy.special import jv

import domain_util
import fourier
import abcoef

from .problem import PizzaProblem

class SingularProblem(PizzaProblem):
    """
    A problem that should be treated as if it has a singularity.

    Homogeneous regularization is done by the solver. inhomogeneous
    regularization is done by the problem.
    """


class SingularKnown(SingularProblem):
    """
    Singular problem with known solution.
    """

    m_max = 300

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if 'to_dst' in kwargs:
            self.to_dst = kwargs.pop('to_dst')
        else:
            self.to_dst = lambda th: 0

        self.fft_b_coef = fourier.arc_dst(self.a, self.to_dst)[:self.m_max]
        self.fft_a_coef = abcoef.b_to_a(self.fft_b_coef, self.k,
            self.R, self.nu)


    def eval_expected_polar(self, r, th):
        k = self.k
        a = self.a
        nu = self.nu
        R = self.R

        u = self.eval_expected__no_w(r, th)
        for m in range(1, self.m_max+1):
            ac = self.fft_a_coef[m-1]
            u += ac * jv(m*nu, k*r) * np.sin(m*nu*(th-a))

        return u

    def eval_expected__no_w(self, r, th):
        """
        Expected solution to the regularized problem, not counting
        the Fourier-Bessel series.
        """
        return 0
