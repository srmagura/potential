import numpy as np
from scipy.special import jv, jvp

from problems import RegularizeBc
from .problem import PizzaProblem
from .bdata import BData

class SingularProblem(PizzaProblem):
    """
    A problem that should be treated as if it has a singularity.
    """

    regularize_bc = RegularizeBc.fft

    def __init__(self, **kwargs):
        self.bdata = SingularProblem_BData(self)

    def setup(self):
        if self.expected_known:
            # Fourier coefficients may get calculated twice for dual
            m_max = max(self.M, self.m_max)
        else:
            m_max = self.M

        # TODO get rid of the BData class
        self.fft_b_coef = self.bdata.calc_coef(m_max)
        print(self.fft_b_coef)

        self.fft_a_coef = np.zeros(m_max)
        k = self.k
        R = self.R
        nu = self.nu

        for m in range(1, m_max+1):
            self.fft_a_coef[m-1] = self.fft_b_coef[m-1] / jv(m*nu, k*R)

        #if self.expected_known and not self.silent:
        #    print('[sing-h] fft_b_coef[m_max] =',
        #        self.fft_b_coef[self.m_max-1])

    def set_a_coef(self, a_coef):
        k = self.k
        R = self.R
        nu = self.nu

        self.a_coef = a_coef
        self.b_coef = np.zeros(len(a_coef), dtype=complex)

        for m in range(1, len(a_coef)+1):
            self.b_coef[m-1] = self.a_coef[m-1] * jv(m*nu, k*R)

    def eval_expected_polar(self, r, th):
        k = self.k
        a = self.a
        nu = self.nu
        R = self.R

        # FIXME
        u = 0

        for m in range(self.M+1, self.m_max+1):
            ac = self.fft_a_coef[m-1]
            u += ac * jv(m*nu, k*r) * np.sin(m*nu*(th-a))

        return u

    def eval_bc_extended(self, arg, sid):
        """
        Evaluate extended boundary condition.

        If self.var_compute_a is false, subtract the sines.
        """
        a = self.a
        nu = self.nu

        bc = self.eval_bc__no_reg(arg, sid)
        r, th = self.arg_to_polar(arg, sid)

        if sid == 0:
            if self.regularize_bc != RegularizeBc.none:
                if self.regularize_bc == RegularizeBc.fft:
                    b_coef = self.fft_b_coef[:self.M]
                elif self.regularize_bc == RegularizeBc.known:
                    b_coef = self.b_coef

                for m in range(1, len(b_coef)+1):
                    bc -= b_coef[m-1] * np.sin(m*nu*(th - a))

            return bc

        elif sid == 1:
            return bc
        elif sid == 2:
            return bc


class SingularProblem_BData(BData):

    def __init__(self, problem):
        self.problem = problem

    def eval_phi0(self, th):
        return self.problem.eval_phi0(th)
