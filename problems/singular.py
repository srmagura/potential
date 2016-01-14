import numpy as np
from scipy.special import jv
from scipy.fftpack import dst

from .problem import PizzaProblem

from enum import Enum
RegularizeBc = Enum('RegularizeBc', 'none fft known')

class SingularProblem(PizzaProblem):
    """
    A problem that should be treated as if it has a singularity.
    """

    m_max = 100

    def __init__(self, **kwargs):
        if 'eval_u_asympt' in kwargs:
            self.eval_u_asympt = kwargs.pop('eval_u_asympt')
        else:
            self.eval_u_asympt = lambda r, th: 0

        if 'to_dst' in kwargs:
            self.regularize_bc = RegularizeBc.fft
            self.to_dst = kwargs.pop('to_dst')
        else:
            self.regularize_bc = RegularizeBc.none
            self.to_dst = lambda th: 0

    def setup(self):
        if self.expected_known:
            m_max = max(self.M, self.m_max)
        else:
            m_max = self.M

        self.calc_fft_coef(m_max)

    def print_b(self):
        self.calc_fft_coef(self.m_max)

        np.set_printoptions(precision=4)

        n = 5
        print('Last {} of {} b coefficients:'.
            format(n, len(self.fft_b_coef)))

        print(self.fft_b_coef[-n:])
        print()

    def a_to_b(self, a_coef):
        return self._convert_ab(a_coef, 'a')

    def b_to_a(self, b_coef):
        return self._convert_ab(b_coef, 'b')

    def _convert_ab(self, coef, coef_type):
        assert coef_type in ('a', 'b')

        k = self.k
        R = self.R
        nu = self.nu

        other_coef = np.zeros(len(coef), dtype=complex)

        for m in range(1, len(coef)+1):
            factor = jv(m*nu, k*R)
            if coef_type == 'b':
                factor = 1/factor

            other_coef[m-1] = coef[m-1] * factor

        return other_coef

    def calc_fft_coef(self, m_max):
        fourier_N = 1024

        # The slice at the end removes the endpoints
        th_data = np.linspace(self.a, 2*np.pi, fourier_N+1)[1:-1]

        discrete_phi0 = np.array([self.to_dst(th) for th in th_data])
        self.fft_b_coef = dst(discrete_phi0, type=1)[:m_max] / fourier_N

        self.fft_a_coef = self.b_to_a(self.fft_b_coef)

        print('First {} a coefficients:'.format(self.M))
        np.set_printoptions(precision=15)
        print(self.fft_a_coef[:self.M])
        print()

    def set_a_coef(self, a_coef):
        self.a_coef = a_coef
        self.b_coef = self.a_to_b(a_coef)

    def get_a_error(self):
        return np.max(np.abs(self.fft_a_coef[:self.M] - self.a_coef))

    def eval_expected_polar(self, r, th):
        k = self.k
        a = self.a
        nu = self.nu
        R = self.R

        u = self.eval_expected_polar__no_reg(r, th)
        for m in range(self.M+1, self.m_max+1):
            ac = self.fft_a_coef[m-1]
            u += ac * jv(m*nu, k*r) * np.sin(m*nu*(th-a))

        return u - self.eval_u_asympt(r, th)

    def eval_bc_extended(self, arg, sid):
        """
        Evaluate extended boundary condition.

        If self.var_compute_a is false, subtract the sines.
        """
        a = self.a
        nu = self.nu
        k = self.k
        R = self.R

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

            return bc - self.eval_u_asympt(r, th)

        if sid == 1:
            if th < 1.5*np.pi:
                return 0
            else:
                return bc - self.eval_u_asympt(r, th)

        elif sid == 2:
            return 0
