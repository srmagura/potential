import numpy as np
from scipy.special import jv, jvp

from solver import cart_to_polar

from .problem import PizzaProblem
from .sympy_problem import SympyProblem
from .bdata import BData
import problems.functions as functions


class SingH_BData(BData):

    def __init__(self, problem):
        self.problem = problem

    def eval_phi0(self, th):
        return self.problem.eval_phi0(th)


class SingH(PizzaProblem):
    """
    A singular problem whose singularity only has a homogeneous part.

    The Dirichlet data on segments 1 and 2 is 0. The Dirichlet data on
    segment 0 is given by a function phi0.
    """

    homogeneous = True
    expected_known = False
    silent = False

    M = 7

    def __init__(self, **kwargs):
        self.bdata = SingH_BData(self)

        if self.expected_known:
            m = self.m_max
        else:
            m = self.M

        self.fft_b_coef = self.bdata.calc_coef(m)

        if self.expected_known:
            self.bessel_R = np.zeros(len(self.fft_b_coef))

            for m in range(1, len(self.fft_b_coef)+1):
                self.bessel_R[m-1] = jv(m*self.nu, self.k*self.R)

            if not self.silent:
                print('[sing-h] fft_b_coef[m_max] =',
                    self.fft_b_coef[self.m_max-1])

    def calc_bessel_ratio(self, m, r):
        k = self.k
        R = self.R
        nu = self.nu

        ratio = jv(m*nu, k*r) / self.bessel_R[m-1]
        return float(ratio)

    def eval_expected_polar(self, r, th):
        a = self.a
        nu = self.nu

        u = 0

        for m in range(self.M+1, self.m_max+1):
            b = self.fft_b_coef[m-1]
            ratio = self.calc_bessel_ratio(m, r)
            u += b * ratio * np.sin(m*nu*(th-a))

        return u

    def eval_bc_extended(self, arg, sid):
        """
        Evaluate extended boundary condition.

        If self.var_compute_b is false, subtract the sines.
        """
        a = self.a
        nu = self.nu

        r, th = self.arg_to_polar(arg, sid)

        if sid == 0:
            bc = self.eval_phi0(th)

            if not self.var_compute_b:
                for m in range(1, self.M+1):
                    bc -= self.fft_b_coef[m-1] * np.sin(m*nu*(th - a))

            return bc

        elif sid == 1:
            return 0
        elif sid == 2:
            return 0


class VarProblem:
    """
    A problem where the b coefficients are calculated during the solution
    of the variational formulation.
    """
    var_compute_b = True

    def get_a_error(self):
        k = self.k
        R = self.R
        nu = self.nu

        expected = np.zeros(self.M)

        for m in range(1, self.M+1):
            expected[m-1] = self.fft_b_coef[m-1] / jv(m*nu, k*R)

        actual = self.a_coef
        return np.max(np.abs(expected-actual))


class Sine:

    k = 1.75

    expected_known = True
    m_max = 8

    def eval_phi0(self, th):
        a = self.a
        nu = self.nu
        return np.sin(2*nu*(th-a)) + np.sin(8*nu*(th-a))


class SingH_FFT_Sine(Sine, SingH):

    n_basis_dict = {
        16: (20, 5),
        32: (24, 11),
        64: (41, 18),
        128: (53, 28),
        256: (65, 34),
        None: (80, 34),
    }


class SingH_Var_Sine(VarProblem, Sine, SingH):

    n_basis_dict = {
        16: (21, 9),
        32: (24, 15),
        64: (42, 27),
        128: (55, 31),
        256: (65, 34),
        None: (80, 34),
    }


class Sine8:

    k = 1.75

    expected_known = True
    silent = True
    m_max = 8

    def eval_phi0(self, th):
        a = self.a
        nu = self.nu
        return np.sin(8*nu*(th-a))


class SingH_FFT_Sine8(Sine8, SingH):

    n_basis_dict = {
        16: (20, 5),
        32: (24, 11),
        64: (41, 18),
        128: (53, 28),
        256: (65, 34),
        None: (80, 34),
    }


class SingH_Var_Sine8(VarProblem, Sine8, SingH):

    n_basis_dict = {
        16: (21, 9),
        32: (24, 15),
        64: (42, 27),
        128: (55, 31),
        256: (65, 34),
        None: (80, 34),
    }


class SingH_FFT_Hat(SingH):

    k = 5.5

    n_basis_dict = {
        16: (24, 6),
        32: (33, 8),
        64: (42, 12),
        128: (65, 18),
        256: (80, 24),
        512: (100, 30),
        1024: (120, 35),
    }

    expected_known = False
    m_max = 199

    def eval_phi0(self, th):
        return functions.eval_hat_th(th)


class SingH_FFT_Parabola(SingH):

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
