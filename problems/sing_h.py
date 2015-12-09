import numpy as np
from scipy.special import jv, jvp


from solver import cart_to_polar

from problems import RegularizeBc
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
    regularize_bc = RegularizeBc.fft

    def __init__(self, **kwargs):
        self.bdata = SingH_BData(self)

    def setup(self):
        if self.expected_known:
            # Fourier coefficients may get calculated twice for dual
            m_max = max(self.M, self.m_max)
        else:
            m_max = self.M

        self.fft_b_coef = self.bdata.calc_coef(m_max)

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

        r, th = self.arg_to_polar(arg, sid)

        if sid == 0:
            bc = self.eval_phi0(th)

            if self.regularize_bc != RegularizeBc.none:
                if self.regularize_bc == RegularizeBc.fft:
                    b_coef = self.fft_b_coef[:self.M]
                elif self.regularize_bc == RegularizeBc.known:
                    b_coef = self.b_coef

                for m in range(1, len(b_coef)+1):
                    bc -= b_coef[m-1] * np.sin(m*nu*(th - a))

            return bc

        elif sid == 1:
            return 0
        elif sid == 2:
            return 0

    def get_a_error(self):
        return np.max(np.abs(
            self.a_coef - self.fft_a_coef[:len(self.a_coef)]
        ))


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

    #k = 5.5
    k = 1

    n_basis_dict = {
        16: (17, 3),
        32: (33, 7),
        64: (36, 15),
        128: (39, 21),
        256: (42, 23),
        512: (65, 43),
        1024: (80, 45),
    }

    expected_known = True
    m_max = 199

    def eval_phi0(self, th):
        return -(th - np.pi/6) * (th - 2*np.pi)
