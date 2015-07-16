import itertools as it
import math
import collections
import numpy as np
import matplotlib.pyplot as plt

from solver import Result
from cs.cs import CircleSolver
from cs.debug import CsDebug
import matrices

# Grid size for boundary. Should be divisible by 2
fourier_N = 1024

class CsFourier(CircleSolver, CsDebug):
    """
    Solve over a circular domain using a Fourier basis.
    """

    # Fourier coefficients lower than this number (may) be thrown own
    fourier_lower_bound = 5e-15

    # Side length of square domain on which AP is solved
    AD_len = 2*np.pi

    # Radius of circular region Omega
    R = 2.3

    def get_Q(self, index):
        """
        Returns the matrix Q0 or Q1, depending on the value of `index`
        """
        columns = []

        for J in self.J_dict:
            ext = self.extend_basis(J, index)
            potential = self.get_potential(ext)
            projection = self.get_trace(potential)

            columns.append(projection - ext)

        return np.column_stack(columns)

    def choose_n_basis(self, J_range, c0_raw):
        """Choose which Fourier coefficients to keep"""
        for J in J_range:
            normalized = c0_raw[J] / fourier_N

            if abs(normalized) > self.fourier_lower_bound:
                return abs(J)

        return 0

    def calc_c0(self):
        """
        Use FFT to expand the Dirichlet data with respect to the Fourier
        basis.
        """
        grid = np.arange(0, 2*np.pi, 2*np.pi / fourier_N)
        discrete_phi = [self.problem.eval_bc(th) for th in grid]

        # All of the Fourier coefficients. Most should be nearly 0
        c0_raw = np.fft.fft(discrete_phi)

        # Figure out which basis terms are significant
        while True:
            Jminus = self.choose_n_basis(range(-fourier_N // 2, 1), c0_raw)
            Jplus = self.choose_n_basis(range(fourier_N // 2, 0, -1), c0_raw)
            Jmax = max(Jminus, Jplus)

            # Need to keep the variational formulation overdetermined
            Q_ratio = 2
            if Jmax*2+1 < len(self.gamma) / Q_ratio:
                break
            self.fourier_lower_bound *= 10

        # Make sure we don't take unreasonably few basis functions
        if Jmax < 5:
            Jmax = 5

        self.J_dict = collections.OrderedDict(((J, None) for J in
            range(-Jmax, Jmax+1)))

        i = 0
        self.c0 = np.zeros(len(self.J_dict), dtype=complex)

        for J in self.J_dict:
            self.J_dict[J] = i
            self.c0[i] = c0_raw[J] / fourier_N
            i += 1

    def extend_basis(self, J, index):
        """
        Construct the equation-based extension for the basis function
        $ \psi_{index}^{(J)} $.
        """
        ext = np.zeros(len(self.gamma), dtype=complex)
        for l in range(len(self.gamma)):
            i, j = self.gamma[l]
            r, th = self.get_polar(i, j)
            exp = np.exp(complex(0, J*th))

            if index == 0:
                ext[l] = self.extend_circle(r, exp, 0, -J**2 * exp, 0, J**4 * exp)
            else:
                ext[l] = self.extend_circle(r, 0, exp, 0, -J**2 * exp, 0)

        return ext

    def extend_boundary(self):
        """
        Construct the equation-based extension for the boundary data,
        as approximated by the Fourier coefficients c0 and c1.
        """
        boundary = np.zeros(len(self.gamma), dtype=complex)

        for l in range(len(self.gamma)):
            i, j = self.gamma[l]
            r, th = self.get_polar(i, j)

            xi0 = xi1 = 0
            d2_xi0_th = d2_xi1_th = 0
            d4_xi0_th = 0

            for J, i in self.J_dict.items():
                exp = np.exp(complex(0, J*th))
                xi0 += self.c0[i] * exp
                d2_xi0_th += -J**2 * self.c0[i] * exp
                d4_xi0_th += J**4 * self.c0[i] * exp

                xi1 += self.c1[i] * exp
                d2_xi1_th += -J**2 * self.c1[i] * exp

            boundary[l] = self.extend_circle(r, xi0, xi1, d2_xi0_th, d2_xi1_th,
                d4_xi0_th)
            boundary[l] += self.calc_inhomo_circle(r, th)

        return boundary

    def run(self):
        self.calc_c0()
        self.calc_c1()

        if self.verbose:
            print('Using {} basis functions.'.
                format(len(self.J_dict)))

        ext = self.extend_boundary()
        u_act = self.get_potential(ext) + self.ap_sol_f

        error = self.eval_error(u_act)

        result = Result()
        result.error = error
        result.u_act = u_act
        return result
