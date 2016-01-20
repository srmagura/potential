import itertools as it

import numpy as np
from scipy.fftpack import dst

class APSolver:
    """
    Solves the auxiliary problem and applies the operater L.
    """

    eigenvals = None

    def __init__(self, N, order, AD_len, k):
        self.N = N
        self.order = order
        self.AD_len = AD_len
        self.k = k

    def calc_eigenvals(self):
        N = self.N
        order = self.order
        AD_len = self.AD_len
        k = self.k

        self.eigenvals = np.zeros((N-1, N-1))

        h = AD_len / N

        h2 = h**2
        k2 = k**2

        # Exploit symmetry to speed up the computation:
        # The (j, l) eigenvalue is the same as the (l, j) eigenvalue
        for j in range(1, N):
            for l in range(1, j+1):
                if order == 2:
                    eigenval = k2 + (2*np.cos(np.pi*j/N) +
                        2*np.cos(np.pi*l/N) - 4)/h2
                if order == 4:
                    eigenval = (h2*k2*(np.cos(np.pi*j/N) +
                        np.cos(np.pi*l/N) + 4) +
                        4*np.cos(np.pi*j/N)*np.cos(np.pi*l/N) +
                        8*np.cos(np.pi*j/N) +
                        8*np.cos(np.pi*l/N) - 20)/(6*h2)

                self.eigenvals[j-1, l-1] = eigenval
                self.eigenvals[l-1, j-1] = eigenval

    def _fourier(self, Bf, solve):
        N = self.N
        AD_len = self.AD_len

        assert N == Bf.shape[0] + 1
        h = AD_len / N

        if self.eigenvals is None:
            self.calc_eigenvals()

        scoef = np.zeros((N-1, N-1), dtype=complex)

        for i in range(1, N):
            scoef[i-1, :] = dst(Bf[i-1, :], type=1)
        for j in range(1, N):
            scoef[:, j-1] = dst(scoef[:, j-1], type=1)

        scoef *= h**2 / 2

        if solve:
            # When solving the AP
            fourier_coef_sol = scoef / self.eigenvals
        else:
            # When applying L
            fourier_coef_sol = scoef * self.eigenvals

        u_f = np.zeros((N-1, N-1), dtype=complex)
        for i in range(1, N):
            u_f[i-1, :] = dst(fourier_coef_sol[i-1, :], type=1)
        for j in range(1, N):
            u_f[:, j-1] = dst(u_f[:, j-1], type=1)

        u_f /= (2*AD_len**2)
        return u_f

    def solve(self, Bf):
        return self._fourier(Bf, True)

    def apply_L(self, v):
        return self._fourier(v, False)


def arc_dst(a, func):
    N = 1024

    # The slice at the end removes the endpoints
    th_data = np.linspace(a, 2*np.pi, N+1)[1:-1]

    discrete_func = np.array([func(th) for th in th_data])
    return dst(discrete_func, type=1) / N
