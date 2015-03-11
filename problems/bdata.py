import numpy as np
from scipy.integrate import quad
import collections

a = np.pi/6
nu = np.pi / (2*np.pi - a)


class BData:

    analytic_known = False

    def __init__(self):
        self.quad_error = []

    def calc_fft_exp(self, Jmax):
        fourier_N = 2**10
        th_data = np.linspace(a, 2*np.pi, fourier_N+1)[:-1]

        discrete_phi0 = np.array([self.eval_phi0(th) for th in th_data])

        # Half-wave FFT. It would be more efficient to use an actual 
        # half-wave FFT routine
        discrete_phi0 = np.concatenate((discrete_phi0, -discrete_phi0))
        coef_raw = np.fft.fft(discrete_phi0)

        J_dict = collections.OrderedDict(((J, None) for J in 
            range(-Jmax, Jmax+1)))

        i = 0
        coef = np.zeros(len(J_dict), dtype=complex)

        for J in J_dict:
            J_dict[J] = i
            coef[i] = coef_raw[J] / (fourier_N*2)
            i += 1

        return J_dict, coef

    def calc_fft(self, Jmax):
        J_dict, coef = self.calc_fft_exp(Jmax)

        ccoef = np.zeros(Jmax+1)
        ccoef[0] = coef[J_dict[0]].real
        scoef = np.zeros(Jmax+1)
        for J in range(1, Jmax+1):
            ccoef[J] = (coef[J_dict[J]] + coef[J_dict[-J]]).real
            scoef[J] = (1j*(coef[J_dict[J]] - coef[J_dict[-J]])).real

        assert np.max(np.abs(ccoef)) < 1e-13
        return scoef[1:]

    def calc_coef(self, M):
        if self.analytic_known:
            return self.calc_coef_analytic(M)
        else:
            return self.calc_fft(M)


class Parabola(BData):

    analytic_known = True

    def eval_phi0(self, th):
        return -(th - np.pi/6) * (th - 2*np.pi) 

    def calc_coef_analytic(self, M):
        coef = np.zeros(M)

        for m in range(1, M+1):
            if m % 2 == 0:
                coef[m-1] = 0
            else:
                coef[m-1] = 8/(2*np.pi - a) * 1/(m*nu)**3

        return coef


class Hat(BData):

    def eval_phi0(self, th):
        x = (2*th-(2*np.pi+a))/(2*np.pi-a)

        if abs(abs(x)-1) < 1e-15 or abs(x) > 1:
            return 0
        else:
            arg = -1/(1-abs(x)**2)
            return np.exp(arg)
