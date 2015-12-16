import numpy as np
from scipy.fftpack import dst

from .problem import PizzaProblem

# DELETE
import collections

a = PizzaProblem.a
nu = PizzaProblem.nu

class BData:
    """
    A abstract class that represents Dirichlet data on the outer arc. You should
    make a subclass that defines a function:

        eval_phi0(self, th)

    which gives the Dirichlet data at the angle th on the arc. The purpose of
    this class is to compute the sine Fourier coefficients of the given function
    phi0. The function calc_coef() exposes this functionality.
    """

    def calc_fft_exp(self, Jmax):
        fourier_N = 1024
        th_data = np.linspace(a, 2*np.pi, fourier_N+1)

        discrete_phi0 = np.array([self.eval_phi0(th) for th in th_data])

        # Half-wave FFT. It would be more efficient to use an actual
        # half-wave FFT routine

        # Reverse array and remove last element
        reflected = -discrete_phi0[::-1][:-1]
        discrete_phi0 = np.concatenate((discrete_phi0[:-1], reflected))
        #print(len(discrete_phi0))
        #print('discrete_phi0')
        #print(discrete_phi0)
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

    def calc_coef2(self, M):
        """
        Return the first M sine Fourier series coefficients.
        """
        J_dict, coef = self.calc_fft_exp(M)

        ccoef = np.zeros(M+1)
        ccoef[0] = coef[J_dict[0]].real
        scoef = np.zeros(M+1)
        for J in range(1, M+1):
            ccoef[J] = (coef[J_dict[J]] + coef[J_dict[-J]]).real
            scoef[J] = (1j*(coef[J_dict[J]] - coef[J_dict[-J]])).real

        # The cosine Fourier coefficients should all be 0
        assert np.max(np.abs(ccoef)) < 1e-13

        return scoef[1:]

    def calc_coef(self, m_max):
        fourier_N = 1024

        # Slice at the end removes the endpoints
        th_data = np.linspace(a, 2*np.pi, fourier_N+1)[1:-1]

        discrete_phi0 = np.array([self.eval_phi0(th) for th in th_data])
        return dst(discrete_phi0, type=1)[:m_max] / fourier_N
