import numpy as np
from scipy.integrate import quad
import collections
from .problem import PizzaProblem

a = PizzaProblem.a
nu = np.pi / (2*np.pi - a)

class BData:
    '''
    A abstract class that represents Dirichlet data on the outer arc. You should 
    make a subclass that defines a function:

        eval_phi0(self, th)

    which gives the Dirichlet data at the angle th on the arc. The purpose of
    this class is to compute the sine Fourier coefficients of the given function
    phi0. The function calc_coef() exposes this functionality. 
    
    If you have an analytic expression for the sine Fourier coefficients,
    you should write a function in your subclass:
        
        calc_coef_analytic(self, M):

    that returns the first M of these coefficients.
    '''    

    def calc_fft_exp(self, Jmax):
        fourier_N = 2**10
        th_data = np.linspace(a, 2*np.pi, fourier_N+1)

        discrete_phi0 = np.array([self.eval_phi0(th) for th in th_data])

        # Half-wave FFT. It would be more efficient to use an actual 
        # half-wave FFT routine

        # Reverse array and remove last element
        reflected = -discrete_phi0[::-1][:-1]
        discrete_phi0 = np.concatenate((discrete_phi0[:-1], reflected))
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
        ''' Return the first M sine Fourier series coefficients.'''
        if hasattr(self, 'calc_coef_analytic'):
            return self.calc_coef_analytic(M)
        else:
            return self.calc_fft(M)
