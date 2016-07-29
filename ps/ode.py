import numpy as np

from scipy.integrate import odeint
from scipy.interpolate import interp1d

import abcoef
import fourier

# Recommended: 512
fourier_N = 512

# Recommended: 1e-10
atol = rtol = 1e-10

def my_print(s):
    print('(ode) {}'.format(s))

def calc_z1_fourier(eval_f1, a, nu, k, R, M):
    arc_dst = lambda func: fourier.arc_dst(a, func, N=fourier_N)

    my_print('tol: {}'.format(atol))

    # Estimate the accuracy of the DFT
    r = R
    d1 = arc_dst(lambda th: eval_f1(r, th))[:M]
    d2 = fourier.arc_dst(a, lambda th: eval_f1(r, th), N=8192)[:M]
    my_print('Fourier discretization error: {}'.format(np.max(np.abs(d1-d2))))

    h = R / 256


    global n_dst
    n_dst = 0

    def eval_deriv(Y, r):
        global n_dst
        assert r != 0

        n_dst += 1

        f_fourier = arc_dst(lambda th: eval_f1(r, th))[:M]

        derivs = np.zeros(2*M)
        for i in range(0, 2*M, 2):
            m = i//2+1
            z1 = Y[i]
            z2 = Y[i+1]

            derivs[i] = z2
            derivs[i+1] = -z2/r - (k**2 - (m*nu/r)**2) * z1 + f_fourier[m-1]

        return np.array(derivs)

    sol = odeint(eval_deriv, np.zeros(2*M), [h, R], atol=atol, rtol=rtol)
    my_print('Number of dst: {}'.format(n_dst))

    # Fourier coefficients at r=R
    return sol[1, 0::2]
