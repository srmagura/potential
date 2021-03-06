import numpy as np

from scipy.integrate import odeint
from scipy.interpolate import interp1d

import abcoef
import fourier

# Recommended: 1024
fourier_N = 1024

# Recommended: 1e-12
atol = rtol = 1e-12

# Default: 500?
mxstep = 3000

def my_print(s):
    print('(ode) {}'.format(s))

def calc_z1_fourier(initial_func, d_initial_func, eval_f1, a, nu, k, R0, R, M):
    arc_dst = lambda func: fourier.arc_dst(a, func, N=fourier_N)

    my_print('R0: {}'.format(R0))
    my_print('tol: {}'.format(atol))
    my_print('mxstep: {}'.format(mxstep))

    # Estimate the accuracy of the DFT
    r = R
    d1 = arc_dst(lambda th: eval_f1(r, th))[:M]
    d2 = fourier.arc_dst(a, lambda th: eval_f1(r, th), N=8192)[:M]
    my_print('Fourier discretization error: {}'.format(np.max(np.abs(d1-d2))))

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

    initial_fourier = arc_dst(lambda th: initial_func(R0, th))[:M]
    d_initial_fourier = arc_dst(lambda th: d_initial_func(R0, th))[:M]

    ic = np.zeros(2*M)
    for i in range(0, 2*M, 2):
        m = i//2+1
        ic[i] = initial_fourier[m-1]
        ic[i+1] = d_initial_fourier[m-1]

    sol = odeint(eval_deriv, ic, [R0, R], atol=atol, rtol=rtol, mxstep=mxstep)
    my_print('Number of dst: {}'.format(n_dst))

    # Fourier coefficients at r=R
    return sol[1, 0::2]
