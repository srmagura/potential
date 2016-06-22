import numpy as np

from scipy.integrate import odeint
from scipy.interpolate import interp1d

import abcoef
import fourier

# For best accuracy, would less than 1024 work?
fourier_N = 16#1024

# Recommended: 1e-14
atol = rtol = 1e-5

def calc_z(problem, th_data, M):
    print('Cheating!')
    z_data = np.zeros(len(th_data))
    for i in range(len(th_data)):
        th = th_data[i]
        r = problem.boundary.eval_r(th)

        z_data[i] = problem.eval_expected__no_w(r, th)

    return z_data

    a = problem.a
    nu = problem.nu
    k = problem.k
    R = problem.R

    arc_dst = lambda func: fourier.arc_dst(a, func, N=fourier_N)

    # Estimate the accuracy of the DFT
    r = R
    d1 = arc_dst(lambda th: problem.eval_f_polar(r, th))[:M]
    d2 = fourier.arc_dst(a, lambda th: problem.eval_f_polar(r, th), N=8192)[:M]
    print('Fourier error:', np.max(np.abs(d1-d2)))

    h = R / 256
    r_data = np.array([0, h] + [problem.boundary.eval_r(th) for th in th_data])
    r_data = np.array(sorted(r_data))

    global n_dst
    n_dst = 0

    def eval_deriv(Y, r):
        global n_dst
        assert r != 0

        n_dst += 1
        f_fourier = arc_dst(lambda th: problem.eval_f_polar(r, th))[:M]

        derivs = np.zeros(2*M)
        for i in range(0, 2*M, 2):
            m = i//2+1
            z1 = Y[i]
            z2 = Y[i+1]

            derivs[i] = z2
            derivs[i+1] = -z2/r - (k**2 - (m*nu/r)**2) * z1 + f_fourier[m-1]

        return np.array(derivs)

    sol = odeint(eval_deriv, np.zeros(2*M), r_data[1:], atol=atol, rtol=rtol)

    # cut off the initial condition, we don't want it here
    z_fourier = sol[1:, 0::2]

    print('Number of dst:', n_dst)

    z_data = np.zeros(len(th_data))

    def eval_z(i, th=None):
        if th is None:
            th = th_data[i]

        z = 0
        for m in range(1, M+1):
            z += z_fourier[i, m-1] * np.sin(m*nu*(th-a))

        return z

    for i in range(len(th_data)):
        z_data[i] = eval_z(i)

    # Estimate error incurred by ODE solver
    i = (len(th_data)-1)//2
    r = problem.boundary.eval_r(th_data[i])
    expected = lambda th: problem.eval_expected__no_w(r, th) - eval_z(i, th)
    expected_fourier = fourier.arc_dst(a, expected)
    print(np.abs(expected_fourier[:M]))

    return z_data
