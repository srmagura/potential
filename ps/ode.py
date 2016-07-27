import numpy as np

from scipy.integrate import odeint
from scipy.interpolate import interp1d

import abcoef
import fourier

# For best accuracy, would less than 1024 work?
fourier_N = 128

# Recommended: 1e-14
atol = rtol = 1e-5

def calc_z1_fourier(problem, M):
    eval_f1 = problem.eval_f1

    a = problem.a
    nu = problem.nu
    k = problem.k
    R = problem.R

    arc_dst = lambda func: fourier.arc_dst(a, func, N=fourier_N)

    # Estimate the accuracy of the DFT
    r = R
    d1 = arc_dst(lambda th: problem.eval_f1(r, th))[:M]
    d2 = fourier.arc_dst(a, lambda th: problem.eval_f1(r, th), N=8192)[:M]
    print('Fourier error:', np.max(np.abs(d1-d2)))

    h = R / 256

    global n_dst
    n_dst = 0

    def eval_deriv(Y, r):
        global n_dst
        assert r != 0

        n_dst += 1

        f_fourier = arc_dst(lambda th: problem.eval_f1(r, th))[:M]

        derivs = np.zeros(2*M)
        for i in range(0, 2*M, 2):
            m = i//2+1
            z1 = Y[i]
            z2 = Y[i+1]

            derivs[i] = z2
            derivs[i+1] = -z2/r - (k**2 - (m*nu/r)**2) * z1 + f_fourier[m-1]

        return np.array(derivs)

    sol = odeint(eval_deriv, np.zeros(2*M), [h, R], atol=atol, rtol=rtol)

    # Fourier coefficients at r=R
    z1_fourier = sol[1, 0::2]


    print('Number of dst:', n_dst)

    # Estimate accuracy of z1
    gq_fourier = arc_dst(lambda th: problem.eval_gq(r, th))[:M]
    z_fourier = z1_fourier + gq_fourier

    def eval_z(r, th):
        i = r_data0.index(r)

        q_fourier = arc_dst(lambda th: problem.eval_q(r, th))

        z = 0
        for m in range(1, M+1):
            z += z_fourier[m-1] * np.sin(m*nu*(th-a))
            z += q_fourier[m-1]  * np.sin(m*nu*(th-a))

        return z

    for th in range(len(th_data)):
        z_data[i] = eval_z(R, th_data[i])


    return z1_fourier
