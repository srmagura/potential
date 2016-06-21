import numpy as np

from scipy.integrate import odeint
from scipy.interpolate import interp1d

import abcoef
import fourier

fourier_N = 128  # Fourier error will be approx 1e-15

def calc_z(problem, th_data, M):
    a = problem.a
    nu = problem.nu
    k = problem.k
    R = problem.R

    arc_dst = lambda func: fourier.arc_dst(a, func, N=fourier_N)

    # Estimate the accuracy of the DFT
    #r = 1
    #d1 = arc_dst(lambda th: problem.eval_f_polar(r, th))[:M]
    #d2 = fourier.arc_dst(a, lambda th: problem.eval_f_polar(r, th), N=2048)[:M]
    #print('Fourier error:', np.max(np.abs(d1-d2)))

    h = R / 2048
    r_data = [0, h] + [problem.boundary.eval_r(th) for th in th_data]
    r_data += list(np.linspace(h, R, 256)[1:])
    r_data.sort()
    r_data = np.array(r_data)

    f_fourier_cache = {}
    z_fourier = {}

    rth_map = {}
    for th in th_data:
        z_fourier[th] = np.zeros(M)

        for i in range(r_data):
            if boundary.eval_r(th) == r_data[i]:
                rth_map[th] = r_data[i]

    for m in range(1, M+1):
        def eval_deriv(Y, r):
            assert r != 0

            z1 = Y[0]
            z2 = Y[1]

            if r not in f_fourier_cache:
                f_fourier_cache[r] = arc_dst(lambda th: problem.eval_f_polar(r, th))[:M]

            d_z1_r = z2
            d_z2_r = -z2/r - (k**2 - (m*nu/r)**2) * z1 + f_fourier_cache[r][m-1]
            return np.array([d_z1_r, d_z2_r])

        odeint_options = {'atol': 1e-14}
        sol = odeint(eval_deriv, [0, 0], r_data[1:], **odeint_options).T

        for th in th_data:
            z_fourier[th][m-1] = sol[rth_map[th]-1]

    z_data = np.zeros(len(th_data))

    def eval_z(i, th):
        z = 0
        for m in range(1, M+1):
            z += z_fourier[i, m-1] * np.sin(m*nu*(th-a))

        return z

    for i in range(len(th_data)):
        for j in range(len(r_data)):
            z_data[i] = eval_z(i)

    # Calculate error incurred by ODE solver, for the last th value
    i = len(th_data)-1
    r = problem.boundary.eval_r(th_data[i])
    expected = lambda th: problem.eval_expected__no_w(r, th) - eval_z(i, th)
    expected_fourier = fourier.arc_dst(a, expected)
    print(np.abs(expected_fourier[:M]))

    return z_data
