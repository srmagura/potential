import numpy as np

from scipy.interpolate import interp1d

import abcoef
import fourier

rk_s = 4
rk_a = ((1/2,), (0, 1/2), (0, 0, 1))
rk_b = (1/6, 1/3, 1/3, 1/6)
rk_c = (0, 1/2, 1/2, 1)

def erk4(eval_deriv, Y, x, h):
    """ Explicit Runge-Kutta method of order 4 """
    ksum = np.zeros(2)
    kk = np.zeros((2, rk_s))
    for l in range(rk_s):
        _y = np.zeros(2)
        _y[:] = Y
        for ll in range(l):
            _y += h * rk_a[l-1][ll] * kk[:, ll]

        kk[:, l] = eval_deriv(_y, x + rk_c[l]*h)
        ksum += rk_b[l] * kk[:, l]

    return Y + h*ksum

ab_s = 4
ab_coef = (-3/8, 37/24, -59/24, 55/24)

def ab4(eval_deriv, sol, n, h):
    """ Adams-Bashforth method of order 4 """
    derivs = np.zeros(2)
    for l in range(ab_s):
        x = (n+l)*h
        if n+l < 0:
            _sol = np.zeros((2,))
        else:
            _sol = sol[:, n+l]

        derivs += ab_coef[l] * eval_deriv(_sol, x)

    return sol[:, n+ab_s-1] + h*derivs


fourier_N = 128  # Fourier error will be approx 1e-15
ode_N = 128

def calc_z(problem, M):
    # MEGA FIXME
    #return problem.eval_expected__no_w

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

    xspan = np.linspace(0, k*R, ode_N)
    h = k*R/(ode_N-1)

    f_fourier_cache = {}

    eval_z_fourier = [None]*M

    for m in range(1, M+1):
        def eval_deriv(Y, x):
            if x == 0:
                return np.array([0, 0])

            z1 = Y[0]
            z2 = Y[1]

            if x not in f_fourier_cache:
                r = x/k
                f_fourier_cache[x] = arc_dst(lambda th: problem.eval_f_polar(r, th))[:M]

            d_z1_x = z2
            d_z2_x = (-x * z2 - (x**2 - (m*nu)**2) * z1) / x**2
            d_z2_x += f_fourier_cache[x][m-1] / k**2
            return np.array([d_z1_x, d_z2_x])

        sol = np.zeros((2, ode_N))

        for n in range(1, ab_s-1):
            sol[:, n+1] = erk4(eval_deriv, sol[:, n], xspan[n], h)

        for n in range(0, ode_N-ab_s):
            sol[:, n+ab_s] = ab4(eval_deriv, sol, n, h)

        eval_z_fourier[m-1] = interp1d(xspan/k, sol[0, :], kind='cubic', assume_sorted=True)

    def eval_z(r, th):
        z = 0
        for m in range(1, M+1):
            z += eval_z_fourier[m-1](r) * np.sin(m*nu*(th-a))

        return z

    q = lambda th: problem.eval_expected__no_w(R, th) - eval_z(R, th)
    q_f = arc_dst(q)
    print(q_f[:M])

    return eval_z
