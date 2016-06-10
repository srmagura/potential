import numpy as np

import abcoef
import fourier

def calc_a_coef(problem, M):
    fourier_N = 8192
    ode_N = 256

    a = problem.a
    nu = problem.nu
    k = problem.k
    R = problem.R

    arc_dst = lambda func: fourier.arc_dst(a, func, N=fourier_N)

    xspan = np.linspace(0, k*R, ode_N)
    h = k*R/(ode_N-1)

    def get_x(n):
        return h*n

    f_fourier_cache = {}
    phi_fourier = arc_dst(lambda th: problem.eval_bc(th, 0))

    b_coef = np.zeros(M)

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

        # Explicit Runge-Kutta method of order 4
        s = 4
        rk_a = ((1/2,), (0, 1/2), (0, 0, 1))
        rk_b = (1/6, 1/3, 1/3, 1/6)
        rk_c = (0, 1/2, 1/2, 1)

        for n in range(ode_N-1):
            ksum = np.zeros(2)
            kk = np.zeros((2, s))
            for l in range(s):
                _y = np.zeros(2)
                _y[:] = sol[:, n]
                for ll in range(l):
                    _y += h * rk_a[l-1][ll] * kk[:, ll]

                kk[:, l] = eval_deriv(_y, xspan[n] + rk_c[l]*h)
                ksum += rk_b[l] * kk[:, l]

            sol[:, n+1] = sol[:, n] + h*ksum

        b_coef[m-1] = phi_fourier[m-1] - sol[0, -1]

        print('m=', m)
        #print('bm_exp=', problem.fft_b_coef[m-1])
        print('error=', abs(b_coef[m-1]-problem.fft_b_coef[m-1]))
        print()

    #print('b error:', np.max(np.abs(b_coef - problem.fft_b_coef[:M])))
    a_coef = abcoef.b_to_a(b_coef, k, R, nu)
    #print('a error:', np.max(np.abs(a_coef - problem.fft_a_coef[:M])))
    return a_coef
