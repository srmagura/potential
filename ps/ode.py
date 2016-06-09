import numpy as np

import abcoef
import fourier

def calc_a_coef(problem, M):
    fourier_N = 8192  # best: 8192
    adams_N = 256 # best: 64 for k=1; 256 for k=5.5

    a = problem.a
    nu = problem.nu
    k = problem.k
    R = problem.R

    arc_dst = lambda func: fourier.arc_dst(a, func, N=fourier_N)

    xspan = np.linspace(0, k*R, adams_N)
    h = k*R/(adams_N-1)

    def get_x(n):
        return h*n

    f_fourier = np.zeros((adams_N, M))
    for n in range(adams_N):
        r = xspan[n] / k
        f_fourier[n, :] = arc_dst(lambda th: problem.eval_f_polar(r, th))[:M]

    phi_fourier = arc_dst(lambda th: problem.eval_bc(th, 0))

    b_coef = np.zeros(M)

    for m in range(1, M+1):
        def eval_deriv(Y, x):
            if x <= 0:
                return np.array([0, 0])

            z1 = Y[0]
            z2 = Y[1]

            n = int(np.round(x / h))
            assert x == get_x(n)

            d_z1_x = z2
            d_z2_x = (-x * z2 - (x**2 - (m*nu)**2) * z1) / x**2
            d_z2_x += f_fourier[n, m-1] / k**2
            return np.array([d_z1_x, d_z2_x])

        sol = np.zeros((2, adams_N))

        def get_sol(n):
            if n >= 0:
                return sol[:, n]
            else:
                return np.zeros(2)

        # Adams-Bashforth method of order 5
        s = 5
        coef = (251/720, -637/360, 109/30, -1387/360, 1901/720)
        for n in range(-s+1, adams_N-s):
            derivs = np.zeros(2)
            for l in range(s):
                derivs += coef[l] * eval_deriv(get_sol(n+l), get_x(n+l))

            sol[:, n+s] = sol[:, n+s-1] + h*derivs


        b_coef[m-1] = phi_fourier[m-1] - sol[0, -1]

        #print('m=', m)
        #print('bm_exp=', problem.fft_b_coef[m-1])
        #print('error=', abs(b_coef[m-1]-problem.fft_b_coef[m-1]))

    print('b error:', np.max(np.abs(b_coef - problem.fft_b_coef[:M])))
    a_coef = abcoef.b_to_a(b_coef, k, R, nu)
    #print('a error:', np.max(np.abs(a_coef - problem.fft_a_coef[:M])))
    return a_coef
