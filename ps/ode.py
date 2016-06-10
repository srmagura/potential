import numpy as np

import abcoef
import fourier

def calc_a_coef(problem, M):
    fourier_N = 256  # best: 8192
    adams_N = 64 # best: 64 for k=1; 256 for k=5.5

    a = problem.a
    nu = problem.nu
    k = problem.k
    R = problem.R

    arc_dst = lambda func: fourier.arc_dst(a, func, N=fourier_N)

    xspan = np.linspace(0, k*R, adams_N)
    h = k*R/(adams_N-1)

    def get_x(n):
        return h*n

    #f_fourier = np.zeros((adams_N, M))
    #for n in range(adams_N):
    #    r = xspan[n] / k
    #    f_fourier[n, :] = arc_dst(lambda th: problem.eval_f_polar(r, th))[:M]

    phi_fourier = arc_dst(lambda th: problem.eval_bc(th, 0))

    b_coef = np.zeros(M)

    #for m in range(1, M+1):
    for m in [5]:
        def eval_deriv(Y, x):
            if x <= 0:
                return np.array([0, 0])

            z1 = Y[0]
            z2 = Y[1]

            n = int(np.round(x / h))
            #assert x == get_x(n)

            # FIXME
            r = x/k
            f_fourier = {}
            f_fourier[n, m-1] = arc_dst(lambda th: problem.eval_f_polar(r, th))[m-1]

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

        sol2 = np.zeros((2, adams_N))

        # erk 4
        s = 4
        rk_a = ((1/2,), (0, 1/2), (0, 0, 1))
        rk_b = (1/6, 1/3, 1/3, 1/6)
        rk_c = (0, 1/2, 1/2, 1)

        #s=2
        #rk_a = ((1/2,),)
        #rk_b = (0, 1)
        #rk_c = (0, 1/2)

        #s=1
        #rk_b = (1,)
        #rk_c = (0,)

        for n in range(adams_N-1):
            ksum = np.zeros(2)
            kk = np.zeros((2, s))
            for l in range(s):
                _y = np.zeros(2)
                _y[:] = sol2[:, n]
                for ll in range(l):
                    _y += h * rk_a[l-1][ll] * kk[:, ll]

                kk[:, l] = eval_deriv(_y, xspan[n] + rk_c[l]*h)
                ksum += rk_b[l] * kk[:, l]

            sol2[:, n+1] = sol2[:, n] + h*ksum

        '''def eval_deriv3(x, Y):
            print(x, Y)
            return eval_deriv(Y, x)


        from scipy.integrate import ode as scipy_ode
        od = scipy_ode(eval_deriv3).set_integrator('dopri5')
        od.set_initial_value([0,0], 0)
        xspan3 = []
        sol3 = []
        while od.successful() and od.t <= xspan[-1]:
            xspan3.append(od.t)
            sol3.append(od.integrate(od.t+h)[0])'''


        #print('m=', m)
        #print('bm_exp=', problem.fft_b_coef[m-1])
        #print('error=', abs(b_coef[m-1]-problem.fft_b_coef[m-1]))

        from scipy.special import jv
        import matplotlib.pyplot as plt

        def eval_v(r, th):
            return jv(nu/2, k*r) * np.sin(nu/2*(th-a))

        z_data = []
        j_data = []

        for x in xspan3:
            r = x/k
            z_data.append(arc_dst(lambda th: eval_v(r, th) - problem.eval_g(r, th) - problem.eval_v_asympt(r, th))[m-1])
            j_data.append(jv(m*nu, r))

        #plt.plot(xspan, z_data, label='expected')
        #plt.plot(xspan, j_data, label='bessel')
        #plt.plot(xspan, sol[0, :], label='numerical, adams')
        #plt.plot(xspan, sol2[0, :], label='numerical, rk')
        sol3 = np.array(sol3)
        plt.plot(xspan3, sol3-z_data, label='numerical, scipy')
        #plt.plot(xspan, sol[0,:]-sol2[0,:], label='diff')

        print('diff between solutions:', np.max(np.abs(sol[0,:]-sol2[0,:])))

        print(xspan3[-1])
        print('diff(1-3):', np.max(np.abs(sol[0,-1]-sol3[-1])))

        plt.legend(loc='upper left')
        plt.show()


    #print('b error:', np.max(np.abs(b_coef - problem.fft_b_coef[:M])))
    a_coef = abcoef.b_to_a(b_coef, k, R, nu)
    #print('a error:', np.max(np.abs(a_coef - problem.fft_a_coef[:M])))
    return a_coef
