# To allow __main__ in subdirectory
import sys
sys.path.append(sys.path[0] + '/..')

import numpy as np
import sympy
import matplotlib.pyplot as plt

import chebyshev
import problems.bdata as bdata
import problems.sing_ih as sing_ih

k = 1
a = np.pi/6
R = 2.3
M = 7

nu = np.pi / (2*np.pi - a)

def eval_phi0(th):
    return bdata.eval_hat_th(th)

v_asympt_lambda = sympy.lambdify(sympy.symbols('k R r th'), 
    sing_ih.get_v_asympt_expr())

class MyBData(bdata.BData):

    def eval_phi0(self, th):
        phi0 = eval_phi0(th)
        phi0 -= sing_ih.eval_v(k, R, th)
        return phi0

my_bdata = MyBData()
b_coef = my_bdata.calc_coef(M)

def eval_data(th):
    data = eval_phi0(th)
    data -= v_asympt_lambda(k, R, R, th)

    for m in range(1, M):
        data -= b_coef[m-1] * np.sin(m*nu*(th-a))

    return data

g_eps = 1e-3
g_span = np.pi - .5*a + g_eps
g_center = np.pi + .5*a

def eval_g(t):
    return g_span*t + g_center

def eval_g_inv(th):
    return (th - g_center) / g_span

def do_test(n_basis, prev_error):
    cheb_roots = chebyshev.get_chebyshev_roots(1024)

    to_fit = np.array([eval_data(eval_g(t)) for t in cheb_roots])
    c0 = np.polynomial.chebyshev.chebfit(cheb_roots, to_fit, n_basis-1)

    N = 1024
    th_data = np.linspace(a, 2*np.pi, N)
    exact_data = np.zeros(N)
    expansion_data = np.zeros(N)

    for i in range(N):
        th = th_data[i]
        exact_data[i] = eval_data(th)

        t = eval_g_inv(th)
        for J in range(n_basis):
            expansion_data[i] += c0[J] * chebyshev.eval_T(J, t)

    error = np.max(np.abs(expansion_data - exact_data))

    print('---- n_basis: {} ----'.format(n_basis))
    print('Chebyshev error:', error)
    print()
    return error

def convergence_test():
    prev_error = None

    for n_basis in range(10, 100, 10):
        prev_error = do_test(n_basis, prev_error)

convergence_test()
