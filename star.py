import math
from bessel import mpjv

import numpy as np
import matplotlib.pyplot as plt

from problems.sing_h import H_Hat, H_SineRange, H_Sine8

problem = H_Hat()

k = problem.k
R = problem.R
a = problem.a
nu = problem.nu
print(k)
M = 3

n_nodes = 1024
th_data = np.linspace(a, 2*np.pi, n_nodes + 2)[1:-1]

def eval_phi0(r, th):
    return problem.eval_expected_polar(r, th)

def get_boundary_r(th):
    return R + 0.5*np.sin(nu*(th-a))

def calc_a_coef(m1):
    phi0_data = np.zeros(n_nodes, dtype=complex)
    W = np.zeros((n_nodes, m1), dtype=complex)

    for i in range(n_nodes):
        th = th_data[i]
        r = get_boundary_r(th)

        phi0_data[i] = eval_phi0(r, th)
        for m in range(1, m1+1):
            W[i, m-1] = mpjv(m*nu, k*r) * np.sin(m*nu*(th-a))

    for m in range(M+1, m1+1):
        exp = -math.frexp(mpjv(m*nu, k*R))[1]
        W[:, m-1] = 2**exp * W[:, m-1]

    result = np.linalg.lstsq(W, phi0_data)

    a_coef = result[0]
    #rank = result[2]
    #print('Rank:', rank)
    error = np.max(np.abs(a_coef[:M] - problem.fft_a_coef[:M]))
    return error

def do_plot():
    x_data0 = np.zeros(n_nodes)
    y_data0 = np.zeros(n_nodes)

    x_data1 = np.zeros(n_nodes)
    y_data1 = np.zeros(n_nodes)

    for i in range(n_nodes):
        th = th_data[i]
        r = get_boundary_r(th)

        x_data0[i] = R*np.cos(th)
        y_data0[i] = R*np.sin(th)

        x_data1[i] = r*np.cos(th)
        y_data1[i] = r*np.sin(th)

    plt.plot(x_data0, y_data0, color='gray')
    plt.plot(x_data1, y_data1)
    plt.show()


min_error = float('inf')
for m1 in range(10, 200, 10):
    error = calc_a_coef(m1)

    print('m1={}   error={}'.format(m1, error), end=' ')
    if error < min_error:
        min_error = error
        print('!!!')
    else:
        print()
