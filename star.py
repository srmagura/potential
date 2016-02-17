import math

import numpy as np
from scipy.special import jv
import matplotlib.pyplot as plt

from problems.sing_h import H_Hat, H_SineRange, H_Sine8

problem = H_Hat()
print('[{}]'.format(problem.name))
print('m_max =', problem.m_max)

k = problem.k
R = problem.R
a = problem.a
nu = problem.nu
print('k =', k)
M = 7

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
            W[i, m-1] = jv(m*nu, k*r) * np.sin(m*nu*(th-a))

    for m in range(M+1, m1+1):
        W[:, m-1] = W[:, m-1] / jv(m*nu, k*R)

    result = np.linalg.lstsq(W, phi0_data)

    a_coef = result[0]
    rank = result[2]

    if rank == m1:
        print('Full rank')
    else:
        print('\n!!!! Rank deficient !!!!\n')

    return a_coef[:M]

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

    plt.plot(x_data0, y_data0, linestyle='--', color='gray',
        label=r'$\Gamma_\mathrm{arc}$')
    plt.plot(x_data1, y_data1, label=r'$\Gamma_3$')

    plt.legend(loc='upper right', fontsize=16)
    plt.axes().set_aspect('equal', 'datalim')
    plt.show()

def test_many():
    min_error3 = float('inf')
    for m1 in range(10, 200, 2):
        print('\n----- m1={} -----'.format(m1))
        a_coef = calc_a_coef(m1)
        error3 = np.max(np.abs(a_coef[:3] - problem.fft_a_coef[:3]))

        print('a_coef=', a_coef)
        print('max_error(1-3)={}'.format(error3), end=' ')
        if error3 < min_error3:
            min_error3 = error3
            print('!!!')
        else:
            print()

test_many()
