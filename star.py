import math
import sys

import numpy as np
import scipy

from scipy.special import jv
import bessel

jv = bessel.mpjv

import numpy as np
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

def print_array(array):
    assert np.max(np.abs(scipy.imag(array))) == 0
    for x in scipy.real(array):
        print('{:.15e}'.format(x), end=' ')
    print()


def test_many(m1_list=None):
    min_error7 = float('inf')

    if m1_list is None:
        m1_list = range(2, 200, 2)

    for m1 in m1_list:
        print('\n----- m1={} -----'.format(m1))
        a_coef = calc_a_coef(m1)
        error7 = np.max(np.abs(a_coef - problem.fft_a_coef[:7]))

        print('a_coef=')
        print_array(a_coef)
        print('max_error(a1-a7)={}'.format(error7), end=' ')
        if error7 < min_error7:
            min_error7 = error7
            print('!!!')
        else:
            print()

test_many([144])
