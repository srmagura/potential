# To allow __main__ in subdirectory
import sys
sys.path.append(sys.path[0] + '/..')

import argparse

import numpy as np
import scipy
import matplotlib.pyplot as plt

import io_util
import problems
import problems.boundary

import ps.coordinator

from star.shared import calc_a_coef

parser = argparse.ArgumentParser()

io_util.add_arguments(parser, ('boundary'))
args = parser.parse_args()

R = 2.3
a = np.pi/6
boundary = problems.boundary.boundaries[args.boundary](R)

print('[{}]'.format(boundary.name))
print('R = {}'.format(R))

def do_plot():
    n_nodes = 1024
    th_data = np.linspace(a, 2*np.pi, n_nodes + 2)[1:-1]

    x_data0 = np.zeros(n_nodes)
    y_data0 = np.zeros(n_nodes)

    x_data1 = np.zeros(n_nodes)
    y_data1 = np.zeros(n_nodes)

    for i in range(n_nodes):
        th = th_data[i]
        r = boundary.eval_r(th)

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

do_plot()
