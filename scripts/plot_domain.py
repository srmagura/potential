# To allow __main__ in subdirectory
import sys
sys.path.append(sys.path[0] + '/..')

import numpy as np
import matplotlib.pyplot as plt

import argparse

import ps.ps
import io_util

import problems
from problems.smooth import SmoothH_Sine

def plot_aux():
    """ Plot auxiliary domain. """
    s = solver.AD_len/2
    x_data = [-s, s, s, -s, -s]
    y_data = [s, s, -s, -s, s]
    plt.plot(x_data, y_data, color='gray', linestyle='--')

def plot_arc(color, linestyle):
    th_data = np.linspace(solver.a, 2*np.pi, 512)
    x_data = solver.R * np.cos(th_data)
    y_data = solver.R * np.sin(th_data)

    plt.plot(x_data, y_data, linestyle, color=color)

def plot_Gamma(color='black', extended=True):
    linestyle = '-'

    for sample in solver.get_boundary_sample(extended=True):
        n = len(sample)
        Gamma_x_data = np.zeros(n)
        Gamma_y_data = np.zeros(n)

        for l in range(n):
            p = sample[l]
            Gamma_x_data[l] = p['x']
            Gamma_y_data[l] = p['y']

        plt.plot(Gamma_x_data, Gamma_y_data, linestyle, color=color)

        if not extended:
            break

def nodes_to_plottable(nodes):
    x_data = np.zeros(len(nodes))
    y_data = np.zeros(len(nodes))

    l = 0
    for i, j in nodes:
        x, y = solver.get_coord(i, j)
        x_data[l] = x
        y_data[l] = y
        l += 1

    return x_data, y_data

def plot_Mplus():
    plot_Gamma()

    colors = ('red', 'green', 'blue')
    markers = ('o', 'x', '^')

    for sid in range(3):
        Mplus = solver.all_Mplus[sid]
        x_data, y_data = nodes_to_plottable(Mplus)

        label_text = 'Mplus{}'.format(sid)
        plt.plot(x_data, y_data, markers[sid], label=label_text,
            mfc='none', mec=colors[sid], mew=1)

    plt.xlim(-4,4)
    plt.ylim(-4,4)
    plt.legend(loc=3)
    plt.show()

def plot_gamma():
    plot_Gamma()

    colors = ('red', 'green', 'blue')
    markers = ('o', 'x', '^')

    for sid in range(3):
        gamma = solver.all_gamma[sid]
        x_data, y_data = nodes_to_plottable(gamma)

        label_text = '$\gamma_{}$'.format(sid)
        plt.plot(x_data, y_data, markers[sid], label=label_text,
            mfc='none', mec=colors[sid], mew=1)

    #plt.title('$\gamma$ nodes')
    plt.xlim(-4,4)
    plt.ylim(-4,4)
    plt.legend(loc=3)
    plt.show()


parser = argparse.ArgumentParser()
parser.add_argument('method', choices=('gamma', 'pert'))
io_util.add_arguments(parser, ('boundary', 'N'))
args = parser.parse_args()

# Dummy problem just to get the solver to run
problem = SmoothH_Sine()
boundary = problems.boundary.boundaries[args.boundary](problem.R)
problem.boundary = boundary

options = {'problem': problem, 'boundary': boundary, 'N': args.N,
    'scheme_order': 4}
solver = ps.ps.PizzaSolver(options)


if args.method == 'gamma':
    plot_aux()
    plot_gamma()
    plt.show()
elif args.method == 'pert':
    plot_arc(color='gray', linestyle='--')
    plot_Gamma(color='black', extended=False)
    plt.axes().set_aspect('equal', 'datalim')
    plt.show()
