# To allow __main__ in subdirectory
import sys
sys.path.append(sys.path[0] + '/..')

import numpy as np
import matplotlib.pyplot as plt

import argparse

import ps.ps
import io_util

import problems
from problems.smooth import Smooth_Sine

def plot_aux():
    """ Plot auxiliary domain. """
    s = solver.AD_len/2
    x_data = [-s, s, s, -s, -s]
    y_data = [s, s, -s, -s, s]
    plt.plot(x_data, y_data, color='gray', linestyle='--')

def plot_arc(color, linestyle, linewidth):
    th_data = np.linspace(solver.a, 2*np.pi, 512)
    x_data = solver.R * np.cos(th_data)
    y_data = solver.R * np.sin(th_data)

    plt.plot(x_data, y_data, linestyle, color=color, linewidth=linewidth, label='arc')

def plot_Gamma(sid, color=None, marker=None, label='', ms=1):
    Gamma_x_data = []
    Gamma_y_data = []

    for p in solver.get_boundary_sample():
        if p['sid'] != sid:
            continue

        Gamma_x_data.append(p['x'])
        Gamma_y_data.append(p['y'])

    plt.plot(Gamma_x_data, Gamma_y_data, '-',
        color=color, linewidth=1, marker=marker, markevery=15, label=label, ms=ms)

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


# Dummy problem just to get the solver to run
problem = Smooth_Sine()

dms = 6
data = [
    [
        np.pi/6,
        {'boundary': 'inner-sine', 'color': 'blue', 'marker': 'o', 'ms': dms},
        {'boundary': 'cubic', 'color': 'green', 'marker': '^', 'ms': dms},
    ],
    [
        np.pi/2,
        {'boundary': 'outer-sine', 'color': 'red', 'marker': 'd', 'ms': dms},
        {'boundary': 'sine7', 'color': 'purple', 'marker': '*', 'ms': dms+2.5},
    ]
]

i = 0
for data1 in data:
    a = data1[0]
    b0 = data1[1]
    b1 = data1[2]

    boundary = problems.boundary.boundaries[b0['boundary']](problem.R, a=a)
    problem.boundary = boundary
    problem.a = a

    options = {'problem': problem, 'boundary': boundary, 'N': 16,
        'scheme_order': 4}
    solver = ps.ps.PizzaSolver(options)

    plot_arc(color='gray', linestyle='--', linewidth=1.2)
    plot_Gamma(1, color='black')
    plot_Gamma(2, color='black')
    plot_Gamma(0, color=b0['color'], marker=b0['marker'], label=b0['boundary'],
        ms=b0['ms'])

    options['boundary'] = problems.boundary.boundaries[b1['boundary']](problem.R)
    problem.boundary = options['boundary']
    solver = ps.ps.PizzaSolver(options)

    plot_Gamma(0, color=b1['color'], marker=b1['marker'], label=b1['boundary'],
        ms=b1['ms'])

    plt.axes().set_aspect('equal', 'datalim')
    #plt.axes().get_xaxis().set_visible(False)
    #plt.axes().get_yaxis().set_visible(False)
    plt.axes().axis('off')
    plt.legend(loc='upper right')
    plt.savefig('/Users/sam/Google Drive/research/writeup/images/boundary{}.pdf'.format(i),
        transparent=True)
    #plt.show()
    plt.clf()
    i += 1
