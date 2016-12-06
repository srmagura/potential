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

def plot_aux(options):
    """ Plot auxiliary domain. """
    s = solver.AD_len/2
    x_data = [-s, s, s, -s, -s]
    y_data = [s, s, -s, -s, s]

    #color='gray', linestyle='--'
    plt.plot(x_data, y_data, **options)

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

def plot_allMK(ntype):
    # TODO there is some duplicated functionality between this function
    # and the next one
    plot_Gamma(extended=False)

    colors = ('red', 'black')

    if ntype == 'm':
        node_choices = (solver.global_Mplus, solver.global_Mminus)
        labels = ('$\mathbb{M}^+$', '$\mathbb{M}^-$')
    elif ntype == 'k':
        node_choices = (solver.Kplus, [])
        labels = ('$\mathbb{K}^+$', '')

    markers = ('o', 'x')
    zipped = zip(colors, node_choices, labels, markers)

    for color, nodes, label, marker in zipped:
        x_data, y_data = nodes_to_plottable(nodes)

        plt.plot(x_data, y_data, marker, label=label,
            mfc='none', mec=color, mew=1)

    plt.xlim(-4,4)
    plt.ylim(-4,4)
    plt.legend(loc=3)

    plt.axes().set_aspect('equal', 'datalim')
    plt.axis('off')
    plt.show()

def plot_intext(interior):
    """
    interior -- boolean
    """
    plot_Gamma(extended=False)
    plot_aux(options={'color': 'black'})

    if interior:
        node_choices = (solver.global_Mplus, solver.Nplus)
        labels = ('$\mathbb{M}^+$', '$\mathbb{N}^+$')
        plotname = 'int'
    else:
        node_choices = (solver.global_Mminus, solver.Nminus)
        labels = ('$\mathbb{M}^-$', '$\mathbb{N}^-$')
        plotname = 'ext'

    colors = ('black', 'black')
    markers = ('o', 'o')
    mfc = ['black', 'none']
    ms = [5, 10]
    zipped = zip(node_choices, labels, colors, markers, mfc, ms)

    for nodes, label, color, marker, _mfc, _ms in zipped:
        x_data, y_data = nodes_to_plottable(nodes)

        plt.plot(x_data, y_data, marker, label=label,
            mfc=_mfc, ms=_ms, mec=color, mew=1)

    plt.axes().set_aspect('equal', 'datalim')
    plt.axis('off')

    filename = '/Users/sam/Google Drive/research/writeup_v4/{}.pdf'.format(plotname)
    plt.savefig(filename, transparent=True, bbox_inches='tight', pad_inches=0)
    plt.show()

def plot_gamma():
    plot_Gamma(extended=False)

    if False:
        colors = ('black', 'black', 'black')
        markers = ('o', 'o', 's')
        mfc = ['black', 'none', 'none']
        ms = [5, 10, 12]
    else:
        colors = ('black', 'black', 'black')
        markers = ('o', 'o', 'o')
        mfc = ['black', 'black', 'black']
        ms = [5, 5, 5]

    # Map the code SID to the paper SID
    sidmap = {0: 3, 1: 2, 2: 1}

    for sid in [2, 1, 0]:
        gamma = solver.all_gamma[sid]
        x_data, y_data = nodes_to_plottable(gamma)

        label_text = '$\gamma_{}$'.format(sidmap[sid])
        plt.plot(x_data, y_data, markers[sid], label=label_text,
            mfc=mfc[sid], mec=colors[sid], mew=1,
            ms=ms[sid])

    #plt.title('$\gamma$ nodes')
    #plt.xlim(-4,4)
    #plt.ylim(-4,4)
    #plt.legend(loc=3)


parser = argparse.ArgumentParser()
parser.add_argument('method', choices=('gamma', 'pert', 'm', 'k', 'aux',
    'int', 'ext'))
io_util.add_arguments(parser, ('boundary', 'N'))
args = parser.parse_args()

# Dummy problem just to get the solver to run
problem = Smooth_Sine()
boundary = problems.boundary.boundaries[args.boundary](problem.R)
problem.boundary = boundary

options = {'problem': problem, 'boundary': boundary, 'N': args.N,
    'scheme_order': 4}
solver = ps.ps.PizzaSolver(options)


if args.method == 'gamma':
    plot_aux(options={'color': 'black'})
    plot_gamma()
    plt.axes().set_aspect('equal', 'datalim')
    plt.axis('off')
    plt.savefig('/Users/sam/Google Drive/research/writeup_v4/union_gamma.pdf',
        bbox_inches='tight', transparent=True)
    plt.show()
elif args.method == 'pert':
    plot_arc(color='gray', linestyle='--')
    plot_Gamma(color='black', extended=False)
    plt.axes().set_aspect('equal', 'datalim')
    plt.show()
elif args.method == 'm':
    plot_allMK('m')
elif args.method == 'k':
    plot_allMK('k')
elif args.method == 'int':
    plot_intext(True)
elif args.method == 'ext':
    plot_intext(False)
elif args.method == 'aux':
    plot_aux({'color': 'black'})
    plot_Gamma(color='black', extended=False)
    plt.axes().set_aspect('equal', 'datalim')
    plt.axis('off')
    plt.savefig('/Users/sam/Google Drive/research/writeup_v4/image_sources/Omega0_raw.pdf',
        bbox_inches='tight')
    plt.show()
