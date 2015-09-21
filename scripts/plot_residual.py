# To allow __main__ in subdirectory
import sys
sys.path.append(sys.path[0] + '/..')

import os
import argparse

import numpy as np
from scipy.special import jv
import matplotlib.pyplot as plt
from tabulate import tabulate

import interface

from problems.problem import PizzaProblem
from problems.sine import SinePizza


def eval_expected(m, r, th):
    a = PizzaProblem.a
    nu = PizzaProblem.nu

    if 0 <= th and th < a/2:
        th += 2*np.pi

    return jv(m*nu, k*r) * np.sin(m*nu*(th-a))

def do_test(N):
    ps = fake_problem.get_solver(N)

    ext = np.zeros(len(ps.union_gamma), dtype=complex)

    for l in range(len(ps.union_gamma)):
        node = ps.union_gamma[l]
        r, th = ps.get_polar(*node)
        ext[l] = eval_expected(m, r, th)

    potential = ps.get_potential(ext)
    projection = ps.get_trace(potential)

    bep = projection - ext

    x_data, y_data = ps.nodes_to_plottable(ps.union_gamma)
    r_data = [np.sqrt(x**2 + y**2) for x, y in zip(x_data, y_data)]

    plt.stem(r_data, abs(bep), markerfmt=' ', basefmt='k')

    plt.xlim(xmin=-ps.R/15)

    fs = 14
    plt.xlabel('Polar radius', fontsize=fs)
    plt.ylabel('Absolute value of BEP residual', fontsize=fs)

    if args.s:
        plt.savefig('{}/{}.pdf'.format(dir_name, N))

    if not args.s or args.f:
        plt.show()

    plt.clf()

# Used only to construct a PizzaSolver object
fake_problem = SinePizza()

k = 1
m = 8

parser = argparse.ArgumentParser()
parser.add_argument('-s', action='store_true', help='save plots to file')
parser.add_argument('-f', action='store_true', help='show plots even if '
    'saving to file.')

interface.add_arguments(parser, ('N', 'c'))
args = parser.parse_args()

N_list = interface.get_N_list(args.N, args.c)

print('k={}'.format(k))
print('m={}'.format(m))
print()

if args.s:
    dir_name = 'plot_residual_m={}_k={}'.format(m, k)
    if os.path.exists(dir_name):
        print('Error: `{}` already exists'.format(dir_name))
        sys.exit(1)

    os.mkdir(dir_name)

for N in N_list:
    do_test(N)
