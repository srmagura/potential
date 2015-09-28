"""
Test that a norm has proper normalization.

This script samples a given function f(x,y) on the discrete boundary of
the pizza domain and prints the norm. The grid is refined and the
process is repeated. The value norm should be (mostly) constant, despite
the changing grid size. 
"""
# To allow __main__ in subdirectory
import sys
sys.path.append(sys.path[0] + '/..')

import numpy as np
import argparse

import interface
from ps.ps import PizzaSolver

import norms.sobolev
import norms.linalg

from problems.sine import SinePizza

# Used only to construct a PizzaSolver object
fake_problem = SinePizza()

def eval_f(x, y):
    return np.sqrt(abs(x)) * np.sin(y)

def do_test(N):
    print('---- {0} x {0} ----'.format(N))

    # Create a solver so we can use its grid functionality
    solver = PizzaSolver(fake_problem, N, {'skip_matrix_build': True})

    # Same the function f on the grid
    f = np.zeros(len(solver.union_gamma))
    for l in range(len(solver.union_gamma)):
        i, j = solver.union_gamma[l]
        x, y = solver.get_coord(i, j)
        f[l] = eval_f(x, y)

    h = solver.AD_len / N
    sa = 0.5
    ip_array = norms.sobolev.get_ip_array(h, solver.union_gamma, sa)

    norm_f = norms.linalg.eval_norm(ip_array, f)
    print('Norm: {:.5}'.format(norm_f))
    print()

parser = argparse.ArgumentParser()

interface.add_arguments(parser, ('N', 'c'))
args = parser.parse_args()

N_list = interface.get_N_list(args.N, args.c)
for N in N_list:
    do_test(N)
