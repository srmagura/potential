"""
Script for figuring out how many basis functions needed on the
outer boundary to get a certain accuracy for the Chebyshev series.
"""
# To allow __main__ in subdirectory
import sys
sys.path.append(sys.path[0] + '/..')

import argparse

import numpy as np

import ps.ps

import io_util
import problems
import problems.boundary

import copy
from multiprocessing import Pool

parser = argparse.ArgumentParser()

io_util.add_arguments(parser, ('problem', 'N'))
args = parser.parse_args()

problem = problems.problem_dict[args.problem]()
boundary = problems.boundary.OuterSine(problem.R)
problem.boundary = boundary

# Options to pass to the solver
options = {
    'problem': problem,
    'N': args.N,
    'scheme_order': 4,
}

meta_options = {
    'procedure_name': 'optimize_basis',
}

io_util.print_options(options, meta_options)

def my_print(t):
    print('n_circle={}    n_radius={}    error={}'.format(*t))



# Tweak the following ranges as needed
for n_circle in range(10, 100, 5):
    options['n_circle'] = n_circle
    options['n_radius'] = int(.75*n_circle)
    my_solver = ps.ps.PizzaSolver(options)

    my_solver.calc_c0()
    print('n_circle={}'.format(n_circle))
    my_solver.c0_test(plot=False)
    print()
