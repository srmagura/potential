"""
Script for selecting a good number of basis functions.
Too many or too few basis functions will introduce numerical error.
True solution must be known.

Run the program several times, varying the value of the -N option.

There may be a way to improve on this brute force method.
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
boundary = problems.boundary.Arc(problem.R)
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

def worker(t):
    options['n_circle'] = t[0]
    options['n_radius'] = t[1]
    my_solver = ps.ps.PizzaSolver(options)

    result = my_solver.run()
    t = (t[0], t[1], result.error)
    my_print(t)

    return t

all_options = []
# Tweak the following ranges as needed
for n_circle in range(28, 40, 3):
    for n_radius in range(5, int(.87*n_circle), 2):
        all_options.append((n_circle, n_radius))

with Pool(4) as p:
    results = p.map(worker, all_options)

min_error = float('inf')

for t in results:
    if t[2] < min_error:
        min_error = t[2]
        min_t = t

print()
my_print(min_t)
