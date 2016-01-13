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

import os
import argparse

import numpy as np

import interface
import problems
import ps.dual

parser = argparse.ArgumentParser()

interface.add_arguments(parser, ('problem', 'N', 'o', 'a'))
args = parser.parse_args()

problem = problems.problem_dict[args.problem]\
    (scheme_order=args.o, var_compute_a=args.a)

# Options to pass to the solver
options = {
    'scheme_order': args.o,
    'var_compute_a': args.a,
    'do_dual': args.a# and not self.args.no_dual,
}

min_error = float('inf')
first_time = True

# Tweak the following ranges as needed
for n_circle in range(41, 85, 3):
    for n_radius in range(9, int(.8*n_circle), 2):
        options['n_circle'] = n_circle
        options['n_radius'] = n_radius

        my_solver = ps.dual.DualCoordinator(problem, args.N, options)
        if first_time:
            my_solver.print_info()
            first_time = False

        result = my_solver.run()
        error = result.error

        s =('n_circle={}    n_radius={}    error={}'
            .format(n_circle, n_radius, error))

        if error < min_error:
            min_error = error
            s = '!!!    ' + s

        print(s)
