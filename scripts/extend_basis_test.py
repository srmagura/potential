"""
Verify that extend_basis() and extend_boundary() give the same
result, when c0 and c1 are set appropriately.
"""
# To allow __main__ in subdirectory
import sys
sys.path.append(sys.path[0] + '/..')

import numpy as np
import argparse
import itertools as it

import ps.ps

import matplotlib.pyplot as plt

import problems
import io_util

#prec_str = '{:.5}'

parser = argparse.ArgumentParser()

io_util.add_arguments(parser, ('problem', 'boundary'))

args = parser.parse_args()
problem = problems.problem_dict[args.problem]()
boundary = problems.boundary.boundaries[args.boundary](problem.R)
problem.boundary = boundary

options = {
    'problem': problem,
    'boundary': boundary,
    'scheme_order': 4,
    'N': 16
}

meta_options = {'procedure_name': 'extend_basis_test'}
io_util.print_options(options, meta_options)

solver = ps.ps.PizzaSolver(options)

error = []
for index in (0, 1):
    for JJ in range(len(solver.B_desc)):
        ext1_array = solver.extend_basis(JJ, index)

        solver.c0 = np.zeros(len(solver.B_desc))
        solver.c1 = np.zeros(len(solver.B_desc))

        if index == 0:
            solver.c0[JJ] = 1
        elif index == 1:
            solver.c1[JJ] = 1

        ext2_array = solver.extend_boundary(homogeneous_only=True)
        diff = np.abs(ext1_array - ext2_array)
        error.append(np.max(diff))
        print('index={}  JJ={}  error={}'.format(index, JJ, error[-1]))

        if error[-1] > 1e-14:
            print('FAILURE')
            sys.exit(0)

print('SUCCESS')
