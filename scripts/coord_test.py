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

#parser = argparse.ArgumentParser()
#io_util.add_arguments(parser, ('boundary', 'N'))
#args = parser.parse_args()

# Dummy problem just to get the solver to run
problem = SmoothH_Sine()
boundary = problems.boundary.boundaries['arc'](problem.R)

options = {'problem': problem, 'boundary': boundary, 'N': 64,
    'scheme_order': 4}
solver = ps.ps.PizzaSolver(options)

max_error = 0

for node in solver.all_gamma[0]:
    r1, th1 = solver.get_polar(*node)
    n, th0 = solver.boundary_coord_cache[node]

    error = abs(th1-th0)
    #if error > 1e-14:
    #    print(error)
    if error > max_error:
        max_error = error
        max_node = node

print('Error:', max_error)
print('Node:', solver.get_coord(*max_node), max_node)
