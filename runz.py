"""
Run the zmethod
"""
import sys

import argparse
import numpy as np

import problems
from problems.singular import HReg

import ps.ps
import ps.zmethod

import io_util
from io_util import prec_str

def get_N_list(N0, c):
    if c is None:
        return [N0]

    N_list = []

    N = N0
    while N <= c:
        N_list.append(N)
        N *= 2

    return N_list


"""
Create an ArgumentParser for handling command-line arguments.
"""
parser = argparse.ArgumentParser()
parser = parser

arg_list = ['problem', 'boundary', 'N', 'c', 'o', 'r', 'a']
io_util.add_arguments(parser, arg_list)

parser.add_argument('--acheat', action='store_true',
    help='for testing purposes. Uses the true-ish values of the a/b '
    'coefficients, which really should not be known')

parser.add_argument('--z1cheat', action='store_true',
    help='for testing purposes. Skip the ODE method')

args = parser.parse_args()

problem = problems.problem_dict[args.problem]()
boundary = problems.boundary.boundaries[args.boundary](problem.R, k=problem.k)
problem.set_boundary(boundary)

if args.acheat:
    problem.hreg = HReg.acheat

N_list = get_N_list(args.N, args.c)

# Options to pass to the solver
options = {
    'problem': problem,
    'scheme_order': args.o,
    'acheat': args.acheat,
    'z1cheat': args.z1cheat,
}

meta_options = {
    'N_list': N_list
}

io_util.print_options(options, meta_options)

"""
Perform the convergence test.
"""
do_rel_conv = args.r or not problem.expected_known

u2 = None
u1 = None
u0 = None

z2 = None
z1 = None
z0 = None

# Cache result of ODE method
z1_fourier = None

prev_error = None

for N in N_list:
    options['N'] = N

    print('---- {0} x {0} ----'.format(N))
    my_zmethod = ps.zmethod.ZMethod(options)
    my_zmethod.z1_fourier = z1_fourier

    result = my_zmethod.run()

    solver = result['solver']
    #if result.error is not None:
    #    print('Error: ' + prec_str.format(result.error))

    u2 = u1
    u1 = u0
    #u0 = result.u_act

    z2 = z1
    z1 = z0
    z0 = result['z']

    #if prev_error is not None:
    #    convergence = np.log2(prev_error / result.error)

    #    print('Convergence: ' + prec_str.format(convergence))

    if z2 is not None:
        convergence = solver.calc_rel_convergence(z0, z1, z2)
        print('z rel convergence: ' + prec_str.format(convergence))

    print()
    sys.stdout.flush()

    z1_fourier = my_zmethod.z1_fourier
    #prev_error = result.error
