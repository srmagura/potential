"""
Run the zmethod
"""
import sys

import argparse
import numpy as np

import problems

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

v2 = None
v1 = None
v0 = None

# Cache result of ODE method
z1_fourier = None

prev_u_error = None
prev_v_error = None

for N in N_list:
    options['N'] = N

    print('---- {0} x {0} ----'.format(N))
    my_zmethod = ps.zmethod.ZMethod(options)
    my_zmethod.z1_fourier = z1_fourier

    result = my_zmethod.run()

    polarfd = result['polarfd']
    pert_solver = result['pert_solver']
    u_error = result['u_error']
    v_error = result['v_error']

    v2 = v1
    v1 = v0
    v0 = result['v']

    if v_error is not None:
        print('v error: ' + prec_str.format(v_error))

    if v2 is not None:
        convergence = polarfd.calc_rel_convergence(v0, v1, v2)
        print('v rel convergence: ' + prec_str.format(convergence))

    if prev_v_error is not None:
        convergence = np.log2(prev_v_error / v_error)
        print('v convergence: ' + prec_str.format(convergence))

    print()

    u2 = u1
    u1 = u0
    u0 = result['u']

    if u_error is not None:
        print('Error: ' + prec_str.format(u_error))

    if u2 is not None:
        convergence = pert_solver.calc_rel_convergence(u0, u1, u2)
        print('u rel convergence: ' + prec_str.format(convergence))

    if prev_u_error is not None:
        convergence = np.log2(prev_u_error / u_error)
        print('u convergence: ' + prec_str.format(convergence))

    print()
    sys.stdout.flush()

    z1_fourier = my_zmethod.z1_fourier
    prev_u_error = u_error
    prev_v_error = v_error
