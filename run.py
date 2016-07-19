import sys

import argparse
import numpy as np

import problems
from problems.singular import HReg

import ps.ps

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

parser.add_argument('--cheat-fft', action='store_true',
    help='for testing purposes. Uses the true values of the a/b '
    'coefficients, which really should not be known')

parser.add_argument('--polarfd-N', type=int, default=64,
    help='polarfd: grid size')

parser.add_argument('--polarfd-Nlam', type=int, default=10,
    help='polarfd: number of Chebyshev coefficients')

args = parser.parse_args()

problem = problems.problem_dict[args.problem]()
boundary = problems.boundary.boundaries[args.boundary](problem.R, k=problem.k)
problem.boundary = boundary

if args.cheat_fft:
    problem.hreg = HReg.cheat_fft

N_list = get_N_list(args.N, args.c)

# Options to pass to the solver
options = {
    'problem': problem,
    'scheme_order': args.o,
    'cheat_fft': args.cheat_fft,
    'polarfd_N': args.polarfd_N,
    'polarfd_Nlam': args.polarfd_Nlam,
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

prev_error = None

for N in N_list:
    options['N'] = N

    solver = ps.ps.PizzaSolver(options)

    print('---- {0} x {0} ----'.format(N))
    result = solver.run()
    if result is None:
        continue

    if result.error is not None:
        print('Error: ' + prec_str.format(result.error))

    u2 = u1
    u1 = u0
    u0 = result.u_act

    if prev_error is not None:
        convergence = np.log2(prev_error / result.error)

        print('Convergence: ' + prec_str.format(convergence))

    if do_rel_conv and u2 is not None:
        convergence = solver.calc_rel_convergence(u0, u1, u2)
        print('Rel convergence: ' + prec_str.format(convergence))

    print()
    sys.stdout.flush()

    prev_error = result.error
