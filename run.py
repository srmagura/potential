import argparse
import numpy as np

import problems
import ps.coordinator

from io_util import add_arguments, prec_str

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

arg_list = ['problem', 'N', 'c', 'o', 'r', 'a']
add_arguments(parser, arg_list)
args = parser.parse_args()

problem_name = args.problem
problem = problems.problem_dict[problem_name](scheme_order=args.o)

# Options to pass to the solver
options = {
    'scheme_order': args.o,
}

"""
Perform the convergence test.
"""
N_list = get_N_list(args.N, args.c)

do_rel_conv = args.r or not problem.expected_known

u2 = None
u1 = None
u0 = None

prev_error = None

for N in N_list:
    coord = ps.coordinator.Coordinator(problem, N, options)

    if N == N_list[0]:
        coord.print_info(N_list)

    print('---- {0} x {0} ----'.format(N))
    result = coord.run()
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
        convergence = coord.calc_rel_convergence(u0, u1, u2)
        print('Rel convergence: ' + prec_str.format(convergence))

    print()

    prev_error = result.error
