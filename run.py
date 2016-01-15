from io_util import add_arguments, prec_str

def get_N_list(N0, c, ignore_c=False):
    if c is None or ignore_c:
        N_list = [N0]
    else:
        N_list = []

        N = N0
        while N <= c:
            N_list.append(N)
            N *= 2

    return N_list


"""
Create an ArgumentParser for handling command-line arguments.
"""
problem = kwargs.get('problem', None)
problem_name = kwargs.get('problem_name', '')
arg_string = kwargs.get('arg_string', None)

parser = argparse.ArgumentParser()
parser = parser

arg_list = ['N', 'c', 'o', 'r', 'a']
if problem is None:
    arg_list.append('problem')

add_arguments(parser, arg_list)

parser.add_argument('-p', action='store_true',
    help='run optimize_n_basis')

parser.add_argument('-n', choices=ps.dual.norm_names,
    default=ps.dual.default_norm)
parser.add_argument('--no-dual', action='store_true', default=False,
    help='do not use a higher-order scheme to compute the a '
    'coefficients')
parser.add_argument('--vm', choices=ps.dual.var_methods,
    default=ps.dual.default_var_method)
parser.add_argument('--print-b', action='store_true', default=False,
    help='print the last 5 b coefficients, to get a sense of how '
        'close the (finite) Fourier-Bessel sum will be to its true '
        'value')


"""
Build the list of N values for which the algorithm should be run,
then call run_solver().
"""
if arg_string:
    args = parser.parse_args(arg_string.split())
else:
    args = parser.parse_args()
    problem_name = args.problem

if problem is None:
    problem = problems.problem_dict[problem_name]\
        (scheme_order=args.o, var_compute_a=args.a)

N_list = get_N_list(args.N, args.c, args.p)


"""
Perform the convergence test.
"""
do_rel_conv = args.r or not problem.expected_known

u2 = None
u1 = None
u0 = None

# Options to pass to the solver
options = {
    'scheme_order': args.o,
}

if args.print_b:
    problem.print_b()

for N in N_list:
    coord = ps.coordinator.Coordinator(problem, N, options)

    if N == N_list[0]:
        coord.print_info(N_list)

    print('---- {0} x {0} ----'.format(N))
    result = coord.run()
    if result is None:
        continue

    if result.a_error is not None:
        print('a error: ' + prec_str.format(result.a_error))

        if prev_a_error is not None:
            a_convergence = np.log2(prev_a_error / result.a_error)
            print('a convergence: ' + prec_str.format(a_convergence))

        print()

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
