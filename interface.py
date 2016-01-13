"""
Command-line interface to the solvers
"""
import numpy as np
import sys
import argparse

import problems
import solver
import ps.ps

import ps.dual
prec_str = ps.dual.prec_str

def add_arguments(parser, args):
    problem_choices = problems.problem_dict.keys()
    if 'problem' in args:
        parser.add_argument('problem', metavar='problem',
            choices=problem_choices,
            help='name of the problem to run: ' + ', '.join(problem_choices))

    if 'N' in args:
        parser.add_argument('-N', type=int, default=16,
            help='initial grid size')

    if 'c' in args:
        parser.add_argument('-c', type=int, nargs='?', const=128,
            help='run convergence test, up to the C x C grid. '
            'Default is 128.')

    if 'o' in args:
        parser.add_argument('-o', type=int,
            default=ps.dual.default_primary_scheme_order,
            choices=(2, 4),
            help='order of scheme')

    if 'r' in args:
        parser.add_argument('-r', action='store_true',
            help='show relative convergence, even if the problem\'s '
            'true solution is known')

    if 'a' in args:
        parser.add_argument('-a', action='store_true', default=False,
            help='calculate the a and b coefficients using the variational'
                'formulation')

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

class Interface:

    def __init__(self, **kwargs):
        """
        Create an ArgumentParser for handling command-line arguments.
        """
        self.problem = kwargs.get('problem', None)
        self.problem_name = kwargs.get('problem_name', '')
        self.arg_string = kwargs.get('arg_string', None)

        parser = argparse.ArgumentParser()
        self.parser = parser

        arg_list = ['N', 'c', 'o', 'r', 'a']
        if self.problem is None:
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

        # Broken by dual scheme
        #parser.add_argument('-m', type=int)
        #parser.add_argument('--mmax', type=int)


    def run(self):
        """
        Build the list of N values for which the algorithm should be run,
        then call run_solver().
        """
        if self.arg_string:
            self.args = self.parser.parse_args(self.arg_string.split())
        else:
            self.args = self.parser.parse_args()
            self.problem_name = self.args.problem

        if self.problem is None:
            self.problem = problems.problem_dict[self.problem_name]\
                (scheme_order=self.args.o, var_compute_a=self.args.a)

        N_list = get_N_list(self.args.N, self.args.c, self.args.p)

        self.run_solver(N_list)

    def run_solver(self, N_list):
        """
        Perform the convergence test.
        """
        do_rel_conv = (self.args.r or
            not self.problem.expected_known or
            self.problem.force_relative)
        prev_error = None
        prev_a_error = None

        u2 = None
        u1 = None
        u0 = None

        # Options to pass to the solver
        options = {
            'verbose': (len(N_list) == 1),
            'scheme_order': self.args.o,
            'do_optimize': self.args.p,
            'norm': self.args.n,
            'var_compute_a': self.args.a,
            'var_method': self.args.vm,
            'do_dual': self.args.a and not self.args.no_dual,
            'print_a': True,
        }


        #if self.args.m is not None:
        #    if self.args.mmax is not None:
        #        options['m_list'] = range(self.args.m, self.args.mmax+1)
        #    else:
        #        options['m_list'] = (self.args.m,)

        #    print('m included in variational formulation:', list(options['m_list']))
        #    print()

        if self.args.print_b:
            self.problem.print_b()

        for N in N_list:
            my_solver = ps.dual.DualCoordinator(self.problem, N, options)

            if N == N_list[0]:
                my_solver.print_info(N_list)

            print('---- {0} x {0} ----'.format(N))
            result = my_solver.run()
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
                convergence = my_solver.calc_rel_convergence(u0, u1, u2)
                print('Rel convergence: ' + prec_str.format(convergence))

            print()

            prev_error = result.error
            prev_a_error = result.a_error
