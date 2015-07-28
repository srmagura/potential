"""
Command-line interface to the solvers
"""
import numpy as np
import sys
import argparse

import problems
import solver
import ps.ps

class Interface:

    prec_str = '{:.5}'

    def __init__(self):
        """
        Create an ArgumentParser for handling command-line arguments.
        """
        parser = argparse.ArgumentParser()
        self.parser = parser

        problem_choices = problems.problem_dict.keys()
        parser.add_argument('problem', metavar='problem',
            choices=problem_choices,
            help='name of the problem to run: ' + ', '.join(problem_choices))

        parser.add_argument('-N', type=int, default=16,
            help='initial grid size')
        parser.add_argument('-c', type=int, nargs='?', const=128,
            help='run convergence test, up to the C x C grid. '
            'Default is 128.')
        parser.add_argument('-o', type=int, default=4,
            choices=(2, 4),
            help='order of scheme')

        parser.add_argument('-r', action='store_true',
            help='show relative convergence, even if the problem\'s '
            'true solution is known')
        #parser.add_argument('--tex', action='store_true',
        #    help='print convergence test results in TeX-friendly format')

        # Stuff specific to PizzaSolver
        parser.add_argument('-p', action='store_true',
            help=' PizzaSolver: run optimize_n_basis')
        parser.add_argument('-n', choices=ps.ps.norms, default='l2')

        parser.add_argument('-m', type=int)
        parser.add_argument('-mmax', type=int)

    def run(self):
        """
        Build the list of N values for which the algorithm should be run,
        print basic info about the problem, then call run_solver().
        """
        self.args = self.parser.parse_args()
        self.problem = problems.problem_dict[self.args.problem]\
            (scheme_order=self.args.o)

        if self.args.c is None or self.args.p:
            N_list = [self.args.N]
        else:
            N_list = []

            N = self.args.N
            while N <= self.args.c:
                N_list.append(N)
                N *= 2

        print('[{}]'.format(self.args.problem))
        print('Norm:', self.args.n)
        print('k = ' + self.prec_str.format(float(self.problem.k)))
        print('R = ' + self.prec_str.format(self.problem.R))
        print('AD_len = ' + self.prec_str.format(self.problem.AD_len))
        print()

        if hasattr(self.problem, 'get_n_basis'):
            print('[Basis sets]')
            for N in N_list:
                print('{}: {}'.format(N, self.problem.get_n_basis(N)))

        print()
        self.run_solver(N_list)

    def run_solver(self, N_list):
        """
        Perform the convergence test.
        """
        do_rel_conv = (self.args.r or
            not self.problem.expected_known or
            self.problem.force_relative)
        prev_error = None
        prev_b_error = None

        u2 = None
        u1 = None
        u0 = None

        # Options to pass to the solver
        options = {
            'verbose': (len(N_list) == 1),
            'scheme_order': self.args.o,
            'do_optimize': self.args.p,
            'norm': self.args.n
        }

        if self.args.m is not None:
            if self.args.mmax is not None:
                options['m_list'] = range(self.args.m, self.args.mmax+1)
            else:
                options['m_list'] = (self.args.m,)

            print('m included in variational formulation:', list(options['m_list']))
            print()

        for N in N_list:
            my_solver = self.problem.get_solver(N, options)

            print('---- {0} x {0} ----'.format(N))
            result = my_solver.run()
            if result is None:
                continue

            if result.b_error is not None:
                print('b error: ' + self.prec_str.format(result.b_error))

                if prev_b_error is not None:
                    b_convergence = np.log2(prev_b_error / result.b_error)
                    print('b convergence: ' + self.prec_str.format(b_convergence))

                print()

            if result.error is not None:
                print('Error: ' + self.prec_str.format(result.error))

            u2 = u1
            u1 = u0
            u0 = result.u_act

            if prev_error is not None:
                convergence = np.log2(prev_error / result.error)

                print('Convergence: ' + self.prec_str.format(convergence))

            if do_rel_conv and u2 is not None:
                convergence = my_solver.calc_rel_convergence(u0, u1, u2)
                print('Rel convergence: ' + self.prec_str.format(convergence))

            print()

            prev_error = result.error
            prev_b_error = result.b_error


if __name__ == '__main__':
    interface = Interface()
    interface.run()
