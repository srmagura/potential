'''
Command-line interface to the solvers
'''
import numpy as np
import sys
import argparse

import problems
import solver


class Interface:

    def test_convergence(self):
        '''
        Perform the convergence test. Prints in TeX-friendly
        format if the `--tex` option was given.
        '''
        do_rel_conv = self.args.r or not self.problem.expected_known
        prev_error = None
        
        u2 = None
        u1 = None
        u0 = None

        N = self.args.N
        while N <= self.args.c:
            my_solver = self.problem.get_solver(N, self.args.o)
            result = my_solver.run()

            print('---- {0} x {0} ----'.format(N))
            
            if result.error is not None:
                print('Error:', result.error)
                
            u2 = u1
            u1 = u0
            u0 = result.u_act

            if self.args.tex:            
                tex = '\\grid{' + str(N) + '} &'
                tex += '\\error{' + str(result.error) + '} &'

            if prev_error is not None:
                convergence = np.log2(prev_error / result.error)
                
                if self.args.tex:
                    tex += str(convergence)
                else:
                    print('Convergence:', convergence)
            
            if do_rel_conv and u2 is not None:
                convergence = my_solver.calc_convergence3(u0, u1, u2)
                print('Rel convergence:', convergence)

            if self.args.tex:
                print(tex + r'\\')
            else:
                print()
                          
            prev_error = result.error  
            N *= 2

    def run(self):
        '''
        Main routine. Parse command-line arguments and run a solver
        or start the convergence test.
        '''
        parser = argparse.ArgumentParser()
        parser.add_argument('-p', required=True, 
            choices=problems.problem_dict.keys())
        parser.add_argument('-N', type=int, default=16)
        parser.add_argument('-c', type=int, nargs='?', const=128)
        parser.add_argument('-o', type=int, default=4)

        parser.add_argument('--tex', action='store_true')
        parser.add_argument('-r', action='store_true')

        self.args = parser.parse_args()

        self.problem = problems.problem_dict[self.args.p](scheme_order=self.args.o)

        if self.args.c is None:
            my_solver = self.problem.get_solver(self.args.N, 
                self.args.o, verbose=True)
            result = my_solver.run()
            
            if result.error is not None:
                print('Error:', result.error)
        else:
            self.test_convergence()


if __name__ == '__main__':
    interface = Interface()
    interface.run()
