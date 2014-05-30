import numpy as np
import sys
import argparse

import warnings
from scipy.integrate import IntegrationWarning

import problems
import solver

class Interface:

    def run_solver(self, solver):
        with warnings.catch_warnings():
            #warnings.simplefilter('ignore',
                #category=IntegrationWarning)
            return solver.run()

    def test_convergence(self):
        prev_error = None

        N = 16
        while N <= self.args.c:
            my_solver = self.problem.get_solver(N, self.args.o)
            error = self.run_solver(my_solver)

            print('---- {0} x {0} ----'.format(N)) 
            print('Error:', error)

            if prev_error is not None:
                print('Convergence:', np.log2(prev_error / error))

            prev_error = error
            N *= 2
            print()

    def run(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-p', required=True, 
            choices=problems.problem_dict.keys())
        parser.add_argument('-N', type=int, default=16)
        parser.add_argument('-c', type=int, nargs='?', const=128)
        parser.add_argument('-o', type=int, default=4)
        parser.add_argument('-s', choices=solver.solver_dict.keys())
        self.args = parser.parse_args()

        self.problem = problems.problem_dict[self.args.p]()

        if self.args.s is not None:
            self.problem.solver_class =\
                solver.solver_dict[self.args.s]

        if self.args.c is None:
            my_solver = self.problem.get_solver(self.args.N, 
                self.args.o, verbose=True)
            print('Using `{}`.'.format(
                self.problem.solver_class.__name__))
            error = self.run_solver(my_solver)
            print('Error:', error)
        else:
            self.test_convergence()

if __name__ == '__main__':
    interface = Interface()
    interface.run()
