import numpy as np
import sys
import argparse

class Interface:

    def __init__(self, problem):
        self.problem = problem

    def test_convergence(self):
        prev_error = None

        N = 16
        while N <= self.args.c:
            my_solver = self.problem.get_solver(N, self.args.o)
            error = my_solver.run()

            print('---- {0} x {0} ----'.format(N)) 
            print('Error:', error)

            if prev_error is not None:
                print('Convergence:', np.log2(prev_error / error))

            prev_error = error
            N *= 2
            print()

    def run(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('N', type=int, nargs='?', default=16)
        parser.add_argument('-c', type=int, nargs='?', const=128)
        parser.add_argument('-o', type=int, default=4)
        self.args = parser.parse_args()

        if self.args.c is None:
            my_solver = self.problem.get_solver(self.args.N, 
                self.args.o, verbose=True)
            error = my_solver.run()
            print('Error:', error)
        else:
            self.test_convergence()
