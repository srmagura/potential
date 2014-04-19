import numpy as np
import sys

class Interface:

    def __init__(self, problem):
        self.problem = problem

    def test_convergence(self):
        prev_error = None

        for N in (16, 32, 64, 128, 256): 
            my_solver = self.problem.get_solver(N)
            error = my_solver.run()

            print('---- {0} x {0} ----'.format(N)) 
            print('Error:', error)

            if prev_error is not None:
                print('Convergence:', np.log2(prev_error / error))

            prev_error = error
            print()

    def run(self):
        N = 0
        verbose = True

        if len(sys.argv) > 1:
            try:
                N = int(sys.argv[1])
            except ValueError:
                verbose = False
                self.test_convergence()
                   
        if N <= 0:
            N = 16

        if verbose:
            my_solver = self.problem.get_solver(N, verbose=True)
            error = my_solver.run()
            print('Error:', error)
