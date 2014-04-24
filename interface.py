import numpy as np
import sys

class Interface:

    def __init__(self, problem):
        self.problem = problem

    def test_convergence(self):
        if len(sys.argv) == 3:
            N_max = int(sys.argv[2])
        else:
            N_max = 128

        prev_error = None

        N = 16
        while N <= N_max:
            my_solver = self.problem.get_solver(N)
            error = my_solver.run()

            print('---- {0} x {0} ----'.format(N)) 
            print('Error:', error)

            if prev_error is not None:
                print('Convergence:', np.log2(prev_error / error))

            prev_error = error
            N *= 2
            print()

    def run(self):
        N = 16
        if len(sys.argv) > 1:
            if sys.argv[1] == 'c':
                self.test_convergence()
                N = 0
            else:
                try:
                    N = int(sys.argv[1])
                except ValueError:
                    print('Could not parse integer `{}`.'.format(N))
        
        if N > 0:
            my_solver = self.problem.get_solver(N, verbose=True)
            error = my_solver.run()
            print('Error:', error)
