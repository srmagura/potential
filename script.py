import numpy as np
import sys

import problems

problem = problems.ComplexWave()

def test_convergence():
    global N
    prev_error = None

    for N in (16, 32, 64, 128, 256): 
        my_solver = problem.get_solver(N)
        error = my_solver.run()

        print('---- {0} x {0} ----'.format(N)) 
        print('Error:', error)

        if prev_error is not None:
            print('Convergence:', np.log2(prev_error / error))

        prev_error = error
        print()

if __name__ == '__main__':
    N = 0
    verbose = True

    if len(sys.argv) > 1:
        try:
            N = int(sys.argv[1])
        except ValueError:
            verbose = False
            plot_grid = False
            test_convergence()
               
    if N <= 0:
        N = 16

    if verbose:
        my_solver = problem.get_solver(N, verbose=True)
        error = my_solver.run()
        print('Error:', error)
