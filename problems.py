import numpy as np
import circle_solver

class Problem:
    def get_solver(self, *args, **kwargs):
        return self.solver_class(self, *args, **kwargs)

class ComplexWave(Problem):
    # Wave number
    k = 1 
    kx = .8*k
    ky = .6*k

    R = 2.3
    solver_class = circle_solver.CircleSolver

    # Boundary data
    def eval_bc(self, th):
        return self.eval_expected(self.R*np.cos(th), self.R*np.sin(th))

    # Expected solution
    def eval_expected(self, x, y):
        return np.exp(complex(0, self.kx*x + self.ky*y))

    # Expected normal derivative on boundary
    def eval_expected_derivative(self, th):
        x = complex(0, kx*np.cos(th) + ky*np.sin(th))
        return x*np.exp(R*x)
