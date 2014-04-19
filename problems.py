import numpy as np
from numpy import cos, sin

import solver
import circle_solver

class Problem:
    def get_solver(self, *args, **kwargs):
        return self.solver_class(self, *args, **kwargs)

class Mountain(Problem):
    k = 1
    solver_class = solver.SquareSolver

    def eval_expected(self, x, y):
        return np.cos(x/2)**4 * np.cos(y/2)**4

    def eval_f(self, x, y):
        return (self.k**2*cos(x/2)**4*cos(y/2)**4 + 
            (3*sin(x/2)**2 - cos(x/2)**2)*cos(x/2)**2*cos(y/2)**4 +
            (3*sin(y/2)**2 - cos(y/2)**2)*cos(x/2)**4*cos(y/2)**2)

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

    def eval_f(self, x, y):
        return 0

    # Expected solution
    def eval_expected(self, x, y):
        return np.exp(complex(0, self.kx*x + self.ky*y))

    # Expected normal derivative on boundary
    #def eval_expected_derivative(self, th):
    #    x = complex(0, kx*np.cos(th) + ky*np.sin(th))
    #    return x*np.exp(R*x)
