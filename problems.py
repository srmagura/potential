import numpy as np
from numpy import cos, sin

from potential.solver import SquareSolver
from potential.circle_solver import CircleSolver

class Problem:
    def get_solver(self, *args, **kwargs):
        return self.solver_class(self, *args, **kwargs)

class Mountain(Problem):
    k = 1
    solver_class = SquareSolver

    def eval_expected(self, x, y):
        return np.cos(x/2)**4 * np.cos(y/2)**4

    def eval_f(self, x, y):
        return (self.k**2*cos(x/2)**4*cos(y/2)**4 + 
            (3*sin(x/2)**2 - cos(x/2)**2)*cos(x/2)**2*cos(y/2)**4 +
            (3*sin(y/2)**2 - cos(y/2)**2)*cos(x/2)**4*cos(y/2)**2)

class MountainCircle(Mountain):
    solver_class = CircleSolver

    def eval_bc(self, th):
        return self.eval_expected(self.R*np.cos(th), self.R*np.sin(th))

    def eval_expected_derivative(self, th):
        R = self.R
        return (-2*sin(th)*sin(R*sin(th)/2)*
            cos(R*sin(th)/2)**3*cos(R*cos(th)/2)**4 -
            2*sin(R*cos(th)/2)*cos(th)*
            cos(R*sin(th)/2)**4*cos(R*cos(th)/2)**3)


class ComplexWave(Problem):
    # Wave number
    k = 1 
    kx = .8*k
    ky = .6*k

    solver_class = CircleSolver

    # Boundary data
    def eval_bc(self, th):
        return self.eval_expected(self.R*np.cos(th), self.R*np.sin(th))

    def eval_f(self, x, y):
        return 0

    # Expected solution
    def eval_expected(self, x, y):
        return np.exp(complex(0, self.kx*x + self.ky*y))

    # Expected normal derivative on boundary
    def eval_expected_derivative(self, th):
        x = complex(0, self.kx*np.cos(th) + self.ky*np.sin(th))
        return x*np.exp(self.R*x)
