from numpy import cos, sin

from problems.problem import Problem
from solver import SquareSolver

class Mountain(Problem):
    k = 3
    solver_class = SquareSolver

    def eval_expected(self, x, y):
        return cos(x/2)**4 * cos(y/2)**4

    def eval_f(self, x, y):
        return (self.k**2*cos(x/2)**4*cos(y/2)**4 + 
            (3*sin(x/2)**2 - cos(x/2)**2)*cos(x/2)**2*cos(y/2)**4 +
            (3*sin(y/2)**2 - cos(y/2)**2)*cos(x/2)**4*cos(y/2)**2)
