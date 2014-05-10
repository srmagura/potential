import numpy as np
from numpy import cos, sin

from potential.solver import SquareSolver
from potential.circle_solver import CircleSolver

class Problem:
    homogeneous = False

    def get_solver(self, *args, **kwargs):
        return self.solver_class(self, *args, **kwargs)

    def eval_bc(self, th):
        return self.eval_expected(self.R*np.cos(th), self.R*np.sin(th))

class Mountain(Problem):
    k = 1
    solver_class = SquareSolver

    def eval_expected(self, x, y):
        return np.cos(x/2)**4 * np.cos(y/2)**4

    def eval_f(self, x, y):
        return (self.k**2*cos(x/2)**4*cos(y/2)**4 + 
            (3*sin(x/2)**2 - cos(x/2)**2)*cos(x/2)**2*cos(y/2)**4 +
            (3*sin(y/2)**2 - cos(y/2)**2)*cos(x/2)**4*cos(y/2)**2)

class YCosine(Problem):
    k = 1
    solver_class = CircleSolver

    def eval_expected(self, x, y):
        return y*cos(x)

    def eval_f(self, x, y):
        return self.k**2*y*cos(x) - y*cos(x)

    def eval_d_f_r(self, r, th):
        return -r*self.k**2*sin(th)*sin(r*cos(th))*cos(th) + r*sin(th)*sin(r*cos(th))*cos(th) + self.k**2*sin(th)*cos(r*cos(th)) - sin(th)*cos(r*cos(th))

    def eval_d2_f_r(self, r, th):
        return (-r*self.k**2*cos(th)*cos(r*cos(th)) + r*cos(th)*cos(r*cos(th)) - 2*self.k**2*sin(r*cos(th)) + 2*sin(r*cos(th)))*sin(th)*cos(th)

    def eval_d2_f_th(self, r, th):
        return r*(-r**2*self.k**2*sin(th)**2*cos(r*cos(th)) + r**2*sin(th)**2*cos(r*cos(th)) + 3*r*self.k**2*sin(r*cos(th))*cos(th) - 3*r*sin(r*cos(th))*cos(th) - self.k**2*cos(r*cos(th)) + cos(r*cos(th)))*sin(th)

class ComplexWave(Problem):
    k = 1 
    kx = .8*k
    ky = .6*k

    solver_class = CircleSolver
    homogeneous = True

    def eval_f(self, x, y):
        return 0

    def eval_expected(self, x, y):
        return np.exp(complex(0, self.kx*x + self.ky*y))
