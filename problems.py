import numpy as np
from numpy import cos, sin

from solver import SquareSolver
from circle_solver import CircleSolver
from pizza_solver import PizzaSolver

class Problem:
    homogeneous = False

    def get_solver(self, *args, **kwargs):
        return self.solver_class(self, *args, **kwargs)

    def eval_bc(self, th):
        return self.eval_expected(self.R*cos(th), self.R*sin(th))

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

class Wave(Problem):
    k = 1 
    kx = .8*k
    ky = .6*k

    solver_class = CircleSolver
    homogeneous = True

    def eval_f(self, x, y):
        return 0

    def eval_expected(self, x, y):
        return np.exp(complex(0, self.kx*x + self.ky*y))

    def eval_du_dr(self, th):
        a = self.kx*cos(th) + self.ky*sin(th)
        return 1j*a*np.exp(1j*self.R*a)

class Sine(Problem):
    k = 1 

    solver_class = CircleSolver
    homogeneous = True

    def eval_f(self, x, y):
        return 0

    def eval_expected(self, x, y):
        return sin(x)

    def eval_du_dr(self, th):
        return cos(th) * cos(self.R*cos(th))

class WavePizza(Wave):
    sectorAngle = np.pi / 6
    solver_class = PizzaSolver

    def eval_bc(self, segment_id, arg):
        if segment_id == 0:
            r = self.R
            th = arg
        elif segment_id == 1:
            r = arg
            th = 0
        elif segment_id == 2:
            r = arg
            th = self.sectorAngle

        x = r * cos(th)
        y = r * sin(th)
        return self.eval_expected(x, y)


problem_dict = {
    'mountain': Mountain,
    'ycosine': YCosine,
    'wave': Wave,
    'sine': Sine,
    'wave-pizza': WavePizza
}
