import itertools as it
import math
import numpy as np
import scipy
from scipy.integrate import quad

from solver import Solver
import matrices

import matplotlib.pyplot as plt

def complex_quad(f, a, b):
    def real_func(x):
        return scipy.real(f(x))
    def imag_func(x):
        return scipy.imag(f(x))

    real_integral = quad(real_func, a, b)
    imag_integral = quad(imag_func, a, b)
    return real_integral[0] + 1j*imag_integral[0]

def eval_chebyshev(J, t):
    return np.cos(J * np.arccos(t))

class PizzaSolver(Solver):
    AD_len = 2*np.pi
    R = 2.3

    def __init__(self, problem, N, scheme_order, **kwargs):
        super().__init__(problem, N, scheme_order, **kwargs)
        problem.R = self.R

    # Get the polar coordinates of grid point (i,j)
    def get_polar(self, i, j):
        x, y = self.get_coord(i, j)
        return math.hypot(x, y), math.atan2(y, x)

    # Get the rectangular coordinates of grid point (i,j)
    def get_coord(self, i, j):
        x = (self.AD_len * i / self.N - self.AD_len/2, 
            self.AD_len * j / self.N - self.AD_len/2)
        return x

    def is_interior(self, i, j):
        #FIXME
        return self.get_polar(i, j)[0] <= self.R

    def run(self):
        def w(t):
            return 1 / np.sqrt(1 - t**2)

        def g(t):
            return np.pi*t + np.pi + .5*self.problem.sectorAngle

        def ginv(th):
            return (th - np.pi - .5*self.problem.sectorAngle)/np.pi

        c0 = [[]]
        for J in range(10):
            def integrand(t):
                return (w(t) * self.problem.eval_bc(0, g(t)) *    
                    eval_chebyshev(J, t))

            I = complex_quad(integrand, -1, 1)
            if J == 0:
                c0[0].append(I/np.pi)
            else:
                c0[0].append(2*I/np.pi)

        th_data = np.arange(0, 2*np.pi, .05)
        act_data = np.zeros(len(th_data))
        exp_data = np.zeros(len(th_data))
        for i in range(len(th_data)):
            th = th_data[i]
            for J in range(len(c0[0])):
                act_data[i] += c0[0][J]*eval_chebyshev(J, ginv(th)).real

            exp_data[i] = self.problem.eval_bc(0, th)

        plt.plot(th_data, act_data, label='act')
        plt.plot(th_data, exp_data, 'g^', label='exp')
        plt.legend()
        plt.show()
        return np.max(np.abs(act_data-exp_data))
