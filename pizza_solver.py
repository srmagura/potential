import itertools as it
import math
import numpy as np
import scipy
from scipy.integrate import quad

from solver import Solver
import matrices

import matplotlib.pyplot as plt

N_BASIS = 15

def complex_quad(f, a, b):
    def real_func(x):
        return scipy.real(f(x))
    def imag_func(x):
        return scipy.imag(f(x))

    real_integral = quad(real_func, a, b)
    imag_integral = quad(imag_func, a, b)
    return real_integral[0] + 1j*imag_integral[0]

def w(t):
    return 1 / np.sqrt(1 - t**2)

def eval_chebyshev(J, t):
    return np.cos(J * np.arccos(t))


class PizzaSolver(Solver):
    AD_len = 2*np.pi
    R = 2.3

    def __init__(self, problem, N, scheme_order, **kwargs):
        self.a = problem.sectorAngle
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
        r, th = self.get_polar(i, j)
        return r <= self.R and (th >= self.a or th <= 0) 

    def g(self, segment_id, t):
        if segment_id == 0:
            return np.pi*t + np.pi + .5*self.problem.sectorAngle
        elif segment_id == 1 or segment_id == 2:
            return 6/10*self.R*t + self.R/2

    def g_inv(self, segment_id, arg):
        if segment_id == 0:
            th = arg
            return (th - np.pi - .5*self.problem.sectorAngle)/np.pi
        elif segment_id == 1 or segment_id == 2:
            r = arg
            return 10/(6*self.R)*(r - self.R/2)

    def calc_c0(self, segment_id):
        self.c0.append([])
        for J in range(N_BASIS):
            def integrand(t):
                return (w(t) * self.problem.eval_bc(segment_id, 
                    self.g(segment_id, t)) *    
                    eval_chebyshev(J, t))

            I = complex_quad(integrand, -1, 1)
            if J == 0:
                self.c0[segment_id].append(I/np.pi)
            else:
                self.c0[segment_id].append(2*I/np.pi)
    
    def extend_basis(self, J, index):
        ext = np.zeros(len(self.gamma), dtype=complex)
        for l in range(len(self.gamma)):
            i, j = self.gamma[l]
            r, th = self.get_polar(i, j)
            exp = np.exp(complex(0, J*th))

            if index == 0:
                ext[l] = self.extend(r, th, exp, 0, -J**2 * exp, 0, J**4 * exp)
            else:
                ext[l] = self.extend(r, th, 0, exp, 0, -J**2 * exp, 0)  

        return ext

    def run(self):
        self.c0 = []
        self.calc_c0(0)
        self.calc_c0(1)
        self.calc_c0(2)

        self.do_plots()


    def do_plots(self):
        a = self.problem.sectorAngle

        th_data = np.arange(a, 2*np.pi, .05)
        r_data = np.linspace(0, self.R, 100)

        s_data = np.zeros(len(th_data) + 2*len(r_data))
        act_data = np.zeros(len(s_data), dtype=complex)
        exp_data = np.zeros(len(s_data), dtype=complex)

        Gamma_x_data = np.zeros(len(s_data))
        Gamma_y_data = np.zeros(len(s_data))

        j = 0
        for i in range(len(th_data)):
            th = th_data[i]
            s_data[j] = self.R * (th - a)

            for J in range(len(self.c0[0])):
                t = self.g_inv(0, th)
                act_data[j] += self.c0[0][J]*eval_chebyshev(J, t)

            exp_data[j] = self.problem.eval_bc(0, th)
            Gamma_x_data[j] = self.R * np.cos(th)
            Gamma_y_data[j] = self.R * np.sin(th)
            j += 1

        for i in range(len(r_data)-1, -1, -1):
            r = r_data[i]
            s_data[j] = self.R*(2*np.pi - a + 1) - r 

            for J in range(len(self.c0[1])):
                t = self.g_inv(1, r)
                act_data[j] += self.c0[1][J]*eval_chebyshev(J, t)

            exp_data[j] = self.problem.eval_bc(1, r)
            Gamma_x_data[j] = r 
            j += 1

        for i in range(len(r_data)):
            r = r_data[i]
            s_data[j] = self.R*(2*np.pi - a + 1) + r 

            for J in range(len(self.c0[2])):
                t = self.g_inv(1, r)
                assert -1 <= t and t <= 1
                act_data[j] += self.c0[2][J]*eval_chebyshev(J, t)

            exp_data[j] = self.problem.eval_bc(2, r)
            Gamma_x_data[j] = r*np.cos(a)
            Gamma_y_data[j] = r*np.sin(a)
            j += 1

        plt.plot(s_data, np.real(exp_data), label='Exact',
            color='#AAAAAA', linewidth=5)
        plt.plot(s_data, np.real(act_data), label='Expansion')
        plt.xlabel('s, arc length along boundary')
        plt.ylabel('Real part of boundary data')
        plt.legend()
        plt.show()
        plt.clf()

        plt.plot(Gamma_x_data, Gamma_y_data)

        def convert(points):
            x = []
            y = []

            for p in points:
                x1, y1 = self.get_coord(*p)
                x.append(x1)
                y.append(y1)

            return x, y

        plt.plot(*convert(self.gamma), marker='o', linestyle='')
        plt.show()

        return np.max(np.abs(act_data-exp_data))
