import math
import numpy as np
import scipy
from scipy.integrate import quad
import matplotlib.pyplot as plt

from solver import Solver
import matrices
from chebyshev import *

N_BASIS = 20

def eval_g(t):
    return np.pi*t + np.pi

def eval_g_inv(th):
    return (th - np.pi) / np.pi

class CircleSolver1(Solver):
    # Side length of square domain on which AP is solved
    AD_len = 2*np.pi

    # Radius of circular region Omega
    R = 2.3

    def __init__(self, problem, N, scheme_order, **kwargs):
        super().__init__(problem, N, scheme_order, **kwargs)
        problem.R = self.R

    # Get the polar coordinates of grid point (i,j)
    def get_polar(self, i, j):
        x, y = self.get_coord(i, j)
        r, th = math.hypot(x, y), math.atan2(y, x)

        if th < 0:
            th += 2*np.pi

        return r, th

    # Get the rectangular coordinates of grid point (i,j)
    def get_coord(self, i, j):
        x = (self.AD_len * i / self.N - self.AD_len/2, 
            self.AD_len * j / self.N - self.AD_len/2)
        return x

    def is_interior(self, i, j):
        return self.get_polar(i, j)[0] <= self.R

    # Calculate the difference potential of a function xi 
    # that is defined on gamma.
    def get_potential(self, xi):
        w = np.zeros([(self.N-1)**2], dtype=complex)

        for l in range(len(self.gamma)):
            w[matrices.get_index(self.N, *self.gamma[l])] = xi[l]

        Lw = np.ravel(self.L.dot(w))

        for i,j in self.Mminus:
            Lw[matrices.get_index(self.N, i, j)] = 0

        return w - self.LU_factorization.solve(Lw)

    def get_trace(self, data):
        projection = np.zeros(len(self.gamma), dtype=complex)

        for l in range(len(self.gamma)):
            index = matrices.get_index(self.N, *self.gamma[l])
            projection[l] = data[index]

        return projection

    # Returns the matrix Q0 or Q1, depending on the value of `index`.
    def get_Q(self, index):
        columns = []

        for J in range(N_BASIS):
            ext = self.extend_basis(J, index)
            potential = self.get_potential(ext)
            projection = self.get_trace(potential)

            columns.append(projection - ext)

        return np.column_stack(columns)

    def calc_c1(self):
        Q0 = self.get_Q(0)
        Q1 = self.get_Q(1)

        rhs = -Q0.dot(self.c0)
        self.c1 = np.linalg.lstsq(Q1, rhs)[0]

    def extend(self, r, th, xi0, xi1):
        R = self.R

        derivs = []
        derivs.append(xi0) 
        derivs.append(xi1)

        v = 0
        for l in range(len(derivs)):
            v += derivs[l] / math.factorial(l) * (r - R)**l

        return v

    def extend_basis(self, J, index):
        ext = np.zeros(len(self.gamma), dtype=complex)
        for l in range(len(self.gamma)):
            i, j = self.gamma[l]
            r, th = self.get_polar(i, j)

            t = eval_g_inv(th)
            basis = eval_T(J, t)

            if index == 0:
                ext[l] = self.extend(r, th, basis, 0)
            else:
                ext[l] = self.extend(r, th, 0, basis)  

        return ext

    def calc_c0(self):
        t_data = np.arange(-1, 1, .005)
        boundary_data = [self.problem.eval_bc(eval_g(t)) for t in t_data] 
        self.c0 = np.polynomial.chebyshev.chebfit(t_data, boundary_data, N_BASIS-1)

    def run(self):
        self.calc_c0()
        self.calc_c1()

        #self.c0_test()
        self.c1_test()

    def c0_test(self):
        print('c0')
        print(np.around(self.c0, 2))

        th_data = np.linspace(0, 2*np.pi, 500)

        expansion_data = np.zeros(len(th_data))
        exact_data = np.zeros(len(th_data))

        j = 0

        for i in range(len(th_data)):
            th = th_data[i]
            t = eval_g_inv(th)

            for J in range(len(self.c0)):
                expansion_data[j] += (self.c0[J]*eval_T(J, t)).real

            exact_data[j] = self.problem.eval_bc(th).real
            j += 1

        plt.plot(th_data, exact_data, label='Exact', linewidth=9, color='#BBBBBB')
        plt.plot(th_data, expansion_data, label='Reconstructed from c0')
        plt.xlim(min(th_data)-.2,max(th_data)+.2)
        plt.title('Dirichlet data')
        plt.legend()
        plt.show()

    def c1_test(self):
        print('c1')
        print(np.around(self.c1, 2))

        th_data = np.linspace(0, 2*np.pi, 500)

        expansion_data = np.zeros(len(th_data))
        exact_data = np.zeros(len(th_data))

        j = 0

        for i in range(len(th_data)):
            th = th_data[i]
            t = eval_g_inv(th)

            for J in range(len(self.c1)):
                expansion_data[j] += (self.c1[J]*eval_T(J, t)).real

            exact_data[j] = self.problem.eval_du_dr(th).real
            j += 1

        plt.plot(th_data, exact_data, label='Exact', linewidth=9, color='#BBBBBB')
        plt.plot(th_data, expansion_data, label='Reconstructed from c1')
        plt.xlim(min(th_data)-.2,max(th_data)+.2)
        plt.title('Neumann data')
        plt.legend()
        plt.show()
