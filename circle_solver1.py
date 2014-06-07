import numpy as np
import math
import scipy
from scipy.integrate import quad
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from solver import Solver
import matrices
from chebyshev import *

DELTA = .1

def eval_g(t):
    return (np.pi+DELTA)*t + np.pi

def eval_g_inv(th):
    return (th - np.pi) / (np.pi+DELTA)

d_g_inv_t = 1 / (np.pi + DELTA)

class CircleSolver1(Solver):

    # Side length of square domain on which AP is solved
    AD_len = 2*np.pi

    # Radius of circular region Omega
    R = 2.3

    def __init__(self, problem, N, scheme_order, **kwargs):
        super().__init__(problem, N, scheme_order, **kwargs)
        problem.R = self.R

    def is_interior(self, i, j):
        return self.get_polar(i, j)[0] <= self.R

    def calc_n_basis(self):
        self.n_basis = 55

        if self.verbose:
            print('Using {} basis functions.'.format(self.n_basis))

    # Returns the matrix Q0 or Q1, depending on the value of `index`.
    def get_Q(self, index):
        columns = []

        for J in range(self.n_basis):
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

    def extend(self, r, th, xi0, xi1, d2_xi0_th, d2_xi1_th, d4_xi0_th):
        R = self.R
        k = self.k

        derivs = []
        derivs.append(xi0) 
        derivs.append(xi1)
        derivs.append(-xi1 / R - d2_xi0_th / R**2 - k**2 * xi0)
        derivs.append(2 * xi1 / R**2 + 3 * d2_xi0_th / R**3 -
            d2_xi1_th / R**2 + k**2 / R * xi0 - k**2 * xi1)
        derivs.append(-6 * xi1 / R**3 + 
            (2*k**2 / R**2 - 11 / R**4) * d2_xi0_th +
            6 * d2_xi1_th / R**3 + d4_xi0_th / R**4 -
            (3*k**2 / R**2 - k**4) * xi0 +
            2 * k**2 / R * xi1)

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
            d2_T_th = (d_g_inv_t)**2 * eval_d2_T_t(J, t)
            d4_T_th = (d_g_inv_t)**4 * eval_d4_T_t(J, t)

            if index == 0:
                ext[l] = self.extend(r, th, basis, 0, d2_T_th, 0, d4_T_th)
            else:
                ext[l] = self.extend(r, th, 0, basis, 0, d2_T_th, 0)  

        return ext

    def extend_boundary(self): 
        R = self.R
        k = self.problem.k

        boundary = np.zeros(len(self.gamma), dtype=complex)

        for l in range(len(self.gamma)):
            i, j = self.gamma[l]
            r, th = self.get_polar(i, j)
            t = eval_g_inv(th)

            xi0 = xi1 = 0
            d2_xi0_th = d2_xi1_th = 0
            d4_xi0_th = 0

            for J in range(self.n_basis):
                basis = eval_T(J, t)
                xi0 += self.c0[J] * basis
                xi1 += self.c1[J] * basis

                d2_T_th = (d_g_inv_t)**2 * eval_d2_T_t(J, t)
                d2_xi0_th += self.c0[J] * d2_T_th
                d2_xi1_th += self.c1[J] * d2_T_th

                d4_T_th = (d_g_inv_t)**4 * eval_d4_T_t(J, t)
                d4_xi0_th += self.c0[J] * d4_T_th

            boundary[l] = self.extend(r, th, xi0, xi1,
                d2_xi0_th, d2_xi1_th, d4_xi0_th)

        return boundary

    def calc_c0(self):
        t_data = get_chebyshev_roots(1000)
        boundary_data = [self.problem.eval_bc(eval_g(t)) for t in t_data] 
        self.c0 = np.polynomial.chebyshev.chebfit(
            t_data, boundary_data, self.n_basis-1)
    
    def run(self):
        self.calc_n_basis()
        self.calc_c0()
        self.calc_c1()

        ext = self.extend_boundary()
        u_act = self.get_potential(ext)

        error = self.eval_error(u_act)
        return error


    ## DEBUGGING FUNCTIONS ##

    def calc_c1_exact(self):
        t_data = get_chebyshev_roots(1000)
        boundary_data = [self.problem.eval_d_u_r(eval_g(t)) 
            for t in t_data] 
        self.c1 = np.polynomial.chebyshev.chebfit(
            t_data, boundary_data, self.n_basis-1)

    def extension_test(self):
        self.calc_n_basis()
        self.calc_c0()
        self.calc_c1_exact()

        ext = self.extend_boundary()
        error = np.zeros(len(self.gamma), dtype=complex)
        for l in range(len(self.gamma)):
            x, y = self.get_coord(*self.gamma[l])
            error[l] = self.problem.eval_expected(x, y) - ext[l]

        return np.max(np.abs(error))

    def c0_test(self):
        n = 200
        eps = .01
        th_data = np.linspace(eps, 2*np.pi-eps, n)
        exact_data = np.zeros(n)
        expansion_data = np.zeros(n)

        for l in range(n):
            th = th_data[l]
            t = eval_g_inv(th)

            exact_data[l] = self.problem.eval_bc(th).real
            for J in range(self.n_basis):
                expansion_data[l] += (self.c0[J] * eval_T(J, t)).real

        plt.plot(th_data, exact_data, label='Exact')
        plt.plot(th_data, expansion_data, 'o', label='Expansion')
        plt.legend()
        plt.title('c0')
        plt.show()

    def c1_test(self):
        n = 200
        eps = .5
        th_data = np.linspace(eps, 2*np.pi-eps, n)
        exact_data = np.zeros(n)
        expansion_data = np.zeros(n)

        for l in range(n):
            th = th_data[l]
            t = eval_g_inv(th)

            exact_data[l] = self.problem.eval_d_u_r(th).real
            for J in range(self.n_basis):
                expansion_data[l] += (self.c1[J] * eval_T(J, t)).real

        plt.plot(th_data, exact_data, label='Exact')
        plt.plot(th_data, expansion_data, 'o', label='Expansion')
        plt.legend()
        plt.title('c1')
        plt.show()

    def n_basis_test(self):
        self.n_basis = 100
        self.calc_c0()

        Q0 = self.get_Q(0)
        Q1 = self.get_Q(1)

        while self.n_basis > 3:
            rhs = -Q0.dot(self.c0)
            self.c1 = np.linalg.lstsq(Q1, rhs)[0]

            ext = self.extend_boundary()
            u_act = self.get_potential(ext)

            error = self.eval_error(u_act)
            print('n_basis={}   error={}'.format(self.n_basis, error))

            self.n_basis -= 1
            Q0 = Q0[:,:-1]
            Q1 = Q1[:,:-1]
            self.c0 = self.c0[:-1]

