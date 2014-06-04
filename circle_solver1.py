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
        if self.N < 128:
            N_data = (16, 32, 64, 128) 
            n_basis_data = (15, 22, 33, 45)
            f = interp1d(N_data, n_basis_data)
            self.n_basis = int(f(self.N))
        else:
            self.n_basis = 45

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

    def extend_boundary(self): 
        R = self.R
        k = self.problem.k

        boundary = np.zeros(len(self.gamma), dtype=complex)

        for l in range(len(self.gamma)):
            i, j = self.gamma[l]
            r, th = self.get_polar(i, j)
            t = eval_g_inv(th)

            xi0 = xi1 = 0

            for J in range(self.n_basis):
                basis = eval_T(J, t)
                xi0 += self.c0[J] * basis
                xi1 += self.c1[J] * basis

            boundary[l] = self.extend(r, th, xi0, xi1)

        return boundary

    def calc_c0(self):
        t_data = np.linspace(-1, 1, 1000)
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
