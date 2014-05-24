import itertools as it
import math
import collections
import numpy as np
import scipy
from scipy.integrate import quad
import matplotlib.pyplot as plt

from solver import Solver
import matrices
from chebyshev import *


N_BASIS = 20
DELTA = .05

def complex_quad(f, a, b):
    def real_func(x):
        return scipy.real(f(x))
    def imag_func(x):
        return scipy.imag(f(x))

    real_integral = quad(real_func, a, b)
    imag_integral = quad(imag_func, a, b)
    return real_integral[0] + 1j*imag_integral[0]

def eval_basis(J, sid, t):
    if J < N_BASIS:
        if sid == 0:
            return eval_T(J, t)
    else:
        if sid == 1:
            return eval_T(J - N_BASIS, t)

    return 0

def g(sid, t):
    if sid == 0:
        return (np.pi/2 + DELTA)*t + np.pi/2
    else:
        return (np.pi/2 + DELTA)*t + 3*np.pi/2

def g_inv(sid, th):
    if sid == 0:
        return (th - np.pi/2) / (np.pi/2 + DELTA)
    else:
        return (th - 3*np.pi/2) / (np.pi/2 + DELTA)

def get_sid(th):
    if th <= np.pi:
        return 0
    else:
        return 1


class CircleSolver2(Solver):
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

        for J in range(2*N_BASIS):
            ext = self.extend_basis(J, index)
            potential = self.get_potential(ext)
            projection = self.get_trace(potential)

            columns.append(projection - ext)

        return np.column_stack(columns)

    def calc_c1(self):
        Q0 = self.get_Q(0)
        Q1 = self.get_Q(1)

        self.ap_sol_f = self.LU_factorization.solve(self.B_src_f)
        ext_f = self.extend_inhomogeneous_f()    
        proj_f = self.get_trace(self.get_potential(ext_f))

        rhs = -Q0.dot(self.c0) - self.get_trace(self.ap_sol_f) - proj_f + ext_f
        self.c1 = np.linalg.lstsq(Q1, rhs)[0]

    def extend(self, r, th, xi0, xi1, d2_xi0_th):#, d2_xi1_th, d4_xi0_th):
        R = self.R
        k = self.k

        derivs = []
        derivs.append(xi0) 
        derivs.append(xi1)
        derivs.append(-xi1 / R - d2_xi0_th / R**2 - k**2 * xi0)
        #derivs.append(2 * xi1 / R**2 + 3 * d2_xi0_th / R**3 -
        #    d2_xi1_th / R**2 + k**2 / R * xi0 - k**2 * xi1)
        #derivs.append(-6 * xi1 / R**3 + 
        #    (2*k**2 / R**2 - 11 / R**4) * d2_xi0_th +
        #    6 * d2_xi1_th / R**3 + d4_xi0_th / R**4 -
        #    (3*k**2 / R**2 - k**4) * xi0 +
        #    2 * k**2 / R * xi1)

        v = 0
        for l in range(len(derivs)):
            v += derivs[l] / math.factorial(l) * (r - R)**l

        return v

    # Construct the equation-based extension for the basis function
    # $\psi_{index}^{(J)}$.
    def extend_basis(self, J, index):
        ext = np.zeros(len(self.gamma), dtype=complex)
        for l in range(len(self.gamma)):
            i, j = self.gamma[l]
            r, th = self.get_polar(i, j)
            sid = get_sid(th)

            t = g_inv(sid, th)
            val = eval_basis(J, sid, t)

            if index == 0:
                ext[l] = self.extend(r, th, val, 0)
            else:
                ext[l] = self.extend(r, th, 0, val)  

        return ext

    def extend_inhomogeneous(self, r, th):
        p = self.problem
        if p.homogeneous:
            return 0

        R = self.R
        k = self.k

        x = R * np.cos(th)
        y = R * np.sin(th)

        derivs = [0, 0]
        derivs.append(p.eval_f(x, y))
        derivs.append(p.eval_d_f_r(R, th) - p.eval_f(x, y) / R)
        derivs.append(p.eval_d2_f_r(R, th) - 
            p.eval_d2_f_th(R, th) / R**2 - 
            p.eval_d_f_r(R, th) / R + 
            (3/R**2 - k**2) * p.eval_f(x, y))

        v = 0
        for l in range(len(derivs)):
            v += derivs[l] / math.factorial(l) * (r - R)**l

        return v

    def extend_inhomogeneous_f(self):
        ext = np.zeros(len(self.gamma), dtype=complex)
        for l in range(len(self.gamma)):
            i, j = self.gamma[l]
            r, th = self.get_polar(i, j)
            ext[l] = self.extend_inhomogeneous(r, th)

        return ext

    def extend_src_f(self):
        p = self.problem
        if p.homogeneous:
            return

        for i,j in self.Kplus - self.Mplus:
            R = self.R

            r, th = self.get_polar(i, j)
            x = R * np.cos(th)
            y = R * np.sin(th)

            derivs = [p.eval_f(x, y), p.eval_d_f_r(R, th), 
                p.eval_d2_f_r(R, th)]

            v = 0
            for l in range(len(derivs)):
                v += derivs[l] / math.factorial(l) * (r - R)**l

            self.src_f[matrices.get_index(self.N,i,j)] = v

    # Construct the equation-based extension for the boundary data,
    # as approximated by the coefficients c0 and c1.
    def extend_boundary(self): 
        R = self.R
        k = self.problem.k

        boundary = np.zeros(len(self.gamma), dtype=complex)

        for l in range(len(self.gamma)):
            i, j = self.gamma[l]
            r, th = self.get_polar(i, j)
            sid = get_sid(th)
            t = g_inv(sid, th)

            xi0 = xi1 = 0
            #d2_xi0_th = d2_xi1_th = 0 
            #d4_xi0_th = 0

            for J in range(2*N_BASIS):
                val = eval_basis(J, sid, t)
                xi0 += self.c0[J] * val
                #d2_xi0_th += -J**2 * c0[i] * exp
                #d4_xi0_th += J**4 * c0[i] * exp

                xi1 += self.c1[J] * val
                #d2_xi1_th += -J**2 * c1[i] * exp

            boundary[l] = self.extend(r, th, xi0, xi1)
            #boundary[l] += self.extend_inhomogeneous(r, th)

        return boundary

    def calc_c0(self):
        self.c0 = []

        for sid in (0, 1):
            for J in range(N_BASIS):
                def integrand(t):
                    return (eval_weight(t) * 
                        self.problem.eval_bc(g(sid, t)) *    
                        eval_T(J, t))

                I = complex_quad(integrand, -1, 1)
                if J == 0:
                    self.c0.append(I/np.pi)
                else:
                    self.c0.append(2*I/np.pi)

    def run(self):
        self.calc_c0()
        self.calc_c1()
        ext = self.extend_boundary()
        u_act = self.get_potential(ext) + self.ap_sol_f

        error = self.eval_error(u_act)
        return error

    def c_test(self):
        th_data = np.linspace(0, 2*np.pi, 300)

        expansion_data0 = np.zeros(len(th_data))
        exact_data0 = np.zeros(len(th_data))

        expansion_data1 = np.zeros(len(th_data))
        exact_data1 = np.zeros(len(th_data))

        j = 0

        for i in range(len(th_data)):
            th = th_data[i]
            sid = get_sid(th)
            t = g_inv(sid, th)

            for J in range(len(self.c0)):
                expansion_data0[j] += (self.c0[J]*eval_basis(J, sid, t)).real
                expansion_data1[j] += (self.c1[J]*eval_basis(J, sid, t)).real

            exact_data0[j] = self.problem.eval_bc(th).real
            exact_data1[j] = self.problem.eval_du_dr(th).real
            j += 1

        plt.plot(th_data, exact_data0, label='Exact', linewidth=9, color='#BBBBBB')
        plt.plot(th_data, expansion_data0, label='Expansion')
        plt.title('c0')
        plt.legend()
        plt.show()
        plt.clf()

        plt.plot(th_data, exact_data1, label='Exact', linewidth=9, color='#BBBBBB')
        plt.plot(th_data, expansion_data1, label='Expansion')
        plt.title('c1')
        plt.legend()
        plt.show()

