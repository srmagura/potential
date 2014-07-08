import math
import numpy as np
import matplotlib.pyplot as plt

from cs.cs import CircleSolver
import matrices
from chebyshev import *

N_BASIS = 25
DELTA = .1

N_SEGMENT = 3

segment_desc = []
B_desc = []

for sid in range(N_SEGMENT):
    s_desc = {'n_basis': N_BASIS}
    segment_desc.append(s_desc)

    for J in range(s_desc['n_basis']):
        b_desc = {} 
        b_desc['J'] = J
        b_desc['sid'] = sid
        B_desc.append(b_desc)


def get_sid(th):
    if th <= 2/3*np.pi:
        return 0
    elif th <= 4/3*np.pi:
        return 1
    else:
        return 2

def get_span_center(sid):
    return (np.pi/3 + DELTA, sid*2/3*np.pi + np.pi/3)

def eval_g(sid, t):
    span, center = get_span_center(sid)
    return span*t + center

def eval_g_inv(sid, th):
    span, center = get_span_center(sid)
    return (th - center) / span

d_g_inv_th = 1 / (np.pi/3 + DELTA) 

def _eval_dn_B_th(JJ, th, n, deriv):
    sid = get_sid(th)
    desc = B_desc[JJ]

    if desc['sid'] == sid:
        t = eval_g_inv(sid, th)
        J = B_desc[JJ]['J']
        return (d_g_inv_th)**n * deriv(J, t)
    else:
        return 0

def eval_B(JJ, th):
    return _eval_dn_B_th(JJ, th, 0, eval_T)

def eval_d2_B_th(JJ, th):
    return _eval_dn_B_th(JJ, th, 2, eval_d2_T_t)

def eval_d4_B_th(JJ, th):
    return _eval_dn_B_th(JJ, th, 4, eval_d4_T_t)

class CsChebyshev3(CircleSolver):
    # Side length of square domain on which AP is solved
    AD_len = 2*np.pi

    # Radius of circular region Omega
    R = 2.3

    # Returns the matrix Q0 or Q1, depending on the value of `index`.
    def get_Q(self, index):
        columns = []

        for JJ in range(len(B_desc)):
            ext = self.extend_basis(JJ, index)
            potential = self.get_potential(ext)
            projection = self.get_trace(potential)

            columns.append(projection - ext)

        return np.column_stack(columns)


    def extend_basis(self, JJ, index):
        ext = np.zeros(len(self.gamma), dtype=complex)
        for l in range(len(self.gamma)):
            i, j = self.gamma[l]
            r, th = self.get_polar(i, j)

            B = eval_B(JJ, th)
            d2_B_th = eval_d2_B_th(JJ, th)
            d4_B_th = eval_d4_B_th(JJ, th)

            if index == 0:
                ext[l] = self.extend_circle(r, B, 0, d2_B_th, 0, d4_B_th)
            else:
                ext[l] = self.extend_circle(r, 0, B, 0, d2_B_th, 0)  

        return ext

    def extend_boundary(self): 
        R = self.R
        k = self.problem.k

        boundary = np.zeros(len(self.gamma), dtype=complex)

        for l in range(len(self.gamma)):
            i, j = self.gamma[l]
            r, th = self.get_polar(i, j)

            xi0 = xi1 = 0
            d2_xi0_th = d2_xi1_th = 0
            d4_xi0_th = 0

            for JJ in range(len(B_desc)):
                B = eval_B(JJ, th)
                xi0 += self.c0[JJ] * B
                xi1 += self.c1[JJ] * B

                d2_B_th = eval_d2_B_th(JJ, th)
                d2_xi0_th += self.c0[JJ] * d2_B_th
                d2_xi1_th += self.c1[JJ] * d2_B_th

                d4_B_th = eval_d4_B_th(JJ, th)
                d4_xi0_th += self.c0[JJ] * d4_B_th

            boundary[l] = self.extend_circle(r, xi0, xi1,
                d2_xi0_th, d2_xi1_th, d4_xi0_th)
            boundary[l] += self.extend_inhomogeneous(r, th)

        return boundary

    def calc_c0(self):
        t_data = get_chebyshev_roots(1000)
        self.c0 = []

        for sid in range(N_SEGMENT):
            th_data = [eval_g(sid, t) for t in t_data] 
            boundary_data = [self.problem.eval_bc(th)
                for th in th_data] 

            n_basis = segment_desc[sid]['n_basis']
            self.c0.extend(np.polynomial.chebyshev.chebfit(
                t_data, boundary_data, n_basis-1))

    def run(self):
        self.calc_c0()
        self.calc_c1()

        ext = self.extend_boundary()
        u_act = self.get_potential(ext) + self.ap_sol_f

        error = self.eval_error(u_act)
        return error


    ## DEBUGGING FUNCTIONS ##

    def calc_c1_exact(self):
        t_data = get_chebyshev_roots(1000)
        self.c1 = []

        for sid in range(N_SEGMENT):
            th_data = [eval_g(sid, t) for t in t_data] 
            boundary_data = [self.problem.eval_d_u_r(th)
                for th in th_data] 

            n_basis = segment_desc[sid]['n_basis']
            self.c1.extend(np.polynomial.chebyshev.chebfit(
                t_data, boundary_data, n_basis-1))

    def extension_test(self):
        self.calc_c0()
        self.calc_c1_exact()

        ext = self.extend_boundary()
        error = np.zeros(len(self.gamma), dtype=complex)
        for l in range(len(self.gamma)):
            x, y = self.get_coord(*self.gamma[l])
            error[l] = self.problem.eval_expected(x, y) - ext[l]

        return np.max(np.abs(error))

    def c0_test(self):
        n = 50
        eps = .01
        th_data = np.linspace(eps, 2*np.pi-eps, n)
        exact_data = np.zeros(n)
        expansion_data = np.zeros(n)

        for l in range(n):
            th = th_data[l]
            exact_data[l] = self.problem.eval_bc(th).real

            for JJ in range(len(B_desc)):
                expansion_data[l] +=\
                    (self.c0[JJ] * eval_B(JJ, th)).real

        plt.plot(th_data, exact_data, label='Exact')
        plt.plot(th_data, expansion_data, 'o', label='Expansion')
        plt.legend()
        plt.ylim(-1, 1)
        plt.title('c0')
        plt.show()

    def c1_test(self):
        n = 50
        eps = .01
        th_data = np.linspace(eps, 2*np.pi-eps, n)
        exact_data = np.zeros(n)
        expansion_data = np.zeros(n)

        for l in range(n):
            th = th_data[l]
            exact_data[l] = self.problem.eval_d_u_r(th).real

            for JJ in range(len(B_desc)):
                expansion_data[l] +=\
                    (self.c1[JJ] * eval_B(JJ, th)).real

        plt.plot(th_data, exact_data, label='Exact')
        plt.plot(th_data, expansion_data, 'o', label='Expansion')
        plt.legend()
        plt.ylim(-1, 1)
        plt.title('c1')
        plt.show()
