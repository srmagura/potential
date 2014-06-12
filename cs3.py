import math
import numpy as np
import matplotlib.pyplot as plt

from cs import CircleSolver
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

def eval_B(JJ, th):
    sid = get_sid(th)
    desc = B_desc[JJ]

    if desc['sid'] == sid:
        t = eval_g_inv(sid, th)
        J = B_desc[JJ]['J']
        return eval_T(J, t)
    else:
        return 0

class CsChebyshev3(CircleSolver):
    # Side length of square domain on which AP is solved
    AD_len = 2*np.pi

    # Radius of circular region Omega
    R = 2.3

    # Returns the matrix Q0 or Q1, depending on the value of `index`.
    def get_Q(self, index):
        columns = []

        for J in range(2*N_BASIS):
            ext = self.extend_basis(J, index)
            potential = self.get_potential(ext)
            projection = self.get_trace(potential)

            columns.append(projection - ext)

        return np.column_stack(columns)


    def extend_basis(self, J, index):
        ext = np.zeros(len(self.gamma), dtype=complex)
        for l in range(len(self.gamma)):
            i, j = self.gamma[l]
            r, th = self.get_polar(i, j)

        return ext


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
        self.c0_test()
        return

        self.calc_c1()

        ext = self.extend_boundary()
        u_act = self.get_potential(ext) #+ self.ap_sol_f

        error = self.eval_error(u_act)
        return error

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
