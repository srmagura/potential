import math
import numpy as np
import matplotlib.pyplot as plt

from cs import CircleSolver
import matrices
from chebyshev import *

N_BASIS = 25
DELTA = .1

B_desc = []
for JJ in range(3*N_BASIS):
    desc = {}
    desc['J'] = JJ % N_BASIS
    desc['sid'] = JJ // N_BASIS
    B_desc.append(desc)

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
    t = eval_g_inv(sid, th)
    J = B_desc[JJ]['J']

    return eval_T(J, t)

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
        self.c0 = []

        for sid in (0, 1):
            for J in range(N_BASIS):


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
            t = eval_g_inv(th)

            exact_data[l] = self.problem.eval_bc(th).real
            for J in range(self.n_basis):
                expansion_data[l] += (self.c0[J] * eval_T(J, t)).real

        plt.plot(th_data, exact_data, label='Exact')
        plt.plot(th_data, expansion_data, 'o', label='Expansion')
        plt.legend()
        plt.ylim(-1, 1)
        plt.title('c0')
        plt.show()
