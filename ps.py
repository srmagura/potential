import math
import numpy as np

from solver import Solver
import matrices

import matplotlib.pyplot as plt

from chebyshev import *

n_basis_by_sid = (30, 15, 15)
N_SEGMENT = 3

segment_desc = []
B_desc = []

for sid in range(N_SEGMENT):
    s_desc = {'n_basis': n_basis_by_sid[sid]}
    segment_desc.append(s_desc)

    for J in range(s_desc['n_basis']):
        b_desc = {} 
        b_desc['J'] = J
        b_desc['sid'] = sid
        B_desc.append(b_desc)


class PizzaSolver(Solver):
    AD_len = 2*np.pi
    R = 2.3

    def __init__(self, problem, N, scheme_order, **kwargs):
        self.a = problem.a
        problem.R = self.R
        super().__init__(problem, N, scheme_order, **kwargs)

    def is_interior(self, i, j):
        r, th = self.get_polar(i, j)
        return r <= self.R and (th >= self.a or th <= 0) 

    def get_span_center(self, sid):
        if sid == 0:
            span = np.pi
            center = np.pi + .5*self.a
        elif sid == 1 or sid == 2:
            span = 6/10*self.R
            center = self.R/2

        return span, center

    def eval_g(self, sid, t):
        span, center = self.get_span_center(sid)
        return span*t + center

    def eval_g_inv(self, sid, arg):
        span, center = self.get_span_center(sid)
        return (arg - center) / span

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
            arg_data = [self.eval_g(sid, t) for t in t_data] 
            boundary_points = self.get_boundary_sample_by_sid(sid, arg_data)
            boundary_data = np.zeros(len(arg_data), dtype=complex)
            for l in range(len(boundary_points)):
                p = boundary_points[l]
                boundary_data[l] = self.problem.eval_bc(p['x'], p['y'])

            n_basis = segment_desc[sid]['n_basis']
            self.c0.extend(np.polynomial.chebyshev.chebfit(
                t_data, boundary_data, n_basis-1))
    

    def get_boundary_sample(self):
        th_data = np.arange(self.a, 2*np.pi, .05)
        r_data = np.linspace(0, self.R, 100)

        points = []
        arg_datas = (th_data, r_data, r_data)

        for sid in range(N_SEGMENT):
            points.extend(
                self.get_boundary_sample_by_sid(sid, arg_datas[sid]))

        return points

    def get_boundary_sample_by_sid(self, sid, arg_data):
        points = []
        a = self.a

        if sid == 0:
            for i in range(len(arg_data)):
                th = arg_data[i]

                points.append({'x': self.R * np.cos(th),
                    'y': self.R * np.sin(th),
                    's': self.R * (th - a),
                    'sid': 0})

        elif sid == 1:
            for i in range(len(arg_data)-1, -1, -1):
                r = arg_data[i]

                points.append({'x': r,
                    'y': 0,
                    's': self.R*(2*np.pi - a + 1) - r,
                    'sid': 1})

        elif sid == 2:
            for i in range(len(arg_data)):
                r = arg_data[i]

                points.append({'x': r*np.cos(a),
                    'y': r*np.sin(a),
                    's': self.R*(2*np.pi - a + 1) + r,
                    'sid': 2})

        return points

    def run(self):
        self.calc_c0()
        self.c0_test()

    def c0_test(self):
        sample = self.get_boundary_sample()

        s_data = np.zeros(len(sample))
        exact_data = np.zeros(len(sample))
        expansion_data = np.zeros(len(sample))

        for l in range(len(sample)):
            p = sample[l]
            s_data[l] = p['s']
            exact_data[l] = self.problem.eval_bc(p['x'], p['y']).real

            #for JJ in range(len(B_desc)):
            #    expansion_data[l] +=\
            #        (self.c0[JJ] * eval_B(JJ, th)).real

        plt.plot(s_data, exact_data, label='Exact')
        #plt.plot(th_data, expansion_data, 'o', label='Expansion')
        plt.legend()
        plt.ylim(-1, 1)
        plt.title('c0')
        plt.show()


