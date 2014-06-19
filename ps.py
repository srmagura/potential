import math
import numpy as np

from solver import Solver, cart_to_polar
import matrices

import matplotlib.pyplot as plt

from chebyshev import *

EXTEND_CIRCLE = 0
EXTEND_RADIUS1 = 1
EXTEND_RADIUS2 = 2
EXTEND_OUTER = 3
EXTEND_INNER = 4

ALL_ETYPES = range(5)
ETYPE_NAMES = ('circle', 'radius1', 'radius2', 'outer', 'inner')

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
        self.get_sid = problem.get_sid
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

        if sid == 1:
            span = -span

        return span, center

    def eval_g(self, sid, t):
        span, center = self.get_span_center(sid)
        return span*t + center

    def eval_g_inv(self, sid, arg):
        span, center = self.get_span_center(sid)
        return (arg - center) / span

    def eval_B(self, JJ, r, th):
        sid = self.get_sid(th)
        desc = B_desc[JJ]

        if desc['sid'] == sid:
            if sid == 0:
                arg = th
            elif sid == 1 or sid == 2:
                arg = r

            t = self.eval_g_inv(sid, arg)
            J = B_desc[JJ]['J']
            return eval_T(J, t)
        else:
            return 0

    def get_etype(self, i, j):
        def dist_to_radius(m):
            x1 = (1/(m + 1/m)) * (x/m + y)
            y1 = m*x
            return np.sqrt((x1-x)**2 + (y1-y)**2)

        x, y = self.get_coord(i, j)
        r, th = self.get_polar(i, j)

        dist0 = abs(self.R - r)
        dist1 = abs(y)
        dist2 = dist_to_radius(np.tan(self.a))

        if r > self.R:
            if th < self.a:
                return EXTEND_OUTER
            else:
                return EXTEND_CIRCLE
        elif th > self.a + np.pi/2 and th < 3/2*np.pi:
            if r < dist0:
                raise Exception('EXTEND_INNER')
                return EXTEND_INNER
            else:
                return EXTEND_CIRCLE
        elif th > self.a and dist0 < min(dist1, dist2):
            return EXTEND_CIRCLE
        elif dist1 < dist2:
            return EXTEND_RADIUS1
        else:
            return EXTEND_RADIUS2

        raise Exception('Did not match any etype')

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
        th_data = np.arange(self.a+.1, 2*np.pi, .1)
        r_data = np.linspace(0, self.R, 25)

        points = []
        arg_datas = (th_data, r_data[::-1], r_data)

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
            for i in range(len(arg_data)):
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
        self.plot_gamma()
        #return self.extension_test({EXTEND_CIRCLE})

    def calc_c1_exact(self):
        t_data = get_chebyshev_roots(1000)
        self.c1 = []

        for sid in range(N_SEGMENT):
            arg_data = [self.eval_g(sid, t) for t in t_data] 
            boundary_points = self.get_boundary_sample_by_sid(sid, arg_data)
            boundary_data = np.zeros(len(arg_data), dtype=complex)
            for l in range(len(boundary_points)):
                p = boundary_points[l]
                boundary_data[l] = self.problem.eval_d_u_r(
                    p['x'], p['y'], sid=sid)

            n_basis = segment_desc[sid]['n_basis']
            self.c1.extend(np.polynomial.chebyshev.chebfit(
                t_data, boundary_data, n_basis-1))

    def gamma_filter(self, nodes, etypes):
        result = []
        for i, j in nodes:
            if self.get_etype(i, j) in etypes:
                result.append((i, j))

        return result

    def extension_test(self, etypes=ALL_ETYPES):
        self.calc_c0()
        self.calc_c1_exact()
        ext = self.extend_boundary()

        nodes = self.gamma_filter(self.gamma, etypes)
        error = np.zeros(len(nodes))
        for l in range(len(nodes)):
            x, y = self.get_coord(*nodes[l])
            error[l] = self.problem.eval_expected(x, y) - ext[l]

        return np.max(np.abs(error))

    def c0_test(self):
        sample = self.get_boundary_sample()

        s_data = np.zeros(len(sample))
        exact_data = np.zeros(len(sample))
        expansion_data = np.zeros(len(sample))

        for l in range(len(sample)):
            p = sample[l]
            s_data[l] = p['s']
            exact_data[l] = self.problem.eval_bc(p['x'], p['y']).real

            r, th = cart_to_polar(p['x'], p['y'])
            for JJ in range(len(B_desc)):
                expansion_data[l] +=\
                    (self.c0[JJ] * self.eval_B(JJ, r, th)).real

        plt.plot(s_data, exact_data, label='Exact')
        plt.plot(s_data, expansion_data, 'o', label='Expansion')
        plt.legend()
        plt.ylim(-1, 1)
        plt.title('c0')
        plt.show()

    def c1_test(self):
        sample = self.get_boundary_sample()

        s_data = np.zeros(len(sample))
        exact_data = np.zeros(len(sample))
        expansion_data = np.zeros(len(sample))

        for l in range(len(sample)):
            p = sample[l]
            s_data[l] = p['s']
            exact_data[l] = self.problem.eval_d_u_r(p['x'], p['y']).real

            r, th = cart_to_polar(p['x'], p['y'])
            for JJ in range(len(B_desc)):
                expansion_data[l] +=\
                    (self.c1[JJ] * self.eval_B(JJ, r, th)).real

        plt.plot(s_data, exact_data, label='Exact')
        plt.plot(s_data, expansion_data, 'o', label='Expansion')
        plt.legend()
        plt.ylim(-1, 1)
        plt.title('c1')
        plt.show()

    def plot_Gamma(self):
        sample = self.get_boundary_sample()

        n = len(sample) + 1
        Gamma_x_data = np.zeros(n)
        Gamma_y_data = np.zeros(n)

        for l in range(n):
            p = sample[l % len(sample)]
            Gamma_x_data[l] = p['x']
            Gamma_y_data[l] = p['y']

        plt.plot(Gamma_x_data, Gamma_y_data, color='black')

    def nodes_to_plottable(self, nodes):
        x_data = np.zeros(len(nodes))
        y_data = np.zeros(len(nodes))

        for l in range(len(nodes)):
            x, y = self.get_coord(*nodes[l])
            x_data[l] = x
            y_data[l] = y

        return x_data, y_data

    def plot_gamma(self):
        self.plot_Gamma()

        for etype in ALL_ETYPES:
            nodes = self.gamma_filter(self.gamma, {etype})
            x_data, y_data = self.nodes_to_plottable(nodes)
            plt.plot(x_data, y_data, 'o', label=ETYPE_NAMES[etype])

        plt.xlim(-4,4)
        plt.ylim(-4,4)
        plt.legend(loc=3)
        plt.show()
