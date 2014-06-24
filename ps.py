import math
import numpy as np

from solver import Solver, cart_to_polar
import matrices

import matplotlib.pyplot as plt

from cs import extend_circle
from chebyshev import *

EXTEND_CIRCLE = 0
EXTEND_RADIUS1 = 1
EXTEND_RADIUS2 = 2
EXTEND_OUTER = 3
EXTEND_INNER = 4

ALL_ETYPES = range(5)
ETYPE_NAMES = ('circle', 'radius1', 'radius2', 'outer', 'inner')

n_basis_by_sid = (60, 30, 30)
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

    def eval_d_g_inv_arg(self, sid):
        span, center = self.get_span_center(sid)
        return 1 / span

    def eval_dn_B_arg(self, n, JJ, r, th):
        sid = self.get_sid(th)
        desc = B_desc[JJ]

        if desc['sid'] == sid:
            if sid == 0:
                arg = th
            elif sid == 1 or sid == 2:
                arg = r

            t = self.eval_g_inv(sid, arg)

            J = B_desc[JJ]['J']
            d_g_inv_arg = self.eval_d_g_inv_arg(sid)

            return (d_g_inv_arg)**n * eval_dn_T_t(n, J, t)
        else:
            return 0

    def get_radius_point(self, sid, x, y):
        if sid != 2:
            assert False

        m = np.tan(self.a)
        x1 = (x/m + y)/(m + 1/m)
        y1 = m*x1

        return (x1, y1) 

    def dist_to_radius(self, sid, x, y):
        x1, y1 = self.get_radius_point(sid, x, y)
        return np.sqrt((x1-x)**2 + (y1-y)**2)

    def get_etype(self, i, j):
        x, y = self.get_coord(i, j)
        r, th = self.get_polar(i, j)

        dist0 = abs(self.R - r)
        dist1 = abs(y)
        dist2 = self.dist_to_radius(2, x, y)

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

    def extend_from_radius(self, Y1, xi0, xi1, d2_xi0_X,
        d2_xi1_X, d4_xi0_X):

        k = self.k

        derivs = []
        derivs.append(xi0)
        derivs.append(xi1)
        derivs.append(-(d2_xi0_X + k**2 * xi0))

        derivs.append(-(d2_xi1_X + k**2 * xi1))
        derivs.append(d4_xi0_X + d2_xi0_X + 
            k**2 * (d2_xi0_X + xi0))

        v = 0
        for l in range(len(derivs)):
            v += derivs[l] / math.factorial(l) * Y1**l

        return v

    def extend_basis(self, J, index):
        ext = np.zeros(len(self.gamma), dtype=complex)
        for l in range(len(self.gamma)):
            i, j = self.gamma[l]
            r, th = self.get_polar(i, j)

        return ext

    def extend_boundary(self, nodes=None): 
        if nodes is None:
            nodes = self.gamma

        R = self.R
        k = self.problem.k

        boundary = np.zeros(len(nodes), dtype=complex)
        error= []

        for l in range(len(nodes)):
            i, j = nodes[l]
            x, y = self.get_coord(i, j)
            r, th = self.get_polar(i, j)
            etype = self.get_etype(i, j)

            xi0 = xi1 = 0
            d2_xi0_arg = d2_xi1_arg = 0
            d4_xi0_arg = 0

            for JJ in range(len(B_desc)):
                if etype == EXTEND_CIRCLE:
                    param_r = R
                    param_th = th
                elif etype == EXTEND_RADIUS1:
                    param_r = x
                    param_th = 0
                elif etype == EXTEND_RADIUS2:
                    x0, y0 = self.get_radius_point(2, x, y)
                    param_r = cart_to_polar(x0, y0)[0]
                    param_th = self.a

                B = self.eval_dn_B_arg(0, JJ,
                    param_r, param_th)
                d2_B_arg = self.eval_dn_B_arg(2, JJ,
                    param_r, param_th)
                d4_B_arg = self.eval_dn_B_arg(4, JJ,
                    param_r, param_th)

                xi0 += self.c0[JJ] * B
                xi1 += self.c1[JJ] * B

                d2_xi0_arg += self.c0[JJ] * d2_B_arg
                d2_xi1_arg += self.c1[JJ] * d2_B_arg

                d4_xi0_arg += self.c0[JJ] * d4_B_arg

            if etype == EXTEND_CIRCLE:
                boundary[l] = extend_circle(R, k, r, th, xi0, xi1,
                    d2_xi0_arg, d2_xi1_arg, d4_xi0_arg)

            elif (etype == EXTEND_RADIUS1 or etype == EXTEND_RADIUS2):
                if etype == EXTEND_RADIUS1:
                    Y1 = y
                else:
                    Y1 = self.dist_to_radius(2, x, y)

                    if self.is_interior(i, j):
                        Y1 = -Y1

                boundary[l] = self.extend_from_radius(Y1,
                    xi0, xi1, d2_xi0_arg, d2_xi1_arg, d4_xi0_arg)

            #boundary[l] += self.extend_inhomogeneous(r, th)

        return boundary

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
        self.gen_fake_gamma()
        return self.extension_test({
            EXTEND_RADIUS1, EXTEND_RADIUS2,
            EXTEND_CIRCLE})

    def calc_c1_exact(self):
        t_data = get_chebyshev_roots(1000)
        self.c1 = []

        for sid in range(N_SEGMENT):
            arg_data = [self.eval_g(sid, t) for t in t_data] 
            boundary_points = self.get_boundary_sample_by_sid(sid, arg_data)
            boundary_data = np.zeros(len(arg_data), dtype=complex)
            for l in range(len(boundary_points)):
                p = boundary_points[l]
                boundary_data[l] = self.problem.eval_d_u_outwards(
                    p['x'], p['y'], sid=sid)

            n_basis = segment_desc[sid]['n_basis']
            self.c1.extend(np.polynomial.chebyshev.chebfit(
                t_data, boundary_data, n_basis-1))

    def gen_fake_gamma(self):
        self.gamma = []
        h = self.AD_len / self.N
        a = self.a
        
        for r in np.arange(.1, self.R, .2):
            self.gamma.append(self.get_coord_inv(r, h/5))

            x = r*cos(a) + h*sin(a)
            y = r*sin(a) + h*cos(a)
            self.gamma.append(self.get_coord_inv(x, y))


    def gamma_filter(self, etypes):
        result = []
        for i, j in self.gamma:
            if self.get_etype(i, j) in etypes:
                result.append((i, j))

        return result

    def extension_test(self, etypes=ALL_ETYPES):
        self.calc_c0()
        self.calc_c1_exact()

        nodes = self.gamma_filter(etypes)
        ext = self.extend_boundary(nodes)

        error = np.zeros(len(nodes), dtype=complex)
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
                    (self.c0[JJ] *
                    self.eval_dn_B_arg(0, JJ, r, th)).real

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
            exact_data[l] = self.problem.eval_d_u_outwards(p['x'], p['y']).real

            r, th = cart_to_polar(p['x'], p['y'])
            for JJ in range(len(B_desc)):
                expansion_data[l] +=\
                    (self.c1[JJ] *
                    self.eval_dn_B_arg(0, JJ, r, th)).real

        plt.plot(s_data, exact_data, label='Exact')
        plt.plot(s_data, expansion_data, 'o', label='Expansion')
        plt.legend(loc=4)
        plt.ylim(-1.5, 1.5)
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

    def plot_gamma(self, plot_rpoints=False):
        self.plot_Gamma()

        for etype in ALL_ETYPES:
            nodes = self.gamma_filter({etype})
            x_data, y_data = self.nodes_to_plottable(nodes)
            plt.plot(x_data, y_data, 'o', label=ETYPE_NAMES[etype])

            if plot_rpoints and etype == EXTEND_RADIUS2:
                x_data = np.zeros(len(nodes))
                y_data = np.zeros(len(nodes))

                for l in range(len(nodes)):
                    x, y = self.get_coord(*nodes[l])
                    x1, y1 = self.get_radius_point(2, x, y)
                    x_data[l] = x1
                    y_data[l] = y1

                plt.plot(x_data, y_data, 'o', label='rpoints')

        plt.xlim(-4,4)
        plt.ylim(-4,4)
        plt.legend(loc=3)
        plt.show()
