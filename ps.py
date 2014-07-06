import math
import numpy as np

from solver import Solver, cart_to_polar
import matrices

import matplotlib.pyplot as plt

from chebyshev import *

EXTEND_CIRCLE = 0
EXTEND_RADIUS1 = 1
EXTEND_RADIUS2 = 2
EXTEND_OUTER1 = 3
EXTEND_OUTER2 = 4
EXTEND_INNER = 5

ETYPE_NAMES = ('circle', 'radius1', 'radius2', 
    'outer1', 'outer2', 'inner')
ALL_ETYPES = range(len(ETYPE_NAMES))

TAYLOR_N_DERIVS = 6

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
        return r <= self.R and (th >= self.a or th == 0) 

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

    def eval_dn_B_arg(self, n, JJ, r, th, sid=None):
        if sid is None:
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
            x, y = self.get_coord(i, j)
            etype = self.get_etype(i, j)

            param_r = None

            if etype == EXTEND_CIRCLE:
                param_r = self.R
                param_th = th
            elif etype == EXTEND_RADIUS1:
                param_r = x
                param_th = 0 
            elif etype == EXTEND_RADIUS2:
                x0, y0 = self.get_radius_point(2, x, y)
                param_r = cart_to_polar(x0, y0)[0]
                param_th = self.a

            if param_r is not None:
                B_args = (JJ, param_r, param_th)
                B = self.eval_dn_B_arg(0, *B_args)
                d2_B_arg = self.eval_dn_B_arg(2, *B_args)
                d4_B_arg = self.eval_dn_B_arg(4, *B_args)

                args = (
                    (B, 0, d2_B_arg, 0, d4_B_arg),
                    (0, B, 0, d2_B_arg, 0)
                )

            if etype == EXTEND_CIRCLE:
                ext[l] = self.extend_circle(r, *args[index])

            elif etype == EXTEND_RADIUS1:
                ext[l] = self.extend_from_radius(y, *args[index])

            elif etype == EXTEND_RADIUS2:
                Y = self.signed_dist_to_radius(2, x, y)
                ext[l] = self.extend_from_radius(Y, *args[index])

            elif etype == EXTEND_OUTER1:
                ext[l] = self.do_extend_outer(i, j, 1, JJ, index) 

            elif etype == EXTEND_OUTER2:
                ext[l] = self.do_extend_outer(i, j, 2, JJ, index) 

        return ext

    def get_radius_point(self, sid, x, y):
        assert sid == 2

        m = np.tan(self.a)
        x1 = (x/m + y)/(m + 1/m)
        y1 = m*x1

        return (x1, y1) 

    def dist_to_radius(self, sid, x, y):
        assert sid == 2
        x1, y1 = self.get_radius_point(sid, x, y)
        return np.sqrt((x1-x)**2 + (y1-y)**2)

    def signed_dist_to_radius(self, sid, x, y):
        unsigned = self.dist_to_radius(sid, x, y)

        m = np.tan(self.a)
        if y > m*x:
            return -unsigned
        else:
            return unsigned

    def get_etype(self, i, j):
        x, y = self.get_coord(i, j)
        r, th = self.get_polar(i, j)

        dist0 = abs(self.R - r)
        dist1 = abs(y)
        dist2 = self.dist_to_radius(2, x, y)

        if r > self.R:
            if th < self.a/2:
                return EXTEND_OUTER1
            elif th < self.a:
                return EXTEND_OUTER2
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

    def extend_from_radius(self, Y, xi0, xi1, d2_xi0_X,
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
            v += derivs[l] / math.factorial(l) * Y**l

        return v

    def ext_calc_certain_xi_derivs(self, i, j, param_r, param_th, sid=None):
        xi0 = xi1 = 0
        d2_xi0_arg = d2_xi1_arg = 0
        d4_xi0_arg = 0

        for JJ in range(len(B_desc)):
            B = self.eval_dn_B_arg(0, JJ, param_r, param_th, sid)
            d2_B_arg = self.eval_dn_B_arg(2, JJ, param_r, param_th, sid)
            d4_B_arg = self.eval_dn_B_arg(4, JJ, param_r, param_th, sid)

            xi0 += self.c0[JJ] * B
            xi1 += self.c1[JJ] * B

            d2_xi0_arg += self.c0[JJ] * d2_B_arg
            d2_xi1_arg += self.c1[JJ] * d2_B_arg

            d4_xi0_arg += self.c0[JJ] * d4_B_arg

        return (xi0, xi1, d2_xi0_arg, d2_xi1_arg, d4_xi0_arg)

    def do_extend_circle(self, i, j):
        r, th = self.get_polar(i, j)
        derivs = self.ext_calc_certain_xi_derivs(i, j, self.R, th)
        return self.extend_circle(r, *derivs)

    def do_extend_radius1(self, i, j):
        x, y = self.get_coord(i, j)
        derivs = self.ext_calc_certain_xi_derivs(i, j, x, 0)

        return self.extend_from_radius(y, *derivs)

    def do_extend_radius2(self, i, j):
        x, y = self.get_coord(i, j)
        x0, y0 = self.get_radius_point(2, x, y)

        param_r = cart_to_polar(x0, y0)[0]
        derivs = self.ext_calc_certain_xi_derivs(i, j, param_r, self.a)

        Y = self.signed_dist_to_radius(2, x, y)
        return self.extend_from_radius(Y, *derivs)

    def ext_calc_B_derivs(self, JJ, param_th, segment_sid, index):
        derivs = np.zeros(TAYLOR_N_DERIVS, dtype=complex)

        for n in range(TAYLOR_N_DERIVS):
            derivs[n] = self.eval_dn_B_arg(n, JJ, self.R, param_th, segment_sid)

        return derivs

    def ext_calc_xi_derivs(self, param_th, segment_sid, index): 
        derivs = np.zeros(TAYLOR_N_DERIVS, dtype=complex)

        if index == 0:
            c = self.c0
        elif index == 1:
            c = self.c1

        for n in range(TAYLOR_N_DERIVS):
            for JJ in range(len(B_desc)):
                dn_B_arg = self.eval_dn_B_arg(n, JJ, self.R, param_th, segment_sid)
                derivs[n] += c[JJ] * dn_B_arg

        return derivs

    def do_extend_outer(self, i, j, radius_sid, JJ=None, index=None):
        x, y = self.get_coord(i, j)
        r, th = self.get_polar(i, j)

        dist0 = r - self.R

        if radius_sid == 1:
            dist1 = y
        elif radius_sid == 2:
            dist1 = self.dist_to_radius(radius_sid, x, y)

        if dist0 < dist1:
            taylor_sid = 0

            if radius_sid == 1:
                delta_arg = th
                param_th = 2*np.pi
            elif radius_sid == 2:
                delta_arg = th - self.a
                param_th = self.a
        else:
            taylor_sid = radius_sid

            if radius_sid == 1:
                delta_arg = x - self.R
                param_th = 0
                Y = y
            elif radius_sid == 2:
                x0 = self.R * cos(self.a)
                y0 = self.R * sin(self.a)
                x1, y1 = self.get_radius_point(radius_sid, x, y)
                delta_arg = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
                param_th = self.a
                Y = dist1

        #print('taylor_sid = {}    radius_sid = {}'.format(taylor_sid,radius_sid))

        derivs0 = np.zeros(TAYLOR_N_DERIVS)
        derivs1 = np.zeros(TAYLOR_N_DERIVS)

        if JJ is None:
            derivs0 = self.ext_calc_xi_derivs(param_th, taylor_sid, 0) 
            derivs1 = self.ext_calc_xi_derivs(param_th, taylor_sid, 1) 
        elif index == 0:
            derivs0 = self.ext_calc_B_derivs(JJ, param_th, taylor_sid, 0) 
        elif index == 1:
            derivs1 = self.ext_calc_B_derivs(JJ, param_th, taylor_sid, 1) 

        xi0 = xi1 = 0
        d2_xi0_arg = d2_xi1_arg = 0
        d4_xi0_arg = 0

        for l in range(len(derivs0)):
            f = delta_arg**l / math.factorial(l)
            xi0 += derivs0[l] * f 
            xi1 += derivs1[l] * f

            ll = l - 2
            if ll >= 0:
                f = delta_arg**ll / math.factorial(ll)
                d2_xi0_arg += derivs0[l] * f
                d2_xi1_arg += derivs1[l] * f

            ll = l - 4
            if ll >= 0:
                f = delta_arg**ll / math.factorial(ll)
                d4_xi0_arg += derivs0[l] * f

        ext_params = (xi0, xi1, d2_xi0_arg, d2_xi1_arg, d4_xi0_arg)

        if taylor_sid == 0:
            v = self.extend_circle(r, *ext_params)
        elif taylor_sid in {1, 2}:
            v = self.extend_from_radius(Y, *ext_params)

        return v

    def extend_boundary(self, nodes=None): 
        if nodes is None:
            nodes = self.gamma

        R = self.R
        k = self.problem.k

        boundary = np.zeros(len(nodes), dtype=complex)

        for l in range(len(nodes)):
            i, j = nodes[l]
            x, y = self.get_coord(i, j)
            r, th = self.get_polar(i, j)
            etype = self.get_etype(i, j)

            if etype == EXTEND_CIRCLE:
                boundary[l] = self.do_extend_circle(i, j)

            elif etype == EXTEND_RADIUS1:
                boundary[l] = self.do_extend_radius1(i, j)

            elif etype == EXTEND_RADIUS2:
                boundary[l] = self.do_extend_radius2(i, j)

            elif etype == EXTEND_OUTER1:
                boundary[l] = self.do_extend_outer(i, j, 1)

            elif etype == EXTEND_OUTER2:
                boundary[l] = self.do_extend_outer(i, j, 2)

        return boundary

    def extend_inhomogeneous_f(self, *args):
        ext = np.zeros(len(self.gamma), dtype=complex)
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
        th_data = np.arange(self.a, 2*np.pi, .1)
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
        self.calc_c0()
        self.calc_c1()

        ext = self.extend_boundary()
        u_act = self.get_potential(ext) #+ self.ap_sol_f

        error = self.eval_error(u_act)
        return error


    ## DEBUGGING FUNCTIONS ##

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

    def test_extend_basis(self):
        error = []
        for index in (0, 1):
            for JJ in range(len(B_desc)):
                ext1 = self.extend_basis(JJ, index)

                self.c0 = np.zeros(len(B_desc))
                self.c1 = np.zeros(len(B_desc))

                if index == 0:
                    self.c0[JJ] = 1
                elif index == 1:
                    self.c1[JJ] = 1

                ext2 = self.extend_boundary()
                error.append(np.max(np.abs(ext1-ext2)))
                print('index={}  JJ={}  error={}'.format(index, JJ, error[-1]))

        error = np.array(error)
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
        plt.legend(loc=0)
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

    def gen_fake_gamma(self):
        def ap():
            self.gamma.append(self.get_coord_inv(x, y))

        self.gamma = []
        h = self.AD_len / self.N
        a = self.a
        
        for r in np.arange(.1, self.R, .2):
            x = r
            y = h/5
            ap()

            y = -h/5
            ap()

            x = r*cos(a) + h*sin(a)
            y = r*sin(a) + h*cos(a)
            ap()

            x = r*cos(a) - h*sin(a)/5
            y = r*sin(a) - h*cos(a)/5
            ap()

        for th in np.arange(self.a+.001, 2*np.pi, .1):
            r = self.R + h/2
            x = r*cos(th)
            y = r*sin(th)
            ap()

            r = self.R - h/2
            x = r*cos(th)
            y = r*sin(th)
            ap()
        
        for t in (True, False):
            r = self.R + h
            b = np.pi*h/10
            if t:
                b = self.a - b 
            x = r*cos(b)
            y = r*sin(b)
            ap()

            r = self.R + h
            b = np.pi*h/5
            if t:
                b = self.a - b 
            x = r*cos(b)
            y = r*sin(b)
            ap()

