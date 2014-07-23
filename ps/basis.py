import numpy as np

from solver import cart_to_polar
from chebyshev import eval_dn_T_t, get_chebyshev_roots

class PsBasis:

    n_basis_by_sid = (40, 20, 20)
    N_SEGMENT = 3

    def __init__(self):
        super().__init__()

        self.segment_desc = []
        self.B_desc = []

        for sid in range(self.N_SEGMENT):
            s_desc = {'n_basis': self.n_basis_by_sid[sid]}
            self.segment_desc.append(s_desc)

            for J in range(s_desc['n_basis']):
                b_desc = {} 
                b_desc['J'] = J
                b_desc['sid'] = sid
                self.B_desc.append(b_desc)

    def get_span_center(self, sid):
        if sid == 0:
            span = np.pi - np.pi/24      # 23/24 pi
            center = np.pi + .5*self.a   # 26/24 pi
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

        desc = self.B_desc[JJ]

        if desc['sid'] == sid:
            if sid == 0:
                arg = th
            elif sid == 1 or sid == 2:
                arg = r

            t = self.eval_g_inv(sid, arg)

            J = self.B_desc[JJ]['J']
            d_g_inv_arg = self.eval_d_g_inv_arg(sid)

            return (d_g_inv_arg)**n * eval_dn_T_t(n, J, t)
        else:
            return 0

    def calc_c0(self):
        t_data = get_chebyshev_roots(1000)
        self.c0 = []

        for sid in range(self.N_SEGMENT):
            arg_data = [self.eval_g(sid, t) for t in t_data] 
            boundary_points = self.get_boundary_sample_by_sid(sid, arg_data)
            boundary_data = np.zeros(len(arg_data), dtype=complex)
            for l in range(len(boundary_points)):
                p = boundary_points[l]
                boundary_data[l] = self.problem.eval_bc_extended(p['x'], p['y'], sid)

            n_basis = self.segment_desc[sid]['n_basis']
            self.c0.extend(np.polynomial.chebyshev.chebfit(
                t_data, boundary_data, n_basis-1))

    def get_boundary_sample(self):
        th_data = np.arange(self.a, 2*np.pi, .1)
        r_data = np.linspace(0, self.R, 25)

        points = []
        arg_datas = (th_data, r_data[::-1], r_data)

        for sid in range(self.N_SEGMENT):
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

    def extend_basis(self, JJ, index):
        ext = np.zeros(len(self.gamma), dtype=complex)
        for l in range(len(self.gamma)):
            i, j = self.gamma[l]
            r, th = self.get_polar(i, j)
            x, y = self.get_coord(i, j)
            etype = self.get_etype(i, j)

            param_r = None

            if etype == self.etypes['circle']:
                param_r = self.R
                param_th = th
            elif etype == self.etypes['radius1']:
                param_r = x
                param_th = 0 
            elif etype == self.etypes['radius2']:
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

            if etype == self.etypes['circle']:
                ext[l] = self.extend_circle(r, *args[index])

            elif etype == self.etypes['radius1']:
                ext[l] = self.extend_from_radius(y, *args[index])

            elif etype == self.etypes['radius2']:
                Y = self.signed_dist_to_radius(2, x, y)
                ext[l] = self.extend_from_radius(Y, *args[index])

            elif etype == self.etypes['outer1']:
                ext[l] = self.do_extend_outer(i, j, 1, JJ, index) 

            elif etype == self.etypes['outer2']:
                ext[l] = self.do_extend_outer(i, j, 2, JJ, index) 

        return ext
