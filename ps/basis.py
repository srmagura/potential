import numpy as np

from chebyshev import eval_dn_T_t

class PsBasis:

    n_basis_by_sid = (60, 30, 30)
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
