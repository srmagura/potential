import numpy as np

from solver import cart_to_polar
from chebyshev import eval_dn_T_t, get_chebyshev_roots

class PsBasis:

    N_SEGMENT = 3

    def setup_B_desc(self, n_circle, n_radius):
        self.segment_desc = []
        self.B_desc = []
        self.sid_by_JJ = {}
        
        n_basis_by_sid = (n_circle, n_radius, n_radius)

        JJ = 0
        for sid in range(self.N_SEGMENT):
            s_desc = {'n_basis': n_basis_by_sid[sid]}
            self.segment_desc.append(s_desc)

            for J in range(s_desc['n_basis']):
                b_desc = {} 
                b_desc['J'] = J
                b_desc['sid'] = sid
                self.B_desc.append(b_desc)
                
                self.sid_by_JJ[JJ] = sid
                JJ += 1

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
        '''
        Do Chebyshev fits on the Dirichlet data of each segment to get
        the coefficients c0.
        '''
        t_data = get_chebyshev_roots(1000)
        self.c0 = []

        for sid in range(self.N_SEGMENT):
            boundary_data = np.zeros(len(t_data))
            
            for i in range(len(t_data)):
                arg = self.eval_g(sid, t_data[i])
                boundary_data[i] = self.problem.eval_bc_extended(arg, sid)

            n_basis = self.segment_desc[sid]['n_basis']
            self.c0.extend(np.polynomial.chebyshev.chebfit(
                t_data, boundary_data, n_basis-1))

    def extend_basis(self, JJ, index):
        return self.extend_boundary({'JJ': JJ, 'index': index})
