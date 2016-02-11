import numpy as np
from scipy.fftpack import dst

from chebyshev import eval_dn_T_t, get_chebyshev_roots
import fourier
import domain_util

class PsBasis:
    """
    Functionality related to the basis functions defined on the boundary.
    """

    N_SEGMENT = 3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Used for computing Chebyshev expansions
        self.chebyshev_roots = get_chebyshev_roots(1024)

    def setup_basis(self, n_circle, n_radius):
        """
        Create data structures that describe the basis functions defined
        on the boundary.
        """
        self.segment_desc = []
        self.B_desc = []

        n_basis_by_sid = (n_circle, n_radius, n_radius)

        JJ = 0
        for sid in range(self.N_SEGMENT):
            s_desc = {
                'n_basis': n_basis_by_sid[sid],
                'JJ_list': [],
            }
            self.segment_desc.append(s_desc)

            for J in range(s_desc['n_basis']):
                b_desc = {}
                b_desc['J'] = J
                b_desc['sid'] = sid
                self.B_desc.append(b_desc)

                s_desc['JJ_list'].append(JJ)
                JJ += 1

    def get_span_center(self, sid):
        eps = 1e-3

        if sid == 0:
            span = np.pi - .5*self.a + eps
            center = np.pi + .5*self.a
        elif sid == 1 or sid == 2:
            span = self.R/2 + eps
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

    def eval_dn_B_arg(self, n, JJ, r, th, sid):
        # TODO: change (r, th, sid) to (arg, sid)?
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

    def get_chebyshev_coef(self, sid, func):
        """
        Compute the Chebyshev expansion of a function defined on one of
        the segments of the boundary.

        The function is sampled at the roots of a very high degree Chebyshev
        polynomial, and then the NumPy function chebfit() is used.

        sid -- segment ID for the boundary segment
        func -- the function to be approximated
        """
        t_data = self.chebyshev_roots
        boundary_data = np.zeros(len(t_data), dtype=complex)

        for i in range(len(t_data)):
            arg = self.eval_g(sid, t_data[i])
            boundary_data[i] = func(arg)

        n_basis = self.segment_desc[sid]['n_basis']
        return np.polynomial.chebyshev.chebfit(t_data, boundary_data,
            n_basis-1)


    def calc_c0(self):
        """
        Do Chebyshev fits on the Dirichlet data of each segment to get
        the coefficients c0.
        """
        self.c0 = []

        a = self.a
        nu = self.nu

        b_coef = fourier.arc_dst(a,
            lambda th: self.problem.eval_bc(th, 0)
        )

        regularized_c0 = True

        def eval_phi0_reg(th):
            phi0 = self.problem.eval_bc(th, 0)

            if regularized_c0:
                for m in range(1, self.M+1):
                    phi0 -= b_coef[m-1] * np.sin(m*nu*(th-a))

            return phi0

        if regularized_c0:
            print('regularized-c0')
        else:
            print('full-c0')

        self.c0.extend(self.get_chebyshev_coef(0, eval_phi0_reg))

        for sid in (1, 2):
            def func(arg):
                return self.problem.eval_bc(arg, sid)

            self.c0.extend(self.get_chebyshev_coef(sid, func))

    def extend_basis(self, JJ, index):
        """
        Extend a basis function to the discrete boundary.
        """
        return self.extend_boundary({'JJ': JJ, 'index': index})
