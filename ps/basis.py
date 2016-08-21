import numpy as np
import scipy
from scipy.special import jv

from chebyshev import eval_dn_T_t, get_chebyshev_roots
import abcoef
import domain_util

from problems.singular import HReg

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

    def eval_dn_B_arg(self, n, JJ, arg, sid):
        if self.B_desc[JJ]['sid'] == sid:
            t = self.eval_g_inv(sid, arg)

            J = self.B_desc[JJ]['J']
            d_g_inv_arg = self.eval_d_g_inv_arg(sid)

            return (d_g_inv_arg)**n * eval_dn_T_t(n, J, t)
        else:
            return 0

    def calc_a_coef(self):
        """
        Calculate the coefficients necessary to remove the homogeneous
        part of the singularity, using one of several metods.
        """
        hreg = self.problem.hreg
        eval_bc0 = None

        if hreg == HReg.acheat:
            if hasattr(self.problem, 'fft_a_coef'):
                self.a_coef = self.problem.fft_a_coef[:self.M]
            else:
                def eval_bc0(th):
                    r = self.boundary.eval_r(th)
                    return self.problem.eval_bc__noreg(th, 0) - self.problem.eval_v(r, th)

        elif hreg == HReg.linsys:

            def eval_bc0(th):
                return self.problem.eval_bc(th, 0)

        else:
            assert hreg == HReg.none
            self.a_coef = np.zeros(self.M)


        if eval_bc0 is not None:
            m1 = self.problem.get_m1()
            self.a_coef = abcoef.calc_a_coef(self.problem, self.boundary,
                eval_bc0, self.M, m1)[0]
            self.problem.a_coef = self.a_coef

            if(self.boundary.name == 'arc' and
                hasattr(self.problem, 'fft_a_coef')):
                error = np.max(np.abs(self.a_coef - self.problem.fft_a_coef[:self.M]))
                print('a_coef error:', error)

            np.set_printoptions(precision=15)
            print()
            print('a_coef:')
            print(scipy.real(self.a_coef))
            print()


    def get_chebyshev_coef(self, sid, func, t_data=None):
        """
        Compute the Chebyshev expansion of a function defined on one of
        the segments of the boundary.

        The function is sampled at the roots of a very high degree Chebyshev
        polynomial, and then the NumPy function chebfit() is used.

        sid -- segment ID for the boundary segment
        func -- the function to be approximated
        """
        if t_data is None:
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

        k = self.problem.k
        R = self.problem.R
        a = self.a
        nu = self.nu

        self.calc_a_coef()

        def eval_bc0_reg(th):
            bc0 = self.problem.eval_bc(th, 0)
            r = self.boundary.eval_r(th)

            if self.problem.hreg != HReg.none:
                for m in range(1, self.M+1):
                    bc0 -= self.a_coef[m-1] * jv(m*nu, k*r) * np.sin(m*nu*(th-a))

            return bc0

        self.c0.extend(self.get_chebyshev_coef(0, eval_bc0_reg))

        for sid in (1, 2):
            def func(arg):
                return self.problem.eval_bc(arg, sid)

            self.c0.extend(self.get_chebyshev_coef(sid, func))

     #def update_c0(self):
        """
        Update c0 to reflect the sines that we are subtracting
        for regularization
        """
        """def eval_regularized_bc(th):
            k = self.k
            R = self.R
            a = self.a
            nu = self.nu

            a_coef = self.problem.a_coef

            bc = self.problem.eval_bc_extended(th, 0)
            #for i in range(len(self.m_list)):
            for m in range(1, self.M+1):
                bc -= a_coef[m-1] * jv(m*nu, k*R) * np.sin(m*nu*(th-a))

            return bc

        self.c0[:self.segment_desc[0]['n_basis']] =\
            self.get_chebyshev_coef(0, eval_regularized_bc)"""

    def extend_basis(self, JJ, index):
        """
        Extend a basis function to the discrete boundary.
        """
        return self.extend_boundary(JJ=JJ, basis_index=index)
