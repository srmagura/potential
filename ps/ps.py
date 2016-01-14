import sys
import math
import numpy as np
import scipy
from scipy.special import jv

from solver import Solver, Result, cart_to_polar
from linop import apply_B
from apsolver import APSolver

import norms.linalg
import norms.sobolev
import norms.l2
import norms.weight_func

from ps.basis import PsBasis
from ps.grid import PsGrid
from ps.extend import PsExtend
from ps.inhomo import PsInhomo
from ps.debug import PsDebug

from problems.singular import RegularizeBc

def get_M(scheme_order):
    if scheme_order == 2:
        return 3
    elif scheme_order == 4:
        return 7

def print_a_coef(a_coef):
    if np.max(np.abs(scipy.imag(a_coef))) == 0:
        a_coef_to_print = scipy.real(a_coef)
    else:
        a_coef_to_print = a_coef

    np.set_printoptions(precision=15)

    print('a coefficients:')
    print(a_coef_to_print)


class PizzaSolver(Solver, PsBasis, PsGrid, PsExtend, PsInhomo, PsDebug):


    def __init__(self, problem, N, options={}, **kwargs):
        self.a = problem.a
        self.nu = problem.nu
        self.R = problem.R
        super().__init__(problem, N, options, **kwargs)

        self.norm = options['norm']
        self.var_method = options['var_method']

        self.var_compute_a = options['var_compute_a']
        problem.var_compute_a = self.var_compute_a

        self.do_dual = options['do_dual']

        if(self.var_compute_a): # or self.var_method in fft_test_var_methods):
            problem.regularize_bc = RegularizeBc.none
        elif self.do_dual:
            problem.regularize_bc = RegularizeBc.known

        self.var_compute_a_only = options.get('var_compute_a_only', False)

        self.scheme_order = options['scheme_order']
        self.extension_order = self.scheme_order + 1

        self.M = options.get('M', get_M(self.scheme_order))
        self.problem.M = self.M
        self.problem.setup()

        self.m_list = options.get('m_list', range(1, problem.M+1))

        # For optimizing basis set sizes
        if 'n_circle' in options:
            assert 'n_radius' in options
            self.setup_basis(options['n_circle'], options['n_radius'])

        self.ps_construct_grids(self.scheme_order)

        apsolver = APSolver(self.N, self.scheme_order, self.AD_len, self.k)
        self.apply_L = lambda v: apsolver.apply_L(v)
        self.solve_ap = lambda Bf: apsolver.solve(Bf)

        self.calc_ap_sol_f()

    def calc_ap_sol_f(self):
        N = self.N
        AD_len = self.AD_len
        k = self.k

        self.setup_f()
        Bf = apply_B(self.f)

        for i,j in self.global_Mminus:
            Bf[i-1,j-1] = 0

        self.ap_sol_f = self.solve_ap(Bf)

    def get_sid(self, th):
        return self.problem.get_sid(th)

    def is_interior(self, i, j):
        r, th = self.get_polar(i, j)
        return r <= self.R and th >= self.a

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

    def get_potential(self, ext):
        """
        Compute the difference potential of `ext`.

        Output:
        potential -- shape (N-1, N-1)
        """
        N = self.N
        args = (self.scheme_order, self.problem.AD_len, self.k)
        w = np.zeros((N-1, N-1), dtype=complex)

        for l in range(len(self.union_gamma)):
            i, j = self.union_gamma[l]
            w[i-1, j-1] = ext[l]

        Lw = self.apply_L(w)

        for i,j in self.global_Mminus:
            Lw[i-1, j-1] = 0

        return w - self.solve_ap(Lw)

    def get_trace(self, w):
        trace = np.zeros(len(self.union_gamma), dtype=complex)

        for l in range(len(self.union_gamma)):
            i, j = self.union_gamma[l]
            trace[l] = w[i-1, j-1]

        return trace

    def get_Q(self, index, sid):
        """
        Get one of the three submatrices that are used in the construction of
        V.
        """
        columns = []

        for JJ in self.segment_desc[sid]['JJ_list']:
            ext = self.extend_basis(JJ, index)
            potential = self.get_potential(ext)
            projection = self.get_trace(potential)

            columns.append(projection - ext)

        return np.column_stack(columns)

    def get_d_matrix(self):
        """
        Create matrix whose columns contain the Chebyshev coefficients
        for the functions sin(m*nu*(th-a)) for m=1...M.
        """
        k = self.k
        R = self.R
        a = self.a
        nu = self.nu

        d_columns = []

        for m in self.m_list:
            def func(th):
                return np.sin(m*nu*(th-a))

            d = self.get_chebyshev_coef(0, func)
            d = np.matrix(d.reshape((len(d), 1)))
            d_columns.append(jv(m*nu, k*R) * d)

        return np.column_stack(d_columns)

    def get_fbterm_matrix(self):
        r"""
        Create matrix with M columns, where each column is

            (P_\gamma - I) Tr(\bar{w}_m),

        m=1,...,M, where

            \bar{w}_m = J_{m\nu}(kr) \sin(m\nu(\theta-\alpha))

        and Tr means the trace on the discrete boundary union_gamma.
        """
        k = self.k
        a = self.a
        nu = self.nu

        columns = []

        def eval_fbterm(m, r, th):
            if th < a/2:
                th += 2*np.pi

            return jv(m*nu, k*r) * np.sin(m*nu*(th-a))

        for i in range(len(self.m_list)):
            m = self.m_list[i]
            trace = np.zeros(len(self.union_gamma), dtype=complex)

            for l in range(len(self.union_gamma)):
                node = self.union_gamma[l]
                r, th = self.get_polar(*node)
                trace[l] = eval_fbterm(m, r, th)

            potential = self.get_potential(trace)
            projection = self.get_trace(potential)

            columns.append(projection - trace)

        return np.column_stack(columns)

    def get_var(self):
        """
        Get the matrices V0 and V1, and the RHS of the variational
        formulation.

        V0, V1, and RHS are the main components of the variational formulation,
        an overdetermined linear system used to solve for the unknown Neumann
        coefficients. V0 operates on the (known) Dirichlet Chebyshev
        coefficients, while V1 operates on the vector of unknowns (Neumann
        Chebyshev coefficients and Fourier b coefficients).

        index -- 0 or 1
        """

        # Put the Q-submatrices in a 2x3 multidimensional list. The first row
        # corresponds to V0, while the second row corresponds to V1.
        submatrices = []
        for index in (0, 1):
            row = []
            for sid in range(self.N_SEGMENT):
                row.append(self.get_Q(index, sid))

            submatrices.append(row)

        rhs = np.zeros(len(self.union_gamma), dtype=complex)

        if self.problem.var_compute_a:
            if self.var_method == 'fbterm':
                submatrices[1].append(-self.get_fbterm_matrix())

            elif self.var_method == 'chebsine':
                submatrices[1].append(-submatrices[0][0].dot(
                    self.get_d_matrix()))

        elif self.var_method == 'fft-test-fbterm':
            rhs = self.get_fbterm_matrix().dot(
                self.problem.fft_a_coef[:self.M])

        elif self.var_method == 'fft-test-chebsine':
            Q0_d = submatrices[0][0].dot(self.get_d_matrix())
            Q0_d_a = Q0_d.dot(self.problem.fft_a_coef[:self.M])
            Q0_d_a = np.ravel(Q0_d_a)
            rhs += Q0_d_a

        V0 = np.concatenate(submatrices[0], axis=1)
        V1 = np.concatenate(submatrices[1], axis=1)

        # Compute the RHS of the variational formulation.
        ext_f = self.extend_inhomo_f()
        proj_f = self.get_potential(ext_f)

        term = self.get_trace(proj_f + self.ap_sol_f)
        rhs += -V0.dot(self.c0) + ext_f - term

        return (V0, V1, rhs)

    def solve_var(self):
        """
        Setup and solve the variational formulation.
        """
        V0, V1, rhs = self.get_var()

        if self.norm == 'l2':
            varsol = np.linalg.lstsq(V1, rhs)[0]
        else:
            h = self.AD_len / self.N

            if self.norm == 'sobolev':
                sa = 0.5
                ip_array = norms.sobolev.get_ip_array(h, self.union_gamma, sa)

            elif self.norm == 'l2-wf1':
                radii = [self.get_polar(*node)[0] for node in self.union_gamma]
                ip_array = norms.l2.get_ip_array__radial(h, radii,
                    norms.weight_func.wf1)


            varsol = norms.linalg.solve_var(V1, rhs, ip_array)

        self.handle_varsol(varsol)

    def handle_varsol(self, varsol):
        """
        Break up the solution vector of the variational formulation into
        its meaningful parts.

        Set c1. Update the boundary data if necessary.
        """
        self.c1 = varsol[:len(self.c0)]

        if self.var_compute_a:
            var_a = varsol[len(self.c0):]

            #if self.var_method in fft_test_var_methods:
            #    a_coef = self.problem.fft_a_coef[:self.M]
            #else:
            a_coef = np.zeros(self.M, dtype=complex)

            for i in range(len(self.m_list)):
                m = self.m_list[i]
                a_coef[m-1] = var_a[i]

            if self.var_compute_a_only:
                self.a_coef = a_coef
                return

            self.problem.a_coef = a_coef

            if self.problem.regularize_bc:
                self.update_c0()

    def run(self):
        """
        The main procedure for PizzaSolver.
        """

        '''
        Debugging function for choosing an appropriate number of basis
        functions.
        '''
        if not hasattr(self, 'B_desc'):
            n_basis_tuple = self.problem.get_n_basis(N=self.N,
                scheme_order=self.scheme_order)
            self.setup_basis(*n_basis_tuple)

        #self.plot_gamma()

        self.calc_c0()

        '''
        Uncomment the following line to see a plot of the boundary data and
        its Chebyshev series, which has coefficients c0. There is also an
        analogous function c1_test().
        '''
        #self.c0_test(plot=False)

        self.solve_var()

        if self.var_compute_a_only:
            return self.a_coef

        #self.calc_c1_exact()
        #self.c1_test()

        ext = self.extend_boundary()
        potential = self.get_potential(ext) + self.ap_sol_f
        u_act = potential

        for i, j in self.M0:
            r, th = self.get_polar(i, j)
            u_act[i-1, j-1] += self.problem.get_restore_polar(r, th)

        error = self.eval_error(u_act)

        result = Result()
        result.error = error
        result.u_act = u_act

        if self.problem.var_compute_a or self.do_dual:
            result.a_error = self.problem.get_a_error()

        return result
