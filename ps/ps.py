import math
import numpy as np
import scipy
from scipy.special import jv

from solver import Solver, Result, cart_to_polar
import matrices
import norms.sobolev

from ps.basis import PsBasis
from ps.grid import PsGrid
from ps.extend import PsExtend
from ps.inhomo import PsInhomo
from ps.debug import PsDebug

norms = ('l2', 'sobolev')
var_methods = ('fbterm', 'chebsine')

class PizzaSolver(Solver, PsBasis, PsGrid, PsExtend, PsInhomo, PsDebug):

    M = 7

    def __init__(self, problem, N, options={}, **kwargs):
        self.a = problem.a
        self.nu = problem.nu

        self.R = problem.R
        self.AD_len = problem.AD_len

        self.norm = options.get('norm', 'l2')
        self.var_method = options.get('var_method', 'chebsine')

        self.var_compute_a = options.get('var_compute_a', False)
        problem.var_compute_a = self.var_compute_a

        self.m_list = options.get('m_list', range(1, problem.M+1))
        self.do_optimize = options.get('do_optimize', False)

        super().__init__(problem, N, options, **kwargs)

        self.ps_construct_grids()

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
        w = np.zeros([(self.N-1)**2], dtype=complex)

        for l in range(len(self.union_gamma)):
            w[matrices.get_index(self.N, *self.union_gamma[l])] = ext[l]

        Lw = np.ravel(self.L.dot(w))

        for i,j in self.global_Mminus:
            Lw[matrices.get_index(self.N, i, j)] = 0

        return w - self.LU_factorization.solve(Lw)

    def get_trace(self, w):
        trace = np.zeros(len(self.union_gamma), dtype=complex)

        for l in range(len(self.union_gamma)):
            i, j = self.union_gamma[l]
            index = matrices.get_index(self.N, i, j)
            trace[l] = w[index]

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

    def get_V(self):
        """
        Get the matrices V0 and V1.

        These matrices are the main components of the variational formulation,
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

        if self.problem.var_compute_a:
            if self.var_method == 'fbterm':
                submatrices[1].append(self.get_fbterm_matrix())
            elif self.var_method == 'chebsine':
                submatrices[1].append(-submatrices[0][0].dot(
                    self.get_d_matrix()))

        V0 = np.concatenate(submatrices[0], axis=1)
        V1 = np.concatenate(submatrices[1], axis=1)
        return (V0, V1)

    def get_var_rhs(self, V0):
        """
        Get the RHS of the variational formulation.
        """
        ext_f = self.extend_inhomo_f()
        proj_f = self.get_potential(ext_f)

        term = self.get_trace(proj_f + self.ap_sol_f)
        return -V0.dot(self.c0) + ext_f - term

    def solve_var(self):
        """
        Setup and solve the variational formulation.
        """
        V0, V1 = self.get_V()
        rhs = self.get_var_rhs(V0)

        if self.norm == 'l2':
            varsol = np.linalg.lstsq(V1, rhs)[0]
        elif self.norm == 'sobolev':
            h = self.AD_len / self.N
            sa = 0.5
            varsol = sobolev.solve_var(V1, rhs, h, self.union_gamma, sa)

        self.handle_varsol(varsol)

    def handle_varsol(self, varsol):
        """
        Break up the solution vector of the variational formulation into
        its meaningful parts.

        Set c1. Update the boundary data if necessary.
        """

        self.c1 = sol[:len(self.c0)]

        if self.problem.var_compute_a:
            var_a = sol[len(self.c0):]

            a_coef = np.zeros(self.M, dtype=complex)
            self.problem.a_coef = a_coef

            for i in range(len(self.m_list)):
                m = self.m_list[i]
                a_coef[m-1] = var_a[i]

            if np.max(np.abs(scipy.imag(a_coef))) == 0:
                a_coef_to_print = scipy.real(a_coef)
            else:
                a_coef_to_print = a_coef

            print('a coefficients:')
            print(a_coef_to_print)

            self.update_c0()

    # Set to True when debugging with test_extend_boundary() for
    # significant performance benefit.
    # FIXME
    # skip_matrix_build = False

    def run(self):
        """
        The main procedure for PizzaSolver.
        """
        self.ap_sol_f = self.LU_factorization.solve(self.B_src_f)

        '''
        Debugging function for choosing an appropriate number of basis
        functions.
        '''
        if self.do_optimize:
            self.optimize_n_basis()

        n_basis_tuple = self.problem.get_n_basis(self.N)
        self.setup_basis(*n_basis_tuple)

        #self.plot_gamma()
        #return self.ps_test_extend_src_f()

        '''
        Uncomment to run the extend basis test. Just run the program on a
        single grid --- don't run the convergence test.
        '''
        #return self.test_extend_basis()

        self.calc_c0()
        '''
        Uncomment the following line to see a plot of the boundary data and
        its Chebyshev series, which has coefficients c0. There is also an
        analogous function c1_test().
        '''
        #self.c0_test(plot=False)

        self.solve_var()
        #self.calc_c1_exact()
        #self.c1_test()

        ext = self.extend_boundary()
        potential = self.get_potential(ext) + self.ap_sol_f
        u_act = potential

        for i, j in self.M0:
            index = matrices.get_index(self.N, i, j)
            r, th = self.get_polar(i, j)
            u_act[index] += self.problem.get_restore_polar(r, th)

        error = self.eval_error(u_act)

        result = Result()
        result.error = error
        result.u_act = u_act

        if self.problem.var_compute_a:
            result.a_error = self.problem.get_a_error()

        return result
