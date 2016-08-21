import sys
import math
import numpy as np
import scipy
from scipy.special import jv

from chebyshev import eval_T

# TODO remove
import matplotlib.pyplot as plt

from solver import Solver, Result
from linop import apply_B
import fourier
import abcoef
from problems.singular import HReg

from ps.basis import PsBasis
from ps.grid import PsGrid
from ps.extend import PsExtend
from ps.inhomo import PsInhomo
from ps.debug import PsDebug

def get_M(scheme_order):
    if scheme_order == 2:
        return 3
    elif scheme_order == 4:
        return 7

class PizzaSolver(Solver, PsBasis, PsGrid, PsExtend, PsInhomo, PsDebug):


    def __init__(self, options):
        super().__init__(options)
        self.a = self.problem.a
        self.nu = self.problem.nu
        self.R = self.problem.R

        self.boundary = self.problem.boundary

        self.scheme_order = options['scheme_order']
        self.fake_grid = options.get('fake_grid', False)

        self.M = get_M(self.scheme_order)
        self.problem.M = self.M

        # Get basis set sizes from the Problem
        n_basis_tuple = self.problem.get_n_basis(self.N,
            scheme_order=self.scheme_order)
        self.setup_basis(*n_basis_tuple)

        # For optimizing basis set sizes
        if 'n_circle' in options:
            assert 'n_radius' in options
            self.setup_basis(options['n_circle'], options['n_radius'])

        self.ps_construct_grids(self.scheme_order)

        self.build_boundary_coord_cache()

        apsolver = fourier.APSolver(self.N, self.scheme_order, self.AD_len, self.k)
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

    def is_interior(self, i, j):
        r, th = self.get_polar(i, j)
        rb = self.boundary.eval_r(th)
        return r <= rb and th >= self.a and th <= 2*np.pi

    def get_radius_point(self, sid, x, y):
        assert sid == 2

        a = self.a

        if a == np.pi/2:
            return (0, y)

        if a > .45*np.pi:
            print('warning: a is close to pi/2')

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

        x1, y1 = self.get_radius_point(sid, x, y)
        if x < x1:
            return -unsigned
        else:
            return unsigned

    def get_polar(self, i, j):
        r, th = super().get_polar(i, j)
        if th < self.a/2:
            th += 2*np.pi

        return r, th

    def get_boundary_coord(self, i, j):
        """
        Get the coordinates (n, th) relative to the outside boundary.
        """
        r1, th1 = self.get_polar(i, j)
        return self.boundary.get_boundary_coord(r1, th1)

    def build_boundary_coord_cache(self):
        """
        Build a cache of the boundary coordinates of each node in
        all_gama[0].
        """
        self.boundary_coord_cache = {}
        for i, j in self.all_gamma[0]:
            self.boundary_coord_cache[(i, j)] = self.get_boundary_coord(i, j)

    def get_d_matrix(self):
        """
        Create matrix whose columns contain the Chebyshev coefficients
        for the functions jv*sin(m*nu*(th-a)) for m=1...M.
        """
        # FIXME for perturbed

        k = self.k
        R = self.R
        a = self.a
        nu = self.nu

        d_columns = []

        for m in range(1, self.M+1):
            def func(th):
                return np.sin(m*nu*(th-a))

            d = self.get_chebyshev_coef(0, func)
            d = np.matrix(d.reshape((len(d), 1)))
            d_columns.append(jv(m*nu, k*R) * d)

        return np.column_stack(d_columns)


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

        #if self.problem.var_compute_a:

        submatrices[1].append(-submatrices[0][0].dot(
            self.get_d_matrix()))

        V0 = np.concatenate(submatrices[0], axis=1)
        V1 = np.concatenate(submatrices[1], axis=1)

        # Compute the RHS of the variational formulation.
        ext_f = self.extend_inhomo_f()
        proj_f = self.get_potential(ext_f)

        term = self.get_trace(proj_f + self.ap_sol_f)
        rhs += -V0.dot(self.c0) + ext_f - term

        M = self.M

        """for match_r in np.arange(.001, 1, .01):
            enforce_nodes = []
            for i, j in self.union_gamma:
                r, th = self.get_polar(i, j)
                if .001 < r and r < match_r:
                    if (i, j) in self.all_gamma[1]:
                        enforce_nodes.append((i, j, 1))
                    if (i, j) in self.all_gamma[2]:
                        enforce_nodes.append((i, j, 2))

            if len(enforce_nodes) >= 10:
                break"""

        #print('match_r={}  count={}'.format(match_r, len(enforce_nodes)))

        conper = 4
        constraints = conper*2
        h = .025 #.2*self.AD_len / self.N
        s = .1
        conr = np.linspace(s, s+conper*h, conper)
        #print('conr:', conr)

        new_V1 = np.zeros((V1.shape[0]+constraints, V1.shape[1]), dtype=complex)
        new_V1[:V1.shape[0], :] = V1

        chebyshev_error = []

        for sid in [1,2]:
            t_data = []
            for t in self.chebyshev_roots:
                if self.eval_g(sid, t) > conr[0]:
                    t_data.append(t)

            if sid == 1:
                th = 2*np.pi
            else:
                th = self.a

            nu = self.nu

            for m in range(1, self.M+1):
                def nmn(r):
                    val = jv(m*nu, self.k*r) * m*nu*np.cos(m*nu*(th-self.a))
                    if sid == 1:
                        return val
                    elif sid == 2:
                        return -val

                ccoef = self.get_chebyshev_coef(sid, nmn, t_data=t_data)

                # Test chebyshev error
                #r_data = np.arange(self.get_polar(i,j)[0], 2.3, .0025)
                r_data = np.arange(conr[0], 2.3, .0025)
                series_data = []
                exp_data = []
                for r in r_data:
                    series = 0
                    for J in range(self.segment_desc[sid]['n_basis']):
                        JJ = self.segment_desc[sid]['JJ_list'][J]
                        series += ccoef[J] * self.eval_dn_B_arg(0, JJ, r, sid)

                    chebyshev_error.append(series-nmn(r))
                    #if abs(chebyshev_error[-1]) > .02:
                    #    print('r=', r, 'm=', m, 'sid=', sid)

                    #series_data.append(series)
                    #exp_data.append(nmn(r))

                #plt.plot(r_data, series_data, color='green')
                #plt.plot(r_data, exp_data, color='black')
                #plt.show()

                #sys.exit(0)

                for l in range(conper):
                    r = conr[l]
                    for J in range(self.segment_desc[sid]['n_basis']):
                        JJ = self.segment_desc[sid]['JJ_list'][J]
                        new_V1[V1.shape[0]+l, JJ] += ccoef[J] * self.eval_dn_B_arg(0, JJ, r, sid)

                    new_V1[V1.shape[0]+l, len(self.c0)+(m-1)] = -nmn(r)

        new_rhs = np.zeros(len(rhs)+constraints, dtype=complex)
        new_rhs[:len(rhs)] = rhs

        print('Chebyshev error:', np.max(np.abs(chebyshev_error)))

        return (new_V1, new_rhs)


    def solve_var(self):
        """
        Setup and solve the variational formulation.
        """
        V1, rhs = self.get_var()
        varsol = np.linalg.lstsq(V1, rhs)[0]

        self.handle_varsol(varsol)

    def handle_varsol(self, varsol):
        """
        Break up the solution vector of the variational formulation into
        its meaningful parts.
        Set c1. Update the boundary data if necessary.
        """
        self.c1 = varsol[:len(self.c0)]

        #if self.var_compute_a:
        var_a = varsol[len(self.c0):]

        #if self.var_method in fft_test_var_methods:
        #    a_coef = self.problem.fft_a_coef[:self.M]
        #else:
        #a_coef = np.zeros(self.M, dtype=complex)
        print(np.abs(var_a))
        #print(np.max(np.abs(var_a)))

        """for i in range(len(self.m_list)):
            m = self.m_list[i]
            a_coef[m-1] = var_a[i]

        if self.var_compute_a_only:
            self.a_coef = a_coef
            return

        self.problem.a_coef = a_coef

        if self.problem.regularize_bc:
            self.update_c0()"""


    def get_singular_part(self):
        singular_part = np.zeros((self.N-1, self.N-1), dtype=complex)

        if self.problem.hreg == HReg.none:
            return singular_part

        k = self.k
        R = self.R
        a = self.a
        nu = self.nu

        for i, j in self.global_Mplus:
            r, th = self.get_polar(i, j)
            for m in range(1, self.M+1):
                singular_part[i-1, j-1] += (self.a_coef[m-1] * jv(m*nu, k*r) *
                    np.sin(m*nu*(th-a)))

        return singular_part


    def run(self):
        """
        The main procedure for PizzaSolver.
        """
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
        regular_part = self.get_potential(ext) + self.ap_sol_f

        u_act = regular_part #+ self.get_singular_part()

        error = self.eval_error(u_act)

        result = Result()
        result.error = error
        result.u_act = u_act

        return result
