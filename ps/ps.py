import sys
import math
import numpy as np
import scipy

from solver import Solver, Result
from linop import apply_B
import fourier
import opt

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


    def __init__(self, problem, N, options={}, **kwargs):
        self.a = problem.a
        self.nu = problem.nu
        self.R = problem.R
        super().__init__(problem, N, options, **kwargs)

        self.scheme_order = options['scheme_order']
        self.extension_order = self.scheme_order + 1

        self.M = get_M(self.scheme_order)
        problem.M = self.M

        problem.setup()

        # For optimizing basis set sizes
        if 'n_circle' in options:
            assert 'n_radius' in options
            self.setup_basis(options['n_circle'], options['n_radius'])

        self.ps_construct_grids(self.scheme_order)

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

    def get_Q(self, index):
        columns = []

        for JJ in range(len(self.B_desc)):
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
        Chebyshev coefficients).
        """
        Q0 = self.get_Q(0)
        Q1 = self.get_Q(1)

        # Compute the RHS of the variational formulation.
        ext_f = self.extend_inhomo_f()
        proj_f = self.get_potential(ext_f)

        term = self.get_trace(proj_f + self.ap_sol_f)
        rhs = -Q0.dot(self.c0) + ext_f - term

        return (Q1, rhs)

    def get_var_constraints(self):
        C = np.zeros((self.M, len(self.B_desc)))

        J = 0
        for JJ in self.segment_desc[0]['JJ_list']:
            coef = fourier.arc_dst(self.a, lambda th:
                self.eval_dn_B_arg(1, JJ, self.R, th, 0))

            C[:, J] = coef[:self.M]
            J += 1

        d = np.zeros(self.M)
        return (C, d)

    def solve_var(self):
        """
        Setup and solve the variational formulation.
        """
        Q1, rhs = self.get_var()
        C, d = self.get_var_constraints()

        self.c1 = opt.constrained_lstsq(Q1, rhs, C, d)

    def run(self):
        """
        The main procedure for PizzaSolver.
        """

        if not hasattr(self, 'B_desc'):
            n_basis_tuple = self.problem.get_n_basis(N=self.N,
                scheme_order=self.scheme_order)
            self.setup_basis(*n_basis_tuple)

        self.calc_c0()

        '''
        Uncomment the following line to see a plot of the boundary data and
        its Chebyshev series, which has coefficients c0. There is also an
        analogous function c1_test().
        '''
        #self.c0_test(plot=True)

        self.solve_var()

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

        return result
