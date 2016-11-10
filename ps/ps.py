import sys
import math
import numpy as np
import scipy
from scipy.special import jv

from solver import Solver, Result
from linop import apply_B
import fourier
import abcoef

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

        self.boundary_coord_cache = {}
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


    def get_potential(self, ext):
        """
        Compute the difference potential of `ext`.

        Output:
        potential -- shape (N-1, N-1)
        """
        N = self.N
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

    def solve_var(self):
        """
        Setup and solve the variational formulation.
        """
        Q1, rhs = self.get_var()
        self.c1 = np.linalg.lstsq(Q1, rhs)[0]

    def run(self):
        """
        The main procedure for PizzaSolver.
        """
        #self.Q_residual()
        #result = Result()
        #result.error = 0.
        #return result

        #r_data = np.arange(0, 2.3, .001)
        #f_data = np.array([self.problem.eval_f_polar(r, 1) for r in r_data])
        #import matplotlib.pyplot as plt
        #plt.plot(r_data, f_data)
        #plt.show()

        self.calc_c0()

        '''
        Uncomment the following line to see a plot of the boundary data and
        its Chebyshev series, which has coefficients c0. There is also an
        analogous function c1_test().
        '''
        self.c0_test(plot=False)

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
