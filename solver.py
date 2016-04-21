import itertools as it

import numpy as np

from domain_util import cart_to_polar
from extend import SolverExtend

class Result:
    u_act = None
    a_error = None


class Solver(SolverExtend):
    """
    A generic class representing a difference potentials problem.

    This class doesn't make any assumptions about the shape of the domain
    (except that it is 2D). It only contains functionality that is
    expected to be relevant for all solvers.

    Solvers for specific domains and cases should extend this class.
    """

    def __init__(self, options):
        super().__init__()
        self.problem = options['problem']
        self.k = self.problem.k
        self.AD_len = self.problem.AD_len
        self.N = options['N']

    def get_coord(self, i, j):
        """
        Get the rectangular coordinates of grid point (i,j)
        """
        x = self.AD_len * (i / self.N - 1/2)
        y = self.AD_len * (j / self.N - 1/2)
        return x, y

    def get_coord_inv(self, x, y):
        """
        Inverse of get_coord(). For testing purposes.
        """
        i = self.N * (x / self.AD_len + 1/2)
        j = self.N * (y / self.AD_len + 1/2)
        return i, j

    def get_polar(self, i, j):
        """
        Get the polar coordinates of grid point (i,j)
        """
        x, y = self.get_coord(i, j)
        return cart_to_polar(x, y)

    def construct_grids(self, scheme_order):
        """
        Construct the various grids used in the algorithm.
        """
        self.N0 = set(it.product(range(0, self.N+1), repeat=2))
        self.M0 = set(it.product(range(1, self.N), repeat=2))

        self.global_Mplus = set()
        for i, j  in self.M0:
            if self.is_interior(i, j):
                self.global_Mplus.add((i, j))

        self.global_Mminus = self.M0 - self.global_Mplus

        self.Kplus = set()
        for i, j in self.global_Mplus:
            Nm = set([(i, j)])

            if scheme_order > 2:
              Nm |= set([(i-1, j), (i+1, j), (i, j-1), (i, j+1)])

            self.Kplus |= Nm

    def setup_f(self):
        """
        Create a grid function f that equals the problem's RHS inside
        the domain and is 0 elsewhere.

        The subclass of Solver should define extend_f() to extend
        f to slightly outside the boundary, if necessary.
        """
        N = self.N
        self.f = np.zeros((N+1, N+1), dtype=complex)

        for i,j in self.global_Mplus:
            self.f[i, j] =\
                self.problem.eval_f(*self.get_coord(i, j))

        if hasattr(self, 'extend_f'):
            self.extend_f()

    def eval_error(self, u_act):
        """
        Returns || u_act - u_exp ||_inf, the error between
        actual and expected solutions under the infinity-norm.
        """
        if not self.problem.expected_known:
            return None

        error = []
        max_error = 0

        for i, j in self.global_Mplus:
            x, y = self.get_coord(i, j)
            diff = abs(self.problem.eval_expected(x, y) - u_act[i-1, j-1])
            error.append(diff)

        return max(error)

    def calc_rel_convergence(self, u0, u1, u2):
        """
        Calculate the relative convergence of the sequence (u0, u1, u2).

        In the typical case, u0, u1, and u2 are arrays representing the
        numerical solution on grid sizes N//4, N//2, and N, respectively.

        The relative convergene is, roughly speaking,

           log2(abs(u0-u1) / abs(u1-u2)).

        However, note that when the differences u0-u1 and u1-u2 are
        computed, we can only compare the two solutions at nodes where
        they are both defined.
        """
        if type(u0) is np.ndarray:
            return self._array_calc_rel_convergence(u0, u1, u2)
        else:
            # If the u's are not numpy arrays, then they are required to
            # implement their own calc_rel_convergence function
            return u0.calc_rel_convergence(u1, u2)

    def _array_calc_rel_convergence(self, u0, u1, u2):
        """
        Calculate the relative convergence of the sequence (u0, u1, u2)
        where the u's are numpy arrays.
        """
        N = self.N
        diff12 = []
        diff01 = []

        for i, j in self.global_Mplus:
            k0 = (i-1, j-1)
            k1 = (i//2-1, j//2-1)
            k2 = (i//4-1, j//4-1)

            if i % 4 == 0 and j % 4 == 0:
                diff12.append(abs(u1[k1] - u2[k2]))

            if i % 2 == 0 and j % 2 == 0:
                diff01.append(abs(u0[k0] - u1[k1]))

        return np.log2(max(diff12) / max(diff01))
