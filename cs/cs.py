import math
import numpy as np

from solver import Solver
import matrices

class CircleSolver(Solver):
    """
    Abstract class representing a solver for a circular domain. Does not
    specify the type of basis to be used on the boundary.
    """

    def __init__(self, problem, N, scheme_order, **kwargs):
        super().__init__(problem, N, scheme_order, **kwargs)
        problem.R = self.R
        self.construct_gamma()

    def construct_gamma(self):
        """
        Build the discrete boundary gamma.
        """
        Nplus = set()
        Nminus = set()

        for i, j in self.M0:
            Nm = set([(i, j), (i-1, j), (i+1, j), (i, j-1), (i, j+1)])

            if self.scheme_order > 2:
                Nm |= set([(i-1, j-1), (i+1, j-1), (i-1, j+1),
                    (i+1, j+1)])

            if (i, j) in self.global_Mplus:
                Nplus |= Nm
            else:
                Nminus |= Nm

        gamma_set = Nplus & Nminus
        self.gamma = list(gamma_set)


    def get_potential(self, xi):
        """
        Calculate the difference potential of a function xi
        that is defined on gamma.
        """
        w = np.zeros([(self.N-1)**2], dtype=complex)

        for l in range(len(self.gamma)):
            w[matrices.get_index(self.N, *self.gamma[l])] = xi[l]

        Lw = np.ravel(self.L.dot(w))

        for i,j in self.global_Mminus:
            Lw[matrices.get_index(self.N, i, j)] = 0

        return w - self.LU_factorization.solve(Lw)

    def get_trace(self, data):
        projection = np.zeros(len(self.gamma), dtype=complex)

        for l in range(len(self.gamma)):
            index = matrices.get_index(self.N, *self.gamma[l])
            projection[l] = data[index]

        return projection

    def calc_c1(self):
        Q0 = self.get_Q(0)
        Q1 = self.get_Q(1)

        self.ap_sol_f = self.LU_factorization.solve(self.B_src_f)
        ext_f = self.extend_inhomo_f()
        proj_f = self.get_trace(self.get_potential(ext_f))

        rhs = -Q0.dot(self.c0) - self.get_trace(self.ap_sol_f) - proj_f + ext_f
        self.c1 = np.linalg.lstsq(Q1, rhs)[0]

    def is_interior(self, i, j):
        return self.get_polar(i, j)[0] <= self.R

    def extend_inhomo_f(self):
        ext = np.zeros(len(self.gamma), dtype=complex)
        for l in range(len(self.gamma)):
            i, j = self.gamma[l]
            r, th = self.get_polar(i, j)
            ext[l] = self.calc_inhomo_circle(r, th)

        return ext

    def extend_src_f(self):
        p = self.problem
        if p.homogeneous:
            return

        for i,j in self.Kplus - self.global_Mplus:
            R = self.R

            r, th = self.get_polar(i, j)
            x = R * np.cos(th)
            y = R * np.sin(th)

            derivs = [p.eval_f(x, y), p.eval_d_f_r(R, th),
                p.eval_d2_f_r(R, th)]

            v = 0
            for l in range(len(derivs)):
                v += derivs[l] / math.factorial(l) * (r - R)**l

            self.src_f[matrices.get_index(self.N,i,j)] = v
