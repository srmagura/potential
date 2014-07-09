import itertools as it
import math

import numpy as np
import scipy.sparse.linalg

import matrices

def cart_to_polar(x, y):
    r, th = math.hypot(x, y), math.atan2(y, x)

    if th < 0:
        th += 2*np.pi

    return r, th

class Solver:  

    def __init__(self, problem, N, scheme_order, **kwargs):
        super().__init__()
        self.problem = problem
        self.k = self.problem.k
        self.N = N
        self.scheme_order = scheme_order

        if 'verbose' in kwargs:
            self.verbose = kwargs['verbose']
        else:
            self.verbose = False

        if self.verbose:
            print('Problem is `{}`.'.format(self.problem.__class__.__name__))
            print('Using `{}`.'.format(self.__class__.__name__))
            print('Using scheme of order {}.'.format(self.scheme_order))
            print('Grid is {0} x {0}.'.format(self.N))

        self.construct_grids()
        self.L = matrices.get_L(self.scheme_order, self.N, 
            self.AD_len, self.problem.k)
        self.LU_factorization = scipy.sparse.linalg.splu(self.L)

        self.setup_src_f()

        self.B = matrices.get_B(self.scheme_order, self.N,
            self.AD_len, self.k)
        self.B_src_f = self.B.dot(self.src_f)

        for i,j in self.Mminus:
            self.B_src_f[matrices.get_index(N,i,j)] = 0

    # Get the rectangular coordinates of grid point (i,j)
    def get_coord(self, i, j):
        x = self.AD_len * (i / self.N - 1/2) 
        y = self.AD_len * (j / self.N - 1/2) 
        return x, y

    # For testing
    def get_coord_inv(self, x, y):
        i = self.N * (x / self.AD_len + 1/2)
        j = self.N * (y / self.AD_len + 1/2)
        return i, j

    # Get the polar coordinates of grid point (i,j)
    def get_polar(self, i, j):
        x, y = self.get_coord(i, j)
        return cart_to_polar(x, y)

    # Construct the various grids used in the algorithm.
    def construct_grids(self):
        self.N0 = set(it.product(range(0, self.N+1), repeat=2))
        self.M0 = set(it.product(range(1, self.N), repeat=2))

        self.Mplus = set()
        for i, j  in self.M0:
            if self.is_interior(i, j): 
                self.Mplus.add((i, j))

        self.Mminus = self.M0 - self.Mplus
        self.Nplus = set()
        self.Nminus = set()
               
        for i, j in self.M0:
            Nm = set([(i, j), (i-1, j), (i+1, j), (i, j-1), (i, j+1)])

            if self.scheme_order > 2:
                Nm |= set([(i-1, j-1), (i+1, j-1), (i-1, j+1),
                    (i+1, j+1)])

            if (i, j) in self.Mplus:
                self.Nplus |= Nm
            else:
                self.Nminus |= Nm

        gamma_set = self.Nplus & self.Nminus
        self.gamma = []

        while len(gamma_set) > 0:
            min_th = float('inf')
            for i, j in gamma_set:
                r, th = self.get_polar(i, j)
                if th < min_th:
                    min_th = th
                    min_th_gamma = i, j

            gamma_set.remove(min_th_gamma)
            self.gamma.append(min_th_gamma)

        self.gamma = np.array(self.gamma)

        self.Kplus = set()
        for i, j in self.Mplus:
            Nm = set([(i, j)])

            if self.scheme_order > 2:
                Nm |= set([(i-1, j), (i+1, j), (i, j-1), (i, j+1)])
            
            self.Kplus |= Nm

    # Calculate the difference potential of a function xi 
    # that is defined on gamma.
    def get_potential(self, xi):
        w = np.zeros([(self.N-1)**2], dtype=complex)

        for l in range(len(self.gamma)):
            w[matrices.get_index(self.N, *self.gamma[l])] = xi[l]

        Lw = np.ravel(self.L.dot(w))

        for i,j in self.Mminus:
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
        ext_f = self.extend_inhomogeneous_f()    
        proj_f = self.get_trace(self.get_potential(ext_f))

        rhs = -Q0.dot(self.c0) - self.get_trace(self.ap_sol_f) - proj_f + ext_f
        self.c1 = np.linalg.lstsq(Q1, rhs)[0]

    def setup_src_f(self):
        self.src_f = np.zeros((self.N-1)**2, dtype=complex)

        for i,j in self.Mplus:
            self.src_f[matrices.get_index(self.N, i, j)] =\
                self.problem.eval_f(*self.get_coord(i, j))

        if hasattr(self, 'extend_src_f'):
            self.extend_src_f()

    def extend_circle(self, r, xi0, xi1,
        d2_xi0_th, d2_xi1_th, d4_xi0_th):

        R = self.R
        k = self.k

        derivs = []
        derivs.append(xi0) 
        derivs.append(xi1)
        derivs.append(-xi1 / R - d2_xi0_th / R**2 - k**2 * xi0)

        derivs.append(2 * xi1 / R**2 + 3 * d2_xi0_th / R**3 -
            d2_xi1_th / R**2 + k**2 / R * xi0 - k**2 * xi1)

        derivs.append(-6 * xi1 / R**3 + 
            (2*k**2 / R**2 - 11 / R**4) * d2_xi0_th +
            6 * d2_xi1_th / R**3 + d4_xi0_th / R**4 -
            (3*k**2 / R**2 - k**4) * xi0 +
            2 * k**2 / R * xi1)

        v = 0
        for l in range(len(derivs)):
            v += derivs[l] / math.factorial(l) * (r - R)**l

        return v


    # Returns || u_act - u_exp ||_inf, the error between 
    # actual and expected solutions under the infinity-norm. 
    def eval_error(self, u_act):
        u_exp = np.zeros((self.N-1)**2, dtype=complex)
        for i,j in self.M0:
            u_exp[matrices.get_index(self.N, i,j)] =\
                self.problem.eval_expected(*self.get_coord(i,j))

        error = []
        for i,j in self.Mplus:
            l = matrices.get_index(self.N, i,j)
            error.append(abs(u_exp[l] - u_act[l]))

        return max(error)

class SquareSolver(Solver):

    AD_len = 2*np.pi

    # Get the rectangular coordinates of grid point (i,j)
    def get_coord(self, i, j):
        x = (self.AD_len * i / self.N - self.AD_len/2, 
            self.AD_len * j / self.N - self.AD_len/2)
        return x

    def is_interior(self, i, j):
        return (i != 0 and i != self.N and j != 0 and j != self.N)

    def run(self):
        u_act = self.LU_factorization.solve(self.B_src_f)
        return self.eval_error(u_act)

from cs.csf import CsFourier
from cs.cs1 import CsChebyshev1
from cs.cs3 import CsChebyshev3

solver_dict = {'csf': CsFourier,
    'cs1': CsChebyshev1, 'cs3': CsChebyshev3}
