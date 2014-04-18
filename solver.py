import itertools as it
import numpy as np
import scipy.sparse.linalg

import problems
import matrices

class Solver:  

    def __init__(self, problem, N, **kwargs):
        self.problem = problem
        self.N = N

        if 'verbose' in kwargs:
            self.verbose = kwargs['verbose']
        else:
            self.verbose = False

        self.construct_grids()
        if self.verbose:
            print('Grid is {0} x {0}.'.format(self.N))

        self.L = matrices.get_L(self.N, self.AD_len, self.problem.k)
        if self.verbose:
            print('Sparse matrix has {} entries.'.
                format(self.L.nnz))

        self.LU_factorization = scipy.sparse.linalg.splu(self.L)


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

    # Construct the various grids used in the algorithm.
    def construct_grids(self):
        self.N0 = set(it.product(range(0, self.N+1), repeat=2))
        self.M0 = set(it.product(range(1, self.N), repeat=2))

        self.K0 = set()
        for i, j in it.product(range(0, self.N+1), repeat=2):
            if(i != 0 and i != self.N) or (j != 0 and j != self.N):
                self.K0.add((i, j))

        self.Mplus = {(i, j) for i, j in self.M0 
            if self.get_polar(i, j)[0] <= self.R}
        self.Mminus = self.M0 - self.Mplus

        self.Nplus = set()
        self.Nminus = set()
               
        for i, j in self.M0:
            Nm = set([(i, j), (i-1, j), (i+1, j), (i, j-1), (i, j+1)])

            if (i, j) in self.Mplus:
                self.Nplus |= Nm
            else:
                self.Nminus |= Nm

        self.gamma = list(self.Nplus & self.Nminus)
