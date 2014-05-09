import os
import pickle
import zlib
import itertools as it

import numpy as np
import scipy.sparse.linalg

import potential.matrices as matrices
import potential.util as util

class Solver:  

    enable_caching = False

    def __init__(self, problem, N, scheme_order, **kwargs):
        self.problem = problem
        self.k = self.problem.k
        self.N = N
        self.scheme_order = scheme_order

        if 'verbose' in kwargs:
            self.verbose = kwargs['verbose']
        else:
            self.verbose = False

        self.construct_grids()
        if self.verbose:
            print('Using scheme of order {}.'.format(self.scheme_order))
            print('Grid is {0} x {0}.'.format(self.N))

        if not self.enable_caching:
            kwargs = {'dtype': complex}

        self.L = matrices.get_L(self.scheme_order, self.N, 
            self.AD_len, self.problem.k, **kwargs)

        self.B = matrices.get_B(self.scheme_order, self.N,
            self.AD_len, self.k)
        self.B_src_f = self.B.dot(self.src_f)

        self.setup_LU()

    def setup_LU(self):
        cache_filename = 'LU_cache/L{}_N={}_k={}'.format(
            self.scheme_order, self.N, self.k)

        if not self.enable_caching:
            self.LU_factorization = scipy.sparse.linalg.splu(self.L)
        elif os.path.isfile(cache_filename):
            cache_file = open(cache_filename, 'rb')
            pickle_bytes = zlib.decompress(cache_file.read())
            self.LU_factorization = pickle.loads(pickle_bytes)

            if self.verbose:
                print('Loaded LU from cache.')
        else:
            self.LU_factorization = util.LU_Factorization(self.L)

            if not os.path.isdir('LU_cache'):
                os.mkdir('LU_cache')
            
            pickle_bytes = pickle.dumps(self.LU_factorization)
            compressed = zlib.compress(pickle_bytes)

            cache_file = open(cache_filename, 'wb')
            cache_file.write(compressed)

            if self.verbose:
                print('Wrote LU to cache.')

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

        self.src_f = np.zeros((self.N-1)**2, dtype=complex)

        self.Mplus = set()
        for i, j  in self.M0:
            if self.is_interior(i, j): 
                self.src_f[matrices.get_index(self.N, i, j)] =\
                    self.problem.eval_f(*self.get_coord(i, j))
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

        self.gamma = list(self.Nplus & self.Nminus)

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
