import itertools as it
import math

import numpy as np
import scipy.sparse.linalg

import matrices
from debug import SolverDebug
from extend import SolverExtend

def cart_to_polar(x, y):
    r, th = math.hypot(x, y), math.atan2(y, x)

    if th <= 0:
        th += 2*np.pi

    return r, th
    
class Result:
    pass
    

class Solver(SolverExtend, SolverDebug): 

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

        for i,j in self.global_Mminus:
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

        self.global_Mplus = set()
        for i, j  in self.M0:
            if self.is_interior(i, j): 
                self.global_Mplus.add((i, j))

        self.global_Mminus = self.M0 - self.global_Mplus

        self.Kplus = set()
        for i, j in self.global_Mplus:
            Nm = set([(i, j)])

            if self.scheme_order > 2:
                Nm |= set([(i-1, j), (i+1, j), (i, j-1), (i, j+1)])
            
            self.Kplus |= Nm

    def setup_src_f(self):
        self.src_f = np.zeros((self.N-1)**2, dtype=complex)

        for i,j in self.global_Mplus:
            self.src_f[matrices.get_index(self.N, i, j)] =\
                self.problem.eval_f(*self.get_coord(i, j))

        if hasattr(self, 'extend_src_f'):
            self.extend_src_f()

    # Returns || u_act - u_exp ||_inf, the error between 
    # actual and expected solutions under the infinity-norm. 
    def eval_error(self, u_act):
        if not self.problem.expected_known:
            return None
    
        u_exp = np.zeros((self.N-1)**2, dtype=complex)
        for i,j in self.M0:
            x, y = self.get_coord(i,j)
            u_exp[matrices.get_index(self.N, i,j)] =\
                self.problem.eval_expected(x, y)

        error = []
        max_error = 0
        
        for i,j in self.global_Mplus:
            l = matrices.get_index(self.N, i,j)
            error.append(abs(u_exp[l] - u_act[l]))
            
            if error[-1] > max_error:
                print('i={}    j={}    error={}'.format(i,j,error[-1]))
                print(self.get_coord(i,j))
                max_error = error[-1]

        return max(error)
        
    def calc_convergence3(self, u0, u1, u2):
        N = self.N
        diff12 = []
        diff01 = []
    
        for i, j in self.Mplus:
            k0 = matrices.get_index(N, i, j)
            k1 = matrices.get_index(N//2, i//2, j//2)
            k2 = matrices.get_index(N//4, i//4, j//4)
        
            if i % 4 == 0 and j % 4 == 0:
                diff12.append(abs(u1[k1] - u2[k2]))
                
            if i % 2 == 0 and j % 2 == 0:
                diff01.append(abs(u0[k0] - u1[k1]))
                
        return np.log2(max(diff12) / max(diff01))
