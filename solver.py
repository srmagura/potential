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
    u_act = None
    

class Solver(SolverExtend, SolverDebug): 

    '''Set to True when debugging with test_extend_boundary() for 
    significant performance benefit.'''
    skip_matrix_build = False

    def __init__(self, problem, N, scheme_order, **kwargs):
        super().__init__()
        self.problem = problem
        self.k = self.problem.k
        self.N = N
        self.scheme_order = scheme_order

        # verbose is currently not used
        if 'verbose' in kwargs:
            self.verbose = kwargs['verbose']
        else:
            self.verbose = False

        self.construct_grids()

        if not self.skip_matrix_build:
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
            diff = abs(u_exp[l] - u_act[l])
            error.append(diff)  

        return max(error)
        
    def calc_convergence3(self, u0, u1, u2):
        if type(u0) is np.ndarray:
            return self._array_calc_convergence3(u0, u1, u2)
        else:
            return self._mv_calc_convergence3(u0, u1, u2)

    def _array_calc_convergence3(self, u0, u1, u2):
        N = self.N
        diff12 = []
        diff01 = []
    
        for i, j in self.global_Mplus:
            k0 = matrices.get_index(N, i, j)
            k1 = matrices.get_index(N//2, i//2, j//2)
            k2 = matrices.get_index(N//4, i//4, j//4)
        
            if i % 4 == 0 and j % 4 == 0:
                diff12.append(abs(u1[k1] - u2[k2]))
                
            if i % 2 == 0 and j % 2 == 0:
                diff01.append(abs(u0[k0] - u1[k1]))
                
        return np.log2(max(diff12) / max(diff01))

    def _mv_calc_convergence3(self, u0, u1, u2):
        diff12 = []
        diff01 = []

        for i, j in u0:
            i1, j1 = i//2, j//2
            i2, j2 = i//4, j//4

            if i % 4 == 0 and j % 4 == 0:
                for l in range(len(u2[i2, j2])):
                    data1 = None
                    data2 = u2[i2, j2][l]

                    for ll in range(len(u1[i1, j1])):
                        _data1 = u1[i1, j1][ll]
                        if _data1['setype'] == data2['setype']:
                            data1 = _data1
                            break

                    if data1:
                        diff12.append(abs(data1['value'] - data2['value']))
                
            if i % 2 == 0 and j % 2 == 0:
                for l in range(len(u1[i1, j1])):
                    data0 = None
                    data1 = u1[i1, j1][l]

                    for ll in range(len(u0[i, j])):
                        _data0 = u0[i, j][ll]
                        if _data0['setype'] == data1['setype']:
                            data0 = _data0
                            break
                    
                    if data0:
                        diff01.append(abs(data0['value'] - data1['value']))
                
        return np.log2(max(diff12) / max(diff01))
