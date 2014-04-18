import numpy as np
import scipy.sparse.linalg

import problems
import matrices

class Solver:  

    def __init__(self, problem, N, **kwargs):
        self.problem = problem
        self.domain = self.problem.domain
        self.N = N

        if 'verbose' in kwargs:
            self.verbose = kwargs['verbose']
        else:
            self.verbose = False

    # Run the algorithm. N has already been set at this point
    def run(self):
        self.domain.construct_grids(self.N)

        if self.verbose:
            print('Grid is {0} x {0}.'.format(self.N))

        self.L = matrices.get_L(self.N, self.domain.AD_len, 
            self.problem.k)

        if self.verbose:
            print('Sparse matrix has {} entries.'.
                format(self.L.nnz))

        self.LU_factorization = scipy.sparse.linalg.splu(self.L)

        c0 = self.domain.get_c0()
        if self.verbose:
            print('Using {} basis functions.'.
                format(len(self.domain.J_dict)))

        Q0 = self.get_Q(0)
        Q1 = self.get_Q(1)

        c1 = np.linalg.lstsq(Q1, -Q0.dot(c0))[0]

        ext = self.domain.extend_boundary(c0,c1)
        u_act = self.get_potential(ext)

        error =  self.eval_error(u_act)
        return error

    # Returns || u_act - u_exp ||_inf, the error between 
    # actual and expected solutions under the infinity-norm. 
    def eval_error(self, u_act):
        u_exp = np.zeros((self.N-1)**2, dtype=complex)
        for i,j in self.domain.M0:
            u_exp[matrices.get_index(self.N, i,j)] =\
                self.problem.eval_expected(*self.domain.get_coord(i,j))

        error = []
        for i,j in self.domain.Mplus:
            l = matrices.get_index(self.N, i,j)
            error.append(abs(u_exp[l] - u_act[l]))

        return max(error)

    # Calculate the difference potential of a function xi 
    # that is defined on gamma.
    def get_potential(self, xi):
        w = np.zeros([(self.N-1)**2], dtype=complex)

        for l in range(len(self.domain.gamma)):
            w[matrices.get_index(self.N, *self.domain.gamma[l])] = xi[l]

        Lw = np.ravel(self.L.dot(w))

        for i,j in self.domain.Mminus:
            Lw[matrices.get_index(self.N, i, j)] = 0

        return w - self.LU_factorization.solve(Lw)

    def get_projection(self, potential):
        projection = np.zeros(len(self.domain.gamma), dtype=complex)

        for l in range(len(self.domain.gamma)):
            index = matrices.get_index(self.N, *self.domain.gamma[l])
            projection[l] = potential[index]

        return projection

    # Returns the matrix Q0 or Q1, depending on the value of `index`.
    def get_Q(self, index):
        columns = []

        for J in self.domain.J_dict:
            ext = self.domain.extend_basis(J, index)
            potential = self.get_potential(ext)
            projection = self.get_projection(potential)

            columns.append(projection - ext)

        return np.column_stack(columns)
