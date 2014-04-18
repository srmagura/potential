import itertools as it
import cmath
import math
import collections
import numpy as np

import solver
import matrices

# Grid size for boundary. Should be divisible by 2
fourier_N = 1024

# After performing FFT on boundary data, ignore Fourier series
# coefficients lower than this number
fourier_lower_bound = 5e-15

#plot_grid = False
plot_c1_error = False

class CircleSolver(solver.Solver):
    # Side length of square domain on which AP is solved
    AD_len = 2*np.pi

    # Radius of circular region Omega
    R = 2.3

    def __init__(self, problem, N, **kwargs):
        super().__init__(problem, N, **kwargs)

    # Get the polar coordinates of grid point (i,j)
    def get_polar(self, i, j):
        x, y = self.get_coord(i, j)
        return math.hypot(x, y), math.atan2(y, x)

    # Get the rectangular coordinates of grid point (i,j)
    def get_coord(self, i, j):
        x = (self.AD_len * i / self.N - self.AD_len/2, 
            self.AD_len * j / self.N - self.AD_len/2)
        return x

    # Choose which Fourier coefficients to keep
    def choose_n_basis(self, J_range, c0_raw):
        for J in J_range:
            normalized = c0_raw[J] / fourier_N

            if abs(normalized) > fourier_lower_bound:
                return abs(J)

    # Applies FFT to the boundary data to get c0, then solves the 
    # system Q1 * c1 = -Q0 * c0 by least squares to find c1. 
    def get_c(self):
        grid = np.arange(0, 2*np.pi, 2*np.pi / fourier_N)
        discrete_phi = [self.problem.eval_bc(th) for th in grid]
        c0_raw = np.fft.fft(discrete_phi)

        Jminus = self.choose_n_basis(range(-fourier_N // 2, 1), c0_raw)
        Jplus = self.choose_n_basis(range(fourier_N // 2, 0, -1), c0_raw)
        Jmax = max(Jminus, Jplus)

        self.J_dict = collections.OrderedDict(((J, None) for J in 
            range(-Jmax, Jmax+1)))

        i = 0
        c0 = np.zeros(len(self.J_dict), dtype=complex)

        for J in self.J_dict:
            self.J_dict[J] = i
            c0[i] = c0_raw[J] / fourier_N
            i += 1

        Q0 = self.get_Q(0)
        Q1 = self.get_Q(1)

        c1 = np.linalg.lstsq(Q1, -Q0.dot(c0))[0]
        return c0, c1


    def _v0(self, J, r, th):
        R = self.R
        k = self.problem.k

        x0 = 1
        x1 = 0
        x2 = (J/R)**2 - k**2
        #x3 = -3*J**2/R**3 + k**2/R
        x3 = 0

        return cmath.exp(complex(0, J*th)) *\
                (x0 + x1 * (r-R) + .5 * x2 * (r-R)**2 +
                 1/6 * x3 * (r-R)**3)

    def _v1(self, J, r, th):
        R = self.R
        k = self.problem.k

        x0 = 0
        x1 = 1
        x2 = -1/R
        #x3 = 2/R**2 + (J/R)**2 - k**2
        x3 = 0

        return cmath.exp(complex(0, J*th)) *\
                (x0 + x1 * (r-R) + .5 * x2 * (r-R)**2 +
                 1/6 * x3 * (r-R)**3)

    # Construct the equation-based extension for the basis function
    # $\psi_{index}^{(J)}$.
    def extend_basis(self, J, index):
        ext = np.zeros(len(self.gamma), dtype=complex)
        for l in range(len(self.gamma)):
            i, j = self.gamma[l]
            r, th = self.get_polar(i, j)

            if index == 0:
                ext[l] = self._v0(J, r, th)
            else:
                ext[l] = self._v1(J, r, th)  

        return ext

    # Construct the equation-based extension for the boundary data,
    # as approximated by the Fourier coefficients c0 and c1.
    def extend_boundary(self, c0, c1): 
        R = self.R
        k = self.problem.k

        boundary = np.zeros(len(self.gamma), dtype=complex)

        for l in range(len(self.gamma)):
            i, j = self.gamma[l]
            r, th = self.get_polar(i, j)

            v0 = v1 = xi0d = xi1d = 0 

            for J, i in self.J_dict.items():
                factor = cmath.exp(complex(0, J*th))
                v0 += c0[i] * factor 
                v1 += c1[i] * factor
                xi0d += J**2 * c0[i] * factor
                #xi1d += - J**2 * c1[i] * factor

            v2 = -v1/R + xi0d / R**2 - k**2 * v0

            #v3 = (2*v1 / R**2 + 3*xi0d/R**3 - xi1d/R**2 +
            #     k**2 *v0 / R - k**2 * v1)
            v3 = 0

            boundary[l] = (v0 + v1 * (r-R) + .5 * v2 * (r-R)**2 + 
                1/6 * v3 * (r-R)**3)

        return boundary


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

    def get_projection(self, potential):
        projection = np.zeros(len(self.gamma), dtype=complex)

        for l in range(len(self.gamma)):
            index = matrices.get_index(self.N, *self.gamma[l])
            projection[l] = potential[index]

        return projection

    # Returns the matrix Q0 or Q1, depending on the value of `index`.
    def get_Q(self, index):
        columns = []

        for J in self.J_dict:
            ext = self.extend_basis(J, index)
            potential = self.get_potential(ext)
            projection = self.get_projection(potential)

            columns.append(projection - ext)

        return np.column_stack(columns)

    def run(self):
        c0, c1 = self.get_c()
        if self.verbose:
            print('Using {} basis functions.'.
                format(len(self.J_dict)))

        ext = self.extend_boundary(c0,c1)
        u_act = self.get_potential(ext)

        error = self.eval_error(u_act)
        return error
