import itertools as it
import cmath
import math
import collections
import numpy as np

from potential.solver import Solver
import potential.matrices as matrices

# Grid size for boundary. Should be divisible by 2
fourier_N = 1024

Q_ratio = 2

class CircleSolver(Solver):
    # After performing FFT on boundary data, ignore Fourier series
    # coefficients lower than this number
    fourier_lower_bound = 5e-15

    # Side length of square domain on which AP is solved
    AD_len = 2*np.pi

    # Radius of circular region Omega
    R = 2.3

    def __init__(self, problem, N, **kwargs):
        super().__init__(problem, N, **kwargs)
        problem.R = self.R

    # Get the polar coordinates of grid point (i,j)
    def get_polar(self, i, j):
        x, y = self.get_coord(i, j)
        return math.hypot(x, y), math.atan2(y, x)

    # Get the rectangular coordinates of grid point (i,j)
    def get_coord(self, i, j):
        x = (self.AD_len * i / self.N - self.AD_len/2, 
            self.AD_len * j / self.N - self.AD_len/2)
        return x

    def is_interior(self, i, j):
        return self.get_polar(i, j)[0] <= self.R

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

    # Returns the matrix Q0 or Q1, depending on the value of `index`.
    def get_Q(self, index):
        columns = []

        for J in self.J_dict:
            ext = self.extend_basis(J, index)
            potential = self.get_potential(ext)
            projection = self.get_trace(potential)

            columns.append(projection - ext)

        return np.column_stack(columns)

    # Choose which Fourier coefficients to keep
    def choose_n_basis(self, J_range, c0_raw):
        for J in J_range:
            normalized = c0_raw[J] / fourier_N

            if abs(normalized) > self.fourier_lower_bound:
                return abs(J)

    # Applies FFT to the boundary data to get c0, then solves the 
    # system Q1 * c1 = -Q0 * c0 by least squares to find c1. 
    def get_c(self):
        grid = np.arange(0, 2*np.pi, 2*np.pi / fourier_N)
        discrete_phi = [self.problem.eval_bc(th) for th in grid]
        c0_raw = np.fft.fft(discrete_phi)

        while True: 
            Jminus = self.choose_n_basis(range(-fourier_N // 2, 1), c0_raw)
            Jplus = self.choose_n_basis(range(fourier_N // 2, 0, -1), c0_raw)
            Jmax = max(Jminus, Jplus)

            if Jmax*2+1 < len(self.gamma) / Q_ratio:
                break
            self.fourier_lower_bound *= 10

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

        self.ap_sol_f = self.LU_factorization.solve(self.src_f)
        ext_f = self.extend_inhomogeneous_f()    
        proj_f = self.get_trace(self.get_potential(ext_f))

        rhs = -Q0.dot(c0) - self.get_trace(self.ap_sol_f) - proj_f + ext_f
        c1 = np.linalg.lstsq(Q1, rhs)[0]

        return c0, c1

    def fourier_test(self, c1):
        step = .01
        error = []

        th_range = np.arange(0, 2*np.pi, step)
        for th in th_range:
            x = 0
            for J, i in self.J_dict.items():
                x += c1[i] * np.exp(complex(0, J*th))

            error.append(abs(x - self.problem.eval_expected_derivative(th)))

        print('Fourier error (c1):', max(error))

    def extend(self, r, th, xi0, xi1, d2_xi0_th):
        R = self.R
        k = self.k

        derivs = []
        derivs.append(xi0) 
        derivs.append(xi1)
        derivs.append(-xi1 / R - d2_xi0_th / R**2 - k**2 * xi0)

        v = 0
        for l in range(len(derivs)):
            v += derivs[l] / math.factorial(l) * (r - R)**l

        return v

    # Construct the equation-based extension for the basis function
    # $\psi_{index}^{(J)}$.
    def extend_basis(self, J, index):
        ext = np.zeros(len(self.gamma), dtype=complex)
        for l in range(len(self.gamma)):
            i, j = self.gamma[l]
            r, th = self.get_polar(i, j)
            exp = np.exp(complex(0, J*th))

            if index == 0:
                ext[l] = self.extend(r, th, exp, 0, -J**2 * exp)
            else:
                ext[l] = self.extend(r, th, 0, exp, 0)  

        return ext

    def extend_inhomogeneous(self, r, th):
        x = self.R * np.cos(th)
        y = self.R * np.sin(th)
        return .5 * (r - self.R)**2 * self.problem.eval_f(x, y)

    def extend_inhomogeneous_f(self):
        ext = np.zeros(len(self.gamma), dtype=complex)
        for l in range(len(self.gamma)):
            i, j = self.gamma[l]
            r, th = self.get_polar(i, j)
            ext[l] = self.extend_inhomogeneous(r, th)

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

            xi0 = xi1 = d2_xi0_th = 0 

            for J, i in self.J_dict.items():
                exp = cmath.exp(complex(0, J*th))
                xi0 += c0[i] * exp 
                xi1 += c1[i] * exp
                d2_xi0_th += -J**2 * c0[i] * exp

            boundary[l] = self.extend(r, th, xi0, xi1, d2_xi0_th)
            boundary[l] += self.extend_inhomogeneous(r, th)

        return boundary

    def run(self):
        c0, c1 = self.get_c()
        if self.verbose:
            self.fourier_test(c1)
            print('Using {} basis functions.'.
                format(len(self.J_dict)))

        ext = self.extend_boundary(c0,c1)
        u_act = self.get_potential(ext) + self.ap_sol_f

        error = self.eval_error(u_act)
        return error
