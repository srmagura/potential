import itertools as it
import math
import collections
import numpy as np
import matplotlib.pyplot as plt

from solver import Solver
import matrices

# Grid size for boundary. Should be divisible by 2
fourier_N = 1024

Q_ratio_upper = 2
Q_ratio_lower = 6

class CircleSolver(Solver):
    # After performing FFT on boundary data, ignore Fourier series
    # coefficients lower than this number
    fourier_lower_bound = 5e-15

    # Side length of square domain on which AP is solved
    AD_len = 2*np.pi

    # Radius of circular region Omega
    R = 2.3

    def __init__(self, problem, N, scheme_order, **kwargs):
        super().__init__(problem, N, scheme_order, **kwargs)
        problem.R = self.R

    def is_interior(self, i, j):
        return self.get_polar(i, j)[0] <= self.R

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
    def calc_c(self):
        grid = np.arange(0, 2*np.pi, 2*np.pi / fourier_N)
        discrete_phi = [self.problem.eval_bc(th) for th in grid]
        c0_raw = np.fft.fft(discrete_phi)

        while True: 
            Jminus = self.choose_n_basis(range(-fourier_N // 2, 1), c0_raw)
            Jplus = self.choose_n_basis(range(fourier_N // 2, 0, -1), c0_raw)
            Jmax = max(Jminus, Jplus)

            if Jmax*2+1 < len(self.gamma) / Q_ratio_upper:
                break
            self.fourier_lower_bound *= 10

        if Jmax*2+1 < len(self.gamma) / Q_ratio_lower:
            Jmax = min(15, int(len(self.gamma) / Q_ratio_lower))

        self.J_dict = collections.OrderedDict(((J, None) for J in 
            range(-Jmax, Jmax+1)))

        i = 0
        self.c0 = np.zeros(len(self.J_dict), dtype=complex)

        for J in self.J_dict:
            self.J_dict[J] = i
            self.c0[i] = c0_raw[J] / fourier_N
            i += 1

        Q0 = self.get_Q(0)
        Q1 = self.get_Q(1)

        self.ap_sol_f = self.LU_factorization.solve(self.B_src_f)
        ext_f = self.extend_inhomogeneous_f()    
        proj_f = self.get_trace(self.get_potential(ext_f))

        rhs = -Q0.dot(self.c0) - self.get_trace(self.ap_sol_f) - proj_f + ext_f
        self.c1 = np.linalg.lstsq(Q1, rhs)[0]

    def extend(self, r, th, xi0, xi1, d2_xi0_th, d2_xi1_th, d4_xi0_th):
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

    # Construct the equation-based extension for the basis function
    # $\psi_{index}^{(J)}$.
    def extend_basis(self, J, index):
        ext = np.zeros(len(self.gamma), dtype=complex)
        for l in range(len(self.gamma)):
            i, j = self.gamma[l]
            r, th = self.get_polar(i, j)
            exp = np.exp(complex(0, J*th))

            if index == 0:
                ext[l] = self.extend(r, th, exp, 0, -J**2 * exp, 0, J**4 * exp)
            else:
                ext[l] = self.extend(r, th, 0, exp, 0, -J**2 * exp, 0)  

        return ext

    def extend_inhomogeneous(self, r, th):
        p = self.problem
        if p.homogeneous:
            return 0

        R = self.R
        k = self.k

        x = R * np.cos(th)
        y = R * np.sin(th)

        derivs = [0, 0]
        derivs.append(p.eval_f(x, y))
        derivs.append(p.eval_d_f_r(R, th) - p.eval_f(x, y) / R)
        derivs.append(p.eval_d2_f_r(R, th) - 
            p.eval_d2_f_th(R, th) / R**2 - 
            p.eval_d_f_r(R, th) / R + 
            (3/R**2 - k**2) * p.eval_f(x, y))

        v = 0
        for l in range(len(derivs)):
            v += derivs[l] / math.factorial(l) * (r - R)**l

        return v

    def extend_inhomogeneous_f(self):
        ext = np.zeros(len(self.gamma), dtype=complex)
        for l in range(len(self.gamma)):
            i, j = self.gamma[l]
            r, th = self.get_polar(i, j)
            ext[l] = self.extend_inhomogeneous(r, th)

        return ext

    def extend_src_f(self):
        p = self.problem
        if p.homogeneous:
            return

        for i,j in self.Kplus - self.Mplus:
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

    # Construct the equation-based extension for the boundary data,
    # as approximated by the Fourier coefficients c0 and c1.
    def extend_boundary(self): 
        boundary = np.zeros(len(self.gamma), dtype=complex)

        for l in range(len(self.gamma)):
            i, j = self.gamma[l]
            r, th = self.get_polar(i, j)

            xi0 = xi1 = 0
            d2_xi0_th = d2_xi1_th = 0 
            d4_xi0_th = 0

            for J, i in self.J_dict.items():
                exp = np.exp(complex(0, J*th))
                xi0 += self.c0[i] * exp 
                d2_xi0_th += -J**2 * self.c0[i] * exp
                d4_xi0_th += J**4 * self.c0[i] * exp

                xi1 += self.c1[i] * exp
                d2_xi1_th += -J**2 * self.c1[i] * exp

            boundary[l] = self.extend(r, th, xi0, xi1, d2_xi0_th, d2_xi1_th,
                d4_xi0_th)
            boundary[l] += self.extend_inhomogeneous(r, th)

        return boundary

    def run(self):
        self.calc_c()
        if self.verbose:
            print('Using {} basis functions.'.
                format(len(self.J_dict)))
            self.c1_test()

        ext = self.extend_boundary()
        u_act = self.get_potential(ext) + self.ap_sol_f

        error = self.eval_error(u_act)
        return error


    ## DEBUGGING FUNCTIONS ##

    def c0_test(self):
        n = 200
        th_data = np.linspace(0, 2*np.pi, n)
        exact_data = np.zeros(n)
        expansion_data = np.zeros(n)

        for l in range(n):
            th = th_data[l]

            exact_data[l] = self.problem.eval_bc(th).real
            for J, i in self.J_dict.items():
                exp = np.exp(complex(0, J*th))
                expansion_data[l] += (self.c0[i] * exp).real

        plt.plot(th_data, exact_data, label='Exact')
        plt.plot(th_data, expansion_data, 'o', label='Expansion')
        plt.legend()
        plt.title('c0')
        plt.show()

    def c1_test(self):
        n = 200
        th_data = np.linspace(0, 2*np.pi, n)
        exact_data = np.zeros(n)
        expansion_data = np.zeros(n)

        for l in range(n):
            th = th_data[l]

            exact_data[l] = self.problem.eval_d_u_r(th).real
            for J, i in self.J_dict.items():
                exp = np.exp(complex(0, J*th))
                expansion_data[l] += (self.c1[i] * exp).real

        plt.plot(th_data, exact_data, label='Exact')
        plt.plot(th_data, expansion_data, 'o', label='Expansion')
        plt.legend()
        plt.title('c1')
        plt.show()
