import itertools as it
import cmath
import math
import collections
import numpy as np

# Grid size for boundary. Should be divisible by 2
fourier_N = 1024

# After performing FFT on boundary data, ignore Fourier series
# coefficients lower than this number
fourier_lower_bound = 5e-15

#plot_grid = False
plot_c1_error = False

class CircularDomain:
    # Side length of square domain on which AP is solved
    AD_len = 2*np.pi

    # Radius of circular region Omega
    R = 2.3

    def __init__(self, problem):
        self.problem = problem

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
    def get_c0(self):
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

        return c0

        #if verbose:
        #    fourier_test(c0, 'c0', self.problem.eval_expected)
        #    fourier_test(c1, 'c1', 
        #        self.problem.eval_expected_derivative)

    # Prints the error caused by truncating the Fourier series.
#    def fourier_test(self, c, name, expected):
#        error = []
#        exp = []
#        act = []
#        step = .01

#        th_range = np.arange(0, 2*np.pi, step)
#        for th in th_range:
#            x = 0
#            for J in J_dict:
#                x += c[J_dict[J]] * cmath.exp(complex(0, J*th))

#            error.append(abs(x - expected(th)))
#            act.append(x)
#            exp.append(expected(th))

#        print('Fourier error ({}): '.format(name), max(error))


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

    # Construct the various grids used in the algorithm.
    def construct_grids(self, N):
        self.N = N
        self.N0 = set(it.product(range(0, N+1), repeat=2))
        self.M0 = set(it.product(range(1, N), repeat=2))

        #K0 = set()
        #for i, j in it.product(range(0, N+1), repeat=2):
        #    if(i != 0 and i != N) or (j != 0 and j != N):
        #        K0.add((i, j))

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

#        def convert(points):
#            x = []
#            y = []

#            for p in points:
#                x1, y1 = get_coord(*p)
#                x.append(x1)
#                y.append(y1)

#            return x, y

#        if plot_grid:
#            plt.plot(*convert(M0), marker='o', label='M0', ms=1, linestyle='',
#                color='black')
#            plt.plot(*convert(gamma), marker='^', label='M0', ms=4, linestyle='',
#                color='red')
            #plt.plot(*convert(Mplus), marker='o', label='Mplus', linestyle='',
            #    color='gray')
            #plt.plot(*convert(Nplus), marker='x', label='Nplus', ms=9, linestyle='',
            #    color='black')

#            xdata = []
#            ydata = []
#            for th in np.linspace(0, 4*np.pi, 2000):
#                xdata.append(R*np.cos(th))
#                ydata.append(R*np.sin(th))

#            plt.plot(xdata, ydata, color='blue')
#            plt.xlim(-4,4)
#            plt.ylim(-4,4)
            #plt.legend()
#            plt.show()

