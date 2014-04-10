# Python standard library
import sys
import itertools as it
import cmath
import math
import time
import collections

# NumPy and SciPy
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt

# Grid size for boundary. Should be divisible by 2
fourier_N = 1024

# After performing FFT on boundary data, ignore Fourier series
# coefficients lower than this number
fourier_lower_bound = 5e-15

plot_grid = False
plot_c1_error = False

# Side length of square domain on which AP is solved
AD_len = 2*np.pi

# Radius of circular region Omega
R = 2.3

# Wave number
k = 1 
kx = .8*k
ky = .6*k

# Boundary data
def phi(th):
    return eval_expected(R*np.cos(th), R*np.sin(th))

# Expected solution
def eval_expected(x, y):
    return cmath.exp(complex(0, kx*x + ky*y))

# Expected normal derivative on boundary
def eval_expected_derivative(th):
    x = complex(0, kx*np.cos(th) + ky*np.sin(th))
    return x*cmath.exp(R*x)

# Mapping (i,j) -> k where i and j are in 1 ... N-1 
# and k is in 0 ... (N-1)**2
def get_index(i, j):
    return (i-1)*(N-1) + j-1

# Get the polar coordinates of grid point (i,j)
def get_polar(i, j):
    x, y = get_coord(i, j)
    return math.hypot(x, y), math.atan2(y, x)

# Get the rectangular coordinates of grid point (i,j)
def get_coord(i, j):
    x= (AD_len * i / N - AD_len/2, AD_len * j / N - AD_len/2)
    return x

# Return the discrete differential operator L as a sparse matrix.
def build_L():
    global L
    h = AD_len / N
    H = h**-2

    row_index = []
    col_index = []
    data = []

    for i,j in it.product(range(1, N), repeat=2):
        index = get_index(i, j)

        row_index.append(index)
        col_index.append(index)
        data.append(-4*H + k**2)  

        if j > 1: 
            col_index.append(get_index(i, j-1))
            row_index.append(index)
            data.append(H)
        if j < N-1: 
            col_index.append(get_index(i, j+1))
            row_index.append(index)
            data.append(H)  
        if i > 1: 
            row_index.append(get_index(i-1, j))
            col_index.append(index)
            data.append(H)  
        if i < N-1: 
            row_index.append(get_index(i+1, j))
            col_index.append(index)
            data.append(H)  

    L = scipy.sparse.coo_matrix((data, (row_index, col_index)),
        shape=((N-1)**2, (N-1)**2), dtype=complex)
    L = L.tocsc()

    if verbose:
        print('Sparse matrix has {} entries.'.format(L.nnz))


# Calculate the difference potential of a function xi that is defined on gamma.
def get_potential(xi):
    w = np.zeros([(N-1)**2], dtype=complex)

    for l in range(len(gamma)):
        w[get_index(*gamma[l])] = xi[l]

    Lw = np.ravel(L.dot(w))

    for i,j in Mminus:
        Lw[get_index(i,j)] = 0

    return w - LU_factorization.solve(Lw)

def get_projection(potential):
    projection = np.zeros(len(gamma), dtype=complex)

    for l in range(len(gamma)):
        index = get_index(*gamma[l])
        projection[l] = potential[index]

    return projection

# Returns the matrix Q0 or Q1, depending on the value of `index`.
def get_Q(index):
    columns = []

    for J in J_dict:
        ext = extend_basis(J, index)
        potential = get_potential(ext)
        projection = get_projection(potential)

        columns.append(projection - ext)

    return np.column_stack(columns)

# Choose which Fourier coefficients to keep
def choose_n_basis(J_range, c0_raw):
    for J in J_range:
        normalized = c0_raw[J] / fourier_N

        if abs(normalized) > fourier_lower_bound:
            return abs(J)

# Applies FFT to the boundary data to get c0, then solves the 
# system Q1 * c1 = -Q0 * c0 by least squares to find c1. 
def get_c():
    global J_dict

    grid = np.arange(0, 2*np.pi, 2*np.pi / fourier_N)
    discrete_phi = [phi(th) for th in grid]
    c0_raw = np.fft.fft(discrete_phi)

    Jminus = choose_n_basis(range(-fourier_N // 2, 1), c0_raw)
    Jplus = choose_n_basis(range(fourier_N // 2, 0, -1), c0_raw)
    Jmax = max(Jminus, Jplus)

    J_dict = collections.OrderedDict(((J, None) for J in range(-Jmax, Jmax+1)))

    i = 0
    c0 = np.zeros(len(J_dict), dtype=complex)

    for J in J_dict:
        J_dict[J] = i
        c0[i] = c0_raw[J] / fourier_N
        i += 1

    if verbose:
        print('Using {} basis functions.'.format(len(J_dict)))

    Q0 = get_Q(0)
    Q1 = get_Q(1)

    c1 = np.linalg.lstsq(Q1, -Q0.dot(c0))[0]

    if verbose:
        fourier_test(c0, 'c0', phi)
        fourier_test(c1, 'c1', eval_expected_derivative)

    return c0, c1

# Prints the error caused by truncating the Fourier series.
def fourier_test(c, name, expected):
    error = []
    exp = []
    act = []
    step = .01

    th_range = np.arange(0, 2*np.pi, step)
    for th in th_range:
        x = 0
        for J in J_dict:
            x += c[J_dict[J]] * cmath.exp(complex(0, J*th))

        error.append(abs(x - expected(th)))
        act.append(x)
        exp.append(expected(th))

    if plot_c1_error and expected == eval_expected_derivative:
        plt.plot(th_range, error, marker='o')
        plt.show()

    print('Fourier error ({}): '.format(name), max(error))


def _v0(J, r, th):
    x0 = 1
    x1 = 0
    x2 = (J/R)**2 - k**2
    #x3 = -3*J**2/R**3 + k**2/R
    x3 = 0

    return cmath.exp(complex(0, J*th)) *\
            (x0 + x1 * (r-R) + .5 * x2 * (r-R)**2 +
             1/6 * x3 * (r-R)**3)

def _v1(J, r, th):
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
def extend_basis(J, index):
    ext = np.zeros(len(gamma), dtype=complex)
    for l in range(len(gamma)):
        i, j = gamma[l]
        r, th = get_polar(i, j)

        if index == 0:
            ext[l] = _v0(J, r, th)
        else:
            ext[l] = _v1(J, r, th)  

    return ext

# Construct the equation-based extension for the boundary data,
# as approximated by the Fourier coefficients c0 and c1.
def extend_boundary(c0, c1): 
    boundary = np.zeros(len(gamma), dtype=complex)

    for l in range(len(gamma)):
        i, j = gamma[l]
        r, th = get_polar(i, j)

        v0 = v1 = xi0d = xi1d = 0 

        for J, i in J_dict.items():
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
def construct_grids():
    global N0, M0, Mplus, Mminus, Nplus, Nminus, gamma

    N0 = set(it.product(range(0, N+1), repeat=2))
    M0 = set(it.product(range(1, N), repeat=2))

    #K0 = set()
    #for i, j in it.product(range(0, N+1), repeat=2):
    #    if(i != 0 and i != N) or (j != 0 and j != N):
    #        K0.add((i, j))

    Mplus = {(i, j) for i, j in M0 if get_polar(i, j)[0] <= R}
    Mminus = M0 - Mplus

    Nplus = set()
    Nminus = set()
           
    for i, j in M0:
        Nm = set([(i, j), (i-1, j), (i+1, j), (i, j-1), (i, j+1)])

        if (i, j) in Mplus:
            Nplus |= Nm
        else:
            Nminus |= Nm

    gamma = list(Nplus & Nminus)

    def convert(points):
        x = []
        y = []

        for p in points:
            x1, y1 = get_coord(*p)
            x.append(x1)
            y.append(y1)

        return x, y

    if plot_grid:
        plt.plot(*convert(M0), marker='o', label='M0', ms=1, linestyle='',
            color='black')
        plt.plot(*convert(gamma), marker='^', label='M0', ms=4, linestyle='',
            color='red')
        #plt.plot(*convert(Mplus), marker='o', label='Mplus', linestyle='',
        #    color='gray')
        #plt.plot(*convert(Nplus), marker='x', label='Nplus', ms=9, linestyle='',
        #    color='black')

        xdata = []
        ydata = []
        for th in np.linspace(0, 4*np.pi, 2000):
            xdata.append(R*np.cos(th))
            ydata.append(R*np.sin(th))

        plt.plot(xdata, ydata, color='blue')
        plt.xlim(-4,4)
        plt.ylim(-4,4)
        #plt.legend()
        plt.show()

# Returns || u_act - u_exp ||_inf, the error between actual and expected
# solutions under the infinity-norm. 
def eval_error(u_act):
    u_exp = np.zeros((N-1)**2, dtype=complex)
    for i,j in M0:
        u_exp[get_index(i,j)] = eval_expected(*get_coord(i,j))

    error = []
    error1 = []
    Luact = np.abs(L.dot(u_act))
    for i,j in Mplus:
        l = get_index(i,j)
        error.append(abs(u_exp[l] - u_act[l]))
        error1.append(Luact[get_index(i,j)])

    return max(error)

# Run the algorithm. N has already been set at this point
def run():
    global L, LU_factorization

    construct_grids()
    if verbose:
        print('Grid is {0} x {0}.'.format(N))

    build_L()
    LU_factorization = scipy.sparse.linalg.splu(L)

    c0, c1 = get_c()

    ext = extend_boundary(c0,c1)
    u_act = get_potential(ext)

    error = eval_error(u_act)
    if verbose:
        print('Error:', error)

    return error

# Test the rate of convergence
def test_convergence():
    global N
    prev_error = None

    for N in (16, 32, 64, 128, 256,512, 1024): 
        start_time = time.time()
        error = run()
        time_taken = time.time() - start_time

        print('---- {0} x {0} ----'.format(N)) 
        print('Error:', error)

        if prev_error is not None:
            print('Convergence:', np.log2(prev_error / error))
            # log_2(prev_error/ error)

        #print('Time: {}s'.format(round(time_taken,3)))
        prev_error = error
        print()

# Main function
if __name__ == '__main__':
    N = 0
    verbose = True

    if len(sys.argv) > 1:
        try:
            N = int(sys.argv[1])
        except ValueError:
            verbose = False
            plot_grid = False
            test_convergence()
               
    if N <= 0:
        N = 16

    if verbose:
        run()
