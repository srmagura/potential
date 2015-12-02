import sys
import itertools as it

import numpy as np
from scipy.fftpack import dst

eigenval_params = None

def calc_eigenvals(N, order, AD_len, k):
    global eigenvals, eigenval_params

    eigenvals = np.zeros((N-1, N-1))
    eigenval_params = (N, order, AD_len, k)

    h = AD_len / N

    for j, l in it.product(range(1, N), repeat=2):
        if order == 2:
            eigenvals[j-1, l-1] = k**2 + (2*np.cos(np.pi*j/N) +
                2*np.cos(np.pi*l/N) - 4)/h**2
        if order == 4:
            eigenvals[j-1, l-1] = (h**2*k**2*(np.cos(np.pi*j/N) +
                np.cos(np.pi*l/N) + 4) +
                4*np.cos(np.pi*j/N)*np.cos(np.pi*l/N) + 8*np.cos(np.pi*j/N) +
                8*np.cos(np.pi*l/N) - 20)/(6*h**2)


def _ap(Bf, solve, order, AD_len, k):
    N = Bf.shape[0] + 1
    h = AD_len / N

    if eigenval_params is None:
        calc_eigenvals(N, order, AD_len, k)
    elif (N, order, AD_len, k) != eigenval_params:
        print(N, order, AD_len, k)
        print('Eigenvalue cache problem. Exiting...')
        sys.exit(1)

    scoef = np.zeros((N-1, N-1), dtype=complex)

    for i in range(1, N):
        scoef[i-1, :] = dst(Bf[i-1, :], type=1)
    for j in range(1, N):
        scoef[:, j-1] = dst(scoef[:, j-1], type=1)

    scoef *= h**2 / 2

    if solve:
        fourier_coef_sol = scoef / eigenvals
    else:
        fourier_coef_sol = scoef * eigenvals

    u_f = np.zeros((N-1, N-1), dtype=complex)
    for i in range(1, N):
        u_f[i-1, :] = dst(fourier_coef_sol[i-1, :], type=1)
    for j in range(1, N):
        u_f[:, j-1] = dst(u_f[:, j-1], type=1)

    u_f /= (2*AD_len**2)
    return u_f

def solve(Bf, order, AD_len, k):
    return _ap(Bf, True, order, AD_len, k)

def apply_L(v, order, AD_len, k):
    return _ap(v, False, order, AD_len, k)
