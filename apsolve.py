import numpy as np
from scipy.fftpack import dst

def apsolve(Bf, order, AD_len, k):
    N = Bf.shape[0] + 1
    h = AD_len / N

    scoef = np.zeros((N-1, N-1), dtype=complex)

    for i in range(1, N):
        scoef[i-1, :] = dst(Bf[i-1, :], type=1)
    for j in range(1, N):
        scoef[:, j-1] = dst(scoef[:, j-1], type=1)

    # TODO: optimize arithmetic operations
    scoef *= h**2 / 2

    def get_eigenval(j, l):
        if order == 2:
            return k**2 + (2*np.cos(np.pi*j/N) + 2*np.cos(np.pi*l/N) - 4)/h**2
        if order == 4:
            return (h**2*k**2*(np.cos(np.pi*j/N) + np.cos(np.pi*l/N) + 4) +
                4*np.cos(np.pi*j/N)*np.cos(np.pi*l/N) + 8*np.cos(np.pi*j/N) +
                8*np.cos(np.pi*l/N) - 20)/(6*h**2)

    eigenvals = [[get_eigenval(j, l) for j in range(1, N)]
        for l in range(1, N)]
    fourier_coef_sol = scoef / eigenvals

    u_f = np.zeros((N-1, N-1), dtype=complex)
    for i in range(1, N):
        u_f[i-1, :] = dst(fourier_coef_sol[i-1, :], type=1)
    for j in range(1, N):
        u_f[:, j-1] = dst(u_f[:, j-1], type=1)

    u_f /= 2
    return u_f
