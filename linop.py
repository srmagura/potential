import numpy as np
import itertools as it

def apply_L(v, order, AD_len, k):
    """
    Input:
    v -- shape (N+1, N+1)

    Output:
    Lv -- shape (N-1, N-1)
    """
    N = v.shape[0] - 1
    h = AD_len / N

    Lv = np.zeros((N-1, N-1), dtype=complex)

    for m, n in it.product(range(1, N), repeat=2):
        Lv[m-1, n-1] = 1/h**2*(v[m+1, n] + v[m-1, n] + v[m, n+1] +
            v[m, n-1] - 4*v[m, n]) + k**2 * v[m, n]

        if order == 4:
            Lv[m-1, n-1] += (1/(6*h**2) * (
                v[m+1, n+1] - 2*v[m+1, n] + v[m+1, n-1] -
                2 * (v[m, n+1] - 2*v[m, n] + v[m, n-1]) +
                v[m-1, n+1] - 2*v[m-1, n] + v[m-1, n-1]
                ) +
                k**2/12 * (v[m+1, n] - 2*v[m, n] + v[m-1, n] +
                v[m, n+1] - 2*v[m, n] + v[m, n-1]))

    return Lv

def apply_B(f):
    """
    Input:
    f -- shape (N+1, N+1)

    Output:
    Bf -- shape (N-1, N-1)
    """
    N = f.shape[0] - 1
    Bf = np.zeros((N-1, N-1), dtype=complex)

    for i, j in it.product(range(1, N), repeat=2):
        Bf[i-1, j-1] = f[i, j] + 1/12 * (
            f[i+1, j] - 2*f[i, j] + f[i-1, j] +
            f[i, j+1] - 2*f[i, j] + f[i, j-1]
        )

    return Bf
