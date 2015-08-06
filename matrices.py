"""
Second and fourth order finite difference schemes for the Helholtz
equation.
"""

import itertools as it
import numpy as np
import scipy.sparse

def validate_order(order):
    if order not in {2, 4}:
        raise Exception('Invalid order')

def get_index(N, i, j):
    """
    Mapping (i, j) -> k where i and j are in {1, ..., N-1}
    and k is in {0, ..., (N-1)**2}
    """
    return (i-1)*(N-1) + j-1


def build_matrix(update_local, order, N, AD_len, k):
    """
    Builds a CSC matrix using the given function `update_local()`.
    """
    validate_order(order)

    h = AD_len / N
    h2 = h**2
    k2 = k**2

    row_index = []
    col_index = []
    data = []

    for i,j in it.product(range(1, N), repeat=2):
        index = get_index(N, i, j)

        local = np.zeros([3, 3])
        update_local(order, local, k2, h2)

        for di, dj in it.product((-1, 0, 1), repeat=2):
            i1 = i + di
            j1 = j + dj

            if 0 < i1 and i1 < N and 0 < j1 and j1 < N:
                row_index.append(get_index(N, i1, j1))
                col_index.append(index)
                data.append(local[di, dj])

    L = scipy.sparse.coo_matrix((data, (row_index, col_index)),
        shape=((N-1)**2, (N-1)**2), dtype=complex)
    return L.tocsc()

def get_L(order, N, AD_len, k):
    """
    Get the linear operator for the LHS

    order: order of the scheme, 2 or 4
    N: size of grid
    AD_len: sidelength of auxiliary domain
    k: wavenumber
    """
    return build_matrix(update_local_L, order, N, AD_len, k)

def update_local_L(order, local, k2, h2):
    update_local_L2(local, k2, h2)

    if order == 4:
        update_local_L4(local, k2, h2)

def update_local_L2(local, k2, h2):
    local[1, 0] += 1 / h2
    local[0, 0] += -2 / h2
    local[-1, 0] += 1 / h2

    local[0, 1] += 1 / h2
    local[0, 0] += -2 / h2
    local[0, -1] += 1 / h2

    local[0, 0] += k2

def update_local_L4(local, k2, h2):
    _local = np.zeros([3, 3])
    _local[1, 1] += 1 / h2
    _local[1, 0] += -2 / h2
    _local[1, -1] += 1 / h2

    _local[0, 1] += -2 / h2
    _local[0, 0] += 4 / h2
    _local[0, -1] += -2 / h2

    _local[-1, 1] += 1 / h2
    _local[-1, 0] += -2 / h2
    _local[-1, -1] += 1 / h2

    local += _local / 6

    _local = np.zeros([3, 3])
    _local[1, 0] += k2
    _local[0, 0] += -2 * k2
    _local[-1, 0] += k2

    _local[0, 1] += k2
    _local[0, 0] += -2 * k2
    _local[0, -1] += k2

    local += _local / 12

def get_B(order, N, AD_len, k):
    """
    Get the linear operator for the RHS

    order: order of the scheme, 2 or 4
    N: size of grid
    AD_len: sidelength of auxiliary domain
    k: wavenumber
    """
    return build_matrix(update_local_B, order, N, AD_len, k)

def update_local_B(order, local, k2, h2):
    local[0, 0] = 1

    if order == 4:
        update_local_B4(local, k2, h2)

def update_local_B4(local, k2, h2):
    _local = np.zeros([3, 3])
    _local[1, 0] += 1
    _local[0, 0] += -2
    _local[-1, 0] += 1

    _local[0, 1] += 1
    _local[0, 0] += -2
    _local[0, -1] += 1

    local += _local / 12
