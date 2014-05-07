import itertools as it
import numpy as np
import scipy.sparse

# Mapping (i,j) -> k where i and j are in 1 ... N-1 
# and k is in 0 ... (N-1)**2
def get_index(N, i, j):
    return (i-1)*(N-1) + j-1

def get_L(order, N, AD_len, k):
    h = AD_len / N
    h2 = h**2

    row_index = []
    col_index = []
    data = []

    for i,j in it.product(range(1, N), repeat=2):
        index = get_index(N, i, j)

        local = np.zeros([3, 3])
        update_local2(local, k, h2)

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

def update_local2(local, k, h2):
    local[1, 0] += 1 / h2
    local[0, 0] += -2 / h2
    local[-1, 0] += 1 / h2

    local[0, 1] += 1 / h2
    local[0, 0] += -2 / h2
    local[0, -1] += 1 / h2

    local[0, 0] += k**2
