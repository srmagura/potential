import itertools as it
import scipy.sparse

# Mapping (i,j) -> k where i and j are in 1 ... N-1 
# and k is in 0 ... (N-1)**2
def get_index(N, i, j):
    return (i-1)*(N-1) + j-1

def get_L(N, AD_len, k):
    h = AD_len / N
    H = h**-2

    row_index = []
    col_index = []
    data = []

    for i,j in it.product(range(1, N), repeat=2):
        index = get_index(N, i, j)

        row_index.append(index)
        col_index.append(index)
        data.append(-4*H + k**2)  

        if j > 1: 
            col_index.append(get_index(N, i, j-1))
            row_index.append(index)
            data.append(H)
        if j < N-1: 
            col_index.append(get_index(N, i, j+1))
            row_index.append(index)
            data.append(H)  
        if i > 1: 
            row_index.append(get_index(N, i-1, j))
            col_index.append(index)
            data.append(H)  
        if i < N-1: 
            row_index.append(get_index(N, i+1, j))
            col_index.append(index)
            data.append(H)  

    L = scipy.sparse.coo_matrix((data, (row_index, col_index)),
        shape=((N-1)**2, (N-1)**2), dtype=complex)
    return L.tocsc()
