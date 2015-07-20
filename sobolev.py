import numpy as np

def get_ip_array(h, nodes, sa):
    r"""
    Get the SPD matrix M as that satisfies

    .. math:: \Vert v \Vert**2 = v^* M v,

    where the norm is the Sobolev norm.

    M is actually returned as an ndarray for convenience. The name
    ip_array comes from "inner product array".
    """
    n = len(nodes)
    ip_array = np.zeros((n, n))

    for k in range(n):
        i, j = nodes[k]
        n_adjacent = 0

        for i1, j1 in ((i-1, j), (i+1, j), (i, j-1), (i, j+1)):
            if (i1, j1) in nodes:
                n_adjacent += 1
                l = nodes.index((i1, j1))
                ip_array[k, l] += -sa/h**2

        ip_array[k, k] += 1 + n_adjacent*sa/h**2

    return h*ip_array

def solve_var(A, b, h, nodes, sa):
    """
    Find the optimal solution to the overdetermined linear system
    Ax = b.

    TODO optimal in the sense of a new norm we define
    """
    evals, U = np.linalg.eig(get_ip_array(h, nodes, sa))
    sqrt_lambda = np.matrix(np.diag(np.sqrt(evals)))
    #sqrt_lambda1 = np.matrix(np.diag(1/np.sqrt(evals)))
    U = np.matrix(U)
    B = U * sqrt_lambda * U.getH()
    #B_inv = U.getH() * sqrt_lambda1 * U
    A = np.matrix(A)
    b = np.matrix(b).T

    x = np.linalg.lstsq(B*A, B*b)[0]
    x = np.ravel(x)
    return x
