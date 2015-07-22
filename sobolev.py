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

    This function returns the x that minimizes the Sobolev norm
    of the residual.

    A -- 2D numpy array, left-hand side
    b -- 1D numpy array, right-hand side
    h -- cell width for Cartesian grid with square cells
    nodes -- set of nodes on which Ax and b are defined
    sa -- scaling coefficient used in calculating the Sobolev norm
    """
    A = np.matrix(A)
    b = np.matrix(b).T

    evals, U = np.linalg.eig(get_ip_array(h, nodes, sa))
    sqrt_lambda = np.matrix(np.diag(np.sqrt(evals)))
    U = np.matrix(U)
    B = U * sqrt_lambda * U.getH()

    x = np.linalg.lstsq(B*A, B*b)[0]
    return np.ravel(x)
