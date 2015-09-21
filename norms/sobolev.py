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

    return ip_array
