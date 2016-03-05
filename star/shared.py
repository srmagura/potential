import numpy as np
from scipy.special import jv

def calc_a_coef(eval_phi0, m1, a):
    n_nodes = 1024
    th_data = np.linspace(a, 2*np.pi, n_nodes + 2)[1:-1]

    phi0_data = np.zeros(n_nodes, dtype=complex)
    W = np.zeros((n_nodes, m1), dtype=complex)

    for i in range(n_nodes):
        th = th_data[i]
        r = boundary.eval_r(th)

        phi0_data[i] = eval_phi0(r, th)
        for m in range(1, m1+1):
            W[i, m-1] = jv(m*nu, k*r) * np.sin(m*nu*(th-a))

    for m in range(M+1, m1+1):
        W[:, m-1] = W[:, m-1] / jv(m*nu, k*R)

    result = np.linalg.lstsq(W, phi0_data)

    a_coef = result[0]
    rank = result[2]

    if rank != m1:
        print('\n!!!! Rank deficient !!!!\n')

    return a_coef[:M]
