import numpy as np
from scipy.special import jv

def calc_a_coef(problem, boundary, eval_phi0, M, m1):
    k = problem.k
    R = problem.R
    a = problem.a
    nu = problem.nu

    n_nodes = 2048
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

    a_coef, residuals, rank, s = np.linalg.lstsq(W, phi0_data)

    if rank != m1:
        print('\n!!!! Rank deficient !!!!\n')

    return a_coef[:M], s
