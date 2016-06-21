import warnings

import numpy as np
from scipy.special import jv

from problems.problem import PizzaProblem

n_nodes = 512
th_data = np.linspace(PizzaProblem.a, 2*np.pi, n_nodes + 2)[1:-1]

def calc_a_coef(problem, boundary, eval_bc0, M, m1, to_subtract=None):
    k = problem.k
    R = problem.R
    a = problem.a
    nu = problem.nu

    phi0_data = np.zeros(n_nodes, dtype=complex)
    W = np.zeros((n_nodes, m1), dtype=complex)

    for i in range(n_nodes):
        th = th_data[i]
        r = boundary.eval_r(th)

        phi0_data[i] = eval_bc0(th)
        for m in range(1, m1+1):
            W[i, m-1] = jv(m*nu, k*r) * np.sin(m*nu*(th-a))

    if to_subtract is not None:
        phi0_data -= to_subtract

    for m in range(M+1, m1+1):
        W[:, m-1] = W[:, m-1] / jv(m*nu, k*R)

    a_coef, residuals, rank, s = np.linalg.lstsq(W, phi0_data)

    if rank != m1:
        warnings.warn('Rank deficient')

    return a_coef[:M], s

def a_to_b(a_coef, k, R, nu):
    return _convert_ab(a_coef, 'a', k, R, nu)

def b_to_a(b_coef, k, R, nu):
    return _convert_ab(b_coef, 'b', k, R, nu)

def _convert_ab(coef, coef_type, k, R, nu):
    assert coef_type in ('a', 'b')

    other_coef = np.zeros(len(coef), dtype=complex)

    for m in range(1, len(coef)+1):
        factor = jv(m*nu, k*R)
        if coef_type == 'b':
            factor = 1/factor

        other_coef[m-1] = coef[m-1] * factor

    return other_coef
