import numpy as np
from scipy.special import jv

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
