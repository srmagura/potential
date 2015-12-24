# To allow __main__ in subdirectory
import sys
sys.path.append(sys.path[0] + '/..')

import numpy as np
from scipy.fftpack import dst
from scipy.special import jv

from problems.problem import PizzaProblem

def b_to_a(b_coef):
    return _convert_ab(b_coef, 'b')

def _convert_ab(coef, coef_type):
    assert coef_type in ('a', 'b')
    other_coef = np.zeros(len(coef), dtype=complex)

    for m in range(1, len(coef)+1):
        factor = jv(m*nu, k*R)
        if coef_type == 'b':
            if factor == 0:
                continue

            factor = 1/factor

        other_coef[m-1] = coef[m-1] * factor

    return other_coef

def get_a_coef(fourier_N):
    # The slice at the end removes the endpoints
    th_data = np.linspace(a, 2*np.pi, fourier_N+1)[1:-1]

    discrete_phi0 = np.array([eval_phi0(th) for th in th_data])
    b_coef = dst(discrete_phi0, type=1) / fourier_N

    return b_to_a(b_coef)

def get_error(m_max):
    th = np.pi/4
    u_act = 0

    for m in range(1, m_max+1):
        ac = a_coef[m-1]
        u_act += ac * jv(m*nu, k*R) * np.sin(m*nu*(th-a))

    u_exp = eval_phi0(th)
    return abs(u_act - u_exp)

def eval_phi0(th):
    return -(th - np.pi/6) * (th - 2*np.pi)

k = 1
R = 2.3
a = PizzaProblem.a
nu = PizzaProblem.nu

a_coef = get_a_coef(2048)
for m_max in range(5, 500, 5):
    error = get_error(m_max)
    print('{}   |    {}'.format(m_max, error))
