"""
Test ps.polarfd
"""
import sys
sys.path.append(sys.path[0] + '/../..')

import numpy as np
from scipy.special import jv

from ps.polarfd import PolarFD

from scripts.polarfd.sym import eval_v_asympt, eval_f, eval_d_f_r, eval_d2_f_r, eval_d2_f_th, _k
k = _k

a = np.pi/6
nu = np.pi / (2*np.pi - a)

def eval_u(r, th):
    return jv(nu/2, k*r) * np.sin(nu/2*(th-a)) - eval_v_asympt(r, th)

def get_N_list(N0, c):
    if c is None:
        return [N0]

    N_list = []

    N = N0
    while N <= c:
        N_list.append(N)
        N *= 2

    return N_list

N_list = get_N_list(16, 128)

R1 = 2.3

def eval_phi0(th):
    return eval_u(R1, th)

def eval_phi1(r):
    return eval_u(r, 2*np.pi)

def eval_phi2(r):
    return eval_u(r, a)

def convert(u):
    N = my_polarfd.N
    new = np.zeros((N+1, N+1))
    for m in range(N+1):
        for l in range(N+1):
            new[m, l] = u[my_polarfd.get_index(m, l)]

    return new
"""
Perform the convergence test.
"""
prev_error = None

u2 = None
u1 = None
u0 = None

for N in N_list:
    print('---- {0} x {0} ----'.format(N))
    my_polarfd = PolarFD()
    u = my_polarfd.solve(N, k, a, nu, R1, eval_phi0, eval_phi1, eval_phi2,
        eval_f, eval_d_f_r, eval_d2_f_r, eval_d2_f_th)

    exp = np.zeros((N+1)**2)
    for m in range(N+1):
        r = my_polarfd.get_r(m)
        for l in range(N+1):
            th = my_polarfd.get_th(l)
            exp[my_polarfd.get_index(m, l)] = eval_u(r, th)

    error = np.max(np.abs(u-exp))

    print('Error:', error)

    if prev_error is not None:
        convergence = np.log2(prev_error/error)
        print('Convergence:', convergence)

    u2 = u1
    u1 = u0
    u0 = convert(u)

    if u2 is not None:
        print('Rel convergence:', my_polarfd.calc_rel_convergence(u0, u1, u2))

    print()
    sys.stdout.flush()

    prev_error = error
