"""

"""
# To allow __main__ in subdirectory
import sys
sys.path.append(sys.path[0] + '/../..')

import numpy as np

from ps.zmethod import ZMethod
from ps.polarfd import PolarFD

a = np.pi / 6
nu = np.pi / (2*np.pi - a)

R = 2.3

def eval_v(r, th):
    return np.exp(r) * np.sin(th)

prev_error = None

class Boundary:
    pass

boundary = Boundary()
boundary.R = R
boundary.bet = 0

for N in [16, 32, 64, 128]:
    my_zmethod = ZMethod({})
    my_zmethod.N = N
    my_zmethod.boundary = boundary

    my_polarfd = PolarFD(N, a, nu, R)
    my_zmethod.polarfd = my_polarfd

    v = np.zeros([N+1, N+1])
    for m in range(N+1):
        r = my_polarfd.get_r(m)
        for l in range(N+1):
            th = my_polarfd.get_th(l)
            v[m, l] = eval_v(r, th)

    my_zmethod.v = v
    v_interp = my_zmethod.create_v_interp()

    N2 = 4*N
    diff = []
    for r in np.linspace(R*.75, R, N2):
        for th in np.linspace(a, 2*np.pi, N2):
            diff.append(v_interp(r, th) - eval_v(r, th))

    error = np.max(np.abs(diff))
    print('N:', N)
    print('error:', error)

    if prev_error is not None:
        con = np.log2(prev_error/error)
        print('convergence:', con)

    print()

    prev_error = error
