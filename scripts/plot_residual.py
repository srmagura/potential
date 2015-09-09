# To allow __main__ in subdirectory
import sys
sys.path.append(sys.path[0] + '/..')

import numpy as np
from scipy.special import jv
import matplotlib.pyplot as plt
from tabulate import tabulate

from problems.problem import PizzaProblem
from problems.sine import SinePizza

# Used only to construct a PizzaSolver object
fake_problem = SinePizza()

k = 1
m = 1

def eval_expected(m, r, th):
    a = PizzaProblem.a
    nu = PizzaProblem.nu

    if 0 <= th and th < a/2:
        th += 2*np.pi

    return jv(m*nu, k*r) * np.sin(m*nu*(th-a))

def do_test(N):
    ps = fake_problem.get_solver(N)

    ext = np.zeros(len(ps.union_gamma), dtype=complex)

    for l in range(len(ps.union_gamma)):
        node = ps.union_gamma[l]
        r, th = ps.get_polar(*node)
        ext[l] = eval_expected(m, r, th)

    potential = ps.get_potential(ext)
    projection = ps.get_trace(potential)

    bep = projection - ext

    x_data, y_data = ps.nodes_to_plottable(ps.union_gamma)
    r_data = [np.sqrt(x**2 + y**2) for x, y in zip(x_data, y_data)]

    plt.stem(r_data, abs(bep), markerfmt=' ', basefmt='k')

    h = ps.AD_len / (ps.N+1)
    plt.xlim(xmin=-1.5*h)

    fs = 14
    plt.xlabel('Polar radius', fontsize=fs)
    plt.ylabel('Absolute value of BEP residual', fontsize=fs)
    plt.show()


print('k={}'.format(k))
print('m={}'.format(m))
print()

do_test(64)
#N_list = (16, 32, 64, 128,)# 256, 512, 1024)
#for N in N_list:
#    do_test_N(N, N == N_list[0])