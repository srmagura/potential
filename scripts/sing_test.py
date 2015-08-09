# To allow __main__ in subdirectory
import sys
sys.path.append(sys.path[0] + '/..')

import numpy as np
from scipy.special import jv
from tabulate import tabulate

import sobolev
from problems.problem import PizzaProblem
from problems.sine import SinePizza
from problems.fbterm import FbTerm

# Used only to construct a PizzaSolver object
fake_problem = SinePizza()

k = 1
m_list = [10]

l2_prev_norm = np.zeros(len(m_list))
sobolev_prev_norm = np.zeros(len(m_list))

def eval_expected(m, r, th):
    a = PizzaProblem.a
    nu = PizzaProblem.nu

    return jv(m*nu, k*r) * np.sin(m*nu*(th-a))

def do_test_N(N, first_time):
    ps = fake_problem.get_solver(N)

    def do_test_m(m):
        problem2 = FbTerm()
        problem2.m = m

        ps2 = problem2.get_solver(N)
        n_basis_tuple = problem2.get_n_basis(N)
        ps2.setup_basis(*n_basis_tuple)

        ps2.calc_c0()
        ps2.calc_c1_exact()

        ext2 = ps2.extend_boundary()

        ext = np.zeros(len(ps.union_gamma), dtype=complex)
        print(np.max(np.abs(ext-ext2)))
        for l in range(len(ps.union_gamma)):
            node = ps.union_gamma[l]
            r, th = ps.get_polar(*node)
            ext[l] = eval_expected(m, r, th)

        potential = ps.get_potential(ext)
        projection = ps.get_trace(potential)

        bep = projection - ext

        h = ps.AD_len / N
        sa = 0.5

        l2_norm = np.sqrt(np.vdot(bep, bep)).real
        sobolev_norm = sobolev.eval_norm(h, ps.union_gamma, sa, bep).real

        return (l2_norm, sobolev_norm)

    table = []

    for i in range(len(m_list)):
        m = m_list[i]
        l2_norm, sobolev_norm = do_test_m(m)

        if first_time:
            l2_conv = ''
            sobolev_conv = ''
        else:
            l2_conv = np.log2(l2_prev_norm[i]/l2_norm)
            sobolev_conv = np.log2(sobolev_prev_norm[i]/sobolev_norm)

        l2_prev_norm[i] = l2_norm
        sobolev_prev_norm[i] = sobolev_norm

        table.append([m, l2_norm, l2_conv, sobolev_norm, sobolev_conv])

    print('N={}'.format(N))
    headers = ('m', 'l2', 'l2 conv', 'Sobolev', 'Sobolev conv')
    #print(tabulate(table, headers=headers))
    print()


print('k={}'.format(k))
print()

N_list = (16, 32, 64, 128,)# 256, 512, 1024)
for N in N_list:
    do_test_N(N, N == N_list[0])
