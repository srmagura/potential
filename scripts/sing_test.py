# To allow __main__ in subdirectory
import sys
sys.path.append(sys.path[0] + '/..')

import numpy as np
from tabulate import tabulate

import sobolev
from problems.sing_test import SingTest

m_max = 10

l2_prev_norm = np.zeros(m_max)
sobolev_prev_norm = np.zeros(m_max)

def do_test_N(N, first_time):
    def do_test_m(m):
        problem = SingTest()
        problem.m = m

        ps = problem.get_solver(N)
        n_basis_tuple = problem.get_n_basis(N)
        ps.setup_basis(*n_basis_tuple)

        ps.calc_c0()
        ps.calc_c1_exact()

        ext = ps.extend_boundary()
        potential = ps.get_potential(ext)
        projection = ps.get_trace(potential)

        bep = projection - ext

        h = ps.AD_len / N
        sa = 0.5

        l2_norm = np.sqrt(np.vdot(bep, bep)).real
        sobolev_norm = sobolev.eval_norm(h, ps.union_gamma, sa, bep).real

        return (l2_norm, sobolev_norm)

    table = []

    for m in range(1, m_max+1):
        l2_norm, sobolev_norm = do_test_m(m)

        if first_time:
            l2_conv = ''
            sobolev_conv = ''
        else:
            l2_conv = np.log2(l2_prev_norm[m-1]/l2_norm)
            sobolev_conv = np.log2(sobolev_prev_norm[m-1]/sobolev_norm)

        l2_prev_norm[m-1] = l2_norm
        sobolev_prev_norm[m-1] = sobolev_norm

        table.append([m, l2_norm, l2_conv, sobolev_norm, sobolev_conv])

    print('N={}'.format(N))
    headers = ('m', 'l2', 'l2 conv', 'Sobolev', 'Sobolev conv')
    print(tabulate(table, headers=headers))
    print()

N_list = (16, 32, 64, 128)
for N in N_list:
    do_test_N(N, N == N_list[0])
