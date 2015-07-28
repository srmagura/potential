# To allow __main__ in subdirectory
import sys
sys.path.append(sys.path[0] + '/..')

import numpy as np
from tabulate import tabulate

import sobolev
from problems.sing_h import SingH_Var_Sine8

m_max = 7

def do_test_N(N):
    def do_test_m(m):
        def do_test_norm(norm):
            problem = SingH_Var_Sine8()

            ps = problem.get_solver(N, {'m_list': [m], 'norm': norm})
            n_basis_tuple = problem.get_n_basis(N)
            ps.setup_basis(*n_basis_tuple)
            ps.ap_sol_f = ps.LU_factorization.solve(ps.B_src_f)

            ps.calc_c0()
            ps.solve_var()

            return problem.b_coef[m-1]

        return (do_test_norm('l2'), do_test_norm('sobolev'))

    table = []

    for m in range(1, m_max+1):
        b_l2, b_sobolev = do_test_m(m)
        assert b_l2.imag == 0
        assert b_sobolev.imag == 0
        table.append([m, b_l2.real, b_sobolev.real, abs(b_sobolev / b_l2)])

    print('N={}'.format(N))
    headers = ('m', 'b_m (l2)', 'b_m (Sobolev)', 'Sobolev / l2')
    print(tabulate(table, headers=headers))
    print()

N_list = (16, 32, 64, 128, 256)
for N in N_list:
    do_test_N(N)
