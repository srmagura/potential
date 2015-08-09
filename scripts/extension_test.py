"""
Test the convergence rate of the extension.

usage: extension_test.py [-h] [-N N] [-c C] problem
"""
# To allow __main__ in subdirectory
import sys
sys.path.append(sys.path[0] + '/..')

import numpy as np
import argparse
import itertools as it

import problems
import interface

setypes = None

def set_setypes(ps):
    if args.s is None:
        return

    global setypes
    etype_dict = {
        'l': ps.etypes['left'],
        's': ps.etypes['standard'],
        'r': ps.etypes['right'],
    }

    setypes = set()
    for se in args.s:
        s = int(se[0])
        e = etype_dict[se[1]]
        setypes.add((s, e))

    return setypes

def run_test(N):
    ps = problem.get_solver(N)

    if setypes is None:
        set_setypes(ps)

    n_basis_tuple = problem.get_n_basis(N)
    ps.setup_basis(*n_basis_tuple)

    ps.calc_c0()
    ps.calc_c1_exact()

    mv_ext = ps.mv_extend_boundary()
    error = []

    for node in ps.union_gamma:
        r, th = ps.get_polar(*node)

        for data in mv_ext[node]:
            if setypes is None or data['setype'] in setypes:
                #exp = problem.eval_expected_polar(r, th, setype=data['setype'])
                exp = problem.eval_expected_polar(r, th)
                diff = abs(exp - data['value'])
                error.append(diff)

    return np.max(np.abs(error))


prec_str = '{:.5}'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    problem_choices = problems.problem_dict.keys()
    interface.add_arguments(parser, ('problem', 'N'))
    parser.add_argument('-c', type=int, default=128,
        help='run convergence test, up to the C x C grid. '
                'Default is 128.')

    setype_choices = []
    for s, e in it.product('012', 'lsr'):
        setype_choices.append(s + e)

    parser.add_argument('-s', nargs='*',
        choices=setype_choices,
        help='setypes to check when computing the error'
    )

    args = parser.parse_args()
    problem = problems.problem_dict[args.problem]()

    N = args.N
    prev_error = None

    while N <= args.c:
        print('---- {0} x {0} ----'.format(N))

        error = run_test(N)
        print('Error: ' + prec_str.format(error))

        if prev_error is not None:
            convergence = np.log2(prev_error / error)
            print('Convergence: ' + prec_str.format(convergence))

        print()

        prev_error = error
        N *= 2
