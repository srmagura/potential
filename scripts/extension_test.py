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

import ps.ps
from ps.extend import EType

import matplotlib.pyplot as plt

import problems
import io_util

setypes = None

def set_setypes():
    if args.s is None:
        return

    global setypes
    etype_dict = {
        'l': EType.left,
        's': EType.standard,
        'r': EType.right,
    }

    setypes = set()
    for se in args.s:
        s = int(se[0])
        e = etype_dict[se[1]]
        setypes.add((s, e))

    return setypes

def run_test(N):
    options['N'] = N
    solver = ps.ps.PizzaSolver(options)

    if setypes is None:
        set_setypes()

    solver.calc_c0()
    solver.c0_test()
    solver.calc_c1_exact()
    solver.c1_test()

    mv_ext = solver.mv_extend_boundary()

    th_list = []
    error = []

    for node in solver.union_gamma:
        r, th = solver.get_polar(*node)

        for data in mv_ext[node]:
            setype = data['setype']
            sid, etype = setype

            if setypes is None or setype in setypes:
                # Need correction for non-2pi-periodic solutions
                #if(sid == 1 and (etype == EType.standard or
                #    etype == EType.right) and th < np.pi):
                #    th += 2*np.pi

                exp = problem.eval_expected_polar(r, th)

                diff = abs(exp - data['value'])
                error.append(diff)
                th_list.append(th)

                #if diff > 1e-1:
                #    print('r={}  th={}  diff={}'.format(r, th, diff))
                #    print('exp={}   act={}'.format(exp, data['value']))

    #plt.plot(th_list, np.log10(error), 'o')
    #plt.show()

    return np.max(np.abs(error))


prec_str = '{:.5}'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    io_util.add_arguments(parser, ('problem', 'boundary', 'N'))

    parser.add_argument('-c', type=int, default=128,
        help='run convergence test, up to the C x C grid. '
        'Default is 128.')

    setype_choices = []
    for s, e in it.product('012', 'lsr'):
        setype_choices.append(s + e)

    parser.add_argument('-s', nargs='+',
        choices=setype_choices,
        help='setypes to check when computing the error'
    )

    args = parser.parse_args()
    problem = problems.problem_dict[args.problem]()
    boundary = problems.boundary.boundaries[args.boundary](problem.R)
    problem.boundary = boundary

    options = {
        'problem': problem,
        'scheme_order': 4
    }

    meta_options = {'procedure_name': 'extension_test'}
    io_util.print_options(options, meta_options)

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
