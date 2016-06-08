"""
Test the accuracy of extend_f(), which extends the source term to slightly
outside the domain.
"""
# To allow __main__ in subdirectory
import sys
sys.path.append(sys.path[0] + '/..')
import argparse

import numpy as np

import ps.ps
#from ps.extend import EType

from problems.smooth import Smooth_YCos
from problems.boundary import Arc

import io_util

#setypes = None

"""
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

    return setypes"""

def run_test(N):
    options['N'] = N
    solver = ps.ps.PizzaSolver(options)

    error = []

    for node in solver.Kplus:
        if solver._extend_f_get_sid(*node) in allowed_sid:
            r, th = solver.get_polar(*node)
            diff = abs(problem.eval_f_polar(r, th) - solver.f[node])

            error.append(diff)

            #if diff > 1e-1:
            #    x, y = solver.get_coord(*node)
            #    print('x={}  y={}   diff={}'.format(x, y, diff))
            #    print(problem.eval_f_polar(r, th), solver.f[node])
            #    print()

    return np.max(np.abs(error))


prec_str = '{:.5}'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    sid_choices = list(map(str, range(3)))

    parser.add_argument('-s', nargs='+',
        choices=sid_choices,
        help='setypes to check when computing the error',
        default=sid_choices
    )

    args = parser.parse_args()
    allowed_sid = list(map(int, args.s))

    problem = Smooth_YCos()
    boundary = Arc(problem.R)
    problem.boundary = boundary

    options = {
        'problem': problem,
        'scheme_order': 4
    }

    meta_options = {'procedure_name': 'test_extend_f'}
    io_util.print_options(options, meta_options)

    N = 16
    c = 256
    prev_error = None

    while N <= c:
        print('---- {0} x {0} ----'.format(N))

        error = run_test(N)
        print('Error: ' + prec_str.format(error))

        if prev_error is not None:
            convergence = np.log2(prev_error / error)
            print('Convergence: ' + prec_str.format(convergence))
        print()

        prev_error = error
        N *= 2
