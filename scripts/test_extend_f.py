"""
Test the accuracy of extend_f(), which extends the source term to slightly
outside the domain.
"""
# To allow __main__ in subdirectory
import sys
sys.path.append(sys.path[0] + '/..')

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

    #if setypes is None:
    #    set_setypes()

    th_list = []
    error = []

    for node in solver.union_gamma:
        r, th = solver.get_polar(*node)

        #for data in mv_ext[node]:
        #    setype = data['setype']
        #    sid, etype = setype

        #    if setypes is None or setype in setypes:
        exp = problem.eval_f_polar(r, th)

        diff = abs(exp - solver.f[node])
        error.append(diff)

    return np.max(np.abs(error))


prec_str = '{:.5}'

if __name__ == '__main__':
    #parser = argparse.ArgumentParser()

    #setype_choices = []
    #for s, e in it.product('012', 'lsr'):
    #    setype_choices.append(s + e)

    #parser.add_argument('-s', nargs='+',
    #    choices=setype_choices,
    #    help='setypes to check when computing the error'
    #)

    #args = parser.parse_args()
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
    c = 128
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
