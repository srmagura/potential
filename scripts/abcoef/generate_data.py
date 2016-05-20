# To allow __main__ in subdirectory
import sys
sys.path.append(sys.path[0] + '/../..')

import json
import itertools as it

import numpy as np
import scipy

import problems
import problems.boundary

from abcoef import calc_a_coef

M = 7

output = {}

problem_list = ('h-hat', 'h-parabola', 'h-line-sine',)
boundary_list = problems.boundary.boundaries.keys()

m1_list = range(8, 250, 2)
n_tests = len(problem_list) * len(boundary_list) * len(m1_list)

def to_basic_list(array):
    assert np.linalg.norm(scipy.imag(array)) == 0
    return list(scipy.real(array))

def eval_phi0(th):
    r = boundary.eval_r(th)
    return problem.eval_expected_polar(r, th)

n_done = 0
for problem_name in problem_list:
    problem = problems.problem_dict[problem_name]()
    problem_results = {'k': problem.k}
    output[problem_name] = problem_results

    for boundary_name in boundary_list:
        boundary = problems.boundary.boundaries[boundary_name](problem.R)

        boundary_results = {
            'bet': boundary.bet,
        }
        problem_results[boundary_name] = boundary_results

        min_error7 = float('inf')

        for m1 in m1_list:
            a_coef, singular_vals = calc_a_coef(problem, boundary,
                eval_phi0, M, m1)
            error7 = np.max(np.abs(a_coef - problem.fft_a_coef[:M]))

            if error7 < min_error7:
                min_error7 = error7
                boundary_results['m1'] = m1
                boundary_results['a_coef'] = to_basic_list(a_coef)
                boundary_results['fft_a_coef'] = to_basic_list(problem.fft_a_coef[:M])
                boundary_results['error'] = to_basic_list(a_coef - problem.fft_a_coef[:M])
                boundary_results['error7'] = error7

            n_done += 1
            progress = n_done / n_tests
            print('\r{}%'.format(int(100*progress)), end='')

print('\n')

outfile = open(sys.path[0] + '/acoef.dat', 'w')
json.dump(output, outfile)
print('Wrote acoef.dat')
