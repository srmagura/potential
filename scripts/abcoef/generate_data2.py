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

import sys

M = 7
n_done = 0

def go(problem_name, boundary_name):
    global n_done
    output = {}

    def to_basic_list(array):
        assert np.linalg.norm(scipy.imag(array)) == 0
        return list(scipy.real(array))

    def eval_phi0(th):
        r = boundary.eval_r(th)
        return problem.eval_expected_polar(r, th)

    problem = problems.problem_dict[problem_name]()

    boundary = problems.boundary.boundaries[boundary_name](problem.R)
    problem.boundary = boundary

    output = {
        'problem': problem_name,
        'm_max': problem.m_max,
        'boundary': boundary_name,
        'k': problem.k,
        'a': problem.a,
        'bet': boundary.bet,
        'm1_data': [],
        'error': [],
        'singular_vals': []
    }

    for m1 in m1_list:
        a_coef, singular_vals = calc_a_coef(problem, boundary,
            eval_phi0, M, m1)
        error = a_coef - problem.fft_a_coef[:M]

        output['m1_data'].append(m1)
        output['error'].append(to_basic_list(np.abs(error)))
        output['singular_vals'].append(to_basic_list(singular_vals))

        n_done += 1
        progress = n_done / n_tests
        print('\r{}%'.format(int(100*progress)), end='')

    fname = 'm1trend_{}_{}_{}.dat'.format(problem_name, boundary_name, problem.k)
    outfile = open(sys.path[0] + '/' + fname, 'w')
    json.dump(output, outfile)

_problems = ['trace-line-sine']#['trace-hat', 'trace-line-sine', 'trace-parabola']
#_boundaries = ['outer-sine', 'sine7']
_boundaries = ['inner-sine', 'cubic']
m1_list = range(8, 275, 5)
n_tests = len(m1_list) * len(_problems) * len(_boundaries)

for problem_name in _problems:
    for boundary_name in _boundaries:
        go(problem_name, boundary_name)

print()
