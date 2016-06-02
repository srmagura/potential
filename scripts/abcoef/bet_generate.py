# To allow __main__ in subdirectory
import sys
sys.path.append(sys.path[0] + '/../..')

import argparse

import numpy as np
import scipy

import io_util
import problems
import problems.boundary

from abcoef import calc_a_coef

import json

M = 7

parser = argparse.ArgumentParser()

io_util.add_arguments(parser, ('problem', 'boundary'))
args = parser.parse_args()

problem = problems.problem_dict[args.problem]()
boundary_cls = problems.boundary.boundaries[args.boundary]

m1 = None

output = {}
output['k'] = problem.k
output['a'] = problem.a
output['M'] = M
output['results'] = []

k = problem.k
R = problem.R
a = problem.a
nu = problem.nu

def eval_phi0(th):
    r = boundary.eval_r(th)
    return problem.eval_expected_polar(r, th)

def test_many():
    global m1, boundary
    bet = .5

    max_i = 100
    for i in range(max_i):
        result_dict = {}
        result_dict['bet'] = bet

        boundary = boundary_cls(R, bet)
        problem.boundary = boundary

        if m1 is None:
            m1 = problem.get_m1()
            output['m1'] = m1

        a_coef, singular_vals = calc_a_coef(problem, boundary, eval_phi0, M, m1)

        result_dict['a_error'] = list(np.abs(a_coef - problem.fft_a_coef[:7]))
        result_dict['singular_vals'] = list(singular_vals)
        bet *= .95

        output['results'].append(result_dict)
        print('\r{}%'.format(int(100*i/max_i)), end='')


test_many()
print()

outfile = open(sys.path[0] + '/bet.dat', 'w')
json.dump(output, outfile)
print('Wrote bet.dat')
