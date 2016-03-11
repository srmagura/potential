# To allow __main__ in subdirectory
import sys
sys.path.append(sys.path[0] + '/..')

import argparse

import numpy as np
import scipy

import io_util
import problems
import problems.boundary

import ps.coordinator

from star.shared import calc_a_coef

import json

parser = argparse.ArgumentParser()

io_util.add_arguments(parser, ('problem', 'boundary'))
args = parser.parse_args()

problem = problems.problem_dict[args.problem]()
boundary_cls = problems.boundary.boundaries[args.boundary]

#coord = ps.coordinator.Coordinator(problem, None, {'boundary': boundary})
#coord.print_info()

m1 = 140
M = 7

output = {}
output['m1'] = m1
output['M'] = M
output['results'] = []

k = problem.k
R = problem.R
a = problem.a
nu = problem.nu

def eval_phi0(r, th):
    return problem.eval_expected_polar(r, th)

def test_many():
    bet = boundary_cls.bet0

    for i in range(3):
        result_dict = {}
        result_dict['bet'] = bet

        boundary = boundary_cls(R, bet)
        a_coef, singular_vals = calc_a_coef(problem, boundary, eval_phi0, M, m1)

        result_dict['a_error'] = list(np.abs(a_coef - problem.fft_a_coef[:7]))
        result_dict['singular_vals'] = list(singular_vals)
        bet *= .95

        output['results'].append(result_dict)

test_many()
print(json.dumps(output))
