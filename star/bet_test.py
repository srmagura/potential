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

parser = argparse.ArgumentParser()

io_util.add_arguments(parser, ('problem', 'boundary'))
args = parser.parse_args()

problem = problems.problem_dict[args.problem]()
boundary_cls = problems.boundary.boundaries[args.boundary]

#coord = ps.coordinator.Coordinator(problem, None, {'boundary': boundary})
#coord.print_info()

m1 = 140
M = 7

print('k =', problem.k)
print('m1 =', m1)
print('m_max =', problem.m_max)
print()

k = problem.k
R = problem.R
a = problem.a
nu = problem.nu

def eval_phi0(r, th):
    return problem.eval_expected_polar(r, th)

def print_array(array):
    assert np.max(np.abs(scipy.imag(array))) == 0
    for x in scipy.real(array):
        print('{:.15e}'.format(x), end=' ')
    print()

def test_many():
    bet = boundary_cls.bet0
    for i in range(200):
        boundary = boundary_cls(R, bet)

        a_coef = calc_a_coef(problem, boundary, eval_phi0, M, m1)
        print_array([bet] + list(np.abs(a_coef - problem.fft_a_coef[:7])))

        bet *= 8/9

test_many()
