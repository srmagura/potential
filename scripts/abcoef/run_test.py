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

parser = argparse.ArgumentParser()

io_util.add_arguments(parser, ('problem', 'boundary'))
args = parser.parse_args()

problem = problems.problem_dict[args.problem]()
boundary = problems.boundary.boundaries[args.boundary](problem.R)

M = 7

#print('m1 =', m1)
print('m_max =', problem.m_max)
print()

k = problem.k
R = problem.R
a = problem.a
nu = problem.nu

def eval_phi0(th):
    r = boundary.eval_r(th)
    return problem.eval_expected_(r, th)

def print_array(array):
    assert np.max(np.abs(scipy.imag(array))) == 0
    for x in scipy.real(array):
        print('{:.15e}'.format(x), end=' ')
    print()

def do_test(m1):
    a_coef = calc_a_coef(problem, boundary, eval_phi0, M, m1)
    errors = np.abs(problem.fft_a_coef[:M] - a_coef)
    error7 = np.max(errors)

    print('a_coef(expected):')
    print_array(problem.fft_a_coef[:M])
    print()
    print('a_coef(actual):')
    print_array(a_coef)
    print()
    print('errors:')
    print_array(errors)
    print()
    print('max_error(a1-a7)={}'.format(error7))

    '''print('-'*20)
    print()
    print_array(problem.fft_a_coef[:M])
    print_array(a_coef)
    print_array(list(errors) + [0] + [error7])'''

def test_many(m1_list=None):
    min_error7 = float('inf')

    if m1_list is None:
        m1_list = range(10, 250, 2)

    for m1 in m1_list:
        print('\n----- m1={} -----'.format(m1))

        a_coef, singular_vals = calc_a_coef(problem, boundary,
            eval_phi0, M, m1)
        error7 = np.max(np.abs(a_coef - problem.fft_a_coef[:7]))

        print('a_coef=')
        print_array(a_coef)
        print('max_error(a1-a7)={}'.format(error7), end=' ')
        if error7 < min_error7:
            min_error7 = error7
            print('!!!')
        else:
            print()

#do_test(140)
test_many()
