# To allow __main__ in subdirectory
import sys
sys.path.append(sys.path[0] + '/../..')

import numpy as np
from scipy.special import jv
import matplotlib.pyplot as plt

import problems
import problems.boundary

from abcoef import calc_a_coef

problem_list = ('trace-sine8', 'trace-hat', 'trace-parabola', 'trace-line-sine',)

m_max = 500

'''def a_expan(th):
    r = boundary.eval_r(th)
    expan = 0

    for m in range(1, m_max+1):
        ac = problem.fft_a_coef[m-1]
        expan += ac * jv(m*nu, k*r) * np.sin(m*nu*(th-a))

    return expan'''

def b_expan(th):
    expan = 0

    for m in range(1, m_max+1):
        b = problem.fft_b_coef[m-1]
        expan += b * np.sin(m*nu*(th-a))

    return expan

for problem_name in problem_list:
    print('---- {} ----'.format(problem_name))
    problem = problems.problem_dict[problem_name]()
    k = problem.k
    a = problem.a
    nu = problem.nu

    boundary = problems.boundary.Arc(problem.R)
    problem.boundary = boundary

    #a_error_list = []
    b_error_list = []
    for th in np.linspace(problem.a, 2*np.pi, 512):
        #a_error_list.append(problem.eval_bc(th, 0) - a_expan(th))
        b_error_list.append(problem.eval_phi0(th) - b_expan(th))

    #print('error(a): {}'.format(np.max(np.abs(a_error_list))))
    print('error(b): {}'.format(np.max(np.abs(b_error_list))))
    print()

    #plt.plot(range(m_max), np.log10(np.abs(problem.fft_b_coef[:m_max])), 'o')
    #fs=16
    #plt.ylabel('log10(abs(fourier coefficients))', fontsize=fs)
    #plt.show()
