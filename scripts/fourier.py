# To allow __main__ in subdirectory
import sys
sys.path.append(sys.path[0] + '/..')

import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

from problems.bdata import Parabola, Hat
from problems.shc_bessel import ShcBesselKnown

k = 1

a = np.pi/6
nu = np.pi / (2*np.pi - a)

def do_test(M, prev_error):
    th_data = np.linspace(a, 2*np.pi, 1000)
    exact_data = np.array([bdata.eval_phi0(th) for th in th_data])
    
    expansion_data = np.zeros(len(th_data))
    coef = bdata.calc_coef(M) 

    for i in range(len(th_data)):
        th = th_data[i]

        phi0 = 0 
        for m in range(1, M+1):
            phi0 += coef[m-1] * np.sin(m*nu*(th - a))

        expansion_data[i] = phi0

    error = np.max(np.abs(expansion_data - exact_data))

    print('---- M = {} ----'.format(M))
    print('Fourier error:', error)

    if prev_error is not None:
        conv = np.log2(prev_error / error)
        print('Convergence:', conv)


    print()

    plt.plot(th_data, exact_data, linewidth=5, color='#BBBBBB', label='Exact')
    plt.plot(th_data, expansion_data, label='Expansion')
    plt.show()

    return error

def convergence_test():
    print('Boundary data: `{}`'.format(bdata.__class__.__name__))
    M = 4
    prev_error = None

    while M <= 128:
        prev_error = do_test(M, prev_error)
        M *= 2

def numerical_test():
    M = 7
    coef = bdata.calc_coef_analytic(M)
    coef_numerical = bdata.calc_coef_numerical(M)

    print(np.max(np.abs(coef - coef_numerical)))

problem = ShcBesselKnown()
bdata = problem.bdata
do_test(32, None)
#convergence_test()
