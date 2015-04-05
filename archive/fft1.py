# To allow __main__ in subdirectory
import sys
sys.path.append(sys.path[0] + '/..')

import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

from problems.bdata import Parabola, Hat

k = 1

a = np.pi/6
nu = np.pi / (2*np.pi - a)

def do_test(Jmax, prev_error):
    th_data = np.linspace(a, 2*np.pi, 1024)
    exact_data = np.array([bdata.eval_phi0(th) for th in th_data])
    
    expansion_data = np.zeros(len(th_data))
    coef = bdata.calc_fft(Jmax) 

    for l in range(len(th_data)):
        th = th_data[l]

        phi0 = 0 
        for J in range(1, Jmax+1):
            phi0 += coef[J-1] * np.sin(J*nu*(th-a))

        expansion_data[l] = phi0

    error = np.max(np.abs(expansion_data - exact_data))

    print('---- #coef = {} ----'.format(len(coef)))
    print('Fourier error:', error)

    if prev_error is not None:
        conv = np.log2(prev_error / error)
        print('Convergence:', conv)

    print()

    #plt.plot(th_data, exact_data, linewidth=5, color='#BBBBBB', label='Exact')
    #plt.plot(th_data, expansion_data, label='Expansion')
    #plt.show()

    return error

def convergence_test():
    print('Boundary data: `{}`'.format(bdata.__class__.__name__))
    Jmax = 4
    prev_error = None

    while Jmax <= 1024:
        prev_error = do_test(Jmax, prev_error)
        Jmax *= 2

bdata = Parabola()
convergence_test()
