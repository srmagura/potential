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
    J_dict, coef = bdata.calc_fft(Jmax) 

    period = 2*np.pi - a

    for l in range(len(th_data)):
        th = th_data[l]
        t = 2*np.pi/period * (th-a)

        phi0 = 0 
        for J, i in J_dict.items():
            exp = np.exp(complex(0, J*t))
            phi0 += (coef[i] * exp).real

        expansion_data[l] = phi0

    error = np.max(np.abs(expansion_data - exact_data))

    print('---- #coef = {} ----'.format(len(J_dict)))
    print('Fourier error:', error)

    #if prev_error is not None:
    #    conv = np.log2(prev_error / error)
    #    print('Convergence:', conv)


    print()

    plt.plot(th_data, exact_data, linewidth=5, color='#BBBBBB', label='Exact')
    plt.plot(th_data, expansion_data, label='Expansion')
    plt.show()

    return error

def convergence_test():
    print('Boundary data: `{}`'.format(bdata.__class__.__name__))
    Jmax = 4
    prev_error = None

    while Jmax <= 128:
        prev_error = do_test(Jmax, prev_error)
        Jmax *= 2

bdata = Hat()
#convergence_test()
do_test(200, None)
