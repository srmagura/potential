# To allow __main__ in subdirectory
import sys
sys.path.append(sys.path[0] + '/..')

import numpy as np
import matplotlib.pyplot as plt

from problems.shc_bessel import *

def plot_traces():
    r_data = np.linspace(0, R, 1000)

    for th in (2/3*np.pi, np.pi): 
        u_reg_data = [p.eval_expected_polar(r, th) for r in r_data]
        plt.plot(r_data, u_reg_data, label='th={}'.format(round(th,2)))

    plt.xlabel('r')
    plt.ylabel('u_reg')
    plt.legend(loc=3)
    plt.show()


p = ShcBesselKnown()
a = p.a
R = p.R

error = []
for th in np.arange(a, 2*np.pi, .01):
    diff = abs(p.eval_expected_polar(R, th) - p.eval_bc_extended(th, 0))
    error.append(diff)

for sid, th in ((1, 2*np.pi), (2, a)):
    for r in np.arange(0, R, .01):
        diff = abs(p.eval_expected_polar(r, th) -
            p.eval_bc_extended(r, sid))
        error.append(diff)

print('Diff between expected solution and BC:')
print(np.max(error))
print()
print(p.many_shc_coef)

#plot_traces()
