# To allow __main__ in subdirectory
import sys
sys.path.append(sys.path[0] + '/..')

import numpy as np
import matplotlib.pyplot as plt

from problems.shc_bessel import *

problem = ShcBesselKnown()

r_data = np.linspace(0, problem.R, 1000)

for th in (2/3*np.pi, np.pi): 
    u_reg_data = [problem.eval_expected_polar(r, th) for r in r_data]
    plt.plot(r_data, u_reg_data, label='th={}'.format(round(th,2)))

plt.xlabel('r')
plt.ylabel('u_reg')
plt.legend(loc=3)
plt.show()
