# To allow __main__ in subdirectory
import sys
sys.path.append(sys.path[0] + '/..')

import numpy as np
import matplotlib.pyplot as plt

from problems.fourier_bessel import FourierBessel

problem = FourierBessel()
th_data = np.linspace(problem.a, 2*np.pi, 1024)

for r in [problem.R, 1.5]:
    u_data = [problem.eval_expected_polar(r, th, True) for th in th_data]

    plt.plot(th_data, u_data, label='r={}'.format(r))

plt.legend()
plt.show()
