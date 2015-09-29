# To allow __main__ in subdirectory
import sys
sys.path.append(sys.path[0] + '/..')

import numpy as np
import matplotlib.pyplot as plt
import argparse

import norms.weight_func

r_data = np.linspace(0, 1.1, 1024)
wf_data = [norms.weight_func.wf1(r) for r in r_data]

plt.plot(r_data, wf_data, 'r')

fs = 15
plt.xlabel('r / R = (polar radius) / (domain radius)', fontsize=fs)
plt.ylabel('Weight (unnormalized)', fontsize=fs)
plt.ylim(ymin=0)
plt.show()
