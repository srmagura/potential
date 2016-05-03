# To allow __main__ in subdirectory
import sys
sys.path.append(sys.path[0] + '/..')

import numpy as np
import matplotlib.pyplot as plt

from problems.boundary import OuterSine

boundary = OuterSine(2.3)

th_data = np.linspace(0, 2*np.pi, 512)
curv_data = list(map(boundary.eval_curv, th_data))

plt.plot(th_data, curv_data)
plt.show()
