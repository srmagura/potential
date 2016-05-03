# To allow __main__ in subdirectory
import sys
sys.path.append(sys.path[0] + '/..')

import numpy as np
import matplotlib.pyplot as plt

from problems.boundary import Boundary

a = 2
b = 1

class Ellipse(Boundary):
    name = 'ellipse'
    r_expr_str = '_a*_b/(sqrt((_b*cos(th))**2 + (_a*sin(th))**2))'
    bet0 = 0.05
    additional_params = {'_a': a, '_b': b}

boundary = Ellipse(2.3)

th_data = np.linspace(0, 2*np.pi, 512)
r_data = list(map(boundary.eval_r, th_data))

'''ax = plt.subplot(111, projection='polar')
ax.plot(th_data, r_data)
plt.show()'''

def eval_curv1(r, th):
    x = r * np.cos(th)
    y = r * np.sin(th)
    return -1/(a**2 * b**2) * (x**2 / a**4 + y**2 / b**4)**(-3/2)

error = []
for r, th in zip(r_data, th_data):
     curv1 = eval_curv1(r, th)
     curv2 = boundary.eval_curv(th)
     #print(curv1, curv2)
     error.append(abs(curv1-curv2))

print(np.max(error))
