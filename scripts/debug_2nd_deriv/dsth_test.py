# To allow __main__ in subdirectory
import sys
sys.path.append(sys.path[0] + '/..')

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipeinc
from scipy.integrate import quad
from problems.boundary import Boundary

a = 3.05
b = 2.23

class Ellipse(Boundary):
    name = 'ellipse'
    r_expr_str = '_a*_b/(sqrt((_b*cos(th))**2 + (_a*sin(th))**2))'
    bet0 = 0.05
    additional_params = {'_a': a, '_b': b}

boundary = Ellipse(5)

e = np.sqrt(1-(b/a)**2)

d_s_th = lambda th: 1/boundary.eval_d_th_s(th)

error = []
for th1 in np.linspace(np.pi/12, np.pi/3, 50):
    t = np.arctan(a/b*np.tan(th1))
    C1 = quad(lambda x: np.sqrt(a**2*np.sin(x)**2 + b**2*np.cos(x)**2), 0, t)[0]
    C2 = quad(d_s_th, 0, th1)[0]
    #print('C1=', C1,'C2=',C2)
    error.append(C1-C2)

print(np.max(np.abs(error)))
