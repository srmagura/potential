# To allow __main__ in subdirectory
import sys
sys.path.append(sys.path[0] + '/..')

import numpy as np
import matplotlib.pyplot as plt

from problems.shc_bessel import *

def get_v_reg_expr():
    '''
    SymPy expression for expected solution to the regularized problem.
    '''
    k, R, r, th = symbols('k R r th')
    
    a = pi/6
    nu = pi / (2*pi - a)
    
    # True solution to original problem
    v = besselj(nu/2, k*r) * sin(nu*(th-a)/2)
    
    # Inhomogeneous part of asymptotic expansion
    v -= get_u_asympt_expr()

    return v
            

v_reg_expr = get_v_reg_expr()
v_reg_lambda = lambdify(symbols('k R r th'), v_reg_expr)

k = 1
R = 1

x_data = np.linspace(-R, 0, 1000)
v_reg_data = [v_reg_lambda(k, R, -x, np.pi) for x in x_data]

plt.plot(x_data, v_reg_data)
plt.xlabel('x')
plt.ylabel('v_reg, y=0')
plt.show()
