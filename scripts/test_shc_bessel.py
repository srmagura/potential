# To allow __main__ in subdirectory
import sys
sys.path.append(sys.path[0] + '/..')

import numpy as np
import matplotlib.pyplot as plt

from problems.shc_bessel import *

u_reg_expr = get_u_reg_expr()
u_reg_lambda = lambdify(symbols('k R r th'), u_reg_expr)

k = 1
R = 1

x_data = np.linspace(-R, 0, 1000)
u_reg_data = [u_reg_lambda(k, R, -x, np.pi) for x in x_data]

plt.plot(x_data, u_reg_data)
plt.xlabel('x')
plt.ylabel('Expected solution to regularized problem on y=0')
plt.show()
