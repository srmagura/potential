# To allow __main__ in subdirectory
import sys
sys.path.append(sys.path[0] + '/..')

import numpy as np
from sympy import *

import ps.ps

from problems.boundary import Boundary, OuterSine
from problems.smooth import SmoothH_Sine

problem = SmoothH_Sine()
boundary = OuterSine(problem.R)
problem.boundary = boundary

assert problem.k == 1

options = {
    'problem': problem,
    'boundary': boundary,
    'scheme_order': 4,
    'N': 16
}

solver = ps.ps.PizzaSolver(options)

th = symbols('th')
r = boundary.r_expr.subs(boundary.subs_dict)
xi0 = sin(r*cos(th))
d2_xi0_th = diff(xi0, th, 2)
print(d2_xi0_th)


th_data = np.linspace(0, 2*np.pi, 512)
r_data = list(map(boundary.eval_r, th_data))

for r, th in zip(r_data, th_data):
     ext_val = self.ext_calc_xi_derivs(((2, 0),), th, options) 
