# To allow __main__ in subdirectory
import sys
sys.path.append(sys.path[0] + '/..')

import numpy as np
from sympy import *

import ps.ps

from problems.boundary import Boundary, OuterSine, Arc
from problems.smooth import SmoothH_Sine

problem = SmoothH_Sine()
boundary = OuterSine(problem.R)
problem.boundary = boundary

assert problem.k == 1

options = {
    'problem': problem,
    'boundary': boundary,
    'scheme_order': 4,
    'N': 256
}

solver = ps.ps.PizzaSolver(options)
solver.calc_c0()
solver.calc_c1_exact()

th = symbols('th')
r = boundary.r_expr.subs(boundary.subs_dict)
xi0 = sin(r*cos(th))
d2_xi0_th = diff(xi0, th, 2)

eps=1e-2
th_data = np.linspace(solver.a+eps, 2*np.pi-eps, 128)
r_data = list(map(boundary.eval_r, th_data))

max_error = 0

for r, th in zip(r_data, th_data):
    deriv_types = (
        (0, 0), (0, 1), (2, 0),
    )
    ext_val = solver.ext_calc_xi_derivs(deriv_types, th, {'sid': 0})['d2_xi0_arg']
    sym_val = d2_xi0_th.subs('th', th)

    error = abs(ext_val-sym_val)
    if error > max_error:
        max_error = error
        print('Error of {} at th={}'.format(error, th))

print('Max error:', max_error)
