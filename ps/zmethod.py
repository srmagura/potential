import numpy as np
import sympy

import matplotlib.pyplot as plt

from problems.sympy_problem import SympyProblem

import ps.ps

class ZMethod():

    def __init__(self, options):
        self.problem = options['problem']
        self.scheme_order = options['scheme_order']

    def run(self):
        self.do_algebra()

        solver = ps.ps.PizzaSolver({})
        result = solver.run()

    def do_algebra(self):
        g = self.problem.g
        v_asympt = self.problem.v_asympt

        diff = sympy.diff
        pi = sympy.pi
        k, a, r, th, x = sympy.symbols('k a r th x')

        def apply_helmholtz_op(u):
            d_u_r = diff(u, r)
            return diff(d_u_r, r) + d_u_r / r + diff(u, th, 2) / r**2 + k**2 * u

        f0 = -apply_helmholtz_op(g+v_asympt)

        logistic = 1 / (1 + sympy.exp(-90*(x-0.5)))

        # TODO insert q1
        q2 = (r**2 * 1/2 * (th-2*pi)**2 * f0.subs(th, 2*pi) *
            logistic.subs(x, (th-2*pi)/(2*pi-a)+1))

        q = q2
        f1 = f0 - apply_helmholtz_op(q)


        subs_dict = self.problem.sympy_subs_dict
        lambdify_modules = SympyProblem.lambdify_modules

        def my_lambdify(expr):
            return sympy.lambdify((r, th),
                expr.subs(subs_dict),
                modules=lambdify_modules)

        v_asympt_lambda = my_lambdify(v_asympt)
        def eval_v_asympt(r, th):
            if r > 0:
                return v_asympt_lambda(r, th)
            else:
                return 0

        self.eval_f0 = my_lambdify(f0)

        self.eval_v_asympt = eval_v_asympt

        self.eval_gq = my_lambdify(g+q)
        self.eval_f1 = my_lambdify(f1)
