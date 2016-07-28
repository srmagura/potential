import numpy as np
import sympy

import matplotlib.pyplot as plt

from problems.problem import PizzaProblem
from problems.sympy_problem import SympyProblem
from problems.boundary import Arc
from problems.singular import HReg

import ps.ps
import ps.ode

from fourier import arc_dst
import domain_util

class ZMethod():

    M = 7

    def __init__(self, options):
        self.problem = options['problem']
        self.boundary = self.problem.boundary
        self.options = options

        self.a = PizzaProblem.a
        self.nu = PizzaProblem.nu
        self.k = self.problem.k
        self.R = self.boundary.R + self.boundary.bet

    def run(self):
        a = self.a
        nu = self.nu
        k = self.k
        R = self.R

        self.do_algebra()

        self.z1_fourier = ps.ode.calc_z1_fourier(
            self.eval_f1, a, nu, k, R, self.M
        )
        self.check_ode_accuracy()

        self.do_z_BVP()

        result = {
            'z': self.z
        }

    def do_algebra(self):
        g = self.problem.g
        v_asympt = self.problem.v_asympt

        diff = sympy.diff
        pi = sympy.pi
        k, a, r, th, x = sympy.symbols('k a r th x')

        def apply_helmholtz_op(u):
            d_u_r = diff(u, r)
            return diff(d_u_r, r) + d_u_r / r + diff(u, th, 2) / r**2 + k**2 * u

        f = -apply_helmholtz_op(v_asympt)
        f0 = f - apply_helmholtz_op(g)

        logistic = 1 / (1 + sympy.exp(-90*(x-0.5)))

        # TODO insert q1
        q2 = (r**2 * 1/2 * (th-2*pi)**2 * f0.subs(th, 2*pi) *
            logistic.subs(x, (th-2*pi)/(2*pi-a)+1))

        q = q2
        f1 = f0 - apply_helmholtz_op(q)

        subs_dict = {
            'k': self.k,
            'a': self.a,
            'nu': self.nu,
        }
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

        self.eval_v_asympt = eval_v_asympt
        self.f_expr = f

        self.eval_gq = my_lambdify(g+q)
        self.eval_f1 = my_lambdify(f1)

    def do_z_BVP(self):
        a = self.a
        nu = self.nu
        k = self.k
        R = self.R
        M = self.M

        f_expr = self.f_expr
        eval_phi1 = self.problem.eval_phi1
        eval_phi2 = self.problem.eval_phi2

        eval_v_asympt = self.eval_v_asympt
        eval_gq = self.eval_gq

        z1_fourier = self.z1_fourier

        class z_BVP(SympyProblem, PizzaProblem):

            hreg = HReg.none

            def __init__(self, **kwargs):
                kwargs['f_expr'] = f_expr
                self.k = k

                super().__init__(**kwargs)

            def eval_bc(self, arg, sid):
                r, th = domain_util.arg_to_polar(
                    self.boundary, a, arg, sid
                )

                v_asympt = eval_v_asympt(r, th)
                gq = eval_gq(r, th)

                if sid == 0:
                    fourier_series = 0
                    for m in range(1, M+1):
                        fourier_series += z1_fourier[m-1] * np.sin(m*nu*(th-a))

                    return fourier_series + gq
                elif sid == 1:
                    return eval_phi1(r) - v_asympt
                elif sid == 2:
                    return eval_phi2(r) - v_asympt


        my_z_BVP = z_BVP()
        my_z_BVP.boundary = Arc(self.R)

        options = {}
        options.update(self.options)
        options['problem'] = my_z_BVP

        # Save the solver so we can use its grid
        self.solver = ps.ps.PizzaSolver(options)

        self.z = self.solver.run().u_act

    def check_ode_accuracy(self):
        """
        Estimate accuracy of the z1 fourier coefficients returned
        by the ODE method

        Compute the Fourier coefficients of (z1 + g + q + v_asympt) - v,
        which should be zero, since the quantity in parenthesis
        should equal v
        """
        R = self.R

        def gqv(th):
            return (self.eval_gq(R, th) + self.eval_v_asympt(R, th)
                - self.problem.eval_v(R, th)
            )

        # This should be close to 0
        error_fourier = self.z1_fourier + arc_dst(self.a, gqv)[:self.M]
        print('ODE error:')
        print(error_fourier)
