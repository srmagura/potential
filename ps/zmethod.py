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
        self.z1cheat = options['z1cheat']
        self.boundary = self.problem.boundary
        self.options = options

        self.a = PizzaProblem.a
        self.nu = PizzaProblem.nu
        self.k = self.problem.k
        self.R = self.boundary.R + self.boundary.bet

    def run(self):
        self.do_algebra()
        self.calc_z1_fourier()
        self.do_z_BVP()

        result = {
            'z': self.z,
            'solver': self.solver,
        }

        return result

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

        # Polynomial with p(0) = p'(0) = p"(0) = 0,
        # p(1) = 1 and p'(1) = p"(1) = 0
        p = 10 * x**3 - 15 * x**4 + 6 * x**5

        # TODO insert q1
        q2 = (r**2 * 1/2 * (th-2*np.pi)**2 * f0.subs(th, 2*pi) *
            p.subs(x, (th-2*pi)/(2*pi-a)+1))

        q = q2
        f1 = f0 - apply_helmholtz_op(q)

        subs_dict = {
            'k': self.k,
            'a': self.a,
            'nu': self.nu,
        }

        lambdify_modules = SympyProblem.lambdify_modules

        def my_lambdify(expr):
            # Assuming function is 0 at r=0 ... this may not
            # be necessary
            lam = sympy.lambdify((r, th),
                expr.subs(subs_dict),
                modules=lambdify_modules)


            def newfunc(r, th):
                if r == 0:
                    return 0
                else:
                    return lam(r, th)

            return newfunc


        self.f_expr = f
        self.eval_v_asympt = my_lambdify(v_asympt)

        self.eval_gq = my_lambdify(g+q)
        self.eval_f1 = my_lambdify(f1)

        #f1_data = [self.eval_f1(r, 2*np.pi) for r in np.linspace(0, self.R, 512)]
        #f1_max = np.max(np.abs(f1_data))
        #print('f1_max:', f1_max)

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

        _n_basis_dict = self.problem.n_basis_dict

        class z_BVP(SympyProblem, PizzaProblem):

            hreg = HReg.none
            n_basis_dict = _n_basis_dict

            def __init__(self, **kwargs):
                kwargs['f_expr'] = f_expr

                self.k = k

                super().__init__(**kwargs)

            def eval_bc(self, arg, sid):
                r, th = domain_util.arg_to_polar(
                    self.boundary, a, arg, sid
                )

                if sid == 0:
                    fourier_series = 0
                    for m in range(1, M+1):
                        fourier_series += z1_fourier[m-1] * np.sin(m*nu*(th-a))

                    return fourier_series + eval_gq(r, th)

                elif sid == 1:
                    if r >= 0:
                        v_asympt = eval_v_asympt(r, th)
                        return eval_phi1(r) - v_asympt

                elif sid == 2:
                    if r >= 0:
                        v_asympt = eval_v_asympt(r, th)
                        return eval_phi2(r) - v_asympt

                return 0


        my_z_BVP = z_BVP()
        my_z_BVP.boundary = Arc(self.R)

        options = {}
        options.update(self.options)
        options['problem'] = my_z_BVP

        # Save the solver so we can use its grid
        self.solver = ps.ps.PizzaSolver(options)

        self.z = self.solver.run().u_act


    def calc_z1_fourier(self):
        a = self.a
        nu = self.nu
        k = self.k
        R = self.R
        
        expected_z1_fourier = self.calc_expected_z1_fourier()

        if self.z1cheat:
            self.z1_fourier = expected_z1_fourier
        elif self.z1_fourier is not None:
            # Cached from a previous run
            pass
        else:
            self.z1_fourier = ps.ode.calc_z1_fourier(
                self.eval_f1, a, nu, k, R, self.M
            )

            print('ODE error:')
            print(np.abs(self.z1_fourier - expected_z1_fourier))


    def calc_expected_z1_fourier(self):
        """
        Compute the expected Fourier coefficients of z1 from
        z1 = v - (v_asympt g + q)

        For checking the accuracy of the ODE method or
        skipping the ODE method during testing
        """
        R = self.R

        def z1_expected(th):
            return (self.problem.eval_v(R, th) -
                (self.eval_gq(R, th) + self.eval_v_asympt(R, th))
            )

        # This should be close to 0
        return arc_dst(self.a, z1_expected)[:self.M]
