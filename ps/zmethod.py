import numpy as np
import sympy

import matplotlib.pyplot as plt

from scipy.interpolate import interp2d

from problems.problem import PizzaProblem
from problems.sympy_problem import SympyProblem
from problems.boundary import Arc
from problems.singular import HReg

import ps.ps
import ps.ode
from ps.polarfd import PolarFD

from fourier import arc_dst
import domain_util

class ZMethod():

    M = 7

    def __init__(self, options):
        self.problem = options['problem']
        self.z1cheat = options['z1cheat']
        self.boundary = self.problem.boundary
        self.N = options['N']
        self.options = options

        self.a = PizzaProblem.a
        self.nu = PizzaProblem.nu
        self.k = self.problem.k
        self.arc_R = self.boundary.R + 2*self.boundary.bet

    def run(self):
        self.do_algebra()
        self.calc_z1_fourier()
        self.do_z_BVP()
        #self.do_w_BVP()

        #u = self.v + self.w

        result = {
            'v': self.v,
            #'u': u,
            'polarfd': self.polarfd,
            #'pert_solver': self.pert_solver,
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

    def calc_z1_fourier(self):
        expected_z1_fourier = self.calc_expected_z1_fourier()

        if self.z1cheat:
            self.z1_fourier = expected_z1_fourier
        elif self.z1_fourier is not None:
            # Cached from a previous run
            pass
        else:
            self.z1_fourier = ps.ode.calc_z1_fourier(
                self.eval_f1, self.a, self.nu, self.k,
                self.arc_R, self.M
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
        R = self.arc_R

        def z1_expected(th):
            return (self.problem.eval_v(R, th) -
                (self.eval_gq(R, th) + self.eval_v_asympt(R, th))
            )

        # This should be close to 0
        return arc_dst(self.a, z1_expected)[:self.M]

    def do_z_BVP(self):
        a = self.a
        nu = self.nu
        k = self.k
        M = self.M

        f_expr = self.f_expr

        class DummySympyProblem(SympyProblem, PizzaProblem):
            def __init__(self):
                self.k = k
                super().__init__(f_expr=f_expr)

        dummy = DummySympyProblem()

        def eval_phi0(th):
            fourier_series = 0
            for m in range(1, M+1):
                fourier_series += self.z1_fourier[m-1] * np.sin(m*nu*(th-a))

            return fourier_series + self.eval_gq(self.arc_R, th)

        def eval_phi1(r):
            if r >= 0:
                v_asympt = self.eval_v_asympt(r, 2*np.pi)
                return self.problem.eval_phi1(r) - v_asympt

        def eval_phi2(r):
            if r >= 0:
                v_asympt = self.eval_v_asympt(r, a)
                return self.problem.eval_phi2(r) - v_asympt

        self.polarfd = PolarFD()
        z = self.polarfd.solve(self.N, k, a, nu, self.arc_R,
            eval_phi0, eval_phi1, eval_phi2,
            dummy.eval_f_polar, dummy.eval_d_f_r,
            dummy.eval_d2_f_r, dummy.eval_d2_f_th
        )

        # Add back v_asympt
        N = self.N
        self.v = np.zeros((N+1, N+1), dtype=complex)

        for m in range(N+1):
            for l in range(N+1):
                r = self.polarfd.get_r(m)
                th = self.polarfd.get_th(l)
                index = self.polarfd.get_index(m, l)
                self.v[m, l] = z[index] + self.eval_v_asympt(r, th)

    def create_v_interp(self):
        r_data = []
        th_data = []
        v_real_data = []
        v_imag_data = []

        rmin = self.boundary.R - self.boundary.bet

        for i, j in self.arc_solver.global_Mplus:
            r, th = self.arc_solver.get_polar(i, j)


            r_data.append(r)
            th_data.append(th)

            v_real_data.append(self.v[i-1, j-1].real)
            v_imag_data.append(self.v[i-1, j-1].imag)

        real_interp = interp2d(r_data, th_data, v_real_data, kind='quintic')
        imag_interp = interp2d(r_data, th_data, v_imag_data, kind='quintic')

        def interp(r, th):
            return complex(real_interp(r, th), imag_interp(r, th))

        return interp


    def do_w_BVP(self):
        a = self.a
        nu = self.nu
        k = self.k
        arc_R = self.arc_R
        M = self.M

        _n_basis_dict = self.problem.n_basis_dict

        eval_phi0 = self.problem.eval_phi0

        v_interp = self.create_v_interp()

        class w_BVP(PizzaProblem):

            homogeneous = True

            hreg = HReg.linsys
            n_basis_dict = _n_basis_dict

            def __init__(self, **kwargs):
                self.k = k
                super().__init__(**kwargs)

            def eval_bc(self, arg, sid):
                r, th = domain_util.arg_to_polar(
                    self.boundary, a, arg, sid
                )

                if sid == 0:
                    return eval_phi0(th) - v_interp(r, th)

                return 0


        my_w_BVP = w_BVP()
        my_w_BVP.boundary = self.problem.boundary

        options = {}
        options.update(self.options)
        options['problem'] = my_w_BVP

        # Save the solver so we can use its grid
        self.pert_solver = ps.ps.PizzaSolver(options)

        self.w = self.pert_solver.run().u_act
