import numpy as np
import sympy

#import matplotlib.pyplot as plt

import scipy
from scipy.interpolate import interp2d
from scipy.special import jv
from scipy.fftpack import dst

from problems.problem import PizzaProblem
from problems.sympy_problem import SympyProblem
from problems.boundary import Arc

import ps.ps
import ps.ode
from ps.polarfd import PolarFD

from fourier import arc_dst
import domain_util
import abcoef

class ZMethod:

    M = 7
    R0 = .1

    def __init__(self, options):
        self.options = options

    def run(self):
        self.problem = self.options['problem']
        self.z1cheat = self.options['z1cheat']
        self.acheat = self.options['acheat']
        self.boundary = self.problem.boundary
        self.N = self.options['N']

        self.a = PizzaProblem.a
        self.nu = PizzaProblem.nu
        self.k = self.problem.k

        # FIXME
        self.arc_R = self.boundary.R + self.boundary.bet # + 1

        self.do_algebra()
        self.calc_z1_fourier()
        self.do_z_BVP()
        self.create_v_interp()
        self.calc_a_coef()
        self.do_u_BVP()

        result = {
            'v': self.v,
            'v_error': self.get_v_error(),
            'u': self.u,
            'u_error': self.u_error,
            'polarfd': self.polarfd,
            'pert_solver': self.pert_solver,
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
        f0 = -apply_helmholtz_op(g)

        # Polynomial with p(0) = p'(0) = p"(0) = 0,
        # p(1) = 1 and p'(1) = p"(1) = 0
        p = 10 * x**3 - 15 * x**4 + 6 * x**5

        q1 = (r**2 * 1/2 * (th-a)**2 * f0.subs(th, a) *
            p.subs(x, (a-th)/(2*pi-a)+1))

        q2 = (r**2 * 1/2 * (th-2*np.pi)**2 * f0.subs(th, 2*pi) *
            p.subs(x, (th-2*pi)/(2*pi-a)+1))

        q = q1 + q2
        f1 = f0 - apply_helmholtz_op(q)

        subs_dict = {
            'k': self.k,
            'a': self.a,
            'nu': self.nu,
        }

        initial_func = self.problem.v_series - (g+q)
        d_initial_func = diff(initial_func, r)

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

        v_series_lambda = my_lambdify(self.problem.v_series)
        R0 = self.R0
        v_series_error = [self.problem.eval_v(R0, th) - v_series_lambda(R0, th)
            for th in np.arange(self.a, 2*np.pi, .1)]
        print('v_series_error:', np.max(np.abs(v_series_error)))

        self.f_expr = f
        self.eval_v_asympt = my_lambdify(v_asympt)

        self.eval_initial_func = my_lambdify(initial_func)
        self.eval_d_initial_func = my_lambdify(d_initial_func)
        self.eval_v_series = my_lambdify(self.problem.v_series)

        eval_g = my_lambdify(g)
        self.eval_gq = my_lambdify(g+q)
        self.eval_f1 = my_lambdify(f1)

        """r_data = np.linspace(0, .1, 512)
        f1_data = [self.eval_f1(r, 2*np.pi) for r in r_data]
        f1_data += [self.eval_f1(r, self.a) for r in r_data]
        f1_max = np.max(np.abs(f1_data))
        print('f1_max:', f1_max)

        bc_data1 = [self.problem.eval_phi1(r) - self.eval_v_asympt(r, 2*np.pi)
            - eval_g(r, 2*np.pi) for r in r_data]
        bc_max1 = np.max(np.abs(bc_data1))
        print('bc_max1:', bc_max1)

        bc_data2 = [self.problem.eval_phi2(r) - self.eval_v_asympt(r, self.a)
            - eval_g(r, self.a) for r in r_data]
        bc_max2 = np.max(np.abs(bc_data2))
        print('bc_max2:', bc_max2)"""

        #th = np.pi
        #z_data = [self.problem.eval_v(r, th) - self.eval_v_asympt(r, th)
        #    for r in r_data]
        #plt.plot(r_data, z_data)
        #plt.show()



    def calc_z1_fourier(self):
        expected_z1_fourier = self.get_expected_z1_fourier()

        if self.z1cheat:
            self.z1_fourier = expected_z1_fourier
        elif self.z1_fourier is not None:
            # Cached from a previous run
            pass
        else:
            self.z1_fourier = ps.ode.calc_z1_fourier(
                self.eval_initial_func,
                self.eval_d_initial_func,
                self.eval_f1, self.a, self.nu, self.k,
                self.R0, self.arc_R, self.M
            )

            print('ODE error:')
            print(np.abs(self.z1_fourier - expected_z1_fourier))


    def get_expected_z1_fourier(self):
        """
        Compute the expected Fourier coefficients of z1 from
        z1 = v - (v_asympt g + q)

        For checking the accuracy of the ODE method or
        skipping the ODE method during testing
        """
        R = self.arc_R

        def z1_expected(th):
            return self.problem.eval_v(R, th) - self.eval_gq(R, th)

        # This should be close to 0
        return arc_dst(self.a, z1_expected)[:self.M]

    def do_z_BVP(self):
        a = self.a
        nu = self.nu
        k = self.k
        M = self.M
        N = self.N

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

        self.polarfd = PolarFD(N, a, nu, self.R0, self.arc_R)

        if not self.acheat:
            zero = lambda r, th: 0

            z = self.polarfd.solve(k,
                lambda th: self.eval_v_series(self.R0, th),
                eval_phi0,
                self.problem.eval_phi1,
                self.problem.eval_phi2,
                zero, zero, zero, zero
                #dummy.eval_f_polar, dummy.eval_d_f_r,
                #dummy.eval_d2_f_r, dummy.eval_d2_f_th
            )
        else:
            z = np.zeros((N+1)**2)

        # Add back v_asympt
        self.v = np.zeros((N+1, N+1), dtype=complex)

        for m in range(N+1):
            for l in range(N+1):
                r = self.polarfd.get_r(m)
                th = self.polarfd.get_th(l)
                index = self.polarfd.get_index(m, l)
                self.v[m, l] = z[index]# + self.eval_v_asympt(r, th)

    def calc_a_coef(self):
        a = self.a
        nu = self.nu
        k = self.k
        arc_R = self.arc_R
        M = self.M

        fft_a_coef = self.get_expected_a_coef()

        if self.acheat:
            self.a_coef = fft_a_coef
        else:
            eval_phi0 = self.problem.eval_phi0

            def eval_bc0(th):
                r = self.boundary.eval_r(th)
                return eval_phi0(th) - self.v_interp(r, th)

            a_coef, singvals = abcoef.calc_a_coef(self.problem,
                self.boundary, eval_bc0, self.M, self.problem.get_m1())

            self.a_coef = a_coef

            np.set_printoptions(precision=4)
            if self.boundary.name == 'arc':
                error = np.abs(self.a_coef - fft_a_coef)
                print('a_coef error:', error)
                print()
            elif self.problem.name == 'iz-bessel':
                print('a_coef:', scipy.real(a_coef))
                print()

    def get_expected_a_coef(self):
        if self.problem.name == 'iz-bessel':
            return np.zeros(self.M)

        # Arc only
        r = self.problem.R

        def eval_bc0(th):
            return self.problem.eval_phi0(th) - self.problem.eval_v(r, th)

        def eval_diff(th):
            return self.problem.eval_v(r, th) - self.v_interp(r, th)

        interp_error = arc_dst(self.a, eval_diff)[:self.M]
        print('b interp error:')
        print(interp_error)
        print()

        b_coef = arc_dst(self.a, eval_bc0)[:self.M]
        return abcoef.b_to_a(b_coef, self.k, self.problem.R, self.nu)

    def create_v_interp(self):
        """
        Create 6th order accurate interpolating function for v
        """
        N = self.N

        r_data = []
        th_data = []
        v_real_data = []
        v_imag_data = []

        # Don't need to interpolate far away from boundary
        if N >= 128:
            rmin = (self.boundary.R - self.boundary.bet) * (2/3)
        else:
            rmin = 0

        for m in range(N+1):
            r = self.polarfd.get_r(m)
            if r < rmin:
                continue

            r_data.append(r)

        for l in range(N+1):
            th = self.polarfd.get_th(l)
            th_data.append(th)

        for m in range(N+1):
            r = self.polarfd.get_r(m)
            if r < rmin:
                continue

            for l in range(N+1):
                th = self.polarfd.get_th(l)

                v_real_data.append(self.v[m, l].real)
                v_imag_data.append(self.v[m, l].imag)


        real_interp = interp2d(r_data, th_data, v_real_data, kind='quintic')
        imag_interp = interp2d(r_data, th_data, v_imag_data, kind='quintic')

        def interp(r, th):
            return complex(real_interp(r, th), imag_interp(r, th))

        self.v_interp = interp

    def do_u_BVP(self):
        a = self.a
        nu = self.nu
        k = self.k
        M = self.M

        eval_v_asympt = self.eval_v_asympt
        f_expr = self.f_expr

        eval_phi0 = self.problem.eval_phi0
        eval_phi1 = self.problem.eval_phi1
        eval_phi2 = self.problem.eval_phi2
        eval_expected_polar = self.problem.eval_expected_polar

        a_coef = self.a_coef
        _n_basis_dict = self.problem.n_basis_dict

        expected_known = self.problem.name == 'iz-bessel'

        class u_BVP(SympyProblem, PizzaProblem):

            n_basis_dict = _n_basis_dict

            def __init__(self):
                self.k = k
                self.expected_known = expected_known
                super().__init__(f_expr=f_expr)

            def eval_bc(self, arg, sid):
                r, th = domain_util.arg_to_polar(self.boundary, a, arg, sid)

                if sid == 0:
                    bc = eval_phi0(th) - eval_v_asympt(r, th)

                    for m in range(1, M+1):
                        bc -= a_coef[m-1] * jv(m*nu, k*r) * np.sin(m*nu*(th-a))

                    return bc

                elif sid == 1 and r >= 0:
                    return eval_phi1(r) - eval_v_asympt(r, 2*np.pi)

                elif sid == 2 and r >= 0:
                    return eval_phi2(r) - eval_v_asympt(r, a)

                return 0

            def eval_expected_polar(self, r, th):
                return eval_expected_polar(r, th) - eval_v_asympt(r, th)

        my_u_BVP = u_BVP()
        my_u_BVP.boundary = self.problem.boundary

        options = {}
        options.update(self.options)
        options['problem'] = my_u_BVP

        # Save the solver so we can use its grid
        self.pert_solver = ps.ps.PizzaSolver(options)

        result = self.pert_solver.run()
        self.u = result.u_act
        self.u_error = result.error

    def get_v_error(self):
        N = self.N
        errors = []

        for m in range(1, N):
            r = self.polarfd.get_r(m)
            expected = np.zeros(N-1)

            for l in range(1, N):
                th = self.polarfd.get_th(l)
                expected[l-1] = self.problem.eval_v(r, th)

            to_dst = expected - self.v[m, 1:N]
            fourier = dst(to_dst, type=1)[:self.M] / N
            errors.append(np.max(np.abs(fourier)))

        return max(errors)
