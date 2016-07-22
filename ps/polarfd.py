import numpy as np
import itertools as it
from scipy.interpolate import interp1d, CloughTocher2DInterpolator
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

import matlab
import matlab.engine

import abcoef
from chebyshev import get_chebyshev_roots, eval_T


class PolarFD:

    # rec: .05
    R0 = .05

    # Default: .2
    # for 1024, rank deficient for .25 and .3
    match_r = .2

    @classmethod
    def my_print(cls, x):
        print('(polarfd) ' + str(x))

    def get_r(self, m):
        return self.R0 + m*self.hr

    def get_th(self, l):
        return self.a + l*self.hth

    def get_index(self, m, l):
        return m*(self.N+1) + l

    def get_index_b(self, J):
        return self.dim0 + J

    def eval_B(self, J, th):
        return eval_T(J, self.eval_g_inv(th))

    def calc_N_var(self):
        """ Calculate variables that depend on N """
        self.hr = (self.R2-self.R0)/self.N
        self.hth = (2*np.pi - self.a) / self.N

        self.dim0 = (self.N+1)**2
        self.dim = self.dim0 + self.Nlam

    def new_system(self):
        """ Create variables for making a new sparse system """
        self.rhs = []
        self.row = 0

        self.data = []
        self.row_ind = []
        self.col_ind = []

    def set_bc_origin(self):
        """
        Set BC near origin
        """
        N = self.N
        for l in range(1, N):
            i = self.get_index(0, l)

            self.data.append(1)
            self.row_ind.append(self.row)
            self.col_ind.append(i)

            self.rhs.append(0)
            self.row += 1

    def set_bc_wedge(self):
        """
        Set BC's on wedge
        Be careful not to have any overlap when setting
        boundary and initial conditions
        """
        N = self.N

        for m in range(N+1):
            r = self.get_r(m)

            i = self.get_index(m, 0)

            self.data.append(1)
            self.row_ind.append(self.row)
            self.col_ind.append(i)

            self.rhs.append(self.problem.eval_bc(r, 2))
            self.row += 1

            i = self.get_index(m, N)
            self.data.append(1)
            self.row_ind.append(self.row)
            self.col_ind.append(i)

            self.rhs.append(self.problem.eval_bc(r, 1))
            self.row += 1

    def set_bc_arc(self, known):
        # Set BC at arc
        N = self.N

        for l in range(1, N):
            self.data.append(1)
            self.row_ind.append(self.row)
            self.col_ind.append(self.get_index(N, l))

            th = self.get_th(l)
            rhs_value = 0

            for J in range(self.Nlam):
                if known:
                    rhs_value += self.lam[J] * self.eval_B(J, th)
                else:
                    self.data.append(-self.eval_B(J, th))
                    self.row_ind.append(self.row)
                    self.col_ind.append(self.get_index_b(J))

            self.rhs.append(rhs_value)
            self.row += 1

    def set_chebyshev_match(self):
        """
        Match Chebyshev series to data on wedge
        """
        N = self.N
        for l in (0, N):
            self.data.append(1)
            self.row_ind.append(self.row)
            self.col_ind.append(self.get_index(N, l))

            th = self.get_th(l)

            for J in range(self.Nlam):
                self.data.append(-self.eval_B(J, th))
                self.row_ind.append(self.row)
                self.col_ind.append(self.get_index_b(J))

            self.rhs.append(0)
            self.row += 1

    def set_match_0(self):
        # Solution = 0 near tip
        N = self.N

        for m in range(1, N):
            if self.get_r(m) > self.match_r and m > 3:
                break

            for l in range(1, N):
                self.data.append(1)
                self.row_ind.append(self.row)
                self.col_ind.append(self.get_index(m, l))

                self.rhs.append(0)
                self.row += 1

    def set_fd2(self, m, l):
        """
        Create equation for second order finite difference scheme
        at the node (m, l)
        """
        r_1 = self.get_r(m-1/2)
        r = self.get_r(m)
        r1 = self.get_r(m+1/2)

        hr = self.hr
        hth = self.hth
        k = self.k

        local = np.zeros([3, 3])

        local[-1, -1] += 0
        local[0, -1] += 1/(hth**2*r**2)
        local[1, -1] += 0

        local[-1, 0] += r_1/(hr**2*r)
        local[0, 0] += k**2 - 2/(hth**2*r**2) - r1/(hr**2*r) - r_1/(hr**2*r)
        local[1, 0] += r1/(hr**2*r)

        local[-1, 1] += 0
        local[0, 1] += 1/(hth**2*r**2)
        local[1, 1] += 0

        th = self.get_th(l)
        f = self.problem.eval_f_polar(r, th)

        # 4th order
        self.rhs.append(f)

        self.copy_local(m, l, local)
        self.row += 1


    def set_fd4(self, m, l):
        """
        Create equation for fourth order finite difference scheme
        at the node (m, l)
        """
        r_2 = self.get_r(m-1)
        r_1 = self.get_r(m-1/2)
        r = self.get_r(m)
        r1 = self.get_r(m+1/2)
        r2 = self.get_r(m+1)

        hr = self.hr
        hth = self.hth
        k = self.k

        local = np.zeros([3, 3])

        # 4th order
        local[-1, -1] += (
            -hr/(24*hth**2*r*r_2**2) + 1/(12*hth**2*r_2**2)
            + r_1/(12*hr**2*r)
        )

        local[0, -1] += (
            hr**2/(12*hth**2*r**4)
            + k**2/12 + 5/(6*hth**2*r**2) - r1/(12*hr**2*r)
            - r_1/(12*hr**2*r)
        )

        local[1, -1] += (
            hr/(24*hth**2*r*r2**2) + 1/(12*hth**2*r2**2)
            + r1/(12*hr**2*r)
        )


        local[-1, 0] += (
            -hr*k**2/(24*r) - hr/(12*r**3)
            + hr/(12*hth**2*r*r_2**2) + k**2/12 - 1/(6*hth**2*r_2**2)
            + 5*r_1/(6*hr**2*r)
        )

        local[0, 0] += (hr**2*k**2/(12*r**2)
            - hr**2/(6*hth**2*r**4) + 2*k**2/3
            - 5/(3*hth**2*r**2) - 5*r1/(6*hr**2*r)
            - 5*r_1/(6*hr**2*r)
        )

        local[1, 0] += (
            hr*k**2/(24*r) + hr/(12*r**3)
            - hr/(12*hth**2*r*r2**2) + k**2/12 - 1/(6*hth**2*r2**2)
            + 5*r1/(6*hr**2*r)
        )

        local[-1, 1] += (
            -hr/(24*hth**2*r*r_2**2)
            + 1/(12*hth**2*r_2**2) + r_1/(12*hr**2*r)
        )

        local[0, 1] += (hr**2/(12*hth**2*r**4)
            + k**2/12 + 5/(6*hth**2*r**2) - r1/(12*hr**2*r)
            - r_1/(12*hr**2*r)
        )

        local[1, 1] += (
            hr/(24*hth**2*r*r2**2)
            + 1/(12*hth**2*r2**2) + r1/(12*hr**2*r)
        )

        th = self.get_th(l)
        f = self.problem.eval_f_polar(r, th)
        d_f_r = self.problem.eval_d_f_r(r, th)
        d2_f_r = self.problem.eval_d2_f_r(r, th)
        d2_f_th = self.problem.eval_d2_f_th(r, th)

        # 4th order
        self.rhs.append(d2_f_r*hr**2/12
            + d2_f_th*hth**2/12
            + d_f_r*hr**2/(12*r)
            + f*hr**2/(12*r**2)
            + f
        )

        self.copy_local(m, l, local)
        self.row += 1

    def copy_local(self, m, l, local):
        for dm, dl in it.product((-1, 0, 1), repeat=2):
            m1 = m + dm
            l1 = l + dl

            self.data.append(local[dm, dl])
            self.row_ind.append(self.row)
            self.col_ind.append(self.get_index(m1, l1))

    def get_z_interp(self, N, N2, Nlam, staple, problem, R1, R2,
        eval_g, eval_g_inv):
        """
        Return an interpolating function that approximates z

        N -- between 16 and 1024
        R1 -- smallest value of r for which the interpolating function must be
            accurate
        R2 -- largest value of r for which the interpolating function must be
            accurate
        """
        self.N = N
        self.Nlam = Nlam
        self.problem = problem

        self.eval_g = eval_g
        self.eval_g_inv = eval_g_inv

        self.k = problem.k
        self.a = problem.a
        self.nu = problem.nu

        self.R1 = R1
        self.R2 = R2

        self.calc_N_var()
        dim0 = self.dim0

        self.my_print('N = {}'.format(N))
        self.my_print('Nlam = {}'.format(Nlam))
        self.my_print('R0 = {}'.format(self.R0))
        self.my_print('R1 = {}'.format(self.R1))
        self.my_print('R2 = {}'.format(self.R2))
        self.my_print('match_r = {}'.format(self.match_r))

        # Calculate Chebyshev coefficients of the expected solution
        # (for comparison only)
        t_roots = get_chebyshev_roots(1024)
        boundary_data = np.zeros(len(t_roots))

        for i in range(len(t_roots)):
            th = eval_g(t_roots[i])
            boundary_data[i] = problem.eval_expected__no_w(R2, th)

        true_lam = np.polynomial.chebyshev.chebfit(t_roots, boundary_data, Nlam)
        self.my_print('First chebyshev coef not included: {}'.format(true_lam[-1]))

        true_lam = true_lam[:-1]

        self.new_system()

        self.set_bc_origin()
        self.set_bc_wedge()
        self.set_bc_arc(known=False)

        self.set_chebyshev_match()
        self.set_match_0()

        # Finite difference scheme
        for m in range(1, N):
            for l in range(1, N):
                self.set_fd4(m, l)

        # Extra equations to make sure system is full rank
        if staple:
            for m in range(N//2, N//2+3):
                for l in range(1, N):
                    self.set_fd2(m, l)

        shape = (max(self.row_ind)+1, max(self.col_ind)+1)
        self.my_print('System shape: {} x {}'.format(*shape))

        eng = matlab.engine.start_matlab()

        eng.workspace['row_ind'] = matlab.double([i+1 for i in self.row_ind])
        eng.workspace['col_ind'] = matlab.double([j+1 for j in self.col_ind])
        eng.workspace['data'] = matlab.double(self.data)
        eng.workspace['rhs'] = matlab.double(self.rhs)

        sol = eng.eval('sparse(row_ind, col_ind, data) \ rhs.\'')
        sol = np.array(sol._data)
        u = sol[:dim0]
        self.lam = sol[dim0:]

        eng.quit()

        # Calculate linear system residual
        M = csc_matrix((self.data, (self.row_ind, self.col_ind)))
        self.rhs = np.array(self.rhs)

        exp_sol = np.zeros(self.dim)
        for m in range(N+1):
            for l in range(N+1):
                r = self.get_r(m)
                th = self.get_th(l)
                exp_sol[self.get_index(m, l)] = self.problem.eval_expected__no_w(r, th)

        for J in range(Nlam):
            exp_sol[self.get_index_b(J)] = true_lam[J]

        residual = M.dot(exp_sol) - self.rhs
        self.my_print('residual(exp): {}'.format(np.max(np.abs(residual))))

        residual = M.dot(sol) - self.rhs
        self.my_print('residual(matlab): {}'.format(np.max(np.abs(residual))))

        self.my_print('Max norm of sol: {}'.format(np.max(np.abs(u))))

        # Measure error on arc
        error = []
        for m in range(N+1):
            for l in range(N+1):
                exp = problem.eval_expected__no_w(self.get_r(m), self.get_th(l))
                error.append(abs(u[self.get_index(m, l)] - exp))

        self.my_print('Error on arc: {}'.format(np.max(error)))

        if False:
            # Now that we know the boundary data, solve the system again
            # with a higher value of N
            N = self.N = N2
            self.my_print('N2 = {}'.format(N))

            self.calc_N_var()
            self.new_system()

            self.set_bc_origin()
            self.set_bc_wedge()
            self.set_bc_arc(known=True)

            # Finite difference scheme
            for m in range(1, N):
                for l in range(1, N):
                    self.set_fd4(m, l)

            # Solve the system exactly
            M = csc_matrix((self.data, (self.row_ind, self.col_ind)))
            self.rhs = np.array(self.rhs)

            u = spsolve(M, self.rhs)

        # Create interpolating function
        points = []
        data = []

        if self.problem.boundary.name == 'arc':
            # 1-dimensional interpolation
            for l in range(N+1):
                i = self.get_index(N, l)
                points.append(self.get_th(l))
                data.append(u[i])

            raw_interpolator = interp1d(points, data, kind='cubic')
            interpolator = lambda r, th: float(raw_interpolator(th))

        else:
            # 2-dimensional interpolation
            for m in range(N+1):
                # Don't bother interpolating far away from the
                # perturbed boundary
                if self.get_r(m+3) < self.R1:
                    continue

                for l in range(N+1):
                    i = self.get_index(m, l)
                    points.append((self.get_r(m), self.get_th(l)))
                    data.append(u[i])

            raw_interpolator = CloughTocher2DInterpolator(points, data)
            interpolator = lambda r, th: float(raw_interpolator(r, th))

        # Measure error on perturbed boundary
        interpolator_boundary = []
        u_boundary = []

        for th in abcoef.th_data:
            r = self.problem.boundary.eval_r(th)
            #if np.isnan(interpolator(r, th)): print(r,th)
            interpolator_boundary.append(interpolator(r, th))
            u_boundary.append(problem.eval_expected__no_w(r, th))

        error = np.max(np.abs(np.array(interpolator_boundary) - u_boundary))
        self.my_print('Error on perturbed boundary: {}'.format(error))

        return interpolator
