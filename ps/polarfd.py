import numpy as np
import itertools as it
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve


class PolarFD:

    # rec: .05
    R0 = .05

    @classmethod
    def my_print(cls, x):
        pass
        #print('(polarfd) ' + str(x))

    def get_r(self, m):
        return self.R0 + m*self.hr

    def get_th(self, l):
        return self.a + l*self.hth

    def get_index(self, m, l):
        return m*(self.N+1) + l

    def calc_N_var(self):
        """ Calculate variables that depend on N """
        self.hr = (self.R1-self.R0)/self.N
        self.hth = (2*np.pi - self.a) / self.N

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

            self.rhs.append(self.eval_phi2(r))
            self.row += 1

            i = self.get_index(m, N)
            self.data.append(1)
            self.row_ind.append(self.row)
            self.col_ind.append(i)

            self.rhs.append(self.eval_phi1(r))
            self.row += 1

    def set_bc_arc(self):
        # Set BC at arc
        N = self.N

        for l in range(1, N):
            self.data.append(1)
            self.row_ind.append(self.row)
            self.col_ind.append(self.get_index(N, l))

            th = self.get_th(l)

            self.rhs.append(self.eval_phi0(th))
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
        f = self.eval_f(r, th)
        d_f_r = self.eval_d_f_r(r, th)
        d2_f_r = self.eval_d2_f_r(r, th)
        d2_f_th = self.eval_d2_f_th(r, th)

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

    def solve(self, N, k, a, nu, R1, eval_phi0, eval_phi1, eval_phi2,
        eval_f, eval_d_f_r, eval_d2_f_r, eval_d2_f_th):
        """

        """
        self.N = N

        self.k = k
        self.a = a
        self.nu = nu

        self.R1 = R1

        self.eval_phi0 = eval_phi0
        self.eval_phi1 = eval_phi1
        self.eval_phi2 = eval_phi2

        self.eval_f = eval_f
        self.eval_d_f_r = eval_d_f_r
        self.eval_d2_f_r = eval_d2_f_r
        self.eval_d2_f_th = eval_d2_f_th

        self.calc_N_var()

        self.my_print('N = {}'.format(N))
        self.my_print('R0 = {}'.format(self.R0))
        self.my_print('R1 = {}'.format(self.R1))

        self.new_system()

        self.set_bc_origin()
        self.set_bc_wedge()
        self.set_bc_arc()

        # Finite difference scheme
        for m in range(1, N):
            for l in range(1, N):
                self.set_fd4(m, l)

        shape = (max(self.row_ind)+1, max(self.col_ind)+1)
        self.my_print('System shape: {} x {}'.format(*shape))

        # Calculate linear system residual
        M = csc_matrix((self.data, (self.row_ind, self.col_ind)))
        self.rhs = np.array(self.rhs)
        return spsolve(M, self.rhs)

    def calc_rel_convergence(self, u0, u1, u2):
        """
        Calculate the relative convergence of the sequence (u0, u1, u2)
        where the u's are numpy arrays.
        """
        N = self.N
        diff12 = []
        diff01 = []

        for m in range(1, N):
            for l in range(1, N):
                k0 = (m, l)
                k1 = (m//2, l//2)
                k2 = (m//4, l//4)

                if m % 4 == 0 and l % 4 == 0:
                    diff12.append(abs(u1[k1] - u2[k2]))

                if m % 2 == 0 and l % 2 == 0:
                    diff01.append(abs(u0[k0] - u1[k1]))

        return np.log2(max(diff12) / max(diff01))
