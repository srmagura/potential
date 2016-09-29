import numpy as np
from numpy import cos, sin
import scipy
from scipy.special import jv

import itertools as it

from solver import Result
import domain_util
from chebyshev import get_chebyshev_roots

class PsDebug:
    """
    A collection of functions for debugging PizzaSolver.

    In a normal execution of PizzaSolver, none of these functions are used.
    I.e. there's no core functionality in this module.
    """

    def get_boundary_sample(self, n=100, extended=False):
        ep = 1e-5
        th_data = np.linspace(self.a+ep, 2*np.pi-ep, 3*n)

        r_data = np.linspace(ep, self.R, n)

        points = []
        arg_datas = (th_data, r_data[::-1], r_data)
        for sid in range(self.N_SEGMENT):
            points.extend(
                self.get_boundary_sample_by_sid(sid, arg_datas[sid]))

        if not extended:
            return points

        dth = self.a/4
        extension1 = self.get_boundary_sample_by_sid(0,
            np.linspace(2*np.pi, 2*np.pi+dth, n))
        extension2 = self.get_boundary_sample_by_sid(0,
            np.linspace(self.a-dth, self.a, n))

        return (points, extension1, extension2)

    def get_boundary_sample_by_sid(self, sid, arg_data):
        a = self.a
        R = self.R

        points = []

        if sid == 0:
            for i in range(len(arg_data)):
                th = arg_data[i]
                r = self.boundary.eval_r(th)

                points.append({
                    'arg': th,
                    'x': r * np.cos(th),
                    'y': r * np.sin(th),
                    's': R * (th - a), # rough approximation of arclength
                    'sid': sid
                })

        elif sid == 1:
            for i in range(len(arg_data)):
                r = arg_data[i]

                points.append({
                    'arg': r,
                    'x': r,
                    'y': 0,
                    's': R*(2*np.pi - a + 1) - r,
                    'sid': sid
                })

        elif sid == 2:
            for i in range(len(arg_data)):
                r = arg_data[i]

                points.append({
                    'arg': r,
                    'x': r*np.cos(a),
                    'y': r*np.sin(a),
                    's': R*(2*np.pi - a + 1) + r,
                    'sid': sid
                })

        return points

    def calc_c1_exact(self):
        """
        Do Chebyshev fits on the Neumann data of each segment to get
        the coefficients c1.

        WILL NOT WORK for singular problems
        """
        self.c1 = []

        for sid in range(self.N_SEGMENT):
            def func(arg):
                return self.problem.eval_d_u_outwards(arg, sid)

            self.c1.extend(self.get_chebyshev_coef(sid, func))

    def etype_filter(self, sid, nodes, etypes):
        result = []
        for i, j in nodes:
            if self.get_etype(sid, i, j) in etypes:
                result.append((i, j))

        return result

    def c0_test(self, plot=True):
        """
        Plot the Dirichlet data along with its Chebyshev expansion. If there
        is more than a minor difference between the two, something is wrong.
        """
        k = self.k
        a = self.a
        nu = self.nu

        sample = self.get_boundary_sample()

        s_data = np.zeros(len(sample))
        exact_data = np.zeros(len(sample))
        expansion_data = np.zeros(len(sample))

        exact_data_outer = []
        expansion_data_outer = []

        for l in range(len(sample)):
            p = sample[l]
            s_data[l] = p['s']

            r, th = domain_util.cart_to_polar(p['x'], p['y'])
            sid = domain_util.get_sid(self.a, th)

            exact_data[l] = self.problem.eval_bc(p['arg'], sid).real

            if sid == 0:
                arg = th
            else:
                arg = r

            for JJ in range(len(self.B_desc)):
                expansion_data[l] +=\
                    (self.c0[JJ] *
                    self.eval_dn_B_arg(0, JJ, arg, sid)).real

            if sid == 0:
                exact_data_outer.append(exact_data[l])
                expansion_data_outer.append(expansion_data[l])

        print('c0 error:', np.max(np.abs(exact_data - expansion_data)))

        exact_data_outer = np.array(exact_data_outer)
        expansion_data_outer = np.array(expansion_data_outer)
        print('c0 error outer:', np.max(np.abs(exact_data_outer - expansion_data_outer)))

        #for l in range(len(exact_data)):
        #    diff = abs(exact_data[l] - expansion_data[l])
        #    if diff > .06:
        #        print(sample[l], 'diff:', diff)

        if not plot:
            return

        import matplotlib.pyplot as plt

        plt.plot(s_data, exact_data, linewidth=5, color='#BBBBBB', label='Exact')
        plt.plot(s_data, expansion_data, label='Expansion')

        plt.legend(loc=0)
        plt.title('c0')
        plt.xlabel('Arclength s')
        plt.ylabel('Dirichlet data')
        plt.show()


    def print_c1(self):
        sid = 0
        i = 0

        print()
        print('print_c1: not showing complex part.')
        for desc in self.segment_desc:
            n_basis = desc['n_basis']
            print('c1(segment {}): {}'.format(sid, n_basis))

            print(scipy.real(np.round(self.c1[i:i+n_basis], 2)))
            print()

            i += n_basis
            sid += 1

    def c1_test(self, plot=True):
        """
        Plot the reconstructed Neumann data, as approximated by a Chebyshev
        expansion. If the Chebyshev expansion shows "spikes" near the
        interfaces of the segments, something is probably wrong.
        """
        sample = self.get_boundary_sample()

        do_exact = hasattr(self.problem, 'eval_d_u_outwards')

        s_data = np.zeros(len(sample))
        expansion_data = np.zeros(len(sample))

        if do_exact:
            exact_data = np.zeros(len(sample))

        for l in range(len(sample)):
            p = sample[l]
            s_data[l] = p['s']

            if do_exact:
                exact_data[l] = self.problem.eval_d_u_outwards(p['arg'], p['sid']).real

            r, th = domain_util.cart_to_polar(p['x'], p['y'])
            sid = domain_util.get_sid(self.a, th)

            if sid == 0:
                arg = th
            else:
                arg = r

            for JJ in range(len(self.B_desc)):
                expansion_data[l] +=\
                    (self.c1[JJ] *
                    self.eval_dn_B_arg(0, JJ, arg, sid)).real

        import matplotlib.pyplot as plt

        if do_exact:
            error = np.max(np.abs(exact_data - expansion_data))
            print('c1 error: {}'.format(error))

            plt.plot(s_data, exact_data, linewidth=5, color='#BBBBBB', label='Exact')

        if not plot:
            return

        plt.plot(s_data, expansion_data, label='Expansion')
        plt.legend(loc=2)
        #plt.ylim(-1.5, 1.5)
        plt.title('c1')
        plt.xlabel('Arclength s')
        plt.ylabel('Reconstructed Neumann data')
        plt.show()

    def Q_residual(self):
        self.calc_c0()
        self.calc_c1_exact()

        gamma_info = []
        for node in self.union_gamma:
            r, th = self.get_polar(*node)
            gamma_info.append([node[0], node[1], r, th])

        np.savetxt('gamma_info.dat', np.array(gamma_info))
        Q1, rhs = self.get_var()
        residual = np.abs(Q1.dot(self.c1) - rhs)
        np.savetxt('{}_Qresidual.dat'.format(self.N), residual)

        print(residual)
        print()
        print('Max(residual):', np.max(residual))


    def color_plot(self, u=None):
        N = self.N

        x_range = np.zeros(N-1)
        y_range = np.zeros(N-1)

        for i in range(1, N):
            x_range[i-1] = y_range[i-1] = self.get_coord(i, 0)[0]

        Z = np.zeros((N-1, N-1))
        for i, j in self.global_Mplus:
            if u is not None:
                Z[i-1, j-1] = u[matrices.get_index(N, i, j)].real
            else:
                x, y = self.get_coord(i, j)
                Z[i-1, j-1] = self.problem.eval_expected(x, y)

        X, Y = np.meshgrid(x_range, y_range, indexing='ij')

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        p = ax.pcolor(X, Y, Z,
            vmin=np.min(Z),
            vmax=np.max(Z)
        )
        ax.axis('tight')

        cb = fig.colorbar(p, ax=ax)

        self.plot_Gamma()
        plt.show()
