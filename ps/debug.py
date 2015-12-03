import numpy as np
from numpy import cos, sin
import scipy

import itertools as it

import matplotlib
import matplotlib.pyplot as plt

from solver import cart_to_polar, Result
from chebyshev import get_chebyshev_roots

class PsDebug:
    """
    A collection of functions for debugging PizzaSolver.

    In a normal execution of PizzaSolver, none of these functions are used.
    I.e. there's no core functionality in this module.
    """

    def get_boundary_sample(self, n=100):
        ep = 1e-5
        th_data = np.linspace(self.a+ep, 2*np.pi-ep, 3*n)

        r_data = np.arange(ep, self.R, self.R/n)

        points = []
        arg_datas = (th_data, r_data[::-1], r_data)
        for sid in range(self.N_SEGMENT):
            points.extend(
                self.get_boundary_sample_by_sid(sid, arg_datas[sid]))

        return points

    def get_boundary_sample_by_sid(self, sid, arg_data):
        a = self.a
        R = self.R

        points = []

        if sid == 0:
            for i in range(len(arg_data)):
                th = arg_data[i]

                points.append({
                    'arg': th,
                    'x': R * np.cos(th),
                    'y': R * np.sin(th),
                    's': R * (th - a),
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

    def test_extend_basis(self):
        error = []
        for index in (0, 1):
            for JJ in range(len(self.B_desc)):
                ext1_array = self.extend_basis(JJ, index)

                self.c0 = np.zeros(len(self.B_desc))
                self.c1 = np.zeros(len(self.B_desc))

                if index == 0:
                    self.c0[JJ] = 1
                elif index == 1:
                    self.c1[JJ] = 1

                ext2_array = self.extend_boundary({'homogeneous_only': True})
                diff = np.abs(ext1_array - ext2_array)
                error.append(np.max(diff))
                print('index={}  JJ={}  error={}'.format(index, JJ, error[-1]))

        error = np.array(error)

        result = Result()
        result.error = np.max(np.abs(error))
        return result

    def ps_test_extend_src_f(self, setypes=None):
        error = []

        for node in self.Kplus:
            x, y = self.get_coord(*node)

            sid = self._extend_src_f_get_sid(*node)
            etype = self.get_etype(sid, *node)

            if setypes is None or (sid, etype) in setypes:
                l = matrices.get_index(self.N, *node)
                diff = abs(self.problem.eval_f(x, y) - self.src_f[l])
                error.append(diff)

        result = Result()
        result.error = max(error)
        return result

    def c0_test(self, plot=True):
        """
        Plot the Dirichlet data along with its Chebyshev expansion. If there
        is more than a minor difference between the two, something is wrong.
        """
        sample = self.get_boundary_sample()

        s_data = np.zeros(len(sample))
        exact_data = np.zeros(len(sample))
        expansion_data = np.zeros(len(sample))

        for l in range(len(sample)):
            p = sample[l]
            s_data[l] = p['s']

            r, th = cart_to_polar(p['x'], p['y'])
            sid = self.get_sid(th)

            exact_data[l] = self.problem.eval_bc_extended(p['arg'], sid).real

            for JJ in range(len(self.B_desc)):
                expansion_data[l] +=\
                    (self.c0[JJ] *
                    self.eval_dn_B_arg(0, JJ, r, th)).real

        print('c0 error:', np.max(np.abs(exact_data - expansion_data)))

        if not plot:
            return

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

            r, th = cart_to_polar(p['x'], p['y'])
            for JJ in range(len(self.B_desc)):
                expansion_data[l] +=\
                    (self.c1[JJ] *
                    self.eval_dn_B_arg(0, JJ, r, th)).real

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

    def plot_Gamma(self):
        sample = self.get_boundary_sample()

        n = len(sample) + 1
        Gamma_x_data = np.zeros(n)
        Gamma_y_data = np.zeros(n)

        for l in range(n):
            p = sample[l % len(sample)]
            Gamma_x_data[l] = p['x']
            Gamma_y_data[l] = p['y']

        plt.plot(Gamma_x_data, Gamma_y_data, color='black')

    def nodes_to_plottable(self, nodes):
        x_data = np.zeros(len(nodes))
        y_data = np.zeros(len(nodes))

        l = 0
        for i, j in nodes:
            x, y = self.get_coord(i, j)
            x_data[l] = x
            y_data[l] = y
            l += 1

        return x_data, y_data

    def plot_gamma(self):
        self.plot_Gamma()

        colors = ('red', 'green', 'blue')
        markers = ('o', 'x', '^')

        for sid in range(3):
            gamma = self.all_gamma[sid]
            x_data, y_data = self.nodes_to_plottable(gamma)

            label_text = '$\gamma_{}$'.format(sid)
            plt.plot(x_data, y_data, markers[sid], label=label_text,
                mfc='none', mec=colors[sid], mew=1)

        #plt.title('$\gamma$ nodes')
        plt.xlim(-4,4)
        plt.ylim(-4,4)
        plt.legend(loc=3)
        plt.show()

    # Old?
    def plot_union_gamma(self):
        self.plot_Gamma()

        for sid in range(3):
            gamma = self.all_gamma[sid]
            x_data, y_data = self.nodes_to_plottable(gamma)

            plt.plot(x_data, y_data, 'o', mfc='purple')

        plt.title('union_gamma')
        plt.xlim(-3,3)
        plt.ylim(-3,3)
        plt.show()

    def optimize_n_basis(self):
        """
        Utility function for selecting a good number of basis functions.
        Too many or too few basis functions will introduce numerical error.
        True solution must be known.

        Run without the -c flag. Run the program several times, varying
        the value of the -N option.

        There may be a way to improve on this brute force method.
        """
        min_error = float('inf')

        # Tweak the following ranges as needed
        for n_circle in range(43, 70, 3):
            for n_radius in range(27, int(.7*n_circle), 2):
                self.setup_basis(n_circle, n_radius)

                self.calc_c0()
                self.solve_var()

                ext = self.extend_boundary()
                u_act = self.get_potential(ext) + self.ap_sol_f

                error = self.eval_error(u_act)
                s =('n_circle={}    n_radius={}    error={}'
                    .format(n_circle, n_radius, error))

                if error < min_error:
                    min_error = error
                    s = '!!!    ' + s

                print(s)

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

        fig, ax = plt.subplots()
        p = ax.pcolor(X, Y, Z,
            vmin=np.min(Z),
            vmax=np.max(Z)
        )
        ax.axis('tight')

        cb = fig.colorbar(p, ax=ax)

        self.plot_Gamma()
        plt.show()
