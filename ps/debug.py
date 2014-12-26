import numpy as np
from numpy import cos, sin
import scipy

import itertools as it

import matplotlib.pyplot as plt

from solver import cart_to_polar, Result
from chebyshev import get_chebyshev_roots
import matrices

class PsDebug:
    
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
        '''
        Do Chebyshev fits on the analytically-known Neumann data of 
        each segment to get the "exact" values of the coefficients c1. 
        '''
        t_data = get_chebyshev_roots(1000)
        self.c1 = []

        for sid in range(self.N_SEGMENT):
            boundary_data = np.zeros(len(t_data))
            
            for i in range(len(t_data)):
                arg = self.eval_g(sid, t_data[i])
                boundary_data[i] = self.problem.eval_d_u_outwards(arg, sid)

            n_basis = self.segment_desc[sid]['n_basis']
            self.c1.extend(np.polynomial.chebyshev.chebfit(
                t_data, boundary_data, n_basis-1))

    def etype_filter(self, sid, nodes, etypes):
        result = []
        for i, j in nodes:
            if self.get_etype(sid, i, j) in etypes:
                result.append((i, j))

        return result

    def test_extend_boundary(self, setypes=None):
        self.calc_c0()
        self.calc_c1_exact()
        
        error = []
        mv_ext = self.mv_extend_boundary()

        for node in mv_ext:
            x, y = self.get_coord(*node)
            exp = self.problem.eval_expected(x, y)  
            
            for data in mv_ext[node]:
                if setypes is None or data['setype'] in setypes:
                    diff = abs(exp - data['value'])
                    error.append(diff)
        
        if setypes is None:
            reduced = self.mv_reduce(mv_ext)
            
            for l in range(len(self.union_gamma)):
                node = self.union_gamma[l]
                x, y = self.get_coord(*node)
                exp = self.problem.eval_expected(x, y)  
            
                diff = abs(exp - reduced[l])
                error.append(diff)

        result = Result()
        result.error = max(error)
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

                ext2_array = self.extend_boundary() #- self.extend_inhomo_f()
                diff = np.abs(ext1_array - ext2_array)
                error.append(np.max(diff))
                print('index={}  JJ={}  error={}'.format(index, JJ, error[-1]))

        error = np.array(error)
        
        result = Result()
        result.error = np.max(np.abs(error))
        return result
        
    def ps_test_extend_src_f(self, pairs):
        error = []
        
        for sid, etype_name in pairs:
            etype = self.etypes[etype_name]
        
            for i, j in self.Kplus:
                x, y = self.get_coord(i, j)
                
                sid1 = self._extend_src_f_get_sid(i, j)
                etype1 = self.get_etype(sid, i, j)
                
                if sid1 == sid and etype1 == etype:
                    l = matrices.get_index(self.N, i, j)         
                    diff = abs(self.problem.eval_f(x, y) - self.src_f[l])
                    error.append(diff)                   
                        
        result = Result()
        result.error = max(error)
        return result

    def c0_test(self):
        '''
        Plot the Dirichlet data along with its Chebyshev expansion. If there
        is more than a minor difference between the two, something is wrong.
        '''
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

    def c1_test(self):
        '''
        Plot the reconstructed Neumann data, as approximated by a Chebyshev 
        expansion. If the Chebyshev expansion shows "spikes" near the
        interfaces of the segments, something is probably wrong.
        '''
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
            
        plt.plot(s_data, expansion_data, label='Expansion')
        plt.legend(loc=1)
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
        min_error = float('inf')

        start_circle = 40
        start_radius = 24
        
        for n_circle in range(start_circle, 50, 3):
            for n_radius in range(start_radius, n_circle, 3):
                self.setup_B_desc(n_circle, n_radius)
                
                self.calc_c0()
                self.calc_c1()

                ext = self.extend_boundary()
                u_act = self.get_potential(ext) #+ self.ap_sol_f

                error = self.eval_error(u_act)
                s ='n_circle={}    n_radius={}    error={}'.format(n_circle, n_radius, error)
                
                if error < min_error:
                    min_error = error
                    s = '!!!    ' + s
                    
                print(s)
                
    def plot_contour(self, u):
        N = self.N
    
        x_range = np.zeros(N-1)
        y_range = np.zeros(N-1)
        
        for i in range(1, N):
            x_range[i-1] = y_range[i-1] = self.get_coord(i, 0)[0]
        
        Z = np.zeros((N-1, N-1))
        for i, j in self.global_Mplus:            
            Z[i-1, j-1] = u[matrices.get_index(N, i, j)].real
    
        X, Y = np.meshgrid(x_range, y_range, indexing='ij')
        fig = plt.contourf(X, Y, Z)
        
        plt.colorbar(fig)
        
        self.plot_Gamma()
        plt.show()
