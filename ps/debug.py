import numpy as np
from numpy import cos, sin
import scipy

import itertools as it

import matplotlib.pyplot as plt

from solver import cart_to_polar, Result
from chebyshev import get_chebyshev_roots
import matrices

class PsDebug:

    def calc_c1_exact(self):
        t_data = get_chebyshev_roots(1000)
        self.c1 = []

        for sid in range(self.N_SEGMENT):
            arg_data = [self.eval_g(sid, t) for t in t_data] 
            boundary_points = self.get_boundary_sample_by_sid(sid, arg_data)
            boundary_data = np.zeros(len(arg_data), dtype=complex)
            for l in range(len(boundary_points)):
                p = boundary_points[l]
                boundary_data[l] = self.problem.eval_d_u_outwards(
                    p['x'], p['y'], sid=sid)

            n_basis = self.segment_desc[sid]['n_basis']
            self.c1.extend(np.polynomial.chebyshev.chebfit(
                t_data, boundary_data, n_basis-1))

    def etype_filter(self, sid, nodes, etypes):
        result = []
        for i, j in nodes:
            if self.get_etype(sid, i, j) in etypes:
                result.append((i, j))

        return result

    def test_extend_boundary(self, _pairs=None):
        pairs = set()
    
        if _pairs is None:
            for sid in range(3):
                for etype in self.etypes.values():
                    pairs.add((sid, etype))
        else:
            for sid, etype_name in _pairs:
                pairs.add((sid, self.etypes[etype_name]))

        self.calc_c0()
        self.calc_c1_exact()
        
        all_error = 0
        all_ext = self.extend_boundary()

        for sid, pair_etype in pairs:
            gamma = self.all_gamma[sid]
            ext = all_ext[sid]
            error = np.zeros(len(gamma), dtype=complex)
            
            for l in range(len(gamma)):
                i, j = gamma[l]
                x, y = self.get_coord(i, j)
                etype = self.get_etype(sid, i, j)
                
                if etype == pair_etype:
                    error[l] = self.problem.eval_expected(x, y) - ext[l]
                
            segment_error = np.max(np.abs(error))
            
            if segment_error > all_error:
                all_error = segment_error

        result = Result()
        result.error = all_error
        result.u_act = None
        return result

    def test_extend_basis(self):
        error = []
        for index in (0, 1):
            for JJ in range(len(self.B_desc)):
                ext1 = self.extend_basis(JJ, index)

                self.c0 = np.zeros(len(self.B_desc))
                self.c1 = np.zeros(len(self.B_desc))

                if index == 0:
                    self.c0[JJ] = 1
                elif index == 1:
                    self.c1[JJ] = 1

                ext2 = self.extend_boundary() - self.extend_inhomo_f()
                diff = np.abs(ext1-ext2)
                error.append(np.max(diff))
                print('index={}  JJ={}  error={}'.format(index, JJ, error[-1]))
                
                for l in range(len(self.gamma)):
                    if diff[l] > 1e-13:
                        i, j = self.gamma[l]
                        x, y = self.get_coord(i, j)
                        print('at {}   ext1={}    ext2={}'.format((x, y), ext1[l], ext2[l]))

        error = np.array(error)
        
        result = Result()
        result.error = np.max(np.abs(error))
        result.u_act = None
        return result

    def test_extend_src_f_etype(self, etypes=None):
        if etypes is None:
            etypes = self.etypes.values()

        errors = []
        for i, j in self.Kplus - self.Mplus:
            x, y = self.get_coord(i, j)
            etype = self.get_etype(i, j)

            l = matrices.get_index(self.N, i, j)
            if etype in etypes:
                a = self.problem.eval_f(x, y)
                b = self.src_f[l]
                errors.append(abs(a-b))

        return max(errors)

    def c0_test(self):
        sample = self.get_boundary_sample()

        s_data = np.zeros(len(sample))
        exact_data = np.zeros(len(sample))
        expansion_data = np.zeros(len(sample))

        for l in range(len(sample)):
            p = sample[l]
            s_data[l] = p['s']
            exact_data[l] = self.problem.eval_bc(p['x'], p['y']).real          
            
            r, th = cart_to_polar(p['x'], p['y'])
            for JJ in range(len(self.B_desc)):
                expansion_data[l] +=\
                    (self.c0[JJ] *
                    self.eval_dn_B_arg(0, JJ, r, th)).real

        plt.plot(s_data, exact_data, linewidth=5, color='#BBBBBB', label='Exact')
        plt.plot(s_data, expansion_data, label='Expansion')
        plt.legend(loc=0)
        #plt.ylim(-1, 1)
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
                exact_data[l] = self.problem.eval_d_u_outwards(p['x'], p['y']).real

            r, th = cart_to_polar(p['x'], p['y'])            
            for JJ in range(len(self.B_desc)):                             
                expansion_data[l] +=\
                    (self.c1[JJ] *
                    self.eval_dn_B_arg(0, JJ, r, th)).real        

        if do_exact:
            plt.plot(s_data, exact_data, linewidth=5, color='#BBBBBB', label='Exact')
            
        plt.plot(s_data, expansion_data, label='Expansion')
        plt.legend(loc=4)
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

        for l in range(len(nodes)):
            x, y = self.get_coord(*nodes[l])
            x_data[l] = x
            y_data[l] = y

        return x_data, y_data

    def plot_gamma(self, plot_rpoints=False):
        self.plot_Gamma()

        #for etype_name, etype_int in self.etypes.items():
        #    nodes = self.gamma_filter({etype_int})
        
        colors = ('red', 'green', 'blue')
        markers = ('o', 'x', '^')
        
        for sid in range(3):
            gamma = self.all_gamma[sid]
            x_data, y_data = self.nodes_to_plottable(gamma)
            
            label_text = '$\gamma$ seg{}'.format(sid)
            plt.plot(x_data, y_data, markers[sid], label=label_text,
                mfc='none', mec=colors[sid], mew=1)


        plt.title('$\gamma$ nodes')
        plt.xlim(-4,4)
        plt.ylim(-4,4)
        plt.legend(loc=3)
        plt.show()

    def gen_fake_gamma(self):
        def ap():
            self.gamma.append(self.get_coord_inv(x, y))

        self.gamma = []
        h = self.AD_len / self.N
        a = self.a
        
        for r in np.arange(.1, self.R, .2):
            x = r
            y = h/5
            ap()

            y = -h/5
            ap()

            x = r*cos(a) + h*sin(a)
            y = r*sin(a) + h*cos(a)
            ap()

            x = r*cos(a) - h*sin(a)/5
            y = r*sin(a) - h*cos(a)/5
            ap()

        for th in np.arange(self.a+.001, 2*np.pi, .1):
            r = self.R + h/2
            x = r*cos(th)
            y = r*sin(th)
            ap()

            r = self.R - h/2
            x = r*cos(th)
            y = r*sin(th)
            ap()
        
        for t in (True, False):
            r = self.R + h
            b = np.pi*h/10
            if t:
                b = self.a - b 
            x = r*cos(b)
            y = r*sin(b)
            ap()

            r = self.R + h
            b = np.pi*h/5
            if t:
                b = self.a - b 
            x = r*cos(b)
            y = r*sin(b)
            ap()
            
    def optimize_n_basis(self):
        min_error = float('inf')
        
        for n_circle in range(5, 50):
            for n_radius in range(max(5,n_circle-5), n_circle+1):
                self.setup_B_desc(n_circle, n_radius)
                
                self.calc_c0()
                self.calc_c1()

                ext = self.extend_boundary()
                u_act = self.get_potential(ext) + self.ap_sol_f

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
        for i, j in self.Mplus:            
            Z[i-1, j-1] = u[matrices.get_index(N, i, j)].real
    
        X, Y = np.meshgrid(x_range, y_range, indexing='ij')
        fig = plt.contourf(X, Y, Z)
        
        plt.colorbar(fig)
        
        self.plot_Gamma()
        plt.show()
        
    def _test_extend_basis_not(self, index):
        Q = self.get_Q(index, ext_only=True)
        error = []
        
        for JJ in range(len(self.B_desc)):
            for l in range(len(self.gamma)):
                JJ_sid = self.sid_by_JJ[JJ]
                l_sid = self.sid_by_gamma_l[l]
                
                i, j = self.gamma[l]
                etype = self.get_etype(i,j)
                x, y = self.get_coord(i, j)
                
                if JJ_sid != l_sid:
                    if abs(Q[l, JJ]) > 1e-5:
                        print('Basis on {}, node on {}'.format(JJ_sid, l_sid))
                        
                    error.append(abs(Q[l, JJ]))
                else:
                    if abs(Q[l, JJ]) < 1e-15 and index != 1:
                        print('sid={}  x,y={}'.format(JJ_sid, (x,y)))
                    
        print('Q{} error: {}'.format(index, max(error)))
           
    def test_extend_basis_not(self):
        self._test_extend_basis_not(0)
        self._test_extend_basis_not(1)
        
    def _test_with_c1(self):
        Q0 = self.get_Q(0)
        Q1 = self.get_Q(1)
        
        result = Q0.dot(self.c0) + Q1.dot(self.c1)
        result = np.abs(result)
        
        return result
        
    def test_with_c1_exact(self):
        self.calc_c1_exact()       
        result_exact = self._test_with_c1()
        
        self.calc_c1()
        result_actual = self._test_with_c1()
       
        for l in range(len(self.gamma)):
            i, j = self.gamma[l]
            x, y = self.get_coord(i, j)
            etype = self.get_etype(i, j)
            etypes = {self.etypes['radius2'], self.etypes['outer2']}
            
            if etype in etypes and result_exact[l] > 1e-4:
                print('error={}   at {}'.format(result_exact[l], (x,y)))
        
        print('test_with_c1 exact error: {}'.format(np.max(result_exact)))
        print('test_with_c1 actual error: {}'.format(np.max(result_actual)))
                
        
