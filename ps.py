import math
import numpy as np

from solver import Solver
import matrices

import matplotlib.pyplot as plt


class PizzaSolver(Solver):
    AD_len = 2*np.pi
    R = 2.3

    def __init__(self, problem, N, scheme_order, **kwargs):
        self.a = problem.sectorAngle
        super().__init__(problem, N, scheme_order, **kwargs)
        problem.R = self.R

    def is_interior(self, i, j):
        r, th = self.get_polar(i, j)
        return r <= self.R and (th >= self.a or th <= 0) 

    def g(self, sid, t):
        if sid == 0:
            return np.pi*t + np.pi + .5*self.a
        elif sid == 1 or sid == 2:
            return 6/10*self.R*t + self.R/2

    def g_inv(self, sid, arg):
        if sid == 0:
            th = arg
            return (th - np.pi - .5*self.a)/np.pi
        elif sid == 1 or sid == 2:
            r = arg
            return 10/(6*self.R)*(r - self.R/2)

    def calc_c0(self, segment_id):
        self.c0.append([])
        for J in range(N_BASIS):
            def integrand(t):
                return (w(t) * self.problem.eval_bc(segment_id, 
                    self.g(segment_id, t)) *    
                    eval_chebyshev(J, t))

            I = complex_quad(integrand, -1, 1)
            if J == 0:
                self.c0[segment_id].append(I/np.pi)
            else:
                self.c0[segment_id].append(2*I/np.pi)
    
    def extend_basis(self, J, index):
        ext = np.zeros(len(self.gamma), dtype=complex)
        for l in range(len(self.gamma)):
            i, j = self.gamma[l]
            r, th = self.get_polar(i, j)

        return ext

    def run(self):
        self.c0_test()

    def c0_test(self):
        sample = self.get_boundary_sample()

        s_data = np.zeros(len(sample))
        exact_data = np.zeros(len(sample))
        expansion_data = np.zeros(len(sample))

        for l in range(len(sample)):
            p = sample[l]
            s_data[l] = p['s']
            exact_data[l] = self.problem.eval_bc(p['x'], p['y']).real

            #for JJ in range(len(B_desc)):
            #    expansion_data[l] +=\
            #        (self.c0[JJ] * eval_B(JJ, th)).real

        plt.plot(s_data, exact_data, label='Exact')
        #plt.plot(th_data, expansion_data, 'o', label='Expansion')
        plt.legend()
        plt.ylim(-1, 1)
        plt.title('c0')
        plt.show()


    def get_boundary_sample(self):
        a = self.a
        th_data = np.arange(a, 2*np.pi, .05)
        r_data = np.linspace(0, self.R, 100)
        points = []

        for i in range(len(th_data)):
            th = th_data[i]

            points.append({'x': self.R * np.cos(th),
                'y': self.R * np.sin(th),
                's': self.R * (th - a),
                'sid': 0})

        for i in range(len(r_data)-1, -1, -1):
            r = r_data[i]

            points.append({'x': r,
                'y': 0,
                's': self.R*(2*np.pi - a + 1) - r,
                'sid': 1})

        for i in range(len(r_data)):
            r = r_data[i]

            points.append({'x': r*np.cos(a),
                'y': r*np.sin(a),
                's': self.R*(2*np.pi - a + 1) + r,
                'sid': 2})

        return points
