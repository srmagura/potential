import numpy as np
import matplotlib.pyplot as plt

class CsDebug:

    def c0_test(self):
        n = 200
        th_data = np.linspace(0, 2*np.pi, n)
        exact_data = np.zeros(n)
        expansion_data = np.zeros(n)

        for l in range(n):
            th = th_data[l]

            exact_data[l] = self.problem.eval_bc(th).real
            for J, i in self.J_dict.items():
                exp = np.exp(complex(0, J*th))
                expansion_data[l] += (self.c0[i] * exp).real

        plt.plot(th_data, exact_data, label='Exact')
        plt.plot(th_data, expansion_data, 'o', label='Expansion')
        plt.legend()
        plt.title('c0')
        plt.show()

    def c1_test(self):
        n = 200
        th_data = np.linspace(0, 2*np.pi, n)
        exact_data = np.zeros(n)
        expansion_data = np.zeros(n)

        for l in range(n):
            th = th_data[l]

            exact_data[l] = self.problem.eval_d_u_r(th).real
            for J, i in self.J_dict.items():
                exp = np.exp(complex(0, J*th))
                expansion_data[l] += (self.c1[i] * exp).real

        plt.plot(th_data, exact_data, label='Exact')
        plt.plot(th_data, expansion_data, 'o', label='Expansion')
        plt.legend()
        plt.title('c1')
        plt.show()

    def calc_c1_exact(self):
        th_data = np.arange(0, 2*np.pi, 2*np.pi / fourier_N)
        boundary_data = [self.problem.eval_d_u_r(th) 
            for th in th_data] 
        c1_raw = np.fft.fft(boundary_data)

        self.c1 = np.zeros(len(self.c0), dtype=complex)
        for J, i in self.J_dict.items():
            self.c1[i] = c1_raw[J] / fourier_N

    def extension_test(self):
        self.calc_c0()
        self.calc_c1_exact()

        ext = self.extend_boundary()
        error = np.zeros(len(self.gamma), dtype=complex)
        for l in range(len(self.gamma)):
            x, y = self.get_coord(*self.gamma[l])
            error[l] = self.problem.eval_expected(x, y) - ext[l]

        return np.max(np.abs(error))
        
    def plot_Gamma(self):
        n = 256
        Gamma_x_data = np.zeros(n)
        Gamma_y_data = np.zeros(n)

        l = 0
        for th in np.linspace(0, 2*np.pi, n):
            Gamma_x_data[l] = self.R*np.cos(th)
            Gamma_y_data[l] = self.R*np.sin(th)
            l += 1

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
                
        x_data, y_data = self.nodes_to_plottable(self.gamma)         
        plt.plot(x_data, y_data, 'o')
        
        plt.axes().set_aspect('equal', 'datalim')
        plt.show()
