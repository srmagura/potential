import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

k = 1

a = np.pi/6
nu = np.pi / (2*np.pi - a)

quad_error = []

class BData:

    analytic_known = False

    def calc_coef_numerical(self, M):
        def eval_integrand(th):
            phi = self.eval_phi0(th)
            return phi * np.sin(m*nu*(th - a))

        coef = np.zeros(M)

        for m in range(1, M+1):
            quad_result = quad(eval_integrand, a, 2*np.pi,
                limit=100, epsabs=1e-11)
            quad_error.append(quad_result[1])
            
            c = quad_result[0]
            c *= 2 / (2*np.pi - a)

            coef[m-1] = c

        return coef

    def calc_coef(self, M):
        if self.analytic_known:
            return self.calc_coef_analytic(M)
        else:
            return self.calc_coef_numerical(M)


class Parabola(BData):

    analytic_known = True

    def eval_phi0(self, th):
        return -(th - np.pi/6) * (th - 2*np.pi) 

    def calc_coef_analytic(self, M):
        coef = np.zeros(M)

        for m in range(1, M+1):
            if m % 2 == 0:
                coef[m-1] = 0
            else:
                coef[m-1] = 8/(2*np.pi - a) * 1/(m*nu)**3

        return coef

class Hat(BData):

    def eval_phi0(self, th):
        x = (2*th-(2*np.pi+a))/(2*np.pi-a)

        if abs(abs(x)-1) < 1e-7:
            return 0
        else:
            return np.exp(-1/(1-abs(x)**2))


def do_test(M, prev_error):
    th_data = np.linspace(a, 2*np.pi, 1000)
    exact_data = np.array([bdata.eval_phi0(th) for th in th_data])
    
    expansion_data = np.zeros(len(th_data))
    coef = bdata.calc_coef(M) 

    for i in range(len(th_data)):
        th = th_data[i]

        phi0 = 0 
        for m in range(1, M+1):
            phi0 += coef[m-1] * np.sin(m*nu*(th - a))

        expansion_data[i] = phi0

    error = np.max(np.abs(expansion_data - exact_data))

    print('---- M = {} ----'.format(M))
    print('Fourier error:', error)

    if prev_error is not None:
        conv = np.log2(prev_error / error)
        print('Convergence:', conv)


    print()

    #plt.plot(th_data, exact_data, linewidth=5, color='#BBBBBB', label='Exact')
    #plt.plot(th_data, expansion_data, label='Expansion')
    #plt.show()

    return error

def convergence_test():
    print('Boundary data: `{}`'.format(bdata.__class__.__name__))
    M = 4
    prev_error = None

    while M <= 128:
        prev_error = do_test(M, prev_error)
        M *= 2

def numerical_test():
    M = 7
    coef = bdata.calc_coef_analytic(M)
    coef_numerical = bdata.calc_coef_numerical(M)

    print(np.max(np.abs(coef - coef_numerical)))

bdata = Hat()
convergence_test()
