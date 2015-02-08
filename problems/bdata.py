import numpy as np
from scipy.integrate import quad

a = np.pi/6
nu = np.pi / (2*np.pi - a)


class BData:

    analytic_known = False

    def __init__(self):
        self.quad_error = []

    def calc_coef_numerical(self, M):
        def eval_integrand(th):
            phi = self.eval_phi0(th)
            return phi * np.sin(m*nu*(th - a))

        coef = np.zeros(M)

        for m in range(1, M+1):
            quad_result = quad(eval_integrand, a, 2*np.pi,
                limit=100, epsabs=1e-11)
            self.quad_error.append(quad_result[1])
            
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

        if abs(abs(x)-1) < 1e-7 or abs(x) > 1:
            return 0
        else:
            arg = -1/(1-abs(x)**2)
            return np.exp(arg)
