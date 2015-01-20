import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

from solver import cart_to_polar

from .problem import PizzaProblem
from .sympy_problem import SympyProblem
    
class FourierBessel(PizzaProblem):
    
    k = 1
    homogeneous = True
    expected_known = False
    
    shc_coef_len = 7

    def __init__(self, **kwargs): 
        super().__init__(**kwargs)

        self.nu = np.pi / (2*np.pi - self.a)
        self.calc_exact_shc_coef()
        self.test_shc_coef()

    def eval_phi0(self, th):
        return -(th - np.pi/6) * (th - 2*np.pi) 
        #return th-self.a
        
    def eval_bc_extended(self, arg, sid):
        a = self.a
        nu = self.nu
        
        r, th = self.arg_to_polar(arg, sid)
        
        if sid == 0:
            bc = self.eval_phi0(th)
            
            for m in range(1, len(self.shc_coef)+1):
                bc -= self.shc_coef[m-1] * np.sin(m*nu*(th - a))
            
            return bc
            
        elif sid == 1:
            return 0 
        elif sid == 2:
            return 0
        
    def calc_exact_shc_coef(self):
        k = self.k
        R = self.R
        a = self.a
        nu = self.nu
        
        def eval_integrand(th):
            phi = self.eval_phi0(th)
            return phi * np.sin(m*nu*(th - a))
    
        self.shc_coef = np.zeros(self.shc_coef_len)
        quad_error = []

        for m in range(1, len(self.shc_coef)+1):
            quad_result = quad(eval_integrand, a, 2*np.pi,
                epsabs=1e-11)
            quad_error.append(quad_result[1])
            
            coef = quad_result[0]
            coef *= 2 / (2*np.pi - a)

            self.shc_coef[m-1] = coef

        print('quad error:', max(quad_error))

    def test_shc_coef(self):
        a = self.a
        nu = self.nu

        th_data = np.linspace(self.a, 2*np.pi, 1000)
        exact_data = [self.eval_phi0(th) for th in th_data]
        
        expansion_data = []

        for th in th_data:
            phi0 = 0    
            for m in range(1, len(self.shc_coef)+1):
                phi0 += self.shc_coef[m-1] * np.sin(m*nu*(th - a))

            expansion_data.append(phi0)

        diff = np.max(np.abs(np.array(expansion_data) -
            np.array(exact_data)))
        print('shc_coef diff:', diff)

        plt.plot(th_data, exact_data, linewidth=5, color='#BBBBBB', label='Exact')
        plt.plot(th_data, expansion_data, label='Expansion')

        plt.show()
