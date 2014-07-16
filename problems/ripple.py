import numpy as np
from numpy import sin, cos, sqrt

from .problem import Problem, Pizza
from cs.csf import CsFourier

class Ripple(Problem):
    k = 1
    solver_class = CsFourier

    def eval_expected_polar(self, r, th):
        return cos(self.k*r)

    def eval_d_u_r(self, th):
        k = self.k
        return -k*sin(k*self.R)

    def eval_f_polar(self, r, th):
        k = self.k
        if r != 0:
            return -k * sin(k*r)/r
        else:
            return -k**2

    def eval_d_f_r(self, r, th):
        k = self.k
        return -k**2 * cos(k*r)/r + k*sin(k*r)/r**2

    def eval_d2_f_r(self, r, th):
        k = self.k
        return k*(k**2*sin(k*r) + 2*k*cos(k*r)/r - 2*sin(k*r)/r**2)/r

    def eval_d_f_th(self, r, th):
        return 0

    def eval_d2_f_th(self, r, th):
        return 0

    def eval_d2_f_r_th(self, r, th):
        return 0

    def eval_grad_f(self, x, y):
        k = self.k
        grad = (-k**2*x*cos(k*sqrt(x**2 + y**2))/(x**2 + y**2) +
            k*x*sin(k*sqrt(x**2 + y**2))/(x**2 + y**2)**(3/2),
            -k**2*y*cos(k*sqrt(x**2 + y**2))/(x**2 + y**2) + 
            k*y*sin(k*sqrt(x**2 + y**2))/(x**2 + y**2)**(3/2))
        return np.array(grad)

    def eval_hessian_f(self, x, y):
        k = self.k
        hessian = ((k*(k**2*x**2*sin(k*sqrt(x**2 + y**2))/(x**2 + y**2)**(3/2) + 3*k*x**2*cos(k*sqrt(x**2 + y**2))/(x**2 + y**2)**2 - k*cos(k*sqrt(x**2 + y**2))/(x**2 + y**2) - 3*x**2*sin(k*sqrt(x**2 + y**2))/(x**2 + y**2)**(5/2) + sin(k*sqrt(x**2 + y**2))/(x**2 + y**2)**(3/2)), k*x*y*(k**2*sin(k*sqrt(x**2 + y**2))/(x**2 + y**2)**(3/2) + 3*k*cos(k*sqrt(x**2 + y**2))/(x**2 + y**2)**2 - 3*sin(k*sqrt(x**2 + y**2))/(x**2 + y**2)**(5/2))), (k*x*y*(k**2*sin(k*sqrt(x**2 + y**2))/(x**2 + y**2)**(3/2) + 3*k*cos(k*sqrt(x**2 + y**2))/(x**2 + y**2)**2 - 3*sin(k*sqrt(x**2 + y**2))/(x**2 + y**2)**(5/2)), k*(k**2*y**2*sin(k*sqrt(x**2 + y**2))/(x**2 + y**2)**(3/2) + 3*k*y**2*cos(k*sqrt(x**2 + y**2))/(x**2 + y**2)**2 - k*cos(k*sqrt(x**2 + y**2))/(x**2 + y**2) - 3*y**2*sin(k*sqrt(x**2 + y**2))/(x**2 + y**2)**(5/2) + sin(k*sqrt(x**2 + y**2))/(x**2 + y**2)**(3/2))))
        return np.array(hessian)
    
    
class RipplePizza(Pizza, Ripple):

    def eval_d_u_outwards(self, x, y, **kwargs):
        r, th = cart_to_polar(x, y)

        if 'sid' in kwargs:
            sid = kwargs['sid']
        else:
            sid = self.get_sid(th)

        a = self.a
        k = self.k

        if sid == 0:
            return super().eval_d_u_r(th)
        elif sid == 1 or sid == 2:
            return 0
