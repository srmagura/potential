import numpy as np
from numpy import cos, sin

from solver import SquareSolver, cart_to_polar
from cs.csf import CsFourier
from ps import PizzaSolver

class Problem:
    homogeneous = False

    def get_solver(self, *args, **kwargs):
        return self.solver_class(self, *args, **kwargs)

    def eval_bc(self, th):
        return self.eval_expected(self.R*cos(th), self.R*sin(th))

    def eval_expected(self, x, y):
        return self.eval_expected_polar(*cart_to_polar(x, y))

    def eval_f(self, x, y):
        return self.eval_f_polar(*cart_to_polar(x, y))

class Mountain(Problem):
    k = 3
    solver_class = SquareSolver

    def eval_expected(self, x, y):
        return np.cos(x/2)**4 * np.cos(y/2)**4

    def eval_f(self, x, y):
        return (self.k**2*cos(x/2)**4*cos(y/2)**4 + 
            (3*sin(x/2)**2 - cos(x/2)**2)*cos(x/2)**2*cos(y/2)**4 +
            (3*sin(y/2)**2 - cos(y/2)**2)*cos(x/2)**4*cos(y/2)**2)

class Ripple(Problem):
    k = 1
    solver_class = CsFourier

    def eval_expected_polar(self, r, th):
        return cos(r)

    def eval_d_u_r(self, th):
        return -sin(self.R)

    def eval_f_polar(self, r, th):
        if r != 0:
            return self.k**2*cos(r) - cos(r) - sin(r)/r
        else:
            return self.k**2*cos(r) - cos(r) - 1

    def eval_d_f_r(self, r, th):
        return -self.k**2*sin(r) + sin(r) - cos(r)/r + sin(r)/r**2

    def eval_d2_f_r(self, r, th):
        return -self.k**2*cos(r) + cos(r) + sin(r)/r + 2*cos(r)/r**2 - 2*sin(r)/r**3

    def eval_d2_f_th(self, r, th):
        return 0

class YCosine(Problem):
    k = 2/3
    solver_class = CsFourier

    def eval_expected(self, x, y):
        return y*cos(x)

    def eval_f(self, x, y):
        return self.k**2*y*cos(x) - y*cos(x)

    def eval_d_f_r(self, r, th):
        return -r*self.k**2*sin(th)*sin(r*cos(th))*cos(th) + r*sin(th)*sin(r*cos(th))*cos(th) + self.k**2*sin(th)*cos(r*cos(th)) - sin(th)*cos(r*cos(th))

    def eval_d2_f_r(self, r, th):
        return (-r*self.k**2*cos(th)*cos(r*cos(th)) + r*cos(th)*cos(r*cos(th)) - 2*self.k**2*sin(r*cos(th)) + 2*sin(r*cos(th)))*sin(th)*cos(th)

    def eval_d2_f_th(self, r, th):
        return r*(-r**2*self.k**2*sin(th)**2*cos(r*cos(th)) + r**2*sin(th)**2*cos(r*cos(th)) + 3*r*self.k**2*sin(r*cos(th))*cos(th) - 3*r*sin(r*cos(th))*cos(th) - self.k**2*cos(r*cos(th)) + cos(r*cos(th)))*sin(th)

class Wave(Problem):
    k = 1 
    kx = .8*k
    ky = .6*k

    solver_class = CsFourier
    homogeneous = True

    def eval_f(self, x, y):
        return 0

    def eval_expected(self, x, y):
        return np.exp(complex(0, self.kx*x + self.ky*y))

    def eval_d_u_r(self, th):
        a = self.kx*cos(th) + self.ky*sin(th)
        return 1j*a*np.exp(1j*self.R*a)

class Sine(Problem):
    k = 1 

    solver_class = CsFourier
    homogeneous = True

    def eval_f(self, x, y):
        return 0

    def eval_expected(self, x, y):
        return sin(self.k*x) #+ sin(self.k*y)

    def eval_d_u_r(self, th):
        k = self.k
        R = self.R
        return k*cos(th) * cos(k*R*cos(th))


class Pizza:
    a = np.pi / 6
    solver_class = PizzaSolver

    def eval_bc(self, x, y):
        return self.eval_expected(x, y)

    def get_sid(self, th):
        tol = 1e-12

        if th > self.a:
            return 0
        elif abs(th) < tol:
            return 1
        elif abs(th - self.a) < tol:
            return 2

        assert False


class SinePizza(Pizza, Sine):

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
        elif sid == 1:
            return 0
        elif sid == 2:
            return k*cos(k*x)*cos(a - np.pi/2)

class WavePizza(Pizza, Wave):
    pass

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

problem_dict = {
    'mountain': Mountain,
    'ripple': Ripple,
    'ripple-pizza': RipplePizza,
    'ycosine': YCosine,
    'wave': Wave,
    'wave-pizza': WavePizza,
    'sine': Sine,
    'sine-pizza': SinePizza
}
