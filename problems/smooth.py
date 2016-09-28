import numpy as np
from scipy.special import jv, jvp
import sympy

import domain_util

from .problem import PizzaProblem
from .sympy_problem import SympyProblem
from .singular import HReg

class SmoothProblem(PizzaProblem):

    expected_known = True
    hreg = HReg.none

    k = 1

    n_basis_dict = {
        16: (24, 6),
        32: (33, 8),
        64: (42, 12),
        128: (65, 18),
        256: (80, 30),
        512: (80, 45),
        1024: (80, 45),
    }

    def eval_bc(self, arg, sid):
        """
        Evaluate extended boundary condition.
        """
        r, th = domain_util.arg_to_polar_abs(self.boundary, self.a, arg, sid)
        return self.eval_expected_polar(r, th)

    def eval_d_u_outwards(self, arg, sid):
        """
        Only for extension test.
        """
        k = self.k
        a = self.a

        r, th = domain_util.arg_to_polar_abs(self.boundary, a, arg, sid)
        x, y = r*np.cos(th), r*np.sin(th)

        grad = self.get_grad(x, y)

        if sid == 0:
            normal = self.boundary.eval_normal(th)
        elif sid == 1:
            normal = (0, 1)
        elif sid == 2:
            normal = (np.sin(a), -np.cos(a))

        return np.dot(grad, normal)



class Smooth_Sine(SmoothProblem):

    homogeneous = True

    def eval_expected(self, x, y):
        return np.sin(self.k*x)

    def get_grad(self, x, y):
        k = self.k
        return (k*np.cos(k*x), 0)


class Smooth_E(SmoothProblem):

    k = 5.5
    homogeneous = True

    def eval_expected(self, x, y):
        k = self.k

        self.kx = kx = .8*k
        self.ky = ky = .6*k

        return np.exp(1j*(kx*x + ky*y))

    def get_grad(self, x, y):
        kx = self.kx
        ky = self.ky

        u = np.exp(1j*(kx*x + ky*y))
        return (1j*kx*u, 1j*ky*u)


class Smooth_YCos(SympyProblem, SmoothProblem):
    """
    Inhomogeneous problem.
    """

    # WARNING: homogeneous if k=1
    k = 5.5

    def __init__(self, **kwargs):
        k, r, th = sympy.symbols('k r th')
        sin = sympy.sin
        cos = sympy.cos
        kwargs['f_expr'] = (k**2 - 1) * r * sin(th) * cos(r * cos(th))
        super().__init__(**kwargs)

    def eval_expected(self, x, y):
        return y*np.cos(x)

    def get_grad(self, x, y):
        return (-y*np.sin(x), np.cos(x))

class Smooth_YCos2(SympyProblem, SmoothProblem):
    """
    Inhomogeneous problem.
    """

    # WARNING: homogeneous if k=1
    k = 3

    def __init__(self, **kwargs):
        k, r, th = sympy.symbols('k r th')
        sin = sympy.sin
        cos = sympy.cos
        kwargs['f_expr'] = (k**2 - 1) * r * sin(th) * cos(r * cos(th)) + k**2*r**8 + 64*r**6
        super().__init__(**kwargs)

    def eval_expected(self, x, y):
        r, th = domain_util.cart_to_polar(x, y)
        return y*np.cos(x) + r**8

    def get_grad(self, x, y):
        grad = np.array([-y*np.sin(x), np.cos(x)])
        grad += np.array([8*x*(x**2 + y**2)**3, 8*y*(x**2 + y**2)**3])

        return grad
