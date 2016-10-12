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

    k = 6.75

    n_basis_dict = {
        16: (115, 30),
        #32: (33, 8),
        #64: (42, 12),
        #128: (65, 18),
        #256: (80, 30),
        #512: (80, 45),
        #1024: (80, 45),
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

    def __init__(self, **kwargs):
        k, r, th = sympy.symbols('k r th')
        sin = sympy.sin
        cos = sympy.cos
        kwargs['f_expr'] = (k**2 - 1) * r * sin(th) * cos(r * cos(th))
        super().__init__(**kwargs)

    def eval_expected(self, x, y):
        r, th = domain_util.cart_to_polar(x, y)
        return y*np.cos(x) + np.sin(self.k*x)

    #def get_grad(self, x, y):
    #    grad = np.array([-y*np.sin(x), np.cos(x)])
    #    return grad

class Smooth_YCos3(SympyProblem, SmoothProblem):
    """
    Inhomogeneous problem.
    """

    # WARNING: homogeneous if k=1

    def __init__(self, **kwargs):
        k, r, th = sympy.symbols('k r th')
        sin = sympy.sin
        cos = sympy.cos
        kwargs['f_expr'] = (k**2 - 1) * r * sin(th) * cos(r * cos(th))
        super().__init__(**kwargs)

    def eval_expected(self, x, y):
        return y*np.cos(x) + np.sin(self.k*x)

class Smooth_One(SympyProblem, SmoothProblem):

    def __init__(self, **kwargs):
        k, r, th = sympy.symbols('k r th')
        sin = sympy.sin
        cos = sympy.cos
        kwargs['f_expr'] = sympy.sympify(1)
        super().__init__(**kwargs)

    def eval_expected(self, x, y):
        return np.sin(self.k*x) + 1/self.k**2

    def get_grad(self, x, y):
        k = self.k
        return (k*np.cos(k*x) + 1/k**2, 0)

class Smooth_Two(SympyProblem, SmoothProblem):

    def __init__(self, **kwargs):
        r, k= sympy.symbols('r k')
        sin = sympy.sin
        cos = sympy.cos
        kwargs['f_expr'] = k**2*r + 1/r
        super().__init__(**kwargs)

    def eval_expected(self, x, y):
        r = np.sqrt(x**2 + y**2)
        return np.sin(self.k*x) + r

    #def get_grad(self, x, y):
    #    k = self.k
    #    return (k*np.cos(k*x) + 1/k**2, 0)

class Smooth_F0(SmoothProblem):

    def __init__(self, **kwargs):
        zero = lambda r, th: 0
        self.eval_d_f_r = zero
        self.eval_d2_f_r = zero
        self.eval_d_f_th = zero
        self.eval_d2_f_th = zero
        self.eval_d2_f_r_th = zero

    def eval_expected(self, x, y):
        return y*np.cos(x)

    def eval_f_polar(self, r, th):
        k = self.k
        return (k**2 - 1) * r * np.sin(th) * np.cos(r * np.cos(th))

    def get_grad(self, x, y):
        return (-y*np.sin(x), np.cos(x))

    def eval_grad_f(self, x, y):
        return self.eval_grad_f_polar(*domain_util.cart_to_polar(x, y))

    def eval_grad_f_polar(self, r, th):
        return np.array((0, 0), dtype=float)

    def eval_hessian_f(self, x, y):
        return self.eval_hessian_f_polar(*domain_util.cart_to_polar(x, y))

    def eval_hessian_f_polar(self, r, th):
        return np.zeros((2, 2), dtype=float)

class Smooth_F1(SympyProblem, SmoothProblem):

    def __init__(self, **kwargs):
        k, r, th = sympy.symbols('k r th')
        sin = sympy.sin
        cos = sympy.cos
        kwargs['f_expr'] = (k**2 - 1) * r * sin(th) * cos(r * cos(th))
        super().__init__(**kwargs)

        zero = lambda r, th: 0
        self.eval_d2_f_r = zero
        self.eval_d2_f_th = zero
        self.eval_d2_f_r_th = zero

    def eval_expected(self, x, y):
        return y*np.cos(x)

    def eval_f_polar(self, r, th):
        k = self.k
        return (k**2 - 1) * r * np.sin(th) * np.cos(r * np.cos(th))

    def get_grad(self, x, y):
        return (-y*np.sin(x), np.cos(x))

    def eval_hessian_f(self, x, y):
        return self.eval_hessian_f_polar(*domain_util.cart_to_polar(x, y))

    def eval_hessian_f_polar(self, r, th):
        return np.zeros((2, 2), dtype=float)
