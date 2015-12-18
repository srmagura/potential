from sympy import *
import numpy as np

from solver import cart_to_polar

class SympyProblem:
    """
    Use SymPy to symbolically differentiate the source function f.
    Calculates up to 2nd order derivatives of f WRT r and th. Uses
    the chain rule to compute the gradient and Hessian of f from its
    polar derivatives.
    """

    def __init__(self, **kwargs):
        """
        Requires a keyword argument f which is a SymPy expression
        in k, R, r, and th.
        """
        f = kwargs.pop('f_expr')

        args = symbols('k R r th')
        k, R, r, th = args

        # lambdify keyward arguments
        if 'lambdify_module' in kwargs:
            self.lkw = {'modules': kwargs['lambdify_module']}
        else:
            self.lkw = {}

        self.f_polar_lambda = lambdify(args, f, **self.lkw)

        # If using 2nd order scheme, don't need derivatives of f
        if kwargs['scheme_order'] > 2:
            self.do_diff(f)

        super().__init__(**kwargs)

    def do_diff(self, f):
        """
        Find polar derivatives symbolically.
        """
        args = symbols('k R r th')
        k, R, r, th = args

        d_f_r = diff(f, r)
        self.d_f_r_lambda = lambdify(args, d_f_r, **self.lkw)

        d2_f_r = diff(f, r, 2)
        self.d2_f_r_lambda = lambdify(args, d2_f_r, **self.lkw)

        d_f_th = diff(f, th)
        self.d_f_th_lambda = lambdify(args, d_f_th, **self.lkw)

        d2_f_th = diff(f, th, 2)
        self.d2_f_th_lambda = lambdify(args, d2_f_th, **self.lkw)

        d2_f_r_th = diff(f, r, th)
        self.d2_f_r_th_lambda = lambdify(args, d2_f_r_th, **self.lkw)

    def eval_f_polar(self, r, th):
        """Evaluate f at (r, th)"""
        return self.f_polar_lambda(self.k, self.R, r, th)

    def eval_d_f_r(self, r, th):
        """Partial of f WRT r"""
        return self.d_f_r_lambda(self.k, self.R, r, th)

    def eval_d2_f_r(self, r, th):
        """2nd partial of f WRT to r"""
        return self.d2_f_r_lambda(self.k, self.R, r, th)

    def eval_d_f_th(self, r, th):
        """Partial of f WRT to th"""
        return self.d_f_th_lambda(self.k, self.R, r, th)

    def eval_d2_f_th(self, r, th):
        """2nd partial of f WRT to th"""
        return self.d2_f_th_lambda(self.k, self.R, r, th)

    def eval_d2_f_r_th(self, r, th):
        """Partial of f WRT r and th"""
        return self.d2_f_r_th_lambda(self.k, self.R, r, th)

    def eval_grad_f(self, x, y):
        """
        Evaluate gradient of f at (x, y), using the chain rule. Returns the
        result as a NumPy array.
        """
        r, th = cart_to_polar(x, y)
        d_f_r = self.eval_d_f_r(r, th)
        d_f_th = self.eval_d_f_th(r, th)

        d_f_x = d_f_r * np.cos(th) - d_f_th * np.sin(th) / r
        d_f_y = d_f_r * np.sin(th) + d_f_th * np.cos(th) / r

        return np.array((d_f_x, d_f_y))

    def eval_hessian_f(self, x, y):
        """
        Evaluate Hessian of f at (x, y), using the chain rule. Returns the
        result as a 2x2 NumPy array.
        """
        # Get derivatives of f WRT r and th
        r, th = cart_to_polar(x, y)
        d_f_r = self.eval_d_f_r(r, th)
        d_f_th = self.eval_d_f_th(r, th)

        d2_f_r = self.eval_d2_f_r(r, th)
        d2_f_r_th = self.eval_d2_f_r_th(r, th)
        d2_f_th = self.eval_d2_f_th(r, th)

        # Calculate d2_f_x
        d_f_x_r = d2_f_r * np.cos(th)
        d_f_x_r += d2_f_r_th * (-np.sin(th) / r)
        d_f_x_r += d_f_th * np.sin(th) / r**2

        d_f_x_th = d2_f_r_th * np.cos(th)
        d_f_x_th += d_f_r * (-np.sin(th))
        d_f_x_th += d2_f_th * (-np.sin(th) / r)
        d_f_x_th += d_f_th * (-np.cos(th) / r)

        d2_f_x = d_f_x_r * np.cos(th) + d_f_x_th * (-np.sin(th) / r)

        # Calculate d2_f_x_y
        d2_f_x_y = d_f_x_r * np.sin(th) + d_f_x_th * np.cos(th) / r

        # Calculate d2_f_y
        d_f_y_r = d2_f_r * np.sin(th)
        d_f_y_r += d2_f_r_th * np.cos(th) / r
        d_f_y_r += d_f_th * (-np.cos(th) / r**2)

        d_f_y_th = d2_f_r_th * np.sin(th)
        d_f_y_th += d_f_r * np.cos(th)
        d_f_y_th += d2_f_th * np.cos(th) / r
        d_f_y_th += d_f_th * (-np.sin(th) / r)

        d2_f_y = d_f_y_r * np.sin(th) + d_f_y_th * np.cos(th) / r

        return np.array(((d2_f_x, d2_f_x_y), (d2_f_x_y, d2_f_y)))
