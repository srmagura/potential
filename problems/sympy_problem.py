from sympy import *
import numpy as np
import scipy.special


from domain_util import cart_to_polar

class SympyProblem:
    """
    Use SymPy to symbolically differentiate the source function f.
    Calculates up to 2nd order derivatives of f WRT r and th. Uses
    the chain rule to compute the gradient and Hessian of f from its
    polar derivatives.
    """

    lambdify_modules = ('numpy', {
        'gamma': scipy.special.gamma,
        'besselj': scipy.special.jv
    })

    def __init__(self, **kwargs):
        """
        Requires a keyword argument f which is a SymPy expression
        in k, R, r, and th.
        """
        f = kwargs.pop('f_expr')
        super().__init__(**kwargs)

        r, th = symbols('r th')
        args = r, th

        # lambdify keyword arguments
        self.lkw = {
            'modules': self.lambdify_modules
        }

        self.build_sympy_subs_dict()
        self.f_polar_lambda = lambdify(args, f.subs(self.sympy_subs_dict),
            **self.lkw)

        # If using 2nd order scheme, don't need derivatives of f (TODO?)
        self.do_diff(f)

    def build_sympy_subs_dict(self):
        self.sympy_subs_dict = {
            'k': self.k,
            'a': self.a,
            'nu': self.nu,
        }

    def do_diff(self, f):
        """
        Find polar derivatives symbolically.
        """
        r, th = symbols('r th')
        args = r, th

        subs_dict = self.sympy_subs_dict

        d_f_r = diff(f, r)
        self.eval_d_f_r = lambdify(args, d_f_r.subs(subs_dict), **self.lkw)

        d2_f_r = diff(d_f_r, r)
        self.eval_d2_f_r = lambdify(args, d2_f_r.subs(subs_dict), **self.lkw)

        d_f_th = diff(f, th)
        self.eval_d_f_th = lambdify(args, d_f_th.subs(subs_dict), **self.lkw)

        d2_f_th = diff(d_f_th, th)
        self.eval_d2_f_th = lambdify(args, d2_f_th.subs(subs_dict), **self.lkw)

        d2_f_r_th = diff(d_f_r, th)
        self.eval_d2_f_r_th = lambdify(args, d2_f_r_th.subs(subs_dict), **self.lkw)

    def eval_f_polar(self, r, th):
        """
        Evaluate f.

        Uses a Taylor approximation for f at the origin.
        """
        if r == 0:
            h = self.AD_len / 10000

            hessian_f = self.eval_hessian_f(-h, 0)

            grad_f = self.eval_grad_f(-h, 0)
            grad_f += h * hessian_f.dot((1, 0))

            f = self.eval_f(-h, 0)
            f += h * grad_f.dot((1, 0))
            f += 1/2 * h**2 * hessian_f.dot((1, 0)).dot((1, 0))
            return f
        else:
            return self.f_polar_lambda(r, th)

    def eval_grad_f(self, x, y):
        """
        Evaluate gradient of f at (x, y), using the chain rule. Returns the
        result as a NumPy array.
        """
        return self.eval_grad_f_polar(*cart_to_polar(x, y))

    def eval_grad_f_polar(self, r, th):
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
        return self.eval_hessian_f_polar(*cart_to_polar(x, y))

    def eval_hessian_f_polar(self, r, th):
        # Get derivatives of f WRT r and th
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
